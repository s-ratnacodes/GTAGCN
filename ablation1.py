#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install torch_geometric')
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, InstanceNorm1d, LayerNorm, ReLU, Sequential
from typing import List, Optional
import random


# =============================================
# Device Setup + Seed
# =============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================
# Helper Function
# =============================================
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


# =============================================
# MLP
# =============================================
class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None, bias: bool = True, dropout: float = 0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))
            if i < len(channels) - 1:
                if norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                m.append(ReLU())
                m.append(Dropout(dropout))
        super().__init__(*m)


# =============================================
# TAGConv1
# =============================================
class TAGConv1(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 6,
        bias: bool = True,
        normalize: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        norm: str = 'batch'
        num_layers: int = 2
        expansion: int = 2
        eps: float = 1e-7
        self.normalize = normalize

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        channels = [out_channels]
        for _ in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)

        self.mlp = MLP(channels, norm=norm, bias=bias)
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_weight = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    improved=False,
                    add_self_loops=False,
                    flow=self.flow,
                    dtype=x.dtype
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    add_self_loops=False,
                    flow=self.flow,
                    dtype=x.dtype
                )

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            out = out + lin(x)

        if self.bias is not None:
            out = out + self.bias
        return self.mlp(out)

    def message(self, x_j, edge_weight):
        msg = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return msg.relu() + self.eps

    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, K={self.K})'


# =============================================
# GENTAG Model
# =============================================
class GENTAG(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, K=6):
        super().__init__()
        self.conv1 = TAGConv1(num_features, hidden_channels, K=K)
        self.conv2 = TAGConv1(hidden_channels, num_classes, K=K)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# =============================================
# Train Function
# =============================================
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

    return loss.item(), train_acc


# =============================================
# Test Function
# =============================================
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


# =============================================
# ABLATION STUDY: K = 1, 3, 5, 6, 7
# =============================================
root = os.getcwd()
num_runs = 10
num_epochs = 100
K_values = [1, 3, 5, 6, 7]
k_results = {}

for K in K_values:
    print(f"\n{'='*60}")
    print(f"  Testing K = {K}")
    print(f"{'='*60}")

    best_accuracies = []

    for run in range(num_runs):
        dataset = WebKB(root, name="Cornell", transform=NormalizeFeatures())
        data = dataset[0]
        splitter = RandomNodeSplit(split="train_rest", num_val=0, num_test=0.2)
        data = splitter(data)

        data = data.to(device)

        torch.manual_seed(run)
        model = GENTAG(
            hidden_channels=16,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            K=K
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(1, num_epochs + 1):
            loss, train_acc = train(model, data, optimizer, criterion)
            test_acc = test(model, data)
            if test_acc > best_acc:
                best_acc = test_acc
            print(
                f'K={K} | Run {run+1:02d} | Epoch: {epoch:03d} | '
                f'Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | '
                f'Test Acc: {test_acc*100:.2f}%'
            )

        best_accuracies.append(best_acc)
        print(f'>>> K={K} | Run {run+1:02d} | Best Acc: {best_acc*100:.2f}%')

    mean_acc = np.mean(best_accuracies)
    std_acc = np.std(best_accuracies)
    k_results[K] = {'mean': mean_acc, 'std': std_acc}
    print(f'\n  K={K} Final: {mean_acc*100:.2f} ± {std_acc*100:.2f}%')


# =============================================
# Final Ablation Table
# =============================================
best_k = max(k_results, key=lambda k: k_results[k]['mean'])

print(f"\n{'='*60}")
print(f"  ABLATION STUDY RESULTS")
print(f"  K Values Tested: {K_values}")
print(f"  Runs per K: {num_runs} | Epochs: {num_epochs}")
print(f"{'='*60}")
print(f"  {'K':<10} {'Mean Acc':>10}   {'Std':>8}   {'Report As'}")
print(f"  {'-'*50}")
for K, result in k_results.items():
    mean = result['mean'] * 100
    std = result['std'] * 100
    marker = '   <<<  BEST' if K == best_k else ''
    print(f"  K={K:<8} {mean:>10.2f}%   {std:>6.2f}%   {mean:.2f} ± {std:.2f}%{marker}")
print(f"{'='*60}")
print(f"  Best K = {best_k} with {k_results[best_k]['mean']*100:.2f} ± {k_results[best_k]['std']*100:.2f}%")
print(f"{'='*60}")


# In[2]:


get_ipython().system(' pip install torch_geometric')
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, InstanceNorm1d, LayerNorm, ReLU, Sequential
from typing import List, Optional
import random


# =============================================
# Device Setup + Seed
# =============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================
# Helper Function
# =============================================
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


# =============================================
# MLP
# =============================================
class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None, bias: bool = True, dropout: float = 0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))
            if i < len(channels) - 1:
                if norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                m.append(ReLU())
                m.append(Dropout(dropout))
        super().__init__(*m)


# =============================================
# TAGConv1
# =============================================
class TAGConv1(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 6,
        bias: bool = True,
        normalize: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        norm: str = 'batch'
        num_layers: int = 2
        expansion: int = 2
        eps: float = 1e-7
        self.normalize = normalize

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        channels = [out_channels]
        for _ in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)

        self.mlp = MLP(channels, norm=norm, bias=bias)
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_weight = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    improved=False,
                    add_self_loops=False,
                    flow=self.flow,
                    dtype=x.dtype
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    add_self_loops=False,
                    flow=self.flow,
                    dtype=x.dtype
                )

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            out = out + lin(x)

        if self.bias is not None:
            out = out + self.bias
        return self.mlp(out)

    def message(self, x_j, edge_weight):
        msg = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return msg.relu() + self.eps

    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, K={self.K})'


# =============================================
# GENTAG Model
# =============================================
class GENTAG(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, K=6):
        super().__init__()
        self.conv1 = TAGConv1(num_features, hidden_channels, K=K)
        self.conv2 = TAGConv1(hidden_channels, num_classes, K=K)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# =============================================
# Train Function
# =============================================
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

    return loss.item(), train_acc


# =============================================
# Test Function
# =============================================
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


# =============================================
# ABLATION STUDY: K = 1, 3, 5, 6, 7
# =============================================
root = os.getcwd()
num_runs = 10
num_epochs = 100
K_values = [1, 3, 5, 6, 7]
k_results = {}

for K in K_values:
    print(f"\n{'='*60}")
    print(f"  Testing K = {K}")
    print(f"{'='*60}")

    best_accuracies = []

    for run in range(num_runs):
        dataset = WebKB(root, name="Wisconsin", transform=NormalizeFeatures())
        data = dataset[0]
        splitter = RandomNodeSplit(split="train_rest", num_val=0, num_test=0.2)
        data = splitter(data)

        data = data.to(device)

        torch.manual_seed(run)
        model = GENTAG(
            hidden_channels=16,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            K=K
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(1, num_epochs + 1):
            loss, train_acc = train(model, data, optimizer, criterion)
            test_acc = test(model, data)
            if test_acc > best_acc:
                best_acc = test_acc
            print(
                f'K={K} | Run {run+1:02d} | Epoch: {epoch:03d} | '
                f'Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | '
                f'Test Acc: {test_acc*100:.2f}%'
            )

        best_accuracies.append(best_acc)
        print(f'>>> K={K} | Run {run+1:02d} | Best Acc: {best_acc*100:.2f}%')

    mean_acc = np.mean(best_accuracies)
    std_acc = np.std(best_accuracies)
    k_results[K] = {'mean': mean_acc, 'std': std_acc}
    print(f'\n  K={K} Final: {mean_acc*100:.2f} ± {std_acc*100:.2f}%')


# =============================================
# Final Ablation Table
# =============================================
best_k = max(k_results, key=lambda k: k_results[k]['mean'])

print(f"\n{'='*60}")
print(f"  ABLATION STUDY RESULTS")
print(f"  K Values Tested: {K_values}")
print(f"  Runs per K: {num_runs} | Epochs: {num_epochs}")
print(f"{'='*60}")
print(f"  {'K':<10} {'Mean Acc':>10}   {'Std':>8}   {'Report As'}")
print(f"  {'-'*50}")
for K, result in k_results.items():
    mean = result['mean'] * 100
    std = result['std'] * 100
    marker = '   <<<  BEST' if K == best_k else ''
    print(f"  K={K:<8} {mean:>10.2f}%   {std:>6.2f}%   {mean:.2f} ± {std:.2f}%{marker}")
print(f"{'='*60}")
print(f"  Best K = {best_k} with {k_results[best_k]['mean']*100:.2f} ± {k_results[best_k]['std']*100:.2f}%")
print(f"{'='*60}")


# In[3]:


get_ipython().system(' pip install torch_geometric')
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, InstanceNorm1d, LayerNorm, ReLU, Sequential
from typing import List, Optional
import random


# =============================================
# Device Setup + Seed
# =============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================
# Helper Function
# =============================================
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


# =============================================
# MLP
# =============================================
class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None, bias: bool = True, dropout: float = 0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))
            if i < len(channels) - 1:
                if norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                m.append(ReLU())
                m.append(Dropout(dropout))
        super().__init__(*m)


# =============================================
# TAGConv1
# =============================================
class TAGConv1(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 6,
        bias: bool = True,
        normalize: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        norm: str = 'batch'
        num_layers: int = 2
        expansion: int = 2
        eps: float = 1e-7
        self.normalize = normalize

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        channels = [out_channels]
        for _ in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)

        self.mlp = MLP(channels, norm=norm, bias=bias)
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_weight = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    improved=False,
                    add_self_loops=False,
                    flow=self.flow,
                    dtype=x.dtype
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    add_self_loops=False,
                    flow=self.flow,
                    dtype=x.dtype
                )

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            out = out + lin(x)

        if self.bias is not None:
            out = out + self.bias
        return self.mlp(out)

    def message(self, x_j, edge_weight):
        msg = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return msg.relu() + self.eps

    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, K={self.K})'


# =============================================
# GENTAG Model
# =============================================
class GENTAG(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, K=6):
        super().__init__()
        self.conv1 = TAGConv1(num_features, hidden_channels, K=K)
        self.conv2 = TAGConv1(hidden_channels, num_classes, K=K)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# =============================================
# Train Function
# =============================================
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

    return loss.item(), train_acc


# =============================================
# Test Function
# =============================================
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


# =============================================
# ABLATION STUDY: K = 1, 3, 5, 6, 7
# =============================================
root = os.getcwd()
num_runs = 10
num_epochs = 100
K_values = [1, 3, 5, 6, 7]
k_results = {}

for K in K_values:
    print(f"\n{'='*60}")
    print(f"  Testing K = {K}")
    print(f"{'='*60}")

    best_accuracies = []

    for run in range(num_runs):
        dataset = WebKB(root, name="texas", transform=NormalizeFeatures())
        data = dataset[0]
        splitter = RandomNodeSplit(split="train_rest", num_val=0, num_test=0.2)
        data = splitter(data)

        data = data.to(device)

        torch.manual_seed(run)
        model = GENTAG(
            hidden_channels=16,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            K=K
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(1, num_epochs + 1):
            loss, train_acc = train(model, data, optimizer, criterion)
            test_acc = test(model, data)
            if test_acc > best_acc:
                best_acc = test_acc
            print(
                f'K={K} | Run {run+1:02d} | Epoch: {epoch:03d} | '
                f'Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | '
                f'Test Acc: {test_acc*100:.2f}%'
            )

        best_accuracies.append(best_acc)
        print(f'>>> K={K} | Run {run+1:02d} | Best Acc: {best_acc*100:.2f}%')

    mean_acc = np.mean(best_accuracies)
    std_acc = np.std(best_accuracies)
    k_results[K] = {'mean': mean_acc, 'std': std_acc}
    print(f'\n  K={K} Final: {mean_acc*100:.2f} ± {std_acc*100:.2f}%')


# =============================================
# Final Ablation Table
# =============================================
best_k = max(k_results, key=lambda k: k_results[k]['mean'])

print(f"\n{'='*60}")
print(f"  ABLATION STUDY RESULTS")
print(f"  K Values Tested: {K_values}")
print(f"  Runs per K: {num_runs} | Epochs: {num_epochs}")
print(f"{'='*60}")
print(f"  {'K':<10} {'Mean Acc':>10}   {'Std':>8}   {'Report As'}")
print(f"  {'-'*50}")
for K, result in k_results.items():
    mean = result['mean'] * 100
    std = result['std'] * 100
    marker = '   <<<  BEST' if K == best_k else ''
    print(f"  K={K:<8} {mean:>10.2f}%   {std:>6.2f}%   {mean:.2f} ± {std:.2f}%{marker}")
print(f"{'='*60}")
print(f"  Best K = {best_k} with {k_results[best_k]['mean']*100:.2f} ± {k_results[best_k]['std']*100:.2f}%")
print(f"{'='*60}")

