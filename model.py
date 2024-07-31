import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch.nn import init
from torch.nn.parameter import Parameter

class GatedSelfAttention(nn.Module):
    # 通过一个可学习的参数控制它的影响。创建一个包含门控机制的自注意力模块

    def __init__(self, embed_size):
        super(GatedSelfAttention, self).__init__()
        self.attention = SelfAttention(embed_size)
        self.gate = nn.Parameter(torch.randn(1))  # 初始化门控参数

    def forward(self, x):
        attention_output = self.attention(x)
        gate_value = torch.sigmoid(self.gate)  # 确保门控值在0和1之间
        return gate_value * attention_output + (1 - gate_value) * x
class SelfAttention(nn.Module):

    def __init__(self, hidden_dim):

        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # 定义权重矩阵，用于计算查询（Q）、键（K）和值（V）
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, W):
        Q = self.query(W)  # [T, hidden_dim]
        K = self.key(W)    # [T, hidden_dim]
        V = self.value(W)  # [T, hidden_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)

        # 应用注意力权重
        weighted_W = torch.matmul(attention_weights, V)

        return weighted_W


class MatGRUCell(torch.nn.Module):

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats, out_feats, torch.nn.Sigmoid())

        self.reset = MatGRUGate(in_feats, out_feats, torch.nn.Sigmoid())

        self.htilda = MatGRUGate(in_feats, out_feats, torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class MatGRUGate(torch.nn.Module):

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.Tensor(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(
            self.W.matmul(x) + self.U.matmul(hidden) + self.bias
        )

        return out


class TopK(torch.nn.Module):

    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_parameters()

        self.k = k

    def reset_parameters(self):
        init.xavier_uniform_(self.scorer)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm().clamp(
            min=1e-6
        )
        vals, topk_indices = scores.view(-1).topk(self.k)
        out = node_embs[topk_indices] * torch.tanh(
            scores[topk_indices].view(-1, 1)
        )

        return out.t()



class TGSAN(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        num_layers=2,
    ):

        super(TGSAN, self).__init__()
        self.num_layers = num_layers
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        self.self_attention = GatedSelfAttention(n_hidden)
        self.recurrent_layers.append(
            MatGRUCell(in_feats=in_feats, out_feats=n_hidden)
        )
        self.gcn_weights_list.append(
            Parameter(torch.Tensor(in_feats, n_hidden))
        )
        self.gnn_convs.append(
            GraphConv(
                in_feats=in_feats,
                out_feats=n_hidden,
                bias=False,
                activation=nn.RReLU(),
                weight=False,
            )
        )
        for _ in range(num_layers - 1):
            self.recurrent_layers.append(
                MatGRUCell(in_feats=n_hidden, out_feats=n_hidden)
            )
            self.gcn_weights_list.append(
                Parameter(torch.Tensor(n_hidden, n_hidden))
            )
            self.gnn_convs.append(
                GraphConv(
                    in_feats=n_hidden,
                    out_feats=n_hidden,
                    bias=False,
                    activation=nn.RReLU(),
                    weight=False,
                )
            )


        self.reset_parameters()

    def reset_parameters(self):
        for gcn_weight in self.gcn_weights_list:
            init.xavier_uniform_(gcn_weight)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata["feat"])
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]

            Ws = []
            for j, g in enumerate(g_list):
                W = self.recurrent_layers[i](W)
                Ws.append(W)

            Ws = torch.stack(Ws, dim=0)
            Ws_attended = self.self_attention(Ws)

            for j, g in enumerate(g_list):
                W = self.recurrent_layers[i](W)
                feature_list[j] = self.gnn_convs[i](
                    g, feature_list[j], weight=Ws_attended[j]
                )
        return feature_list[-1]
class DATSAN(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim, fcn_dim, num_classes, device):
        super(DATSAN, self).__init__()
        self.device =device
        self.fcn_dim= fcn_dim
        self.name = 'DATSAN'


        self.GNN_a = TGSAN(input_dim,hidden_dim)

        self.GNN_b = TGSAN(input_dim,hidden_dim)



        self.fc1_a = nn.Linear(hidden_dim*2, fcn_dim)
        self.fc2_a = nn.Linear(fcn_dim, num_classes)
        
        self.fc1_b = nn.Linear(hidden_dim*2, fcn_dim)
        self.fc2_b = nn.Linear(fcn_dim, num_classes)

    def forward(self, g,label_mask, permute=True):
        h_a = self.GNN_a(g)
        h_b = self.GNN_b(g)
        g = g[-1]
        x = g.ndata["feat"]
        label = g.ndata["label"]
        edge_index_tuple = g.edges()
        edge_index = (edge_index_tuple[0].t().contiguous(), edge_index_tuple[1].t().contiguous())
        data = Data(x=x, edge_index=edge_index, y=label)

        data.label_mask = label_mask
        data.train_anm = (label==1)
        data.train_norm = (label==0)

        length = len(label)
        data.train_mask = torch.ones(length, dtype=torch.bool)
        data.val_mask = torch.ones(length, dtype=torch.bool)
        data.test_mask = torch.ones(length, dtype=torch.bool)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.to(device)
        h_back_a = torch.cat((h_a, h_b.detach()), dim=1)
        h_back_b = torch.cat((h_a.detach(), h_b), dim=1)
        
        h_aug_back_a, h_aug_back_b, data = self.permute_operation(data, h_b, h_a, permute)
        
        h_back_a = F.relu(h_back_a)
        h_back_b = F.relu(h_back_b)
        
        h_aug_back_a = F.relu(h_aug_back_a)
        h_aug_back_b = F.relu(h_aug_back_b)
        
        h_back_a = self.fc1_a(h_back_a)
        h_back_a = h_back_a.relu()
        h_back_b = self.fc1_b(h_back_b)
        h_back_b = h_back_b.relu()
        
        h_aug_back_a = self.fc1_a(h_aug_back_a)
        h_aug_back_a = h_aug_back_a.relu()
        h_aug_back_b = self.fc1_b(h_aug_back_b)
        h_aug_back_b = h_aug_back_b.relu()
        
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        pred_org_back_b = F.log_softmax(self.fc2_b(h_back_b), dim=1)
        
        pred_aug_back_a = F.log_softmax(self.fc2_a(h_aug_back_a), dim=1)
        pred_aug_bcak_b = F.log_softmax(self.fc2_b(h_aug_back_b), dim=1)
        
        return pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_bcak_b, data
    
    def permute_operation(self, data, h_b, h_a, permute=True):

        if permute:
            self.indices = np.random.permutation(h_b.shape[0])
        
        indices = self.indices
        h_b_swap = h_b[indices]
        label_swap = data.y[indices]
        data.aug_y = label_swap

        data.aug_train_mask = data.train_mask[indices]
        data.aug_val_mask = data.val_mask[indices]
        data.aug_test_mask = data.test_mask[indices]

        data.aug_train_anm = torch.clone(data.aug_train_mask).detach()
        data.aug_train_norm = torch.clone(data.aug_train_mask).detach()

        temp = data.aug_y == 1
        temp1 = data.aug_train_mask == True
        data.aug_train_anm = torch.logical_and(temp, temp1)

        temp = data.aug_y == 0
        temp1 = data.aug_train_mask == True
        data.aug_train_norm = torch.logical_and(temp, temp1)

        h_aug_back_a = torch.cat((h_a, h_b_swap.detach()), dim=1)
        h_aug_back_b = torch.cat((h_a.detach(), h_b_swap), dim=1)
        
        return h_aug_back_a, h_aug_back_b, data


