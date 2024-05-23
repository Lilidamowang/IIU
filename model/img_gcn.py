import torch
import torch.nn.functional as F
from torch import nn
import dgl
import networkx as nx


class ImageGCN(nn.Module):
    def __init__(self, config, in_dim, out_dim, rel_dim):
        super(ImageGCN, self).__init__()
        self.config = config

        self.gcn1 = ImageGCNLayer(in_dim, out_dim, rel_dim)


    def forward(self, bg):
        bg = self.gcn1(bg)
        return bg


class ImageGCNLayer(nn.Module):
    def __init__(self, in_dims, out_dims, rel_dims):
        super(ImageGCNLayer, self).__init__()
        self.node_fc = nn.Linear(in_dims, in_dims)
        self.rel_fc = nn.Linear(rel_dims, rel_dims)
        self.apply_fc = nn.Linear(in_dims + rel_dims + in_dims, out_dims)

    def forward(self, g):
        g.apply_nodes(func=self.apply_node)
        g.update_all(message_func=self.message, reduce_func=self.reduce)
        return g

    def apply_node(self, nodes):
        h = self.node_fc(nodes.data['h'])
        return {'h': h}

    # 消息传递函数
    def message(self, edges):
        z1 = edges.src['att'] * edges.src['h']
        z2 = edges.data['att'] * self.rel_fc(edges.data['rel'])
        msg = torch.cat([z1, z2], dim=1)
        return {'msg': msg} 

    # 消息聚合函数
    def reduce(self, nodes):
        msg = torch.sum(nodes.mailbox['msg'], dim=1)  # shape(out_dim)
        h = nodes.data['h']  # shape(out_dim)
        h = torch.cat([msg, h], dim=1)  # shape(2*outdim)
        h = nodes.data['att'] * F.relu(self.apply_fc(h))  # shape(outdim)
        return {'h': h}

