import json

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import hues
import networkx as nx



class MemoryRead(nn.Module):
    def __init__(self, query_embed, memory_embed, att_proj):
        super(MemoryRead, self).__init__()
        self.W_att = nn.Linear(query_embed, att_proj)
        self.U_att = nn.Linear(memory_embed, att_proj)
        self.w_att = nn.Linear(att_proj, 1)
        self.W_cat = nn.Linear(query_embed+memory_embed,query_embed)

    def forward(self, query_graph, memory_graph_list):
        # 计算上下文信息 c
        context_vectors = self.cal_attention_from_memory_graph(query_graph, memory_graph_list)  # size=(num_querys, memory_embed)
        # 更新状态信息 h
        concat = torch.cat((context_vectors, query_graph.ndata['h']), dim=1)
        next_query_vectors = F.relu(self.W_cat(concat))  # size=(num_querys, embed_length)
        query_graph.ndata['h'] = next_query_vectors
        return query_graph

    def cal_attention_from_memory_graph(self,query_graph, memory_graph_list):
        query_features = query_graph.ndata['h']  # size=(num_querys, query_embed)  
        query_features = self.W_att(query_features)  # size=(num_querys,att_proj)  W*h
        memory_features_list = [g.ndata['h'] for g in memory_graph_list]  
        memory_features = torch.stack(memory_features_list)
        memory_features_att = self.U_att(memory_features)  # size=(num_querys, num_mem_graph_nodes, att_proj)
        query_features = query_features.unsqueeze(1).repeat(1, memory_graph_list[0].number_of_nodes(), 1)  # shape(num_querys, num_mem_graph_nodes, att_proj)
        att_values = self.w_att(torch.tanh(query_features+memory_features_att)).squeeze()   # size=(num_querys, num_mem_graph_nodes)
        att_values = F.softmax(att_values, dim=-1)
        att_values = att_values.unsqueeze(-1).repeat(1, 1, memory_features.shape[-1])
        context_features = (att_values*memory_features).sum(1)    # size=(num_querys, memory_embed)

        return context_features


class MemoryWrite(nn.Module):
    def __init__(self, memory_size,query_size, relation_size, hidden_size):
        super(MemoryWrite, self).__init__()
        self.W_msg = nn.Linear(memory_size+relation_size, hidden_size)
        self.W_mem = nn.Linear(memory_size, hidden_size)
        self.W_query = nn.Linear(query_size, hidden_size)
        self.W_all = nn.Linear(3*hidden_size, memory_size)

    def forward(self, query_graph, memory_graph_list):
        for i in range(len(memory_graph_list)):  
            query_feature = query_graph.ndata['h'][i]  
            memory_graph = memory_graph_list[i]
            num_memory_graph_nodes = memory_graph.number_of_nodes()
            query_features = query_feature.unsqueeze(0).repeat(num_memory_graph_nodes, 1)
            memory_graph.ndata['q'] = query_features

        memory_graph_batch = dgl.batch(memory_graph_list)
        memory_graph_batch.update_all(message_func=self.message, reduce_func=self.reduce)
        return dgl.unbatch(memory_graph_batch)

    def message(self, edges):
        msg = self.W_msg(torch.cat((edges.src['h'], edges.data['rel']), dim=1))
        return {'msg': msg}

    def reduce(self, nodes):
        neibor_msg = torch.sum(nodes.mailbox['msg'], dim=1)
        new_memory = torch.cat((self.W_query(nodes.data['q']), self.W_mem(nodes.data['h']), neibor_msg), dim=1)
        new_memory = F.relu(self.W_all(new_memory))
        return {'h': new_memory}


class MemoryNetwork(nn.Module):

    
    def __init__(self, query_input_size, memory_size, que_szie, query_hidden_size, memory_relation_size, memory_hidden_size, mem_read_att_proj, T):
        super(MemoryNetwork, self).__init__()
        self.W_query = nn.Linear(query_input_size + que_szie, query_hidden_size)
        self.read_memory = MemoryRead(query_embed=query_hidden_size, memory_embed=memory_size, att_proj=mem_read_att_proj)
        self.write_memory = MemoryWrite(memory_size=memory_size, query_size=query_hidden_size,relation_size=memory_relation_size, hidden_size=memory_hidden_size)
        self.T=T

    
    def forward(self, query_graph, memory_graph, question):
        num_querys = query_graph.number_of_nodes()  
        memory_graph_list = [memory_graph] * num_querys  

        question = question.unsqueeze(0).repeat(num_querys, 1)
        query_feature = torch.cat((query_graph.ndata['h'], question),dim=1)  
        query_feature = F.relu(self.W_query(query_feature)) 
        query_graph.ndata['h'] = query_feature 

        for t in range(self.T): 
            query_graph = self.read_memory(query_graph, memory_graph_list) 
                                                                           
            memory_graph_list = self.write_memory(query_graph, memory_graph_list)  
        return query_graph


class MemoryGate(nn.Module):
    def __init__(self, vis_mem_size, sem_mem_size, node_size, out_size):
        super(MemoryGate, self).__init__()
        self.gate_layer = nn.Linear(vis_mem_size + sem_mem_size + node_size, vis_mem_size + sem_mem_size + node_size)
        self.fuse_layer = nn.Linear(vis_mem_size+sem_mem_size+node_size,out_size)

    def forward(self, fact_graph_batch):
        node_feature = fact_graph_batch.ndata['hh']  
        vis_memory = fact_graph_batch.ndata['vis_mem'] 
        sem_memory = fact_graph_batch.ndata['sem_mem']
        cat = torch.cat((node_feature, vis_memory, sem_memory), dim=1)
        # cat.shape = [784, 900]
        cat = self.gate_layer(cat)
        gate = torch.sigmoid(cat)
        fuse = self.fuse_layer(gate * cat)
        fact_graph_batch.ndata['h'] = fuse
        return fact_graph_batch
