import torch
import torch.nn as nn
import dgl
import hues
import yaml
import json
import os
import numpy as np
import torch.nn.functional as F

from model.memory_network import MemoryNetwork, MemoryGate
from model.img_gcn import ImageGCN
from model.semantic_gcn import SemanticGCN
from model.fact_gcn import FactGCN
from model.global_gcn import GlobalGCN
from util.dynamic_rnn import DynamicRNN
from util.vocabulary import Vocabulary
from util.myFun import collate_fn
from data.okvqa_traindataset import OkvqaTrainDataset
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

'''
获得三种记忆图（多模态异构体）
'''
class MemoryExtractionModule(nn.Module):
    def __init__(self, config, que_vocabulary, glove, device='cpu'):
        super(MemoryExtractionModule, self).__init__()
        self.config = config
        self.device = device
        # question embedding
        self.que_glove_embed = nn.Embedding(len(que_vocabulary), config['model']['glove_embedding_size'])

        self.que_glove_embed.weight.data = glove
        self.que_glove_embed.weight.requires_grad = False

        self.ques_rnn = nn.LSTM(config['model']['glove_embedding_size'],
                                config['model']['lstm_hidden_size'],
                                config['model']['lstm_num_layers'],
                                batch_first=True,
                                dropout=config['model']['dropout'])
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # question guided visual node attention
        self.vis_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_proj_img = nn.Linear(
            config['model']['img_feature_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_value = nn.Linear(
            config['model']['node_att_ques_img_proj_dims'], 1)

        # question guided visual relation attention
        self.vis_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_proj_rel = nn.Linear(
            config['model']['vis_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)

        # question guided semantic node attention
        self.sem_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_proj_sem = nn.Linear(
            config['model']['sem_node_dims'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_value = nn.Linear(
            config['model']['sem_node_att_ques_img_proj_dims'], 1)

        # question guided semantic relation attention
        self.sem_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_proj_rel = nn.Linear(
            config['model']['sem_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)

        # question guided fact node attention
        self.fact_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_proj_node = nn.Linear(
            config['model']['fact_node_dims'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_value = nn.Linear(
            config['model']['fact_node_att_ques_node_proj_dims'], 1)

        # image gcn1
        self.img_gcn1 = ImageGCN(config,
                                 in_dim=config['model']['img_feature_size'],
                                 out_dim=config['model']['image_gcn1_out_dim'],
                                 rel_dim=config['model']['vis_relation_dims'])

        # semantic gcn1
        self.sem_gcn1 = SemanticGCN(config,
                                    in_dim=config['model']['sem_node_dims'],
                                    out_dim=config['model']['semantic_gcn1_out_dim'],
                                    rel_dim=config['model']['sem_relation_dims'])
        # fact gcn1
        self.fact_gcn1 = FactGCN(config,
                                 in_dim=config['model']['fact_node_dims'],
                                 out_dim=config['model']['fact_gcn1_out_dim'])

        self.visual_memory_network = MemoryNetwork(query_input_size=config['model']['fact_gcn1_out_dim'],
                                                   memory_size=config['model']['image_gcn1_out_dim'],
                                                   que_szie=config['model']['lstm_hidden_size'],
                                                   query_hidden_size=config['model']['visual_memory_query_hidden_size'],
                                                   memory_relation_size=config['model']['vis_relation_dims'],
                                                   memory_hidden_size=config['model'][
                                                       'visual_memory_memory_hidden_size'],
                                                   mem_read_att_proj=config['model'][
                                                       'visual_memory_memory_read_att_size'],
                                                   T=config['model']['memory_step'])

        self.semantic_memory_network = MemoryNetwork(query_input_size=config['model']['fact_gcn1_out_dim'],
                                                     memory_size=config['model']['semantic_gcn1_out_dim'],
                                                     que_szie=config['model']['lstm_hidden_size'],
                                                     query_hidden_size=config['model'][
                                                         'semantic_memory_query_hidden_size'],
                                                     memory_relation_size=config['model']['sem_relation_dims'],
                                                     memory_hidden_size=config['model'][
                                                         'semantic_memory_memory_hidden_size'],
                                                     mem_read_att_proj=config['model'][
                                                         'semantic_memory_memory_read_att_size'],
                                                     T=config['model']['memory_step'])

        self.memory_gate = MemoryGate(vis_mem_size=config['model']['visual_memory_query_hidden_size'],
                                      sem_mem_size=config['model']['semantic_memory_query_hidden_size'],
                                      node_size=config['model']['fact_gcn1_out_dim'],
                                      out_size=config['model']['memory_gate_out_dim'])

        self.global_gcn = GlobalGCN(config, in_dim=512, out_dim=512)


    def forward(self, batch):
        batch_size = len(batch['id_list'])

        # image
        images = batch['features_list']  # [(36,2048)]
        images = torch.stack(images).to(self.device)  # (batch,36,2048)

        img_relations = batch['img_relations_list']
        img_relations = torch.stack(img_relations).to(self.device)  # (batch,36,36,7)

        # question
        questions = batch['question_list']  # [(max_length,)]
        questions = torch.stack(questions).to(self.device)  # (batch,max_length)

        questions_len_list = batch['question_length_list']
        questions_len_list = torch.Tensor(questions_len_list).long().to(self.device)

        # semantic graph
        semantic_num_nodes_list = torch.Tensor(batch['semantic_num_nodes_list']).long()

        semantic_n_features_list = batch['semantic_node_features_list']
        semantic_n_features_list = [features.to(self.device) for features in semantic_n_features_list]

        semantic_e1ids_list = batch['semantic_e1ids_list']
        semantic_e1ids_list = [e1ids.to(self.device) for e1ids in semantic_e1ids_list]

        semantic_e2ids_list = batch['semantic_e2ids_list']
        semantic_e2ids_list = [e2ids.to(self.device) for e2ids in semantic_e2ids_list]

        semantic_e_features_list = batch['semantic_edge_features_list']
        semantic_e_features_list = [features.to(self.device) for features in semantic_e_features_list]

        # fact graph
        fact_num_nodes_list = torch.Tensor(batch['facts_num_nodes_list']).long()

        facts_features_list = batch['facts_node_features_list']
        facts_features_list = [features.to(self.device) for features in facts_features_list]

        facts_e1ids_list = batch['facts_e1ids_list']
        facts_e1ids_list = [e1ids.to(self.device) for e1ids in facts_e1ids_list]

        facts_e2ids_list = batch['facts_e2ids_list']
        facts_e2ids_list = [e2ids.to(self.device) for e2ids in facts_e2ids_list]

        facts_answer_list = batch['facts_answer_list']
        facts_answer_list = [answer.to(self.device) for answer in facts_answer_list]

        # ===============================================================================================================
        #                               1. embed questions
        # ===============================================================================================================
        ques_embed = self.que_glove_embed(questions).float()  # shape (batch,question_max_length,300)

        _, (ques_embed, _) = self.ques_rnn(ques_embed, questions_len_list)  # qes_embed shape=(batch,hidden_size)

        # ===============================================================================================================
        #                               2. question guided visual node attention
        # ===============================================================================================================
        # ques_embed : 20:48:45 - INFO - torch.Size([11, 512])
        # images : 20:48:45 - INFO - torch.Size([11, 36, 2048])
        node_att_proj_ques_embed = self.vis_node_att_proj_ques(ques_embed)  # shape (batch,proj_size)
        node_att_proj_img_embed = self.vis_node_att_proj_img(images)  # shape (batch,36,proj_size)
        node_att_proj_ques_embed = node_att_proj_ques_embed.unsqueeze(1).repeat(1, images.shape[1],
                                                                                1)  # shape(batch,36,proj_size)
        node_att_proj_img_sum_ques = torch.tanh(node_att_proj_ques_embed + node_att_proj_img_embed)
        vis_node_att_values = self.vis_node_att_value(node_att_proj_img_sum_ques).squeeze()  # shape(batch,36)
        vis_node_att_values = F.softmax(vis_node_att_values, dim=-1)  # shape(batch,36)

        # ===============================================================================================================
        #                                3. question guided visual relation attention
        # ===============================================================================================================
        rel_att_proj_ques_embed = self.vis_rel_att_proj_ques(ques_embed)  # shape(batch,128)
        rel_att_proj_rel_embed = self.vis_rel_att_proj_rel(img_relations)  # shape(batch,36,36,128)
        rel_att_proj_ques_embed = rel_att_proj_ques_embed.repeat(
            1, 36 * 36).view(
            batch_size, 36, 36, self.config['model']
            ['rel_att_ques_rel_proj_dims'])  # shape(batch,36,36,128)
        rel_att_proj_rel_sum_ques = torch.tanh(rel_att_proj_ques_embed +
                                               rel_att_proj_rel_embed)
        vis_rel_att_values = self.vis_rel_att_value(rel_att_proj_rel_sum_ques).squeeze()  # shape(batch,36,36)

        sem_node_att_val_list = []
        sem_edge_att_val_list = []
        for i in range(batch_size):
            # ===============================================================================================================
            #                                4 question guided semantic node attention
            # ===============================================================================================================
            num_node = semantic_num_nodes_list[i]  # n
            sem_node_features = semantic_n_features_list[i]  # (n,300)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            sem_node_att_proj_ques_embed = self.sem_node_att_proj_ques(q_embed)  # shape (n,p)
            sem_node_att_proj_sem_embed = self.sem_node_att_proj_sem(sem_node_features)  # shape (n,p)
            sem_node_att_proj_sem_sum_ques = torch.tanh(
                sem_node_att_proj_ques_embed + sem_node_att_proj_sem_embed)  # shape (n,p)
            sem_node_att_values = self.sem_node_att_value(sem_node_att_proj_sem_sum_ques)  # shape(n,1)
            sem_node_att_values = F.softmax(sem_node_att_values, dim=0)  # shape(n,1)

            sem_node_att_val_list.append(sem_node_att_values)

            # ===============================================================================================================
            #                                5 question guided semantic relation attention
            # ===============================================================================================================
            num_edge = semantic_e_features_list[i].shape[0]  # n
            sem_edge_features = semantic_e_features_list[i]  # (n,300)
            qq_embed = ques_embed[i]  # (512)
            qq_embed = qq_embed.unsqueeze(0).repeat(num_edge, 1)  # (n,512)
            sem_rel_att_proj_ques_embed = self.sem_rel_att_proj_ques(qq_embed)  # shape (n,p)
            sem_rel_att_proj_rel_embed = self.sem_rel_att_proj_rel(sem_edge_features)  # shape (n,p)
            sem_rel_att_proj_rel_sum_ques = torch.tanh(
                sem_rel_att_proj_ques_embed + sem_rel_att_proj_rel_embed)  # shape (n,p)
            sem_rel_att_values = self.sem_rel_att_value(sem_rel_att_proj_rel_sum_ques)  # shape(n,1)
            sem_rel_att_values = F.softmax(sem_rel_att_values, dim=0)  # shape(n,1)

            sem_edge_att_val_list.append(sem_rel_att_values)

        # ===============================================================================================================
        #                                6 question guided fact node attention
        # ===============================================================================================================
        fact_node_att_values_list = []
        for i in range(batch_size):
            num_node = fact_num_nodes_list[i]  # n
            fact_node_features = facts_features_list[i]  # (n,1024)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            fact_node_att_proj_ques_embed = self.fact_node_att_proj_ques(q_embed)  # shape (n,p)
            fact_node_att_proj_node_embed = self.fact_node_att_proj_node(fact_node_features)  # shape (n,p)
            fact_node_att_proj_node_sum_ques = torch.tanh(
                fact_node_att_proj_ques_embed + fact_node_att_proj_node_embed)  # shape (n,p)
            fact_node_att_values = self.fact_node_att_value(fact_node_att_proj_node_sum_ques)  # shape(n,1)
            fact_node_att_values = F.softmax(fact_node_att_values, dim=0)  # shape(n,1)
            fact_node_att_values_list.append(fact_node_att_values)

        # ===============================================================================================================
        #                             7 Build Image Graph
        # ===============================================================================================================
        # 36 nodes,36*36 edges
        img_graphs = []
        for i in range(batch_size):
            g = dgl.DGLGraph()
            g = g.to(self.device)
            # add nodes
            g.add_nodes(36)
            # add node features
            g.ndata['h'] = images[i]
            g.ndata['att'] = vis_node_att_values[i].unsqueeze(-1)
            g.ndata['batch'] = torch.full([36, 1], i).to(self.device)  # 加入cuda中
            # add edges
            for s in range(36):
                for d in range(36):
                    g.add_edge(s, d)
            # add edge features
            g.edata['rel'] = img_relations[i].view(36 * 36, self.config['model']['vis_relation_dims'])  # shape(36*36,7)
            g.edata['att'] = vis_rel_att_values[i].view(36 * 36, 1)  # shape(36*36,1)
            img_graphs.append(g)
        image_batch_graph = dgl.batch(img_graphs)

        # ===============================================================================================================
        #                                8 Build Semantic Graph
        # ===============================================================================================================
        semantic_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph = graph.to(self.device)
            graph.add_nodes(semantic_num_nodes_list[i])
            graph.add_edges(semantic_e1ids_list[i], semantic_e2ids_list[i])
            graph.ndata['h'] = semantic_n_features_list[i]
            graph.ndata['att'] = sem_node_att_val_list[i]
            graph.edata['rel'] = semantic_e_features_list[i]
            graph.edata['att'] = sem_edge_att_val_list[i]
            semantic_graphs.append(graph)
        semantic_batch_graph = dgl.batch(semantic_graphs)

        # ===============================================================================================================
        #                                9 Build Fact Graph
        # ===============================================================================================================
        fact_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph = graph.to(self.device)
            graph.add_nodes(fact_num_nodes_list[i])
            graph.add_edges(facts_e1ids_list[i], facts_e2ids_list[i])
            graph.ndata['h'] = facts_features_list[i]  # 结点的表示
            graph.ndata['att'] = fact_node_att_values_list[i]
            graph.ndata['batch'] = torch.full([fact_num_nodes_list[i], 1], i).to(self.device)
            graph.ndata['answer'] = facts_answer_list[i]
            fact_graphs.append(graph)
        fact_batch_graph = dgl.batch(fact_graphs)

        # ===============================================================================================================
        #                                8. Intra GCN
        # ===============================================================================================================
    
        image_batch_graph = self.img_gcn1(image_batch_graph)

        semantic_batch_graph = self.sem_gcn1(semantic_batch_graph)

        fact_batch_graph = self.fact_gcn1(fact_batch_graph)
        fact_batch_graph.ndata['hh'] = fact_batch_graph.ndata['h']

        # ===============================================================================================================
        #                                9. Memory network
        # ===============================================================================================================
        image_graph_list = dgl.unbatch(image_batch_graph)
        semantic_graph_list = dgl.unbatch(semantic_batch_graph)
        fact_graph_list = dgl.unbatch(fact_batch_graph)
        new_fact_graph_list = []
        for i, fact_graph in enumerate(fact_graph_list):
            question = ques_embed[i]

            num_fact_nodes = fact_graph.number_of_nodes()
            image_graph = image_graph_list[i]
            semantic_graph = semantic_graph_list[i]

            question = ques_embed[i]
            fact_graph_memory_visual = self.visual_memory_network(fact_graph, image_graph, question)
            fact_graph_memory_semantic = self.semantic_memory_network(fact_graph, semantic_graph, question)
            fact_graph.ndata['vis_mem'] = fact_graph_memory_visual.ndata['h']
            fact_graph.ndata['sem_mem'] = fact_graph_memory_semantic.ndata['h']

            # fact_graph.ndata['hh_cat'] = torch.cat(
            #     (fact_graph.ndata['hh'], question.unsqueeze(0).repeat(fact_graph.ndata['hh'].shape[0], 1)), dim=1)
            new_fact_graph_list.append(fact_graph)

        return new_fact_graph_list, ques_embed

def mem_test():
    config_path = "/home/data/yjgroup/lyl/projects/IIU/config/config_okvqa.yml"
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    glovevocabulary = Vocabulary(config["dataset"]["word_counts_json"],
                                     min_count=config["dataset"]["vocab_min_count"])
    glove = np.load(config['dataset']['glove_vec_path'])
    glove = torch.Tensor(glove)

    model = MemoryExtractionModule(config, glovevocabulary, glove, 'cuda:0').to('cuda:0')
    train_dataset = OkvqaTrainDataset(config, overfit=False)
    train_dataloader = DataLoader(train_dataset,
                                      batch_size=config['solver']['batch_size'],
                                      num_workers=4,
                                      shuffle=True,
                                      collate_fn=collate_fn)
    for i, batch in enumerate(train_dataloader):

        res = model(batch)
        one = res[0]
        hues.info(one.ndata['fact_mem'].shape)
        hues.info(one.ndata['sem_mem'].shape)
        hues.info(one.ndata['vis_mem'].shape)
        one = res[1]
        hues.info(one.ndata['fact_mem'].shape)
        hues.info(one.ndata['sem_mem'].shape)
        hues.info(one.ndata['vis_mem'].shape)
        input('s')
        '''
        fact_mem, sem_mem, vis_mem
        '''