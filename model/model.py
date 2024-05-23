import dgl
import torch
import hues
import torch.nn as nn

from model.RIM import IIUCell
from model.memory_extraction_module import MemoryExtractionModule

class Model(nn.Module):
    def __init__(self, config, que_vocabulary, glove, num_units, k, step, hidden_size,device='cpu'):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.config = config
        self.step = step
        self.num_units = num_units

        self.memory_extraction_module = MemoryExtractionModule(config, que_vocabulary, glove, device)
        self.iiu_cell = IIUCell(device, input_size=300, hidden_size=hidden_size, num_units=num_units, k=k,
                                            rnn_cell="LSTM")
        self.question_update = nn.Sequential(
                nn.Linear(800, 512),
                nn.ReLU(),
                nn.Linear(512, 300),
        )
        self.answer_pred = nn.Sequential(
                nn.Linear(600, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

    def forward(self, batch):
        fact_graph_list, ques_embed = self.memory_extraction_module(batch) # [11, 300]

        ques_embed = ques_embed.unsqueeze(dim=1)
        batch_size = len(fact_graph_list)
        max_node_num = 0
        for g in fact_graph_list:
            if max_node_num < g.ndata['hh'].shape[0]:
                max_node_num = g.ndata['hh'].shape[0]
        x_vis = torch.zeros(batch_size, max_node_num, 300).to(self.device)
        x_sem = torch.zeros(batch_size, max_node_num, 300).to(self.device)
        x_fac = torch.zeros(batch_size, max_node_num, 300).to(self.device)
        for i, g in enumerate(fact_graph_list):
            vis_mem = g.ndata['vis_mem']
            sem_mem = g.ndata['sem_mem']
            fac_nod = g.ndata['hh']
            for j in range(0, fac_nod.shape[0]):
                x_vis[i, j] = vis_mem[j]
                x_sem[i, j] = sem_mem[j]
                x_fac[i, j] = fac_nod[j]
        mems = torch.cat((x_vis, x_sem, x_fac), dim=1).float()  # x.shape = [batch_size, max_node_num*3, 300]
        hs = torch.randn(ques_embed.shape[0], self.num_units, 100).to(self.device)
        cs = torch.randn(ques_embed.shape[0], self.num_units, 100).to(self.device)
        for i in range(self.step):
            hs, cs = self.iiu_cell(ques_embed, mems, hs, cs)
            iiu_h = hs.contiguous().view(ques_embed.size(0), -1)
            ques_embed = ques_embed.squeeze(dim=1)
            ques_embed = self.question_update(torch.cat((ques_embed, iiu_h), dim=1)).unsqueeze(dim=1)
        #iiu_h = hs.contiguous().view(ques_embed.size(0), -1) # [11, 500]

        ques_embed = ques_embed.squeeze(dim=1)
        fact_graph_batch = dgl.batch(fact_graph_list) 
        fact_node_cat_que = torch.zeros(fact_graph_batch.ndata['hh'].shape[0], 300 + 300).to(self.device)
        for i in range(fact_graph_batch.ndata['hh'].shape[0]):
            temp_fact_h = fact_graph_batch.ndata['hh'][i]  
            #temp_batch_rim_h = iiu_h[fact_graph_batch.ndata['batch'][i].item()] 
            temp_question = ques_embed[fact_graph_batch.ndata['batch'][i].item()]  
            cat = torch.cat((temp_fact_h, temp_question))
            fact_node_cat_que[i] = cat
        
        fact_graph_batch.ndata['h'] = self.answer_pred(fact_node_cat_que)

        return fact_graph_batch