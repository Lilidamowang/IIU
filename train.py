import torch
import torch.nn as nn
import dgl
import hues
import yaml
import json
import os
import numpy as np

from tqdm import tqdm
from model.model import Model
from bisect import bisect
from util.dynamic_rnn import DynamicRNN
from util.checkpointing import CheckpointManager, load_checkpoint
from util.vocabulary import Vocabulary
from util.myFun import collate_fn
from data.okvqa_traindataset import OkvqaTrainDataset
from data.okvqa_testdataset import OkvqaTestDataset
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

device = torch.device("cuda:0")
SAVE_PATH = "checkpoint/"
LOAD_PTHPATH = "/home/data/yjgroup/lyl/projects/IIU/checkpoint/checkpoint_6.pth"
CONFIG_PATH = "/home/data/yjgroup/lyl/projects/IIU/config/config_okvqa.yml"
UNIT_NUM = 5
ACTIVE_NUM = 3
STEP = 3
HIDDEN_SIZE=100
config = yaml.load(open(CONFIG_PATH), Loader=yaml.FullLoader)
glovevocabulary = Vocabulary(config["dataset"]["word_counts_json"],
                                    min_count=config["dataset"]["vocab_min_count"])
glove = np.load(config['dataset']['glove_vec_path'])
glove = torch.Tensor(glove)

weight = torch.FloatTensor([0.9, 0.1]).to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=weight)

train_dataset = OkvqaTrainDataset(config, overfit=False)
train_dataloader = DataLoader(train_dataset,
                                    batch_size=config['solver']['batch_size'],
                                    num_workers=4,
                                    shuffle=True,
                                    collate_fn=collate_fn)
val_dataset = OkvqaTestDataset(config, overfit=False, in_memory=True)
val_dataloader = DataLoader(val_dataset,
                            batch_size=config['solver']['batch_size'],
                            num_workers=4,
                            shuffle=True,
                            collate_fn=collate_fn)
model = Model(config, glovevocabulary, glove, UNIT_NUM, ACTIVE_NUM, STEP, HIDDEN_SIZE,device).to(device)

def lr_lambda_fun(current_iteration: int) -> float:
    """Returns a learning rate multiplier.

    Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """
    current_epoch = float(current_iteration) / iterations
    if current_epoch <= config["solver"]["warmup_epochs"]:
        alpha = current_epoch / float(config["solver"]["warmup_epochs"])
        return config["solver"]["warmup_factor"] * (1. - alpha) + alpha
    else:
        idx = bisect(config["solver"]["lr_milestones"], current_epoch)
        return pow(config["solver"]["lr_gamma"], idx)


iterations = len(train_dataloader) // config["solver"]["batch_size"] + 1
optimizer = optim.Adamax(model.parameters(),
                             lr=config["solver"]["initial_lr"])
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
T = iterations * (config["solver"]["num_epochs"] -
                    config["solver"]["warmup_epochs"] + 1)
scheduler2 = lr_scheduler.CosineAnnealingLR(
    optimizer, int(T), eta_min=config["solver"]["eta_min"], last_epoch=-1)

checkpoint_manager = CheckpointManager(model,
                                           optimizer,
                                           SAVE_PATH,
                                           config=config)

def cal_batch_loss(fact_batch_graph, batch, device, loss_fn):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).to(device)

    for i, fact_graph in enumerate(fact_graphs):
        pred = fact_graph.ndata['h']  # (n,2)
        answer = answers[i].long().to(device)
        loss = loss_fn(pred, answer)
        batch_loss = batch_loss + loss

    return batch_loss / len(answers)


def focal_loss(fact_batch_graph, batch, device, alpha=0.5, gamma=2):
    answers = batch['facts_answer_list']
    fact_graphs = dgl.unbatch(fact_batch_graph)
    batch_loss = torch.tensor(0).float().to(device)

    for i, fact_graph in enumerate(fact_graphs):
        pred = fact_graph.ndata['h'].squeeze()
        target = torch.FloatTensor(answers[i]).to(device).squeeze()
        loss = -1 * alpha * ((1 - pred) ** gamma) * target * torch.log(pred) - (1 - alpha) * (target ** gamma) * (
            1 - pred) * torch.log(1 - pred)
        batch_loss = batch_loss+loss.mean()
    return batch_loss/len(answers)


def cal_acc(answers, preds):
    all_num = len(preds)
    acc_num_1 = 0
    

    for i, answer_id in enumerate(answers):
        pred = preds[i]  # (num_nodes)
        try:
            # top@1
            _, idx_1 = torch.topk(pred, k=1)

        except RuntimeError:
            continue
        else:
            if idx_1.item() == answer_id:
                acc_num_1 = acc_num_1 + 1

    return acc_num_1 / all_num

# If loading from checkpoint, adjust start epoch and load parameters.
if LOAD_PTHPATH == "":
    start_epoch = 0
else:
    start_epoch = int(LOAD_PTHPATH.split("_")[-1][:-4])+1

    model_state_dict, optimizer_state_dict = load_checkpoint(
        LOAD_PTHPATH)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    print("Loading resume model from {}...".format(LOAD_PTHPATH))

global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config['solver']['num_epochs']):

    print(f"\nTraining for epoch {epoch}:")

    train_answers = []
    train_preds = []

    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        fact_batch_graph = model(batch)
        batch_loss = cal_batch_loss(fact_batch_graph,
                                    batch,
                                    device,
                                    loss_fn=loss_function)
        
        batch_loss.backward()
        optimizer.step()

        fact_graphs = dgl.unbatch(fact_batch_graph)
        for i, fact_graph in enumerate(fact_graphs):
            train_pred = fact_graph.ndata['h']  # (num_nodes,1)
            train_preds.append(train_pred[:,1])  # [(num_nodes,)]
            train_answers.append(batch['facts_answer_id_list'][i])

        if global_iteration_step <= iterations * config["solver"][
                "warmup_epochs"]:
            scheduler.step(global_iteration_step)
        else:
            global_iteration_step_in_2 = iterations * config["solver"][
                "warmup_epochs"] + 1 - global_iteration_step
            scheduler2.step(int(global_iteration_step_in_2))

        global_iteration_step = global_iteration_step + 1
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # --------------------------------------------------------------------------------------------
    checkpoint_manager.step(epoch)
    train_acc_1 = cal_acc(train_answers, train_preds)
    print("trainacc@1={:.2%} ".format(train_acc_1))



    # Validate and report automatic metrics.
    # if args.validate:
    # model.eval()
    # answers = []  # [batch_answers,...]
    # preds = []  # [batch_preds,...]
    # print(f"\nValidation after epoch {epoch}:")
    # for i, batch in enumerate(tqdm(val_dataloader)):
    #     with torch.no_grad():
    #         fact_batch_graph = model(batch)
    #     batch_loss = cal_batch_loss(fact_batch_graph,
    #                                 batch,
    #                                 device,
    #                                 loss_fn=loss_function)
    #     # batch_loss = focal_loss(fact_batch_graph, batch, device)
    #     fact_graphs = dgl.unbatch(fact_batch_graph)
    #     for i, fact_graph in enumerate(fact_graphs):
    #         pred = fact_graph.ndata['h']  # (num_nodes,1)
    #         preds.append(pred[:,1])  # [(num_nodes,)]
    #         answers.append(batch['facts_answer_id_list'][i])
    # acc_1 = cal_acc(answers, preds)
    # print("acc@1={:.2%}".format(acc_1))
    

    # model.train()
    # torch.cuda.empty_cache()
print('Train finished !!!')