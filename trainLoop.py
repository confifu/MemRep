import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from IPython.display import clear_output
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#============metrics ==================
def MAE(y, ypred) :
    """for period"""
    batch_size = y.shape[0]
    yarr = y.clone().detach().cpu().numpy()
    ypredarr = ypred.clone().detach().cpu().numpy()
    
    ae = np.sum(np.absolute(yarr - ypredarr))
    mae = ae / yarr.flatten().shape[0]
    return mae

def f1score(y, ypred) :
    """for periodicity"""
    batch_size = y.shape[0]
    yarr = y.clone().detach().cpu().numpy()
    ypredarr = ypred.clone().detach().cpu().numpy().astype(bool)
    tp = np.logical_and(yarr, ypredarr).sum()
    precision = tp / (ypredarr.sum() + 1e-6)
    recall = tp / (yarr.sum() + 1e-6)
    if precision + recall == 0:
        fscore = 0
    else :
        fscore = 2*precision*recall/(precision + recall)
    return fscore

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def getPeriodicity(periodLength):
    periodicity = torch.nn.functional.threshold(periodLength, 2, 0)
    periodicity = -torch.nn.functional.threshold(-periodicity, -1, -1)
    return periodicity

def getCount(periodLength):
    frac = 1/periodLength
    frac = torch.nan_to_num(frac, 0, 0, 0)

    count = torch.sum(frac, dim = [1])
    return count

def getStart(periodLength):
    tmp = periodLength.squeeze(2)
    idx = torch.arange(tmp.shape[1], 0, -1)
    tmp2 = tmp * idx
    indices = torch.argmax(tmp2, 1, keepdim=True)
    return indices

def training_loop(n_epochs,
                  model,
                  train_set,
                  val_set,
                  batch_size,
                  num_workers = 4,
                  fpv = 32,
                  lr = 6e-6,
                  ckpt_name = 'ckpt',
                  use_count_error = True,
                  saveCkpt= True,
                  train = True,
                  validate = True,
                  lastCkptPath = None):

    
    
    prevEpoch = 0
    trainLosses = []
    valLosses = []
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
 
    if lastCkptPath != None :
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch']
        trainLosses = checkpoint['trainLosses']
        valLosses = checkpoint['valLosses']

        model.load_state_dict(checkpoint['state_dict'], strict = True)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        del checkpoint
    
        
    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    lossMAE = torch.nn.SmoothL1Loss()
    lossBCE = torch.nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              num_workers=num_workers, 
                              shuffle = True)
    
    val_loader = DataLoader(val_set,
                            batch_size = batch_size,
                            num_workers=num_workers,
                            drop_last = False,
                            shuffle = True)
    
    if validate and not train:
        currEpoch = prevEpoch
    else :
        currEpoch = prevEpoch + 1
        
    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        #train loop
        if train :
            pbar = tqdm(train_loader, total = len(train_loader))
            mae = 0
            mae_count = 0
            i = 1
                
            for X_total, y_total in pbar:
                torch.cuda.empty_cache()

                totalparts = X_total.shape[1]//fpv
                y1 = y_total.to(device).float()
                y2 = getPeriodicity(y1).to(device).float()

                model.init_params(batch_size)
                model.train()
                optimizer.zero_grad()

                y1preds = []
                y2preds = []

                for subidx in range(totalparts):
                    print(subidx)
                    
                    X = X_total[:, subidx*fpv: (subidx+1)*fpv, :, :, :]
                    X = X.transpose(1, 2)
                    X = X.to(device).float()

                    y1pred, y2pred = model(X, subidx)
                    y1preds.append(y1pred)
                    y2preds.append(y2pred)

                y1p = torch.cat(y1preds, dim = 1)
                y2p = torch.cat(y2preds, dim = 1)
                countpred = torch.sum((y2p > 0) / (y1p + 1e-1), 1)
                count = torch.sum((y2 > 0) / (y1 + 1e-1), 1)

                loss1 = lossMAE(y1p, y1)
                loss2 = lossBCE(y2p, y2)
                loss3 = torch.sum(torch.div(torch.abs(countpred - count), (count + 1e-1)))

                loss = loss1 + 5*loss2

                if use_count_error:
                    loss = loss + loss3

                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                del X_total, y_total, y1p, y2p, countpred, count

                train_loss = loss.item()
                trainLosses.append(train_loss)
                mae = mae + loss1.item()
                mae_count = mae_count + loss3.item()
                i = i + 1
                pbar.set_postfix({'Epoch': epoch,
                                'MAE_period': (mae/i),
                                'MAE_count' : (mae_count/i),
                                'Mean Tr Loss':np.mean(trainLosses[-i+1:])})
                
                
        if validate:
            #validation loop
            with torch.no_grad():
                mae = 0
                mae_count = 0
                i = 1
                pbar = tqdm(val_loader, total = len(val_loader))

                for X_total, y_total in pbar:
                    torch.cuda.empty_cache()

                    totalparts = X_total.shape[1]//fpv
                    y1 = y_total.to(device).float()
                    y2 = getPeriodicity(y1).to(device).float()

                    model.init_params(batch_size)
                    model.eval()

                    y1preds = []
                    y2preds = []

                    for subidx in range(totalparts):
                        print(subidx)
                        
                        X = X_total[:, subidx*fpv: (subidx+1)*fpv, :, :, :]
                        X = X.transpose(1, 2)
                        X = X.to(device).float()

                        y1pred, y2pred = model(X, subidx)
                        y1preds.append(y1pred)
                        y2preds.append(y2pred)

                    y1p = torch.cat(y1preds, dim = 1)
                    y2p = torch.cat(y2preds, dim = 1)
                    countpred = torch.sum((y2p > 0) / (y1p + 1e-1), 1)
                    count = torch.sum((y2 > 0) / (y1 + 1e-1), 1)

                    loss1 = lossMAE(y1p, y1)
                    loss2 = lossBCE(y2p, y2)
                    loss3 = torch.sum(torch.div(torch.abs(countpred - count), (count + 1e-1)))

                    loss = loss1 + 5*loss2

                    if use_count_error:
                        loss = loss + loss3

                    torch.cuda.empty_cache()
                    del X_total, y_total, y1p, y2p, countpred, count

                    val_loss = loss.item()
                    valLosses.append(val_loss)
                    mae = mae + loss1.item()
                    mae_count = mae_count + loss3.item()
                    i = i + 1
                    pbar.set_postfix({'Epoch': epoch,
                                    'MAE_period': (mae/i),
                                    'MAE_count' : (mae_count/i),
                                    'Mean Val Loss':np.mean(valLosses[-i+1:])})

        #save checkpoint
        if saveCkpt:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses' : trainLosses,
                'valLosses' : valLosses
            }
            torch.save(checkpoint, 
                       'checkpoint/' + ckpt_name + str(epoch) + '.pt')
        
        #lr_scheduler.step()

    return trainLosses, valLosses


def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    return trainDataset, valDataset

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
   
    ave_grads = []
    max_grads= []
    median_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            median_grads.append(p.grad.abs().median())
            
    width = 0.3
    plt.bar(np.arange(len(max_grads)), max_grads, width, color="c")
    plt.bar(np.arange(len(max_grads)) + width, ave_grads, width, color="b")
    plt.bar(np.arange(len(max_grads)) + 2*width, median_grads, width, color='r')
    
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="r", lw=4)], ['max-gradient', 'mean-gradient', 'median-gradient'])

