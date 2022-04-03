import os
import sys
import argparse
import time
import random
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader,random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# from Kitti import kittidata
from StructureNet import StructureNet

import datetime
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32,
                                            help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=300,
                                            help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                                            help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                                            help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                                            help='seed for random initialisation')
    args = parser.parse_args()
    train(args)


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train_model(model, optimizer, dl_train, dl_valid, batch_size, max_epochs):
    criterion = nn.MSELoss()
    metric_name = 'MSE'
    dfhistory = pd.DataFrame(columns = ["epoch","loss","val_loss"])
    for epoch in range(max_epochs):
        loss_sum = 0
        log_step_freq = 600
        for step, (features,labels) in enumerate(dl_train, 1):
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions,labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            # if step == 100:
            #     print("train steps:")
            #     print(loss)
                # print(labels)
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f") %
                    (step, loss_sum/step))
                    
        # model.eval()
        val_loss_sum = 0.0
        for val_step, (features,labels) in enumerate(dl_train, 1):
            with torch.no_grad():
                predictions = model(features)
                val_loss = criterion(predictions,labels)
                # if val_step == 100:
                #     print("test steps:")
                #     print(val_loss)
                    # print(labels)
            val_loss_sum += val_loss.item()

        info = (epoch, loss_sum/step, val_loss_sum/val_step)
        dfhistory.loc[epoch] = info
        
        print(("\nEPOCH = %d, loss = %.3f, val_loss = %.3f, ") 
            %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
                
    print('Finished Training...')
    time_finished = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dfhistory.to_csv("train/"+time_finished+'.csv')
    torch.save(model.state_dict(), "model/"+time_finished+'.model')
    return model


def evaluate_test_set(model, dl_test):
    model.eval()
    criterion = nn.MSELoss()
    test_loss_sum = 0.0
    test_metric_sum = 0.0
    test_step = 1
    for test_step, (features,labels) in enumerate(dl_test, 1):
        with torch.no_grad():
            predictions = model(features)
            test_loss = criterion(predictions,labels)
            test_metric = criterion(predictions,labels)

        test_loss_sum += test_loss.item()
        test_metric_sum += test_metric.item()
    print("test loss is:", test_loss_sum/test_step)


def train(args):
    random.seed(args.seed)
    BATCH_SIZE = args.batch_size
    """
    need to add dataset

    """
    model = StructureNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = train_model(model, optimizer, dl_train, dl_valid, args.batch_size, args.num_epochs)

    evaluate_test_set(model, dl_test)


if __name__ == '__main__':
    main()