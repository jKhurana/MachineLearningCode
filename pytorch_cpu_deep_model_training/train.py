import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from data import CustomDataset
from data_util import *
from train_util import *
from model import FeedForwardNetwork
from functools import partial
from transformers import AdamW
import resource


def train(training_args):

    print(f'Loading training data.....', flush=True)
    train_data = load_data(training_args['train_dataset_filename'])
    train_dataset = CustomDataset(train_data)
    
    print(f'Loading testing data.....', flush=True)
    eval_data = load_data(training_args['eval_dataset_filename'])
    eval_dataset = CustomDataset(eval_data)

    # train data loader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=training_args['batch_size'],
        num_workers=0,
        drop_last=True
    )

    # test data loader
    eval_dataloader = DataLoader(
            dataset=eval_dataset,
            collate_fn=collate_fn,
            batch_size=training_args['batch_size'],
            num_workers=0,
            drop_last=True,
            shuffle=False            
        )

    # create model object
    model = FeedForwardNetwork(training_args['layer_dims'])
    model.train()

    # create optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(parameters, lr=training_args['lr'])

    # loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f'Starting training.....')   
    for i in range(training_args['epochs']):

        itr = 0
        sum_loss = 0.0
        total = 0

        for features, targets in train_dataloader:

            # model forward pass
            y_pred = model(features)
            y_pred = torch.squeeze(y_pred)

            loss = criterion(y_pred, targets)
            loss.backward()

            optimizer.zero_grad()
            optimizer.step()
            
            sum_loss += loss.item()*targets.shape[0]
            total += targets.shape[0]
            itr += 1

            if itr % training_args['log_every'] == 1:
                print("epoch %d, itr %d : train_loss %.8f" %(i, itr, sum_loss/total), flush=True)
                sum_loss = 0
                total = 0

            if itr % training_args['eval_every'] == 1:
                model.eval()
                metrics = get_eval_metrics(model, eval_dataloader, criterion)
                print("epoch %d, itr %d :" %(i, itr))
                for metric_name, value in metrics.items():
                    print(f'\teval_{metric_name}: {value}', flush=True)
                model.train()

            if i % training_args['save_every'] == 0:
                save_checkpoint(
                    training_args['epoch_checkpoint_dir'],
                    model,
                    optimizer,
                    i,
                    delete_before=i - training_args['save_every'] * training_args['max_checkpoints']
                )


def get_eval_metrics(model, dataloader, criterion):
    '''
    Returns the evaluated metrics on the dataset.
    '''
    with torch.no_grad():
        sum_loss = 0
        total = 0
        correct = 0
        for features, targets in dataloader:

            y_pred = model(features)
            y_pred = torch.squeeze(y_pred)
            print(type(y_pred))
            print(type(targets))
            loss = criterion(y_pred, targets)
            sum_loss += loss.item()*targets.shape[0]
            total += targets.shape[0]
            # compute predecited labels based on threhold 0.5
            predicted_labels = torch.where(y_pred <= 0.5,0,1)
            correct += (targets==predicted_labels).sum().item()

        accuracy = correct / total
        loss = sum_loss / total

        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }

        return metrics



