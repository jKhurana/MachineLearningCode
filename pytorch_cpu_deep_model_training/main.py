import torch
import datetime
from train import *

if __name__ == '__main__':
    
    # memory_limit(0.5) #Limits the maximum memory usage to half

    dt = datetime.datetime.now()

    training_args = {
        'init_checkpoint' : None,
        'layer_dims' : [40,32,16,8,1],
        'log_every' : 10,
        'eval_every' : 5,
        'save_every' : 2,
        'max_checkpoints' : 10,
        'batch_size': 64,
        'epochs' : 40,
        'lr' : 1e-5,
        'train_dataset_filename' : './data/train.csv',
        'eval_dataset_filename' : './data/test.csv',
        'epoch_checkpoint_dir' : './output/checkpoints',
    }

    print('Training with args:', training_args)

    train(training_args)
