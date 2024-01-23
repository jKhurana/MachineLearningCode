import torch
import pandas as pd



# rows is list of series....
def collate_fn (rows):
    '''
    Coverts the data into tensors for passing to the model.
    '''
    x_list = []
    y = []
    for i in range(len(rows)):
        x_list.append(torch.Tensor(rows[i].iloc[:-1].tolist()))
        y.append(rows[i].iloc[-1])

    return torch.stack(x_list), torch.Tensor(y)


def load_data(data_file):
    '''
    Loads tsv into a pandas df
    '''
    data = pd.read_csv(data_file, sep=",")
    data = data.dropna()
    return data

