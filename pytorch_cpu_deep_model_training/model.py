import torch 
import torch.nn as nn 
import sys 

# simple feed forward network
class FeedForwardNetwork(nn.Module):
    def __init__(self,layer_dims):
        super(FeedForwardNetwork, self).__init__()

        layers = []

        for i in range(1,len(layer_dims)):
            l1 = nn.Linear(layer_dims[i-1],layer_dims[i])
            layers.append(l1)
            if i==len(layer_dims): # if last layer
                 layers.append(nn.sigmoid())
            else:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
    
    def forward(self,X):

        out = self.network(X)

        return out


