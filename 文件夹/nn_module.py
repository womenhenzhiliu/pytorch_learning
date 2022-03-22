import torch
from torch import nn


class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()

    def forward(self,input):
        output = input + 1
        return output

jiayi = model()
x = torch.tensor(3)
output = jiayi(x)

print(output)