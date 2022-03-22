import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./CIFAR10dataset", transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset=dataset,batch_size=64,drop_last=True)

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL,self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,input):
        output = self.linear1(input)
        return output


model = MODEL()


for data in dataloader:
    imgs,labels = data
    print(imgs.shape)
    output = torch.reshape(imgs,(1,1,1,-1))
     # or  output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)



