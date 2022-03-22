import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./CIFAR10dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

class CIFAR10_model(nn.Module):
    def __init__(self):
        super(CIFAR10_model,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


loss = nn.CrossEntropyLoss()
model = CIFAR10_model()
optim = torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(5):
    sumloss = 0.0
    for data in dataloader:
        imgs,labels = data
        output = model(imgs)
        result_loss = loss(output,labels)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        # print(result_loss)
        sumloss += result_loss
    print(sumloss)
