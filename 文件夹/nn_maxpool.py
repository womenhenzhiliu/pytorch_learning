
import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root= "./CIFAR10dataset", transform= torchvision.transforms.ToTensor(),download= True)

dataloader = DataLoader(dataset,batch_size=64)



class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, return_indices= False, ceil_mode= False)

    def forward(self,x):
        x = self.maxpool1(x)
        return x

Model = model()
writer = SummaryWriter(log_dir="logs_maxpool")
step = 0
for data in dataloader:
    imgs, labels = data
    output = Model(imgs)
    writer.add_images(tag="input", img_tensor=imgs,global_step=step)
    writer.add_images(tag="output", img_tensor=output,global_step=step)
    step += 1

writer.close()