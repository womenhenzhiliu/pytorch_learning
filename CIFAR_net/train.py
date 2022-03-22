import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import *
# prepare dataset
train_dataset = torchvision.datasets.CIFAR10(root="../CIFAR10dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.CIFAR10(root="../CIFAR10dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)




# lenth of dataset

train_dataset_lenth = len(train_dataset)
test_dataset_lenth = len(test_dataset)
print("the lenth of train_dataset:", train_dataset_lenth)
print("the lenth of test_dataset:", test_dataset_lenth)

# load the data with dataloader

train_dataloader = DataLoader(dataset=train_dataset,batch_size=32)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=32)

# create the model
# in the model.py

model = CIFAR_net()

#create loss function
loss_function = torch.nn.CrossEntropyLoss()

#create optimizer
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=lr)

# setting
total_train_step = 0
total_test_step = 0
epoch =10
writer = SummaryWriter(log_dir="./logs")
start_time = time.time()
for i in range(epoch):
    print("-----the {} epoch is begining-------".format(i+1))
    model.train()
    for data in train_dataloader:
        imgs,labels = data
        output = model(imgs)
        loss = loss_function(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("the{}training step ,loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar(tag="trainloss",scalar_value=loss.item(),global_step=total_train_step)
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            output = model(imgs)
            loss = loss_function(output,labels)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1)==labels).sum()
            total_test_accuracy += accuracy
    print("testloss of the whole test_dataset:{}".format(total_test_loss))
    print("accuracy of the whole test_dataset:{}".format(total_test_accuracy/test_dataset_lenth))
    writer.add_scalar(tag="testloss",scalar_value=total_test_loss,global_step=total_test_step)
    writer.add_scalar(tag="test_accuracy", scalar_value=total_test_accuracy/test_dataset_lenth,global_step=total_test_step)
    total_test_step += 1

writer.close()







