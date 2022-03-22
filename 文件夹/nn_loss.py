
import torch
from torch.nn import L1Loss, MSELoss

input = torch.tensor([1,2,3], dtype=torch.float32)

label = torch.tensor([1,2,5], dtype=torch.float32)


input = torch.reshape(input,(1,1,1,3))

label = torch.reshape(label,(1,1,1,3))

loss = L1Loss()
loss_mse = MSELoss()

result = loss(input,label)
result_mse = loss_mse(input,label)
print(result)
print(result_mse)