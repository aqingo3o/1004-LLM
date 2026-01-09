# plz run this command:
# pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

import torch
print(torch.cuda.is_available())

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
tensor_1 = tensor_1.to("cuda:0")
tensor_2 = tensor_2.to("cuda:0")
print(tensor_1 + tensor_2)