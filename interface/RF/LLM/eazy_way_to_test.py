# plz run this command:
# pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

import torch

# 自動選擇裝置：有 GPU 就用 cuda，沒 GPU 就用 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use:w {device}")

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

# 全部搬到自動選擇的 device
tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)

print(tensor_1 + tensor_2)