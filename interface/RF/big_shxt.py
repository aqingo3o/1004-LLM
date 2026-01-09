import torch
import torch.nn.functional as F

x = torch.tensor([1, 2, 3])
x = x.to("cuda")                # Move tensor to GPU

# create a 2D tensor
tensor_2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
print(tensor_2d)

# check the shape
print(tensor_2d.shape)

# reshape the tensor to 3x2
# method 1
print(tensor_2d.reshape(3, 2))

# **method 2** : use `view`
print(tensor_2d.view(3, 2))

# transpose the tensor
print(tensor_2d.T)

# muptliply two tensors
tensor_a = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
tensor_b = tensor_a.T

# meethod 1
print(tensor_a.matmul(tensor_b))

# method 2: use `@` operator
print(tensor_a @ tensor_b)

class MyLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)    # 定義零件

    def forward(self, x):
        return self.layer1(x)                   # 定義資料流向
    

y = torch.tensor([1.0])         # answer
x1 = torch.tensor([1.1])        # 輸入特徵
w1 = torch.tensor([2.2])        # weights
b = torch.tensor([0.0])         # bias unit

z = x1 * w1 + b         # input
a = torch.sigmoid(z)    # activation function and output

loss = F.binary_cross_entropy(a, y) # 計算損失
print(loss)


from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b 
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True) 
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)

