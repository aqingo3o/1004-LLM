# plz run this command: 
# pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
import torch
import torch.nn.functional as F
from torch.autograd import grad

class MyLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)    # 定義零件

    def forward(self, x):
        return self.layer1(x)                   # 定義資料流向


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

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs): 
        super().__init__()
        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),                    # add activation function

            # 2nd hidden layer
            torch.nn.Linear(30, 20),            # 頭尾數字一樣
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
# 訓練集
X_train = torch.tensor([                # 5個訓練樣本，每個樣本包含2個特徵
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
    ])
y_train = torch.tensor([0, 0, 0, 1, 1]) # 5個訓練樣本對應的標籤

# 測試集
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):           # 初始設定，將準備好的資料(X)(特徵)和答案(y)輸入
        self.features = X
        self.labels = y

    def __getitem__(self, index):       # 取得某一筆資料，告訴index幾，取出幾號資料
        one_x = self.features[index] 
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):                  # 資料總長度
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

from torch.utils.data import DataLoader
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,           # 每次端幾筆資料給model吃
    shuffle=True,           # 洗牌，每次取的資料順序不同，防止模型背答案
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,          # 對於測試集只是要打分數沒必要洗牌
     num_workers=0 
)

# Re: 用CPU執行兩個張量相加
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
# 利用.to()將張量移到GPU上面
tensor_1 = tensor_1.to("cuda:0")
tensor_2 = tensor_2.to("cuda:0")
print(tensor_1 + tensor_2)

# 將訓練迴圈用GPU跑

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
#---------------------------------------------------------#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#---------------------------------------------------------#
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
#---------------------------------------------------------#
        features = features.to(device)
        labels = labels.to(device)
#---------------------------------------------------------#        
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")
        
    model.eval()    # 評估模式on