'''
從 simiarityDemo.py 出發
詳細的建構過程請見 embedder-miniLM_fullComment.py

這邊是在做譯電疑似是 RAG 的東西
讓他讀的文本是 az 的簧文

可以進行 query 並找出語意最相關的段落 index
仍在升級中...
'''

### ----------------------------  Import Models ---------------------------- ###
from pathlib import Path
import time
import torch
import torch.nn.functional
from transformers import AutoTokenizer, AutoModel

timeStart = time.time()

### ---------------------------------  Text --------------------------------- ###
root = Path(__file__).resolve().parents[0]
textPath = f'{root}/In-the-Second-Beginning.txt'
textFile = open(textPath, 'r').read() # open() 打開的是一個通往檔案的管道, 
                                      # 要用 .reaad() 讀了之後才會是字串
                                      # else, AttributeError: '_io.TextIOWrapper' object 
                                      # has no attribute 'split'

### -------------------------------  Load Model ------------------------------- ###
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
ebModel = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

### --------------------------  def Embedder function ------------------------- ###
# 從嵌入到池化都打包了!
def qingsEmbedder(theItem): # theStr: 要轉成embedding的東西
    torch.set_grad_enabled(False)
    inp = tokenizer(theItem, return_tensors="pt")
    out = ebModel(**inp)
    token_emb = out.last_hidden_state
    atMaskP1d = inp['attention_mask'].unsqueeze(-1).expand(token_emb.shape).float()
    torch.set_grad_enabled(True)
    return torch.sum(token_emb * atMaskP1d, dim=1) / torch.clamp(atMaskP1d.sum(dim=1), min=1e-9)

### ---------------------------------  Chunking -------------------------------- ###
chunks = []  # 一塊一塊的, 分割後的產物
tooLong = [] # 先裝起來之後再想辦法吧
for c in textFile.split('\n\n'): # 以雙換行符(\n\n) 為分割服進行分割
    clean_c = c.strip() # remove space
    if len(clean_c) > 400: # 1token ~ 0.75word, 抓個緩衝
        tooLong.append(clean_c) # 不管了先裝起來
    elif len(clean_c) > 20 and len(clean_c) < 500: # 長度太短的應該是廢話
        chunks.append(clean_c)

startt = 0
for c in tooLong: # deal with the list 'tooLong'
    while startt<len(c):
        endd = startt + 400 # 400 = maxLen
        short_c = c[startt:endd] # 我有問題，那為什麼不一開始就這樣切？
        chunks.append(short_c)
        startt += 50 # 一點重疊

### -----------------------  Chunks -> Sentence Embeddings ---------------------- ###
sentenceEmbs = []
for c in chunks:
    sentenceEmbs.append(qingsEmbedder(c))

### ----------------------------  Make the Memory Bank --------------------------- ###
memoryBank = torch.cat(sentenceEmbs, dim=0) # (numChunks, 384)
'''# 這行的解釋
torch.cat 的 cat 是 Concatenate (串接/連結) 的縮寫
想到ㄌ unix command 的 cat/tac 
一樣的英文 但好像用出來的用途不一樣?
總之 torch.cat() 是把呱呱中間的東西沿著 dim 指定的方向堆疊
In 2d tennsor, dim = (0, 1) stamd for (rows, columns)
'''

### ---------------------------------  Try Query --------------------------------- ###
theQuery = 'who is Crowley?' # 隨便打一點東西
queryEmb = qingsEmbedder(theQuery) # (1, 384)

queryEmb_norm = torch.nn.functional.normalize(queryEmb, p=2, dim=1)
memoryBank_norm = torch.nn.functional.normalize(memoryBank, p=2, dim=1)
sim_scores = torch.matmul(queryEmb_norm, memoryBank_norm.t()) # (1, 384) * (384, numChunks) = (1, numChunks)
                                                              # 這邊的 .t() for transpose    
# torch.matmul() 是（或者簡寫成 @ 運算子）會把兩個張量相乘, 打開來講是 matrix multiplication
'''# 這邊這樣做的理由
用正規化(functional.normalize()) + 矩陣乘法(torch.mm()) 比直接用 torhc.cos_sim() 好的原因
是 torch.cos_sim() 每一次都要做一遍正規化, 但是將所有嵌入向量陣列一次性正規化就, 
稍微的節省算力?
因為這樣正規化的動作只需要統一做一次
'''

### -----------------------------------  Top K ----------------------------------- ###
'''# torch.topk()
不用自己寫排序演算法!
回傳值: a tuple of (values, index)
- value: 前k個個最高的值分別是多少
- index:這幾個分數在原本清單中的第幾個位置
'''
topScores, topIndex = torch.topk(sim_scores, k=3)
print(f"最像的 top3 資料編號是: {topIndex[0].tolist()}") # .tolist() 可以刪掉 tensor(顯示)
print(f"相似度分數分別是: {topScores[0].tolist()}")

# another output
print('Top three most simi chunks:', end='\n\n')
for i in range(len(topScores[0])):
    print(f'similarity scores: {topScores[0][i]}')
    print(f'--> {chunks[topIndex[0][i]]}')
    print()

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')