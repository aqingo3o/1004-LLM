'''
從 simiarityDemo.py 出發
詳細的建構過程請見 embedder-miniLM_fullComment.py

接下來要做的是 讀我的簧文
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
textFile = open(textPath, 'r') # Read-only
testText = textFile.read() # textFile 是一個通往檔案的管道, 要用 .reaad() 讀了之後才會是字串
                           # else, AttributeError: '_io.TextIOWrapper' object has no attribute 'split'
'''
也可以是這樣
testArti = textPath.read_text(encoding='utf-8')
加倍優雅
'''
textFile.close()
testString_1 = 'Seren is a physicist.'
testString_2 = ' Kita is a chemist.'
testString_3 = 'Kita'
testString_4 = 'Seren'

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

########################################################################################
# 下面寫得巨醜一大波幹

# Chunking
chunks = [] # 一塊一塊的
tooLong = [] # 先裝起來之後再想辦法吧
for c in testText.split('\n\n'): # 以換行符分割
    clean_c = c.strip() # remove space
    ######
    if len(clean_c) > 400: # 1token ~ 0.75word, 抓個緩衝
        tooLong.append(clean_c)
    elif len(clean_c) > 20 and len(clean_c) < 500: # 長度太短的應該是廢話
        chunks.append(clean_c)

# 這個函式過於精美所以我要好好欣賞一下
# deal with the list 'tooLong'
startt = 0
for c in tooLong:
    while startt<len(c):
        endd = startt + 400 # 400 = maxLen
        short_c = c[startt:endd] # 我有問題，那為什麼不一開始就這樣切？
        # 我覺得做好之後可以比較涼中切法的精確程度
        # 因為切的照個步驟就是在處理句子不要被切得太碎的問題啊 (語意破碎)
        chunks.append(short_c)
        startt += 50 # 一點重疊

sentenceEmbs = []
for c in chunks:
    sentenceEmbs.append(qingsEmbedder(c))

# 進行一個記憶的過程, 但不確定所以先放這邊
memoryBank = torch.cat(sentenceEmbs, dim=0) # (numChunks, 384)
'''# 這行的解釋
torch.cat 的 cat 是 Concatenate (串接/連結) 的縮寫
想到ㄌ unix command 的 cat/tac 
一樣的英文 但好像用出來的用途不一樣?
總之 torch.cat() 是把呱呱中間的東西沿著 dim 指定的方向堆疊
In 2d tennsor, dim = (0, 1) stamd for (rows, columns)
'''

# query
theQuery = 'who is Crowley?' # 隨便打一點東西
queryEmb = qingsEmbedder(theQuery) # (1, 384)

# 這邊不用 torch.cos_sim()
# 用正規化加上矩陣乘法
# 稍微的節省算力，因為這樣正規化的動作只需要統一做一次
queryEmb_norm = torch.nn.functional.normalize(queryEmb, p=2, dim=1)
memoryBank_norm = torch.nn.functional.normalize(memoryBank, p=2, dim=1)
sim_scores = torch.matmul(queryEmb_norm, memoryBank_norm.t()) # (1, 384) * (384, numChunks) = (1, numChunks)
                                                              # 這邊的 .t() for transpose    
# torch.matmul() 是（或者簡寫成 @ 運算子）會把兩個張量相乘, 打開來講是 matrix multiplication
#### 以下的東西並沒有仔細的了解
top_scores, top_indices = torch.topk(sim_scores, k=3)
print(f"最像的 top3 資料編號是: {top_indices[0].tolist()}") # .tolist() 可以刪掉 tensor(顯示)
print(f"相似度分數分別是: {top_scores[0].tolist()}")

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')