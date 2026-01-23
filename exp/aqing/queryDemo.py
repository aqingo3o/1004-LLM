'''
rag-miniLM_fullComment.py 的寫的不要那麼巨醜版
應該是啾呼一樣的功能

這邊能做的是
分隔長文本並檢索與輸入句子**最相關**的段落
可以列出相關段落的 index 以及相似度分數, 但仍然像一坨大便
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
# 從嵌入到池化***到正規化***都打包了!
def qingsEmbedder(theItem, need_normalize): # theStr: 要轉成embedding的東西, need_normalize(bool)
    torch.set_grad_enabled(False)
    inp = tokenizer(theItem, return_tensors="pt")
    out = ebModel(**inp)
    token_emb = out.last_hidden_state
    atMaskP1d = inp['attention_mask'].unsqueeze(-1).expand(token_emb.shape).float()
    sentenceEmbedding = torch.sum(token_emb * atMaskP1d, dim=1) / torch.clamp(atMaskP1d.sum(dim=1), min=1e-9)
    if need_normalize == True:
        sentenceEmbedding = torch.nn.functional.normalize(sentenceEmbedding, p=2, dim=1)
    elif need_normalize == False:
        pass
    else :
        print("'need_normalize' should be a boolean value.")
    torch.set_grad_enabled(True)
    return sentenceEmbedding

### ---------------------------------  Chunking -------------------------------- ###
chunks = []
# This function can deal with too long chunks
def cutLong(tooLongItem, maxLen, overlap): # 與 chunks(list) 相依!!
    startIdx = 0
    while startIdx < len(tooLongItem):
        endIdx = startIdx + maxLen
        chunks.append(tooLongItem[startIdx:endIdx]) # 分割後的東東
        startIdx += overlap # 留一點重疊
    return chunks

for c in textFile.split('\n\n'):
    clean_c = c.strip() # remove space
    if len(clean_c) >= 400: # 1token ~ 0.75word, 抓個緩衝
        cutLong(clean_c, 400, 50)
    elif len(clean_c) > 20: # 長度太短的應該是廢話
        chunks.append(clean_c)

### -----------------------  Chunks -> Sentence Embeddings ---------------------- ###
sentenceEmbs = []
for c in chunks:
    sentenceEmbs.append(qingsEmbedder(c, True)) # normalized

### ----------------------------  Make the Memory Bank --------------------------- ###
memoryBank = torch.cat(sentenceEmbs, dim=0)

### ---------------------------------  Try Query --------------------------------- ###
theQuery = 'who is Crowley?' # 隨便打一點東西
queryEmb = qingsEmbedder(theQuery, True)
sim_scores = torch.matmul(queryEmb, memoryBank.t()) # 已經正規化過了所以可以直接用矩陣乘法
'''# same as ""
sim_scores = queryEmb @ memoryBank.t()
'''

#### 以下的東西並沒有仔細的了解
top_scores, top_indices = torch.topk(sim_scores, k=3)
print(f"The top3 related parts are: {top_indices[0].tolist()}") # .tolist() 可以刪掉 tensor(顯示)
print(f"The similarities are: {top_scores[0].tolist()}")

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')