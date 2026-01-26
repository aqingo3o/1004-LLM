'''
基於 ../../exp/aqing/queryDemo.py 製作的一款腳本
如果順利的話將會把這個東西接在介面上使用

因此額外的新增了一些接口，並移除了更多的註解...  
人終究會變成自己討厭的樣子
'''
print('Loading...', end='\n\n')
### ----------------------------  Import Models ---------------------------- ###
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional
from transformers import AutoTokenizer, AutoModel

### ----------------------------  Set Variables ----------------------------- ###
# 預期這邊的東西都能變成一個使用者能夠決定的東西...
simiLMName = 'sentence-transformers/all-MiniLM-L6-v2' # for sentence similarity
knowledgeSrc = [ #dataset/ 下的文ㄅ
    'what-are-active-galactic-nuclei.txt',
    'mvp-proposal.txt',
]
print('[ Knowledge Sources ]')
for i in knowledgeSrc:
    print(f'- {i}')
print()
textName = input("Pick a file to use as reference >>> ")

### -------------------------------  Load Model ------------------------------- ###
tokenizer = AutoTokenizer.from_pretrained(simiLMName)
ebModel = AutoModel.from_pretrained(simiLMName)
posEmb_max = ebModel.config.max_position_embeddings

### ---------------------------------  Text --------------------------------- ###
root = Path(__file__).resolve().parents[2]
textPath = f'{root}/dataset/{textName}'
textFile = open(textPath, 'r').read()

### --------------------------  def Embedder function ------------------------- ###
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
def cutLong(tooLongItem, maxLen, overlap): # 與 chunks(list) 相依!!
    startIdx = 0
    while startIdx < len(tooLongItem):
        endIdx = startIdx + maxLen
        chunks.append(tooLongItem[startIdx:endIdx])
        startIdx += overlap
    return chunks

chunkLen_max = int(np.floor(posEmb_max*0.75)) # 向下取整
for c in textFile.split('\n\n'):
    clean_c = c.strip() # remove space
    if len(clean_c) >= chunkLen_max:
        cutLong(clean_c, chunkLen_max, 50)
        print('Too long chunks detected, Need more time...')
    elif len(clean_c) > 30:
        chunks.append(clean_c)

### -----------------------  Chunks -> Sentence Embeddings ---------------------- ###
sentenceEmbs = []
for c in chunks:
    sentenceEmbs.append(qingsEmbedder(c, True)) # normalized

### ----------------------------  Make the Memory Bank --------------------------- ###
memoryBank = torch.cat(sentenceEmbs, dim=0)

### ----------------------------  cli is kind of ui... ---------------------------- ###
print()
#print(f'The article you are looking at is {textName}, ')
print(f"The language model for sentence similarity is {simiLMName}", end='\n\n')
while True:
    theQuery = input("Put your query here (or enter 'quit' to quit) >>> ")
    if theQuery=='quit':
        print('byeeee ;))')
        break
    else:
        queryEmb = qingsEmbedder(theQuery, True)
        simiScores = queryEmb @ memoryBank.t()
        topScores, topIndex = torch.topk(simiScores, k=3)
        print(f'In all {len(chunks)} chunks, these are top three most simi:', end='/n/n') # 他媽的英文亂講
        for i in range(len(topScores[0])):
            print(f'Similarity scores: {(topScores[0][i]):.2f}')
            print(f'--> {chunks[topIndex[0][i]]}')
            print()
        print('---------------------------------------------------------------------------------', end='\n\n')
    input("Press 'ENTER' to start a new query.")