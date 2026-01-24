'''
基於 /exp/aqing/queryDemo.py 製作的一款痾腳本
如果順利的話將會把這個東西接在介面上使用
因此額外的新增了一些接口，並移除了更多的註解...  
人終究會變成自己討厭的樣子
'''

### ----------------------------  Import Models ---------------------------- ###
from pathlib import Path
import time
import torch
import torch.nn.functional
from transformers import AutoTokenizer, AutoModel

timeStart = time.time()

### ----------------------------  Set Variables ----------------------------- ###
# 預期這邊的東西都能變成一個使用者能夠決定的東西...
textName = 'what-are-active-galactic-nuclei' # 不包含副檔名
LMName = 'sentence-transformers/all-MiniLM-L6-v2' 

### -------------------------------  Load Model ------------------------------- ###
tokenizer = AutoTokenizer.from_pretrained(LMName)
ebModel = AutoModel.from_pretrained(LMName)
posEmb_max = ebModel.config.max_position_embeddings

### ---------------------------------  Text --------------------------------- ###
root = Path(__file__).resolve().parents[2]
#textPath = f'{root}/In-the-Second-Beginning.txt' # CA qwq
#textPath = f'{root}/{textName}.txt' # agn
textPath = f'{root}/exp/aqing/{textName}.txt' # agn
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

chunkLen_max = posEmb_max*0.75
for c in textFile.split('\n\n'):
    clean_c = c.strip() # remove space
    if len(clean_c) >= chunkLen_max:
        cutLong(clean_c, chunkLen_max, 50)
    elif len(clean_c) > 20:
        chunks.append(clean_c)

### -----------------------  Chunks -> Sentence Embeddings ---------------------- ###
sentenceEmbs = []
for c in chunks:
    sentenceEmbs.append(qingsEmbedder(c, True)) # normalized

### ----------------------------  Make the Memory Bank --------------------------- ###
memoryBank = torch.cat(sentenceEmbs, dim=0)

### ----------------------------  cli is kind of ui... ---------------------------- ###
#theQuery = 'what is the different between seyfert and qusar?'

print(f'The article you are looking at is {textName}.txt, ')
print(f'and the using language mmodel is {LMName}.', end='\n\n')
while True:
    print('Enter your query here')
    theQuery = input("(or 'quit' to exit) >>> ")
    if theQuery=='quit':
        print('byeeee ;))')
        break
    else:
        time.sleep(2)
        queryEmb = qingsEmbedder(theQuery, True)
        simiScores = queryEmb @ memoryBank.t()
        topScores, topIndex = torch.topk(simiScores, k=3)
        print(f'In all {len(chunks)} chunks, these are top three most simi:') # 他媽的英文亂講
        for i in range(len(topScores[0])):
            print(f'Similarity scores: {(topScores[0][i]):.2f}')
            print(f'--> {chunks[topIndex[0][i]]}')
            print()
        print('---------------------------------------------------------------------------------', end='\n\n')

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')