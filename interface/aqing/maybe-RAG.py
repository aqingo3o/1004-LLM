'''
基於 the-R-of-RAG.py 製作的一款腳本
encoder 接 decoder 的一個例子
'''
print('Loading...', end='\n\n')
### ----------------------------  Import Models ---------------------------- ###
import numpy as np
from ollama import chat
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

### -------------------------------  Find GPU ------------------------------- ###
if torch.cuda.is_available():
    dev = torch.device('cuda')
    print(f'Found gpu: {torch.cuda.get_device_name(0)} !')
else:
    dev = torch.device('cpu')
    print('No gpu found, using cpu instead.')
print()

### ----------------------------  Set Variables ----------------------------- ###
simiLMName = 'sentence-transformers/all-MiniLM-L6-v2' # for sentence similarity
decoderName = input("Input the decoder (e.g., Llama2) >>> ") # decoder for 說人話
knowledgeSrc = [ # under dataset/
    'alma-basics.txt',
    'circinus-galaxy.txt',
    'mvp-proposal.txt',
    'what-are-active-galactic-nuclei.txt',
]
print('[ Knowledge Sources ]')
for i in knowledgeSrc:
    print(f'- {i}')
print()
textName = input("Pick a file to use as reference >>> ")

print(f'The language model for sentence similarity is {simiLMName}.')
print(f'The decoder for the "G" of RAG is {decoderName}.')

### -------------------------------  Load Model ------------------------------- ###
tokenizer = AutoTokenizer.from_pretrained(simiLMName)
# tokenizer 這種不涉及張量運算的東西放 cpu 更好
ebModel = AutoModel.from_pretrained(simiLMName)
ebModel = ebModel.to(dev)
posEmb_max = ebModel.config.max_position_embeddings

### ---------------------------------  Text --------------------------------- ###
root = Path(__file__).resolve().parents[2]
textPath = f'{root}/dataset/{textName}'
textFile = open(textPath, 'r').read()

### --------------------------  def Embedder function ------------------------- ###
def qingsEmbedder(theItem, need_normalize): # theStr: 要轉成embedding的東西, need_normalize(bool)
    torch.set_grad_enabled(False)
    inp = tokenizer(theItem, return_tensors="pt")
    inp = inp.to(dev) # to gpu, if have
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

### -------------------------  def Decoder function (ollama) ------------------------ ###
def decoder_OLMA(ref, usrQuery):
    respone = chat(
        model=decoderName,
        messages=[
            {'role': 'system', 
             'content': f'You are a helpful assistant. Answer the question using ONLY these information: /n{ref}'},
            {'role': 'user',
             'content': usrQuery}]
    )
    return respone['message']['content']

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

### ----------------------------  CLI is kind of ui... --------------------------- ###
print()
while True:
    theQuery = input("Put your query here (or enter 'quit' to quit) >>> ")
    print()
    if theQuery=='quit':
        print('byeeee ;))')
        break
    else:
        queryEmb = qingsEmbedder(theQuery, True)
        simiScores = queryEmb @ memoryBank.t()
        topScores, topIndex = torch.topk(simiScores, k=3)

        simiContent = ''
        for i in topIndex[0]:
            simiContent += chunks[i]

        ollamaRe = decoder_OLMA(ref=simiContent, usrQuery=theQuery)
        print(ollamaRe)
        print('---------------------------------------------------------------------------------', end='\n\n')
    input("Press 'ENTER' to start a new query.")