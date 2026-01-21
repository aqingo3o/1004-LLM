'''
函式來自 embedder-miniLM.py
詳細的建構過程請見 embedder-miniLM_fullComment.py

這邊就是進行一個全部大打包進 qingsEmbedder()
'''

from pathlib import Path
import time
import torch
import torch.nn.functional
from transformers import AutoTokenizer, AutoModel

timeStart = time.time()

# Text
root = Path(__file__).resolve().parents[0]
textPath = f'{root}/In-the-Second-Beginning.txt'
textFile = open(textPath, 'r') # Read-only
textFile.close()
testString_1 = 'kiwi bird'
testString_2 = 'feifei is a notebook.'
testString_3 = 'fruit'
testString_4 = 'sweet'

# Load Model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
ebModel = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# def Embedder function
# 從嵌入到池化都打包了!
def qingsEmbedder(theItem): # theStr: 要轉成embedding的東西
    torch.set_grad_enabled(False)
    inp = tokenizer(theItem, return_tensors="pt")
    out = ebModel(**inp)
    token_emb = out.last_hidden_state
    atMaskP1d = inp['attention_mask'].unsqueeze(-1).expand(token_emb.shape).float()
    torch.set_grad_enabled(True)
    return torch.sum(token_emb * atMaskP1d, dim=1) / torch.clamp(atMaskP1d.sum(dim=1), min=1e-9)

# Demo
cos_sim_1 = torch.nn.functional.cosine_similarity(qingsEmbedder(testString_1), qingsEmbedder(testString_2))
cos_sim_2 = torch.nn.functional.cosine_similarity(qingsEmbedder(testString_3), qingsEmbedder(testString_4))
print(cos_sim_1, cos_sim_2)

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')
