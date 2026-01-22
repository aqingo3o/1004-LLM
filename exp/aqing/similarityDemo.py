'''
函式來自 embedder-miniLM.py
詳細的建構過程請見 embedder-miniLM_fullComment.py

這邊就是進行一個全部大打包進 qingsEmbedder()
'''
import time
import torch
import torch.nn.functional
from transformers import AutoTokenizer, AutoModel

timeStart = time.time()

# Text
testString_1 = "I like astronomy."
testString_2 = "I enjoy watching the night sky full of stars."
testString_3 = "I like tomatoes."

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
'''
sentence_emb_3 = qingsEmbedder(testString_3)
sentence_emb_3 = torch.nn.functional.normalize(sentence_emb_3, p=2, dim=1) # Normalize embeddings
'''
cosSim_12 = torch.nn.functional.cosine_similarity(qingsEmbedder(testString_1), qingsEmbedder(testString_2))
cosSim_13 = torch.nn.functional.cosine_similarity(qingsEmbedder(testString_1), qingsEmbedder(testString_3))
print(f'string1 & string2: {cosSim_12[0]:.2f}')
print(f'string1 & string3: {cosSim_13[0]:.2f}') # 成功證明嘻嘻

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')