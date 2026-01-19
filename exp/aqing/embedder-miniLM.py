'''
沒有自動下載文本的功能
所以要把踢叉踢和這支程式一起放進一個資料夾裡
似乎需要聯網才能操作

沒有醜死人註解版
'''

from pathlib import Path
import time
import torch
import torch.nn.functional
from transformers import AutoTokenizer, AutoModel

timeStart = time.time()
# 要處理的文本, 第一次試作獻給az
root = Path(__file__).resolve().parents[0]
textPath = f'{root}/In-the-Second-Beginning.txt'
textFile = open(textPath, 'r') # Read-only
textFile.close()
testString_1 = 'Haha, pi:yan. molecular gas prop in Circinus galaxy!'
testString_2 = 'I trust the universe will always bring me to you'
testString_3 = 'i am Livia'

# 預處理 - 把文本切成 tokens
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inp = tokenizer(testString_3, return_tensors="pt")
# 嵌入
ebModel = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
torch.set_grad_enabled(False)
out = ebModel(**inp)
torch.set_grad_enabled(True)

def meanPooling(eb_model_output, attention_mask):
    token_emb = eb_model_output.last_hidden_state
    atMaskP1d = attention_mask.unsqueeze(-1).expand(token_emb.shape).float()
    return torch.sum(token_emb * atMaskP1d, dim=1) / torch.clamp(atMaskP1d.sum(dim=1), min=1e-9)

sentence_emb = meanPooling(out, inp['attention_mask']) # Perform pooling
sentence_emb = torch.nn.functional.normalize(sentence_emb, p=2, dim=1) # Normalize embeddings

print("Sentence embeddings:")
print(sentence_emb)

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')
