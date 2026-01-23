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
        short_c = c[startt:endd] # 我有問題，那為什麼不一開始就這樣ㄑㄧㄝ
        # 我覺得做好之後可以比較涼中切法的精確程度
        # 因為切切切的過程中就是在處理句子不樣被切得太碎的問題啊
        chunks.append(short_c)
        startt += 50 # 一點重疊


sentenceEmbs = []
for c in chunks:
    sentenceEmbs.append(qingsEmbedder(c))