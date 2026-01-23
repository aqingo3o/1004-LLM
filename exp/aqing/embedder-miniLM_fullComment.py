'''
沒有自動下載文本的功能
所以要把踢叉踢和這支程式一起放進一個資料夾裡
似乎需要聯網才能操作

充滿超級白痴註解版
你問為什麼不用 ipynb 因為註解能對齊比較香啦
'''
### -----------------------  Import Models ----------------------- ###
import time
import torch
from transformers import AutoTokenizer, AutoModel

timeStart = time.time()

### ----------------------------  Text ---------------------------- ###
testString_1 = "I like astronomy."
testString_2 = "I enjoy watching the night sky full of stars."
testString_3 = "I like tomatoes."

### --------------------------  Tokenizer -------------------------- ###
'''
可以使用 re.split() 進行字串分割, 但是有別人做好的超好用小工具於是直接拿來用了
- tiktoken:     openAI 相關
- transformers: 來自HF, 更好地與一些開源模型對接？
'''
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
'''# from_pretrained()
會依序從本地資料夾、快取(~/.cache/huggingface/)、HF 中尋找括號內的東西，
括號內的東西是一整包的, 包含 config.json, tokenizer.json, tokenizer_config.json
vocab.txt, special_tokens_map.json, pytorch_model.bin
config.json 會告訴 AutoTokenizer 要用哪個模型的分詞器
至於要用哪個模型可以到 HF 的網站上看啊廢什麼話
教學:
總之先進到 HF 
選上面的 Model
左邊 Tasks 選 Sentence Similarity, 因為我現在就要做這個
篩選方式選 Most Downloads, 不知道怎麼辦的話就先看大家怎麼做吧
於是就選中了 sentence-transformers/all-MiniLM-L6-v2
可以手動下載放到這個資料夾裡面或者就讓 from_pretrained() 幫我抓下來哈哈屁眼
'''
inp_1 = tokenizer(testString_1, return_tensors="pt")
inp_2 = tokenizer(testString_2, return_tensors="pt")
inp_3 = tokenizer(testString_3, return_tensors="pt") # 'pt' for pytorch
                                                     # 接下來會用到火炬蟒所以把 tensor 打包成 pytorch 能認的格式
                                                     # 也可以是 'tf' for tensorflow
#tokens = tokenizer.convert_ids_to_tokens(inp['input_ids'][0]) # tokenID 還原成字, 超樸實的函數命名我要瘋了
                                                               # 101是[CLS]:Classification Token, 102是[SEP]:Separator Token
'''# print(inp_1)
{'input_ids': tensor([[  101,  1045,  2066, 12799,  1012,   102]]), # 所謂的 tokenID
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])} # 1:有意義的 token; 0: 為了對齊句子長度填上的符號(padding)，沒有意義
'''

### -----------------------  Token Embeddings ----------------------- ###
ebModel = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # 和寫 tokenizer 一樣的邏輯
torch.set_grad_enabled(False) # 不知道為什麼但先關掉梯度
out_1 = ebModel(**inp_1) # **代表全部帶入, inp 裡面有 'input_ids', 'token_type_ids' ...
out_2 = ebModel(**inp_2)
out_3 = ebModel(**inp_3)
torch.set_grad_enabled(True) # 喔喔, 訓練模型(改變)的時候才需要梯度, 只是使用模型的話不用, 所以關掉省電
                             # 開著也沒關係啦但我要善待 feifei
'''# safer :))
with torch.no_grad():
    out = ebModel(**inp)
# 但就是和 with() 不熟, 怎麼知道是這樣寫的啦？
'''
'''# print(out_1)

BaseModelOutputWithPoolingAndCrossAttentions(
         last_hidden_state=tensor([[
         [-0.0361, -0.1036,  0.2475,  ...,  0.0924, -0.8067, -0.0602],
         [ 0.0162, -0.9489,  0.1791,  ...,  0.3235, -0.7657, -0.8752],
         [-0.7430, -1.0482, -0.4274,  ...,  1.4613,  0.7811,  0.8082],
         [-0.2966, -0.1204,  0.2318,  ..., -0.2413, -0.9462, -0.8607],
         [ 0.2942, -0.5316,  0.0226,  ...,  0.2044, -0.6020,  0.3516],
         [ 0.0241, -0.7994,  0.0280,  ...,  0.0899, -0.5316, -0.1024]
         ]]),
         # last_hidden_state: 神經網路最後一層的輸出, 通常來說是語意完整的好東西
         # 5行: 5個token
         # 每行384個元: 每個 token 經過模型後被轉換成 384 維的向量表示, 等於隱藏層的大小(維度, hidden_size, hidden_dim)
         # 代表這句話中，每個 token 在看完整句子後的語意表示, 所以 senEmbedding 一定是用 last_hidden_state 做的

         pooler_output=tensor([[ 
          8.3103e-03,  2.9566e-02,  8.8434e-02,  1.9435e-02, -2.4388e-02,
         ...(總之 384 個元)
         -6.7047e-04,  2.4648e-02, -7.1576e-02,  2.8434e-02
         ]]),
         # 不知道是什麼, 感覺像中間產物或是舊時代遺留的東西?
         # 因為 pooling 是等下我自己要做的啊

         hidden_states=None, # 每一個隱藏層的輸出, 我沒要他算他就沒算
                             # 在 model(..., hidden_states=True) 就可以叫他算
                             # 以下項目同理
         past_key_values=None, 
         attentions=None, 
         cross_attentions=None)
'''

'''# .shape or .size()
print(inp['attention_mask'].shape) # (batch_size, seq_len), seq_len: N 個 token
print(inp['attention_mask'].unsqueeze(-1).shape) # unsqueeze() 他媽哪來的啊? numpy?
                                                 # 明顯並非, torchTensor 和 npArray 看起來就不像一路的東西
token_emb = out.last_hidden_state # 素的取出
print(token_emb.shape) # (batch_size, seq_len, hidden_dim) = (1, N, 384)
'''

def meanPooling(eb_model_output, attention_mask): # 我就是要自己寫然後全部大展開
    token_emb = eb_model_output.last_hidden_state
    atMaskP1d = attention_mask.unsqueeze(-1) # 增加一個維度, 和 token_emb 對齊 (attention mask plus 1 dim)
    atMaskP1d = atMaskP1d.expand(token_emb.shape).float() # 最後一個維度沿著 token_emb.shape 複製, 總之對齊造型
                                                          # float() 因為同是浮點數才能運算, numpy 基操
    efficient_token_emb = token_emb * atMaskP1d # 與遮罩相乘, 遮罩=1 的地方才留下值
    poolingResult = torch.sum(efficient_token_emb, dim=1) # 沿著張量相乘結果的的第1條軸(seq_len)加總, 
                                                          # 即所有真實 token 向量的總和, 耶 pooling
    poolingResult_mean = poolingResult / torch.clamp(atMaskP1d.sum(dim=1), min=1e-9) # 平均
                                         # atMaskP1d.sum(dim=1)是沿著注意力遮罩的第1條軸相加, 
                                         # 因為都是 1, 0 所以相加的值就是有效的 token 的數量
                                         # torch.clamp() 是除數!=0保護機制, 等於零的話就當作 1e-9處理
    return poolingResult_mean

'''# 然後他媽的現在才發現 HF 上有寫好的 demo, 那我在這邊背單字的意義是什麼. 這邊寫的比較美麗吧
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
                                       # 就是在做 token_emb = out.last_hidden_state 一樣的事
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
'''
sentence_emb_1 = meanPooling(out_1, inp_1['attention_mask']) # Perform pooling, 把 N 個 token_emb 壓成一條 sentence_emb
sentence_emb_1 = torch.nn.functional.normalize(sentence_emb_1, p=2, dim=1) # Normalize(L2) embeddings 
sentence_emb_2 = meanPooling(out_2, inp_2['attention_mask'])
sentence_emb_3 = meanPooling(out_3, inp_3['attention_mask'])

print(f"Sentence embeddings: {sentence_emb_3}") # 經過這一坨超複雜步驟, 我們做出了一個句子的向量
                                                # 句子向量只有一個的話不能做什麼, 但是有很多的話就可以! 比較語意啊之類的

### ------------------------------  Simility ------------------------------ ###  
cosSim_11 = torch.nn.functional.cosine_similarity(sentence_emb_1, sentence_emb_1)                                 
cosSim_12 = torch.nn.functional.cosine_similarity(sentence_emb_1, sentence_emb_2)
cosSim_13 = torch.nn.functional.cosine_similarity(sentence_emb_1, sentence_emb_3)
print('cosine')
print(f'string1 & string1: {cosSim_11[0]}') # 因為就是一樣的句子啊屁眼
print(f'string1 & string2: {cosSim_12[0]:.2f}')
print(f'string1 & string3: {cosSim_13[0]:.2f}') # 成功證明嘻嘻

timeEnd = time.time()
print(f'It took {(timeEnd-timeStart):.2f} seconds to finish the work.')