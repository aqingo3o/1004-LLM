# In /tutorial/aqing
這邊放了一些我覺得是面向人類的教程，任何問題都殻以提問！如果我能回答的話我將會告訴你。
包括但不限於建置環境或是跑腳本的時候遇到 FileNotFoundError  

## run-Ollama.md
for Mac, Linux 的如何使用 Ollama 教學。  
這個東西真的有夠簡單，還是別人完整打包的，連指定要不要在在 GPU 上跑都可以自己偵測。  
輕輕鬆鬆在 10 分鐘內就能獲得<mark>**本地運行的**</mark>大語言模型。

## fromZero-to-coarseRAG.ipynb
這是一整套從單句嵌入向量到分割一篇文章 (.txt) 並進行最高相似度搜索的操作指南，集目前在 [/exp/aqing](../../exp/aqing) 下的東西的大成, Especially [simimlarityDemo.py](../../exp/aqing/simimlarityDemo.py) and [queryDemo.py](../../exp/aqing/queryDemo.py).  

過程中省略了學習路上會寫出的很醜程式，整體來說稍微美麗一點。  
相依的文件路經需要與這個腳本放在同一個資料夾下，也可以直接 clone 整個 1004-LLM。更詳細的 setup 指示請見檔案內文。    
總之是一個直接落地實作的好東西。
