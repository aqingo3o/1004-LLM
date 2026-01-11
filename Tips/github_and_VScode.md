## 預防針：不負責任的教程，若遇到任何問題，請搭配C學長或G學長做食用  
# 如何使VScode連接github
前置：
- 請先確認電腦有安裝git，在終端機輸入：
```
git --version
```
如果有跑出`git version x.x.x`代表有，有報錯請直接去官網下載：
```
https://git-scm.com/
```
- 在VS code左下角先登入你的github
1. 打開乾淨的VScode介面(New window)
2. 點選`Clone Git Repository...`
3. 去github網頁複製該repo.的**連結**並貼在VS code裡面
4. 選擇你要在本地端建立該資料夾的位置
5. 之後要打開直接把該資料夾拉到VScode讓他打開即可

# 要怎麼推東西上去
1. 修改完檔案並**儲存**後，在VS code GUI介面左側第三個圖示，應該會有藍圈圈，給他**點下去**
2. 你會在`Changes`那邊看到擬修改過的檔案，滑鼠移到`Changes`，並按下`+`，把更新的東西加到`Staged Changes`  
*PS. 按`Changes`的`+`即可全部加進`Staged Changes`，若單純想加某幾個，選擇該檔案的`+`即可*
3. 加進`Staged Changes`後，按下藍色大框框的`Commit`**的旁邊的選單**，點選`Commit & Push`
4. 按下後會出現**許多綠色字的地方**，直接在最上面打上**Message**，也就是你對於這次更新的訊息 *(必須得打)*，再按右上角的Commit

## Unidentified
`Is qing here, 任何錯誤請大叫你這個瓠瓜給我滾過來處理`   
第一次在本地使用 git 系列工具，可能在 **[ commit & push ]** 這步會遇到**個人資料還未完善**的問題。  
我們將在終端解決這些障礙。即使估以很香但終端一直有種原始的美感...  

先進到剛從 github 抓下來的那個與 repo 同名的本地資料夾中
```
cd ~/(PATHtoREPO)
```

以下是 git 系列的命令  
```
git config user.name "Liv"
git config user.email "universe5961@gmail.com"
```
使用與 github primary email 一樣的郵箱地址可以讓你的 github 頭像出現在 vscode 的 graph 中  
git 認的是**郵件地址**，而不是顯示的名字。

如果你很確定一生就叫這名字了，可以用含有 global 的指令  
（但我才不要）  
```
git config --global user.name "Liv"
git config --global user.email "universe5961@gmail.com"
```
題外話，這些資料會存在 `.git/congig` 中  
可以這樣查看（或修改）
```
cd ~/(PATHtoREPO)
ls -a  # show hidden folders
nano .git/config
```

# 關於建立自己的分支(正在研究中)
如果不特別設定分支的話，你的更新就會在main上做更動，但其他人如果也正在main上更新時整個就會大亂，推上去也容易報錯，因此會建議先設自己的`branch`  
*PS.此部分還在研究...不保證完全正確(其實其他趴也是拉哈哈屁眼)* 
1. 在VS code估以介面中，左下角可以看到`main`，點擊他
2. create new branch
3. 後面應該滿直覺的，可以確認一下左下角變成分支名稱了
4. 一樣是左側第三個圖示，他可能從變成**Publish Branch**，按下去就可以連動到github網站了
5. 你可以去github的網頁看看`Pull requests`那邊有沒有你的東西，有的話也許代表成功
6. 之後更新完就可以把你的分支融合到main裡面，就不會干擾到其他人作業了

# 關於別人新增檔案本地不會更新(正在研究中)
*最暴力的方法就是直接重建一個資料夾、、、*  
VS code左下角有個迴轉箭頭(Synchronize Changes)，如果它顯示向下，代表有新東西拉  
最簡單暴力的方法，還是按左邊第三個圖示，藍色框框顯示`Sync Changes`按下去；此方法會把你的東西推上去，把別人的東西抓下來  
如果你只是要抓別人的檔案，依然是第三個圖示，但這次要按的是上面標題列的`...`，選擇`Pull`
