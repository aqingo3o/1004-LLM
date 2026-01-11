# Start with the NEW server - blackhole
一些普通實用小指令

## ssh
只能使用命令介面 (terminal) 的遠端連線方式  
雖然沒有估以 但勝在穩定又免費

Use **ssh** to start a remote connection. 
```
ssh ooo@140.xxx.xx.xx
```
This command should be type in:
- terminal (Linux/Mac)
- Windows PowerShell (Wins) with 管理員權限  

Remember to replace **"ooo"** to your own user name!  
Full IP address can be find on white boaard in 1004(實體).

Confirm everything that follows.
```
yes/accept/y
```

And enter your personal password.

You can use this command to terminate the remote connection.
```
exit
```

## Observing GPU
Use this command to get more GPU(nvidia) information.
```
nvidia-smi
```
Or more, this command can automatically update the GPU-info once a second.
```
watch -n 1 nvidia-smi
```

