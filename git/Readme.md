# git常用命令

## 添加proxy
```shell
git config --global http.proxy http://username:passwd@ip:port
git config --global https.proxy https://username:passwd@ip:port
```

## 取消SSL校验
```shell
git config --global http.sslVerify false
```

## 克隆
```shell
git clone https://xxxx/xxx.git
```

## 将远程仓库的代码同步到本地
```shell
git pull
```

## 提交本地代码到远程仓库
```shell
git add .
git commit -m '本次提交的若干说明'
git push
```
