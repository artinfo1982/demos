# 将vim打造为IDE

## 安装Vundle
Vundle是一款vim的插件管理器。   
```shell
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```
## 使用Vundle安装插件
安装一般的插件：   
vim进去后，输入 :PluginInstall   
编译ycm   
```text
先下载llvm，下载地址：
http://releases.llvm.org/download.html#4.0.0
文件：
clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-14.04.tar.xz
```
```shell
cd ~/.vim/bundle/YouCompleteMe
./install.sh --clang-completer
```
## 生成ctags
```shell
ctags -R aaa/ bbb/
``` 
