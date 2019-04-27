filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Bundle 'scrooloose/nerdtree'
Bundle 'octol/vim-cpp-enhanced-highlight'
Bundle 'Valloric/YouCompleteMe'
call vundle#end()
filetype plugin indent on
syntax enable
syntax on
filetype on
filetype plugin on
filetype indent on
:inoremap ( ()<ESC>i
:inoremap ) <c-r>=ClosePair(')')<CR>
:inoremap { {<CR>}<ESC>0
:inoremap } <c-r>=ClosePair('}')<CR>
:inoremap [ []<ESC>i
:inoremap ] <c-r>ClosePair(']')<CR>
function! ClosePair(char)
    if getline('.')[col('.') - 1] == a:char
        return "\<Right>"
    else
        return a:char
    endif
endfunction
set completeopt=longest,menu
set number
set smartindent
set tabstop=4
set shiftwidth=4
set background=dark
set cursorline
set ruler
set t_Co=256
set incsearch
set ignorecase
set laststatus=2
set showmatch
set hlsearch
set nowrap
set foldmethod=syntax
set nofoldenable
set tags+=/home/zxh/tensorflow-core/tags
let g:ycm_global_ycm_extra_conf='/home/zxh/.vim/.ycm_extra_conf.py'
let g:ycm_enable_diagnostic_signs=0
let g:ycm_enable_diagnostic_highlighting=0
map <F5> :call CompileRunPython()<CR>
function! CompileRunPython()
    exec "w"
    if &filetype == 'python'
        exec '!python3 %'
    endif
endfunction
map <F1> :NERDTree<CR>
map <F9> 20zh
imap <F9> <ESC>20zhi
map <F10> 20zl
imap <F10> <ESC>20zli