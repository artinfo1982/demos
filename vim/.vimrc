filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Bundle 'scrooloose/nerdtree'
Bundle 'octol/vim-cpp-enhanced-highlight'
Bundle 'Valloric/YouCompleteMe'
Bundle 'taglist.vim'
call vundle#end()
filetype plugin indent on
syntax enable
syntax on
filetype on
filetype plugin on
filetype indent on
inoremap #ifn #ifndef<CR>#endif
inoremap #inc #include <>
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
set tags+=/home/cd/tensorflow-core/tags
let g:ycm_global_ycm_extra_conf='/home/cd/.vim/.ycm_extra_conf.py'
let g:ycm_enable_diagnostic_signs=0
let g:ycm_enable_diagnostic_highlighting=0
let Tlist_Ctags_Cmd='ctags'
let Tlist_Show_One_File=1
let Tlist_File_Fold_Auto_Close=1
let Tlist_WinWidt=32
let Tlist_Exit_OnlyWindow=1
let Tlist_Use_Right_Window=1
let Tlist_Sort_Type='name'
let Tlist_GainFocus_On_ToggleOpen=1
map <F5> :call CompileRunPython()<CR>
function! CompileRunPython()
    exec "w"
    if &filetype == 'python'
        exec '!python3 %'
    endif
endfunction
map <F1> :NERDTree<CR>
map <F2> :TlistToggle<CR>
map <F9> 20zh
imap <F9> <ESC>20zhi
map <F10> 20zl
imap <F10> <ESC>20zli
