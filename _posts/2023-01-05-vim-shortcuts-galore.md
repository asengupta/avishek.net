---
title: "Vim Shortcuts Galore"
author: avishek
usemathjax: true
tags: ["Vim", "Text Editing"]
draft: false
---

This short post lists the Neovim (Vim) shortcuts I am getting used to. I've recently switched to trying the Vim mode for my IDE needs, and having used Vim previously only for very simple tasks, am having a blast practising the basic Vim shortcuts. Ultimately, I will probably move to doing more IDE-related work in native Vim too.

- ```u```/```<C-r>```: Undo/Redo
- ```.```: Repeat last command
- ```w```/```b```: Move forward/backward by a word
- ```i```/```a```/```I```/```A```: Start editing before/after cursor, before start/after end of line
- ```0```/```_```/```$```: Go to starting character / starting non-whitespace character / end of line
- ```d```: Delete (suffix with counter and text object, like ```d2w```, ```dd```) 
- ```c```: Change (suffix with counter and text object, like ```c2w```, ```cc```) 
- ```r```: Replace (suffix with counter and text object, like ```r2w```, ```rr```)
- ```y```: Yank (suffix with counter and text object, like ```y2w```, ```yy```) 
    - **Interesting Use Case**: ```ny$``` yanks from cursor to end of line ```n``` times, so ```n``` lines, starting from the current cursor position.
- ```F-x```/```f-x```: Find character ```x``` before/after
- ```*```: Search forward word under cursor
- ```/``` and ```?```: Find string forward/backward
- ```P```/```p```: Paste before/after cursor
- ```x```: Delete character under cursor
- ```n + G```: Go to line number ```n```
- ```i```: "Everything inside" qualifier used in conjunction with other verbs, like ```diw```, ```ci"```
- ```<C-v>```: Visual Block Mode, use ```I``` to insert en-masse
- ```<C-o>```/```<C-i>```: Go to old/new positions
- ```<C-u>```/```<C-d>```: Move up/down half a page
- ```{```/```}```: Jump forward/back across a contiguous block of text
- ```+```/```-```: Jump to start of next/previous line
