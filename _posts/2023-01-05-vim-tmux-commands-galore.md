---
title: "Vim and TMux Commands Galore"
author: avishek
usemathjax: true
tags: ["Vim", "Text Editing"]
draft: false
---

This short post lists the Neovim (Vim) shortcuts I am getting used to. I've recently switched to trying the Vim mode for my IDE needs, and having used Vim previously only for very simple tasks, am having a blast practising the basic Vim shortcuts. Ultimately, I will probably move to doing more IDE-related work in native Vim too.

I've also added TMux shortcuts because I'm learning to use that too.

### Vim commands
- ```u```/```<C-r>```: Undo/Redo
- ```.```: Repeat last command
- ```w```/```b```: Move forward/backward by a word
- ```s``` refers to a sentence. Thus ```diw``` and ```daw``` deletes a sentence from anywhere inside it, and everything around the sentence, respectively.
- ```(```/```)```: Jump to previous/next sentence
- ```i```/```a```/```I```/```A```: Start editing before/after cursor, before start/after end of line
  - ```i```: Considers whitespace as words too, so ```i2w``` selects a word and any whitespace after it.
  - ```a```: Considers word + whitespace as a text object, so ```a2w``` selects ```"text1 text2 "```.
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
- ```nG```/```ngg```: Go to line number ```n```
- ```G```: Go to end of file
- ```i```: "Everything inside" qualifier used in conjunction with other verbs, like ```diw```, ```ci"```
- ```<C-v>```: Visual Block Mode, use ```I``` to insert en-masse
- ```<C-o>```/```<C-i>```: Go to old/new positions
- ```<C-u>```/```<C-d>```: Move up/down half a page
- ```{```/```}```: Jump forward/back across a contiguous block of text
- ```+```/```-```: Jump to start of next/previous line

### Ex commands

- ```x,y<Command>z```: Defines an inclusive range of lines from ```x``` to ```y``` and performs ```<Command>``` with optional argument ```z```.
  - ```m```/```t```: Move/Copy range of lines to after ```z```. Example: ```10,20m30```. Single line variants like```10m30``` also work.
- ```x;+/-n```: Defines range of ```+/-n``` starting from line ```x```
- ```.```, ```.+/-n```: Refers to the current line / Refers to ```n``` lines after/before current line.
- ```$```, ```$+/-n```: Refers to the last line of the document (Compare to going to last character in line in Vim's Normal mode). ```+/-n``` navigates ```n``` lines after/before last line.
- ```%```: Refers to all lines (same as ```1,$```)
- ```/pattern/``` and ```?pattern?```: Searches forward and backward for ```pattern```. This can be used as a location argument in other commands.
- ```:<C-p>/:<C-n>```: Moves backwards/forwards through command history.

### TMux commands

- ```<C-b>?```: View all keybindings
- ```<C-b>%```: Horizontal split
- ```<C-b>"```: Vertical split
- ```<C-b><Arrow Keys>```: Moves across TMux panes
- ```<C-b>d```: Detaches from current TMux session
- ```<C-b>[```: Enables scroll mode
- ```<C-b><Space>```: Enables highlight mode after entering scroll mode. Press ```<C-b>``` to yank highlighted text.
- ```<C-b>]```: Pastes copied content to another TMux terminal
- ```tmux ls```: Lists running TMux sessions
- ```tmux attach -t <SessionID>```: Attaches to specified TMux session
- ```tmux rename-session -t <OldSessionID> <NewSessionID>```: Renames a TMux session
- ```<C-b>,```: Renames current window
- ```tmux new -s <SessionID>```: Creates a new TMux session with given ```SessionID```
