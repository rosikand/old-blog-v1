---
layout:     post
title:      "My Current LaTeX Workflow: Using LaTeX with VS Code"
date:       2021-05-04 12:01:00
author:     "Rohan Sikand"
header-style: text 
catalog: true
---

## Backstory (for those interested)

I first heard of LaTeX back in high school when I began writing technical reports (lab reports, papers, notes, etc.). Back then, I was using the ever-so-common [Overleaf](https://www.overleaf.com/). I absolutely loved the PDF output produced, but I didn't like the difficulty of use (probably everyone's thoughts after using LaTeX for the first time). So, going forward, I mainly used Google Docs or Notion for writing most things (> 90%)  and only used LaTeX through Overleaf for writing things that needed to be formally typeset. Fast forward to this quarter, we are required to use LaTeX in my *CS 103: Mathematical Foundations for Computing* class for our problem set solutions. They gave us a starter template to use and already specified a bunch of commands for us so actually writing the TeX source code wasn't a problem. What was a problem though, was the overall LaTeX setup—namely the relatively slow speed it takes Overleaf to compile and update the document (and such speeds increase with the length of the document it seems). Additionally, Overleaf runs through the web - not natively - so you are trusting Overleaf to keep your files. 

## The solution

**Overall, I needed these features:**

- Fast compile time to PDF output.
- Semi-autocompile (more optional than necessary but good to have).
    - Note: by semi-autocompile I mean either real time live updates when you type (unlikely to find such a thing with LaTeX) or, more realistically, when saving the file.
- Autocomplete.
- Nice editor (smooth, easy on the eye, customizable size of text).
- PDF preview (important, but can be a different application).
- Runs natively (optional)
    - Also, I am not a Linux or Unix maestro by any means so anything that required Linux (i.e. [Gummi](https://gummi.app/)) wasn't going to work. More specifically, I needed something that would work on Mac OS X.

With those requirements, I did some digging: it seems that the best option at the moment is to pair **VS Code + LaTeX** **extensions** with a Mac OS X LaTeX distribution (namely [**MacTex**](https://tug.org/mactex/)). Such a setup is much easier to get started with than most of the other technical setups (i.e. using [Vim](https://www.vim.org/) with [Zathura](https://pwmt.org/projects/zathura/)). 

**My current LaTeX workflow is as follows:**

- VS Code as my editor. The extensions that make this work are [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) (which powers most of the stuff, such as the PDF viewer) and [LaTeX language support](https://marketplace.visualstudio.com/items?itemName=torn4dom4n.latex-support).
- [Mathpix](https://mathpix.com/) for snapping equations from lecture slides and such. Though, you are only offered a limited number of snips unless you pay a monthly subscription fee.
    - Also occasionally use [this](https://www.codecogs.com/latex/eqneditor.php).
- [MacTex](https://tug.org/mactex/) for native language support.
- [Evan Chen's style file](https://github.com/vEnhance/dotfiles/blob/main/texmf/tex/latex/evan/evan.sty) which I use for my notes.

## **Future**

- Drawing figures using TikZ, PSTricks, Asymptote, [Xy](https://tug.org/applications/Xy-pic/) (even [typesetting automata](https://web.ma.utexas.edu/users/a.debray/lecture_notes/using_xy.pdf)!) or [Inkscape](https://castel.dev/post/lecture-notes-2/).
- Pre-defined [snippets](https://castel.dev/post/lecture-notes-1/) 
- Something that will allow me to add images easily.
- Notion or Markdown to LaTeX converter (that actually works for my purposes).
- Writing LaTeX on this blog. 
- Make a machine learning model that has the functionalities of Mathpix.
    - Maybe this can even be extendable to [handwritten math](https://rosikand.github.io/projects/mse/document.html) too!