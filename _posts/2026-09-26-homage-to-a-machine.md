---
title: "Homage to a Machine"
author: avishek
usemathjax: false
mermaid: false
tags: ["Personal", "Hardware", "Software Engineering"]
---

It arrived with Big Sur and it is being wiped running macOS 26.7 — five years and eleven
major versions on hardware that was supposed to be a first-generation experiment. Apple's
first real silicon, and I gave it COBOL.

It never got a clean start. The first thing it inherited was `/usr/local/Homebrew`,
migrated whole from an Intel machine and already obsolete on arrival. That install stopped
being able to run years ago — `unknown or unsupported macOS version` — and its hundred-odd
formulae have sat there ever since like a previous tenant's furniture. `pandoc` 2.11 from
2021. `gradle` 6.8. A `postgresql@11` that is *still running as a login service*, still
holding live data, quietly serving a machine that moved on to `/opt/homebrew` without ever
telling it.

What it actually did, it did at scale. 344 Homebrew formulae. Seven JDKs, from
AdoptOpenJDK 8 through Temurin 24. Six Pythons under pyenv, two Rubies under rbenv, and RVM
underneath *those* like sediment. OCaml, Lean, Haskell, Rust, Clojure, Scala, Prolog in
three dialects — gnu, SWI, and Scryer. A z390 emulator. An IBM 3270 terminal. Db2 in Docker
on QEMU, emulating x86 on ARM to run a mainframe database, which is roughly the most 2020s
sentence it is possible to write.

Ninety-one git repositories in `~/code`. Red Dragon and its forge, cobble, smojol, codescry,
tape-z, revenger, zfa — an entire body of work on reading dead languages and telling you what
they meant. Tree-sitter grammars for VAX Pascal, PickBASIC, ABAP, Natural, HLASM: languages
most people would say aren't worth the parser. It built them anyway. Somewhere in there is a
grammar called `weird` whose only rule is `func lol`, which proves it had a sense of humour
about all this.

It watched an MSc happen. The LJMU thesis, the transcripts, the certificate, the KSOU
application before that.

It kept two receipts of self-improvement: `.zshrc.omz-uninstalled-2024-10-24` and
`.zshrc.omz-uninstalled-2026-04-02`. Oh My Zsh, installed and renounced, twice. Seven
`.zshrc` backups in total, a sedimentary record of someone who kept meaning to tidy up and
kept finding something more interesting to do instead. That's not a criticism. The tidying
is happening now, at the end, which is when it usually happens.

It was never backed up. Not once — `tmutil` has no destinations and never did. For five
years everything lived here and nowhere else. It held all of it without complaint and lost
none of it. The rest of us should be so reliable on zero redundancy.

Last week it was still working: a call-graph build loading 53,313 classes and running CHA
across 852,628 entry points, the log scrolling past at 16:16 on a Sunday, photographed off
the screen because that was faster than taking a screenshot. Five-year-old laptop,
whole-program analysis, no complaints. Days ago it took a constant-propagation spike across
88 CICS call sites and answered honestly: zero resolved under strict propagation, 59 under a
rule the lattice cannot express. Still producing real results on the way out the door.

Wipe it clean. It earned the rest.
