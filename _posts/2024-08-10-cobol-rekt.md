---
title: "Experiments in COBOL Transpilation"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Reverse Engineering", "COBOL"]
draft: true
---

## Contents

- [Introduction](#introduction)
- [Parser](#parser)
- Control Flow Graph (early version)
- Intermediate representation
  - Translations
- Control Flow Analysis
  - What is reducibility?
  - Tests for reducibility
  - Tests for irreducible regions
  - Future work: Controlled Node Splitting
- Semantics-preserving tree transformations
  - Eliminating GOTOs
- [References](#references)

## Introduction

The source code for all of this work is based on [Cobol-REKT](https://github.com/avishek-sen-gupta/cobol-rekt).

## Parser

The parser is the easier part, if only because I reused what was already available. The [COBOL Language Support](https://github.com/eclipse-che4z/che-che4z-lsp-for-cobol) extension contains an ANTLR grammar which works. Note that it has been en

## References

- Books
  - [Advanced Compiler Design and Implementation by Steven Muchnik](https://www.amazon.in/Advanced-Compiler-Design-Implementation-Muchnick/dp/1558603204)
  - [Compilers: Principles, Techniques, and Tools by Aho, Sethi, Ullman](https://www.amazon.in/Compilers-Principles-Techniques-Tools-Updated/dp/9357054111/)
- General Overview
  - [Structured Program Theorem](https://en.wikipedia.org/wiki/Structured_program_theorem)
  - [Control Flow Analysis slides](http://www.cse.iitm.ac.in/~krishna/cs6013/lecture4.pdf)
- Structural Transformations
  - [Taming Control Flow: A Structured Approach to Eliminating Goto Statements](https://www.researchgate.net/publication/2609386_Taming_Control_Flow_A_Structured_Approach_to_Eliminating_Goto_Statements)
  - [Solving the structured control flow problem once and for all](https://medium.com/leaningtech/solving-the-structured-control-flow-problem-once-and-for-all-5123117b1ee2)
- Dominator Algorithms
    - [Graph-Theoretic Constructs for Program Control Flow Analysis - Allen and Cocke](https://dominoweb.draco.res.ibm.com/reports/rc3923.pdf)
    - [A Fast Algorithm for Finding Dominators in a Flowgraph - Lengauer and Tarjan](https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/a%20fast%20algorithm%20for%20finding.pdf)
    - [A very readable explanation of the Lengauer-Tarjan algorithm](https://fileadmin.cs.lth.se/cs/education/edan75/F02.pdf)
    - [Algorithms for Finding Dominators in Directed Graphs](https://users-cs.au.dk/gerth/advising/thesis/henrik-knakkegaard-christensen.pdf)
- Reducibility
    - [Making Graphs Reducible with Controlled Node Splitting](https://dl.acm.org/doi/pdf/10.1145/267959.269971)
    - [Eliminating go to’s while Preserving Program Structure](https://dl.acm.org/doi/pdf/10.1145/48014.48021)
    - [No More Gotos: Decompilation Using Pattern-Independent Control-Flow Structuring and Semantics-Preserving Transformations](https://github.com/lifting-bits/rellic/blob/master/docs/NoMoreGotos.pdf)
    - [Proper Strongly Connected Components do not imply Reducibility](https://stackoverflow.com/questions/79036830/if-every-strongly-connected-component-has-only-one-incoming-edge-each-from-outsi)
- COBOL References
    - [Examples: numeric data and internal representation](https://www.ibm.com/docs/sk/cobol-zos/6.3?topic=data-examples-numeric-internal-representation)
    - [Enterprise Cobol for Z/OS 6.4 - Language Reference](https://publibfp.dhe.ibm.com/epubs/pdf/igy6lr40.pdf)

