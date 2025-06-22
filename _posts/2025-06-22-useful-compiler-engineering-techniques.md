---
title: "Useful Legacy Code analysis techniques"
author: avishek
usemathjax: true
tags: ["Reverse Engineering", "Compiler Engineering"]
draft: true
---

_This post has not been written or edited by AI._

## Abstract
We look at the criteria for determining whether a program written in a language which supports unstructured programming constructs (like ```GOTO``` in COBOL), can be translated into structured programming representations. To this end, we will talk of the concept of reducibility.
Beyond that, we look at some of the ways of identifying structures which make a program irreducible. As part of that, we will look at how to identify latent structures like loops in COBOL programs, both reducible and irreducible.

## Control Flow Graph

## Basic Blocks

## Reducibility

## T1-T2 Test for Reducibile Flowgraphs

## Strongly Connected Components

## Dominator Analysis

## Loop Characterisation

## Reducible and Irreducible Loops

## Detecting reducible and irreducible loops

### DJ Trees

## Detecting well-behaved procedures

## References
- [Cobol-REKT](https://github.com/avishek-sen-gupta/cobol-rekt)
- Reducibility
  - [Identifying Loops Using DJ Graphs](https://dl.acm.org/doi/pdf/10.1145/236114.236115)
  - [No improper Strongly Connected Components does not imply Reducibility](https://stackoverflow.com/questions/79036830/if-every-strongly-connected-component-has-only-one-incoming-edge-each-from-outsi)
- Dominator Algorithms
    - [Graph-Theoretic Constructs for Program Control Flow Analysis - Allen and Cocke](https://dominoweb.draco.res.ibm.com/reports/rc3923.pdf)
    - [A Fast Algorithm for Finding Dominators in a Flowgraph - Lengauer and Tarjan](https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/a%20fast%20algorithm%20for%20finding.pdf)
    - [A very readable explanation of the Lengauer-Tarjan algorithm](https://fileadmin.cs.lth.se/cs/education/edan75/F02.pdf)
    - [Algorithms for Finding Dominators in Directed Graphs](https://users-cs.au.dk/gerth/advising/thesis/henrik-knakkegaard-christensen.pdf)
    - [Dominator Tree Certification and Independent Spanning Trees](https://arxiv.org/pdf/1210.8303)
