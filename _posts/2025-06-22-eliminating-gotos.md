---
title: "Techniques of Reverse Engineering: Eliminating GOTOs"
author: avishek
usemathjax: false
tags: ["Reverse Engineering", "GOTO"]
draft: true
---

## Abstract
In this post, we talk of how to eliminate unconditional arbitrary jumps (aka GOTOs) by manipulating the Abstract Syntax Tree of a program, in a way that preserves the semantic flow of the program, while making it amenable to structured programming representations (i.e., using control flow constructs like ```while```, ```do...while```). The aim is to transform the program into a form which can make it easier to represent in modern programming languages which use only structured programming constructs.

_This post has not been written or edited by AI._

## The basic principle

## Case 1: Forward Jumps

## Case 2: Backward Jumps

## References
- [Cobol-REKT](https://github.com/avishek-sen-gupta/cobol-rekt)
- [Structured Program Theorem](https://en.wikipedia.org/wiki/Structured_program_theorem)
- [Taming Control Flow: A Structured Approach to Eliminating Goto Statements](https://www.researchgate.net/publication/2609386_Taming_Control_Flow_A_Structured_Approach_to_Eliminating_Goto_Statements)
