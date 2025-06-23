---
title: "Building an HLASM grammar from scratch"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Reverse Engineering", "HLASM", "ANTLR"]
draft: true
---

_This post has not been written or edited by AI._

## Abstract
This post talks about the technique to build an ANTLR grammar for HLASM (mainframe assembler) from scratch, without handwriting the entire instruction set. The technique creates a parser which reads a table of instruction formats from IBM's official documentation, and automates the creation of the actual HLASM grammar based on these instruction formats.

## Existing parsers

## Practical difficulties of hand-writing a large grammar

## The instruction format Meta-Parser

## The HLASM grammar generator

![HLASM Parser/Meta-Parser](/assets/images/tapez-hlasm-parser-metaparser.png)

## Current Limitations

## References

- [Table of all supported HLASM instructions](https://www.ibm.com/docs/en/hla-and-tf/1.6.0?topic=instructions-table-all-supported)
- [Tape/Z](https://github.com/avishek-sen-gupta/tape-z)
