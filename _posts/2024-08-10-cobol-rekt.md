---
title: "Experiments in COBOL Transpilation"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Reverse Engineering", "COBOL"]
draft: true
---

_This post has not been written or edited by AI._

## Contents

- [Introduction](#introduction)
- [Parser](#parser)
- Control Flow Graph (early version)
- [Intermediate representation](#intermediate-representation)
  - [Translation into Syntax Tree](#translation-into-syntax-tree)
  - Translation into Control Flowgraph
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

## Control Flow Graph (early version)

A very naive version of a flowgraph can be built which is still tied to COBOL constructs. The nodes in this flowgraph represent native COBOL constructs. These nodes were originally designed to represent COBOL code as visual flowcharts, like below:

![Example flowchart](/assets/images/example-flowchart.png)

These nodes did not necessarily represent control flow very accurately, because of the following:

- There were always straight-line causal connections between every COBOL sentence. This included statements which unambiguously jumped to other sections/paragraphs of the program. This is because se

## Intermediate representation

### Translation into Syntax Tree
A more language-agnostic way of representing control flows (as well as syntax), is to decompose all COBOL-specific constructs into some simple intermediate representation which is more or less well-supported in most structured programming languages. Examples of such constructs would be:

- IF...THEN...ELSE
- WHILE
- DO...WHILE
- SET
- LOOP (basically, a specialised expression of a looping construct)

The ```LOOP``` semantic is technically not needed, but it does provide a nice way to encapsulate the semantics of a loop (initial value, termination condition, loop update), especially given COBOL has statements like ```PERFORM INLINE```.

A couple of examples of this translation process are shown below.

### Translating ```EVALUATE```
```
EVALUATE TRUE ALSO TRUE
              WHEN SCALED + RESULT < 10 ALSO INVOICE-AMOUNT = 10
                MOVE "CASE 1" TO SOMETHING
              WHEN SCALED + RESULT > 50 ALSO
                INVOICE-AMOUNT = ( SOMETEXT + RESULT ) / SCALED
                MOVE "CASE 2" TO SOMETHING
              WHEN OTHER
                MOVE "CASE OTHER" TO SOMETHING
            END-EVALUATE
```

This becomes:

```
if(and(eq(primitive(true), lt(add(ref('SCALED'), ref('RESULT')), primitive(10.0))), eq(primitive(true), eq(ref('INVOICE-AMOUNT'), primitive(10.0))))) 
 then 
{
	CODE_BLOCK: CODE_BLOCK: set(ref('SOMETHING'), value(primitive("CASE 1"))) 
}
 
else 
{
	if(and(eq(primitive(true), gt(add(ref('SCALED'), ref('RESULT')), primitive(50.0))), eq(primitive(true), eq(ref('INVOICE-AMOUNT'), divide(add(ref('SOMETEXT'), ref('RESULT')), ref('SCALED')))))) 
	 then 
	{
		 CODE_BLOCK: CODE_BLOCK: set(ref('SOMETHING'), value(primitive("CASE 2"))) 
	}
	 
	else 
	{
		 CODE_BLOCK: CODE_BLOCK: set(ref('SOMETHING'), value(primitive("CASE OTHER"))) 
	}
}
```

### Translating ```PERFORM INLINE```

```
PERFORM TEST BEFORE VARYING SOME-PART-1 FROM 1 BY 1
UNTIL SOME-PART-1 > 10
AFTER SOME-PART-2 FROM 1 BY 1 UNTIL SOME-PART-2 > 10
    DISPLAY "GOING " SOME-PART-1 " AND " SOME-PART-2
END-PERFORM.
```

This becomes:

```
loop[loopVariable=ref('SOME-PART-1'), initialValue=primitive(1.0), maxValue=NULL, terminateCondition=gt(ref('SOME-PART-1'), primitive(10.0)), loopUpdate=primitive(1.0), conditionTestTime=BEFORE] 
{
	loop[loopVariable=ref('SOME-PART-2'), initialValue=primitive(1.0), maxValue=NULL, terminateCondition=gt(ref('SOME-PART-2'), primitive(10.0)), loopUpdate=primitive(1.0), conditionTestTime=BEFORE] 
	{
		CODE_BLOCK: print(value(primitive("GOING ")), value(ref('SOME-PART-1')), value(primitive(" AND ")), value(ref('SOME-PART-2')))
	}
}
```

The translation process results in a syntax tree expressed in this intermediate form. Note that ```GO TO``` statements carry over as ```JUMP``` statements; thus, the resulting intermediate program may still not be well-structured enough to be expressible in a modern progreamming language.



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

