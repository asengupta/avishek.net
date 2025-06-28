---
title: "Building a simple Virtual Machine in Prolog"
author: avishek
usemathjax: false
tags: ["Prolog", "Logic Programming", "Virtual Machine", "Symbolic Execution"]
draft: true
---

## Abstract
In this post, I'll talk about how I wrote a small Virtual Machine in Prolog which can both interpret concrete assembly language-like programs, and run symbolic executions, which is useful in data flow analysis of programs.

_This post has not been written or edited by AI._

## Building a simple Virtual Machine


## Atomic Operations from the ground-up

We need a map implementation. SWI-Prolog has the dictionaries, but since we are building everything from scratch, we will write a very naive implementation using only lists. Granted, there are some semantics of a dictionary that can be violated for now - for example, you can start off with duplicate keys, but let's assume the happy path.

```prolog
get2(_,[],empty).
get2(K, [(-(K,VX))|_],VX) :- !.
get2(K, [_|T],R) :- get2(K,T,R).

put2_(-(K,V),[],Replaced,R) :- Replaced->R=[];R=[-(K,V)].
put2_(-(K,V),[-(K,_)|T],_,[-(K,V)|RX]) :- put2_(-(K,V),T,true,RX).
put2_(-(K,V),[H|T],Replaced,[H|RX]) :- put2_(-(K,V),T,Replaced,RX).

put2(-(K,V),Map,R) :- put2_(-(K,V),Map,false,R).
```

To represent entries in a dictionary, we use the `K-V` compound term, which is basically syntax sugar for `-(K,V)`. These entries live inside a list. Both `get2` and `put2` behave in predictable ways, except when the dictionary has duplicate keys. In that case:

- `get2(K,V)` returns the value of the first matching key.
- `put2(-(K,V),InputMap,OutputMap)` modifies all matching keys with the value `V`.

In our current implementation, we will not worry about duplicate entries yet.

We will also need push/pop operations on stacks. This is very simple. Note that the top of the stack is always the leftmost element.

```prolog
push_(V,Stack,UpdatedStack) :- UpdatedStack=[V|Stack].
pop_([],empty,[]).
pop_([H|Rest],H,Rest).
```


## Minimal instruction set

## Registers, Flags, and Pointers

## The concrete interpreter

## Aside: Logging

## Symbolic Execution and World Splits

## Prolog as a Modelling Language

## Unification is so powerful

## References
- [Symbolic Interpreter](https://github.com/asengupta/prolog-exercises/blob/main/ilp/prolog_examples/symbolic_executor.pl)
