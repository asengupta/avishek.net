---
title: "Building a simple Virtual Machine in Prolog"
author: avishek
usemathjax: false
tags: ["Prolog", "Logic Programming", "Virtual Machine", "Symbolic Execution"]
draft: true
---

In this post, I'll talk about how I wrote a small Virtual Machine in Prolog which can both interpret concrete assembly language-like programs, and run symbolic executions, which is useful in data flow analysis of programs.

_This post has not been written or edited by AI._

## Building a simple Virtual Machine


## Foundational Operations from the ground-up

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

To represent entries in a dictionary, we use the `K-V` compound term, which is basically syntactic sugar for `-(K,V)`. These entries live inside a list. Both `get2` and `put2` behave in predictable ways, except when the dictionary has duplicate keys. In that case:

- `get2(K,V)` returns the value of the first matching key.
- `put2(-(K,V),InputMap,OutputMap)` modifies all matching keys with the value `V`.

In our current implementation, we will not worry about duplicate entries yet.

We will also need push/pop operations on stacks. This is very simple. Note that the top of the stack is always the leftmost element.

```prolog
push_(V,Stack,UpdatedStack) :- UpdatedStack=[V|Stack].
pop_([],empty,[]).
pop_([H|Rest],H,Rest).
```

## Logging

We will be logging quite a bit inside the rules. Thus it is important to have a structured way of logging different levels, like `DEBUG`, `INFO`, `WARNING`, etc. This is what a basic logging setup looks like:

```prolog
log_with_level(LogLevel,FormatString,Args) :- format(string(Message),FormatString,Args),format('[~w]: ~w~n',[LogLevel,Message]).

debug(Message) :- log_with_level('DEBUG',Message,[]).
debug(FormatString,Args) :- log_with_level('DEBUG',FormatString,Args).

info(Message) :- log_with_level('INFO',Message,[]).
info(FormatString,Args) :- log_with_level('INFO',FormatString,Args).

warning(Message) :- log_with_level('WARN',Message,[]).
warning(FormatString,Args) :- log_with_level('WARN',FormatString,Args).

error(Message) :- log_with_level('ERROR',Message,[]).
error(FormatString,Args) :- log_with_level('ERROR',FormatString,Args).

dont_log(_).
dont_log(_,_).
```

## Minimal instruction set

The minimal instruction is comprised of the following:

- `mov(reg,reg|constant)`
- `cmp(reg,reg|constant)`
- `label(name)`
- `j(label)`
- `jz(label|address)`
- `jnz(label|address)`
- `push(reg|constant)`
- `pop(reg)`
- `call(label)`
- `ret`
- `hlt`
- `term(string)`
- `nop`
- `inc(reg)`
- `dec(reg)`
- `mul(reg,reg|constant)`

## Registers, Flags, and other Data Structures

We will not have a fixed number of registers for convenience, and thus you can use any symbol as a register. In this respect, we will be treating registers more akin to conventional variables.

There will be one special register called the Instruction Pointer (IP). This will point to the next instruction to be executed. Jump instructions like `j`, `jnz`, and `jz` can can modify the IP to change the flow of the program.

The other useful data structure will be the stack, which is operated by `push`, `pop`, `call`, and `ret` (the last two use it to keep track of the stack when entering and leaving procedures).

There will be one flag called the Zero Flag. This should probably be better named to Equals Flag, because it is set to zero if the two sides of a `cmp` are equal, otherwise -1/+1 depending upon their relative ordering.

## Building the navigation maps


## The concrete interpreter

## Aside: Logging

## Symbolic Execution and World Splits

## Prolog as a Modelling Language

## Unification is so powerful

## References
- [Symbolic Interpreter](https://github.com/asengupta/prolog-exercises/blob/main/ilp/prolog_examples/symbolic_executor.pl)
