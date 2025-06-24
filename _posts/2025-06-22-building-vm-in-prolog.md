---
title: "Building a simple Virtual Machine in Prolog"
author: avishek
usemathjax: false
tags: ["Prolog", "Logic Programming", "Virtual Machine", "Symbolic Execution"]
draft: true
---

_This post has not been written or edited by AI._

## Abstract
In this post, I'll talk about how I wrote a small Virtual Machine in Prolog which can both interpret concrete assembly language-like programs, and run symbolic executions, which is useful in data flow analysis of programs.

## Why Prolog?
I've always been interested in the evolution of AI, and the potential for a synthesis of older techniques (formal and otherwise) and newer ones (deep learning). My sense tells me that combining many of these approaches will only serve to make application of AI more "robust". I use that term in a rather loose sense for the moment. One interpretation might be that to make something robust is to make it more "deterministic" or "reproducible", but this can only cover a subset of qualities of such a system. However, automated reasoning is a principled, methodical way of exploring a subspace of a larger, more ill-defined problem space: hence my interest in this space. After all, this early AI research did give us Lisp :-)

Prolog was the European answer to automating the reasoning process in the 1960's (the American one was Lisp). It belongs to the family of logic programming languages, which is a paradigm distinct from the well-known imperative/functional/object-oriented ones. There are several ideas in here that appealed to me. Not surprisingly, Prolog and its other subsets and derivative approaches (Datalog, Answer Set Programming, etc.) have been used in applications involving different types of reasoning tasks.

From my reading, there are several kinds of reasoning, and the categorisation depends upon different parameters. I only list the types of reasoning which interest me.

- **Deductive:** Incontrovertible conclusions from facts. This is what base Prolog does.
- **Inductive:** Plausible "general" facts from observed facts. Inductive Logic Programming (ILP) systems like Popper build on top of Prolog to provide such capabilities.
- **Abductive:** Generalised "probable" rules from facts. Libraries like ILASP and Potassco's `clingo` provide capabilities like Answer Set Programming.
- **Non-monotonic:** Facts are tentative, and can be invalidated in the face of new facts. ILASP, etc. support non-monotonic reasoning.
- **Probabilistic Logic:** Facts are assigned probabilities, conclusions are probabilistic in nature. Probabilistic Graphical Models embody such logic.

## A very brief introduction to Prolog

Prolog's way of thinking is close to - but not the same as - functional programming. Briefly, these are some of the salient points.

- You don't write statements. You write **facts** and **predicates**. These relate _things_ to other _things_.
- Facts hold unconditionally, i.e., they are _true_.
- Any fact not defined as treated as _false_.
- Predicates hold conditionally, based on whether other predicates or facts hold or not.
- The actual act of computation happens when asserting the truth of these predicates.
- Prolog is superb at pattern matching and symbolic manipulation. This comes partly from the unification mechanism. Personally, I have not seen this level of capability at a language level in any other language I have seen (including OCaml, which is another language I'm currently learning).

So, here's a simple directed graph in Prolog.

```prolog
node(a).
node(b).
node(c).
node(d).
node(e).
node(f).

edge(a,b).
edge(b,c).
edge(b,d).
edge(c,e).
edge(d,e).
```

...and, that's it. Facts (and predicates) can be any combination of letters/numbers (the exact rules are obviously more rigorous, but if you can name a variable in your favourite programming language, you can probably use it as a fact name). The only constraint is that only **variables** can start with an uppercase letter.

Also note that there were no quotes around `a`, `b`, `c`, etc. They are **atoms**, which are basically freeform symbols, that you can use in lieu of string (strings are a different type).

Let's write a simple predicate. This predicate will hold true if `a` is a node.

```prolog
a_exists :- node(a).
```
...and, that's it. Internally, it checks whether the fact `node(a)` exists. Read the `:-` as **'LHS is true if RHS is true'**.

Suppose we want to test if a node with a symbol of our choosing exists. We write:

```prolog
node_exists(N) :- node(N).
```

Here, note that `N` is a variable which we bind to a 'thing' when asking the question. You can also not bind it, but that's a different kind of question, which we will get to in a bit.

Suppose we wish to check if a node has multiple outgoing edges. We can write:

```prolog
has_multiple_outgoing_edges(N) :- edge(N,A), edge(N,B), A \= B.
```

The `,` in this case represents the logical AND. This means that `has_multiple_outgoing_edges(N)` is true if:

- There is an edge from `N` to `A`
- There is an edge from `N` to `B`
- `A` is not the same as `B`. `\=` represents **"not equal to"**.

Note that we have replaced the `a` with `N` which is a variable, which implies that `node_exists` takes in a **thing**. It is important to note that thing can really be anything: an atom, a string, another predicate, etc. As long as Prolog can find a rule which matches the sort of thing we inject into `node_exists`, it will consider this rule for further evaluation.

Now you can ask the Prolog interpreter:

```
?- node_exists(a).
true.

?- node_exists(efgh).
false.
```

Let's check if a node is unconnected or not. We can write:

```prolog
is_unconnected(N) :- \+ edge(N,_).
```

The `\+` negates a condition (logical NOT). The `_` implies that there is a value there, but we don't care about the value enough to put it into a variable.

So, we can try:

```
?- is_unconnected(f).
true.

?- is_unconnected(a).
false.
```

So far, we have been doing forward inference. But we can also ask other questions of our facts. Suppose we want to know which are the outgoing edges of `b`. In any other programming language, you'd want to check the list of outgoing edges from `b`, and so on. In Prolog, you can simply ask for this information.

```
?- edge(b,N).
N = c ;
N = d.
```

No extra programming needed. This is because Prolog leverages the concepts of unification, backtracking and goal resolution.

Here are the last two examples, before we talk about **structural matching**. The first one prints all the elements of a list. Like in functional programming languages, lists are a core supported data structure in Prolog and their heads and tails is the canonical way of representing them. So, we will be using recursive rules.

```prolog
printall([]).
printall([H|T]) :- writeln(H), printall(T).
```

The first rule is the base recursive rule, for when we have an empty list. The second one is the general rule, which separates out a list into its head and tail (car, cdr), writes out the head, and calls `printall` on the tail.

As one last example, let's reverse a list. 

```prolog
reverse2([],Acc,Acc).
reverse2([H|T],Acc,Result) :- reverse2(T,[H|Acc],Result),!.
```

The first argument is the list being passed in (and gradually getting decomposed into successive heads and tails). The second one is the accumulator which keeps getting built up with the result (it always `[]` initially). The final argument is the actual result to which the reversed list will be bound to.

Predicates may look like functions, but they are only ever `true` or `false`. The results of any computation are always bound to any unresolved variables that you specify when invoking them. In this case, `Result` is the unresolved variable.

- The first rule is the simple one: if the list is empty, it binds the third parameter (the result) to whatever the accumulator is at that point. This is an example of unification: very simplistically, you don't explicitly assign values to variable, specifying the value in the slot where a variable sits, is enough to bind it to the variable.
- The second rule once again keeps recursively decomposing the original list, but at the point of recursion, it adds the head to whatever the accumulator is (remember, appending in this case happens at the front of the list). The `!` is called a cut operator, and in this case is not strictly needed for forward inference, but I have it there to demonstrate backward inference, so you can technically ignore it for the moment.

Thus, if we pass in a list `[1,2,3,4,5]`, the accumulator will be appended to (in the front) with `[1]`, `[2,1]`, `[3,2,1]`, `[4,3,2,1]`, and `[5,4,3,2,1]` on each successive recursive call. Also note that I use the term 'append' rather loosely, since there is no mutation: values in Prolog are always immutable.

So we can try:

```
?- reverse2([1,2,3,4,5],[],Reversed).
Reversed = [5, 4, 3, 2, 1].
```

The amazing part is that you can **reverse** this operation, i.e., ask Prolog: if I gave you the reversed list, what was the original list?

```
?- reverse2(Original,[],[5,4,3,2,1]).
Original = [1, 2, 3, 4, 5].
```

## Atomic Operations from the ground-up

## Minimal instruction set

## Registers, Flags, and Pointers

## The concrete interpreter

## Aside: Logging

## Symbolic Execution and World Splits

## Prolog as a Modelling Language

## Unification is so powerful

## References
- [Symbolic Interpreter](https://github.com/asengupta/prolog-exercises/blob/main/ilp/prolog_examples/symbolic_executor.pl)
