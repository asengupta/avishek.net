---
title: "A brief overview of Prolog"
author: avishek
usemathjax: false
tags: ["Prolog", "Logic Programming"]
draft: true
---

_This post has not been written or edited by AI._

## Abstract
In this post, I'll give a brief overview of Prolog and the paradigm of Logic Programming. I'll discuss why I think it makes for such a powerful domain modelling language, and a gateway into the techniques of automated symbolic reasoning.

## Why Prolog?
I've always been interested in the evolution of AI, and the potential for a synthesis of older techniques (formal and otherwise) and newer ones (deep learning). My sense tells me that combining many of these approaches will only serve to make application of AI more "robust". I use that term in a rather loose sense for the moment. One interpretation might be that to make something robust is to make it more "deterministic" or "reproducible", but this can only cover a subset of qualities of such a system. However, automated reasoning is a principled, methodical way of exploring a subspace of a larger, more ill-defined problem space: hence my interest in this space. After all, this early AI research did give us Lisp :-)

Prolog was the European answer to automating the reasoning process in the 1960's (the American one was Lisp). It belongs to the family of logic programming languages, which is a paradigm distinct from the well-known imperative/functional/object-oriented ones. There are several ideas in here that appealed to me. Not surprisingly, Prolog and its other subsets and derivative approaches (Datalog, Answer Set Programming, etc.) have been used in applications involving different types of reasoning tasks.

From my reading, there are several kinds of reasoning, and the categorisation depends upon different parameters. I only list the types of reasoning which interest me.

- **Deductive:** Incontrovertible conclusions from facts. This is what base Prolog does.
- **Inductive:** Plausible "general" facts from observed facts. Inductive Logic Programming (ILP) systems like Popper build on top of Prolog to provide such capabilities.
- **Abductive:** Generalised "probable" rules from facts. Libraries like ILASP and Potassco's `clingo` provide capabilities like Answer Set Programming.
- **Non-monotonic:** Facts are tentative, and can be invalidated in the face of new facts. ILASP, etc. support non-monotonic reasoning.
- **Probabilistic Logic:** Facts are assigned probabilities, conclusions are probabilistic in nature. Probabilistic Graphical Models embody such logic.

For the purposes of this post, I will be using [SWI-Prolog](https://www.swi-prolog.org/, though you can also use other implementations like [Ciao](https://ciao-lang.org/), [GNU Prolog](http://www.gprolog.org/), and others. In an upcoming post, I will go through the design of a simple-but-nontrivial Virtual Machine using Prolog.

## A very brief introduction to Prolog

The way of thinking abou logic programming is close to - but not the same as - functional programming. Briefly, these are some of the salient points.

- Data is immutable. In this it is similar to functional programming.
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

Here is one way you can concatenate two lists:

```prolog
concat([],RHS,RHS) :- !.
concat([H|T],Acc,[H|R]) :- concat(T,Acc,R).
```

The base recursive case is when the left side is empty, so that the concatenation result is simply `RHS`.
The general case is to gradually peel off the head of the left list, and add it as the head of the right side in reverse order.

## Compound Terms: The same thing

Predicates can be pattern-matched too. Write:

```prolog
recognise_letters(some_fn(a)) :- writeln('a was first parameter'),!.
recognise_letters(some_fn(b)) :- writeln('b was first parameter'),!.
recognise_letters(some_fn(_)) :- writeln('Something other than a or b').
```

You will trigger one of the rules above as long as the arity of the given `some_fn` matches, as shown in the examples below. This structural matching of compound terms is extremely powerful and can be as arbitrary as you like.

```
?- recognise_letters(some_fn(a)).
a was first parameter
true.

?- recognise_letters(some_fn(b)).
b was first parameter
true.

?- recognise_letters(some_fn(c)).
Something other than a or b
true.

?- recognise_letters(some_fn(c,12)).
false.
```

## Evaluation does not happen by default

If you write the following, you will see that Prolog doesn't actually evaluate `1+2=3`. This is because `1+2` is actually a compound term `+(1,2)`, and not (yet) an arithmetic expression.

```
?- X=1+2,writeln(X).
1+2
X = 1+2.
```

If you want to actually evaluate, you'd use the `is` operator.

```
?- Y is 1+2.
Y = 3.
```

This allows us to do pass predicates around and write higher-order functions. Here is a (slightly-contrived) example:

```prolog
print_mapped_number(Number,Pred) :- call(Pred, Number, Result),format('Result is: ~w', Result).
add_one(Number, Result) :- Result is Number + 1.
```

Now if you write:
```
?- print_mapped_number(2,add_one).
Result is: 3
true.
```

Effectively, `add_one` is a predicate which isn't evaluated until `call` is used on it.

You can also perform structural construction and deconstruction of compound terms very easily:

```
?- a(1,2)=..X.
X = [a, 1, 2].

?- X=..[a, 1, 2].
X = a(1, 2).
```

This opens up a world of possibilities for metaprogramming, term rewriting, and other applications.

What about more structured data?

You can simply represent more complex structured data, by nesting compound terms, like in the example below. Prolog's unification automatically binds the corresponding variables.

```prolog
print_person((FirstName, LastName),Age,(AddressLine1, AddressLine2)) :- format('First name:~w, Last name: ~w, Age: ~w, AddressLine1: ~w, AddressLne2: ~w', [FirstName, LastName, Age, AddressLine1, AddressLine2]).
```

```
?- print_person(("John", "Doe"),30,("1 Some Street", "Some City")).
First name:John, Last name: Doe, Age: 30, AddressLine1: 1 Some Street, AddressLne2: Some City
true.
```

Let's talk about how you can do things declaratively. For example, suppose you have an array, and you want to check whether it start with the elements `1`, `2`, and ends with `5`, `6`, irrespective of what exists in the middle. In most other languages, you'd want to extract the first 2 elements and the last 2 elements (after some boundary checking), and then check those against the patterns you are looking for. In Prolog, since you can (sort of) run programs both backwards and forwards, you can declare how you would construct a program which shows the desired pattern, and run it backwards.

So, let's say you have three lists `Prefix`, `Middle`, and `Suffix`. How would you construct a list containing all of them?

```prolog
join_3(Prefix, Middle, Suffix, Result) :- concat(Prefix,RHS,Result), concat(Middle,Suffix,RHS).
```

`RHS` is defined here as the concatenation of `Middle` and `Suffix`, and `Result` is defined as the concatenation of `Prefix` and `RHS`. That's pretty logical, and you can test this out like so:

```
?- join_3([1,2],[3,4],[5,6],R).
R = [1, 2, 3, 4, 5, 6].
```

Now, make the result already-determined, and make the middle a _don't-care_ variable, and see what Prolog tells you:

```
?- join_3([1,2],_,[5,6],[1,2,3,4,5,6]).
true.

?- join_3([1,2],_,[5,6],[1,2,3,4,5,6,7]).
false.
```

Here. you're literally asking whether this relation holds given the prefix, the suffix, and the end result: effectively you are performing pattern matching. Note how you did not have to extract out elements manually and do tedious equality checking. You just specified how a list conforming to the structural pattern you are looking for, might be built, and you just run it backwards. This is a very strong motivating example of you instructing the machine WHAT to do, but NOT how to do it.

## Prolog Concepts: Unification

## Prolog Concepts: Backtracking

## My Personal Thoughts on Prolog

## Footnote: How I learned Prolog

This is the first language I've learned for the most part by working with an LLM to design exercises and evaluate my submissions.
