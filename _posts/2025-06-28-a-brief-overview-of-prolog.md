---
title: "A brief overview of Prolog"
author: avishek
usemathjax: false
tags: ["Logic Programming", "Automated Reasoning", "Prolog"]
draft: false
---

In this post, I'll give a brief overview of **Prolog** and the paradigm of **Logic Programming**. I'll discuss why I think it makes for such a powerful domain modelling language, and a gateway into the techniques of automated symbolic reasoning.

_This post has not been written or edited by AI._

## Why Prolog?
I've always been interested in the evolution of AI, and the potential for a synthesis of older techniques (formal and otherwise) and newer ones (deep learning) in my current area of focus (reverse engineering). My intuition tells me that combining many of these approaches will only serve to make application of AI more "robust". I use that term in a rather loose sense for the moment. One interpretation might be that to make something robust is to make it more "deterministic" or "reproducible", but this can only cover a subset of qualities of such a system. However, automated reasoning is a principled, methodical way of exploring a subspace of a larger, more ill-defined problem space: hence my interest in this space. After all, this early AI research did give us Lisp :-)

**Prolog** was the European answer to automating the reasoning process in the 1960's (the American one was Lisp). It belongs to the family of logic programming languages, which is a paradigm distinct from the well-known imperative/functional/object-oriented ones. There are several ideas in here that appealed to me. Not surprisingly, Prolog and its other subsets and derivative approaches (Datalog, Answer Set Programming, etc.) have been used in applications involving different types of reasoning tasks. There is a treasure trove of papers surveying the history of Prolog [here](https://github.com/dtonhofer/prolog_notes/tree/master/other_notes/about_papers_of_interest) if you are interested.

From my reading, there are several kinds of reasoning, and the categorisation depends upon different parameters. I only list the types of reasoning which interest me.

- **Deductive:** Incontrovertible conclusions from facts. This is what base Prolog does.
- **Inductive:** Plausible "general" facts from observed facts. **Inductive Logic Programming (ILP)** systems like Popper build on top of Prolog to provide such capabilities.
- **Abductive:** Generalised "probable" rules from facts. Libraries like ILASP and Potassco's **clingo** provide capabilities like Answer Set Programming.
- **Non-monotonic:** Facts are tentative, and can be invalidated in the face of new facts. ILASP, etc. support non-monotonic reasoning.
- **Probabilistic Logic:** Facts are assigned probabilities, conclusions are probabilistic in nature. **Probabilistic Graphical Models** embody such logic.

Here's a very informative diagram of the logic programming landscape, borrowed from [David Tonhofer's extensive notes on Prolog](https://github.com/dtonhofer/prolog_notes):

![LP Landscape](/assets/images/quick_map_of_lp_landscape.svg)

For the purposes of this post, I will be using [SWI-Prolog](https://www.swi-prolog.org/), though you can also use other implementations like [Ciao](https://ciao-lang.org/), [GNU Prolog](http://www.gprolog.org/), etc. In an upcoming post, I will go through the design of a simple-but-nontrivial **Virtual Machine** together with **symbolic execution** with using Prolog in about **200 lines of code**.

## A very brief introduction to Prolog

The way of thinking about **logic programming** is close to - but not the same as - **functional programming**. Briefly, these are some of the salient points.

- **Data is immutable.** In this it is similar to functional programming.
- You don't write statements. You write **facts** and **predicates**. These relate _things_ to other _things_.
- Facts hold unconditionally, i.e., they are _true_.
- Any fact not defined as treated as _false_.
- **Predicates hold conditionally**, based on whether other predicates or facts hold or not.
- The actual act of **computation happens when asserting the truth** of these predicates.
- Prolog is superb at **pattern matching** and **symbolic manipulation**. This comes partly from its unification mechanism (I discuss it in [here](#prolog-concepts-unification)). Personally, I have not seen this level of capability at a language level in any other language I have seen (including OCaml, which is another language I'm currently learning).

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

The simplest thing you can do with the above facts is query them. For example:

```
?- edge(a,b).
true.
```

The first query simply checks whether there is an edge from `a` to `b`. This is forward inference; but we can also do reverse inference. We can turn a question about our facts on its head: suppose we want to know which are the outgoing edges of `b`. In any other programming language, you'd want to check the list of outgoing edges from `b`, and so on. In Prolog, you can simply ask for this information.

```
?- edge(b,N).
N = c ;
N = d.
```

No extra programming needed. This is because Prolog leverages the concepts of unification, backtracking and goal resolution. **Effectively, you can run your program both forwards and backwards.** This is extremely powerful, and I talk of another example [later](#declarative-structural-matching).

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

Now you can ask the Prolog interpreter:

```
?- node_exists(a).
true.

?- node_exists(efgh).
false.
```

Note that we have replaced the `a` with `N` which is a variable, which implies that `node_exists` takes in a **thing**. It is important to note that thing can really be anything: an atom, a string, another predicate, etc. As long as Prolog can find a rule which matches the sort of thing we inject into `node_exists`, it will consider this rule for further evaluation.

Suppose we wish to check if a node has multiple outgoing edges. We can write:

```prolog
has_multiple_outgoing_edges(N) :- edge(N,A), edge(N,B), A \= B, !.
```

The `,` in this case represents the logical AND. This means that `has_multiple_outgoing_edges(N)` is true if:

- There is an edge from `N` to `A`
- There is an edge from `N` to `B`
- `A` is not the same as `B`. `\=` represents **"not equal to"**.

We can check whether this works:

```
?- has_multiple_outgoing_edges(b).
true.

?- has_multiple_outgoing_edges(a).
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

We can perform more meaningful semantic reasoning though. Suppose, we are interested in knowing whether a node is reachable from another node. We can define a predicate `can_reach` like so:

```prolog
can_reach(From, To) :- edge(From, To), !.
can_reach(From, To) :- edge(From, Z), can_reach(Z, To).
```

The base case is simply that a node can reach another if there is an edge between them.
The general case is a recursive definition: it says that A can reach B, if A has an edge to Z (some undefined node), and if Z in turn can reach B.

Thus for example, if we run `can_reach(a,c)`, we conceptually reason in the following manner:

- Can `a` reach `c`? This can be answered if we can answer if `a` has an edge to some node Z, and Z can reach `c`.
- The only node that `a` has an edge to is `b`, so we want to answer the question of whether `b` can reach `c`.
- This triggers the base case because `b` has a direct `edge` to `c`.
- Therefore, there is an arbitrary node Z (in this case `b`) to which `a` has an edge to, and which can reach `c`.

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

The first argument is the list being passed in (and gradually getting decomposed into successive heads and tails). The second one is the accumulator which keeps getting built up with the result (it always `[]` initially). The final argument is the actual result to which the reversed list will be bound to. You can run this like so:

```
?- reverse2([1,2,3],[],R).
R = [3, 2, 1].
```

A list of common list processing idioms may be found [here](https://github.com/dtonhofer/prolog_notes/tree/master/other_notes/about_list_processing).

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

## Declarative Structural Matching

Let's talk about how you can do non-trivial pattern matching declaratively, and in the process, we will concretise the idea of being able to run Prolog programs backwards and forwards. For example, suppose you have an array, and you want to check whether it start with the elements `1`, `2`, and ends with `5`, `6`, irrespective of what exists in the middle. In most other languages, you'd want to extract the first 2 elements and the last 2 elements (after some boundary checking), and then check those against the patterns you are looking for. In Prolog, since you can (sort of) run programs both backwards and forwards, you can declare how you would construct a program which shows the desired pattern, and run it backwards.

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

Here, **you're literally asking whether this relation holds given the prefix, the suffix, and the end result**: effectively you are performing **pattern matching**. Note how you did not have to extract out elements manually and do tedious equality checking. **You just specified how a list conforming to the structural pattern you are looking for, might be built, and you just run it backwards.** This is a very strong motivating example of you instructing the machine **WHAT to do, but NOT how** to do it.

## Prolog Concepts: Unification

In automated reasoning, unification is a technique used to solve equations where the left and right hand sides of the equation are symbolic expressions. For example, the following are examples of such equations:

```prolog
% The free variable is X and the solution is X=c
f(a,b,X) :- f(a,b,c).

% The free variables are X and Y and the solution is {X=c, Y=a}
f(a,b,X) :- f(Y,b,c).

```

If the right hand side has no free variables, it is basically the same as pattern matching. Consider the earlier example of reversing a list:

```prolog
reverse2([],Acc,Acc).
reverse2([H|T],Acc,Result) :- reverse2(T,[H|Acc],Result),!.
```

Let's trace the process of reversing a small list `[1,2]` and see how unification works. 

### Step 1: General Rule fires but is unresolved
When we run `reverse2([1,2])`, the general rule triggers with the following unifications.

- `Acc` is unified with []
- `H` is unified with 1
- `T` is unified with `[2]`

```prolog
reverse2([1|[2]],[],Result) :- reverse2([2],[1|[]],Result).
```
At this point, this equation cannot be fully solved since `Result` is still unbound. However, this triggers the general rule again when trying to determine the truth of the right hand side.

### Step 2: General Rule fires but is unresolved

A similar situation occurs here too, with the following unifications.

- `Acc` is unified with `[1|[]]`
` `H` is unified with 2
- `T` is unified with `[]`

```prolog
reverse2([2|[]],[1|[]],Result) :- reverse2([],[2|[1|[]]],Result).
```

As in [Step 1](#step-1-general-rule-fires-but-is-unresolved), this equation is also unsolved because of the unbound `Result` variable. To determine the truth of the right hand side this time though, it is the base case which fires.

### Step 3: Base Rule fires and is resolved
The base case is triggered, and here the third parameter (`Result`) is finally unified with the value of `Acc` (which is `[2|[1|[]]]`), since the same `Acc` variable appears in the second and third place.

```prolog
reverse2([],[2|[1|[]]],[2|[1|[]]]).
```

Now that this fact has been determined to be true, it is time to unroll and determine the truth values of the preceding rules in the stack.

### Step 4: General Rule Resolution is resolved
Now the [unresolved rule in Step 2](#step-2-general-rule-fires-but-is-unresolved) can be solved, since `Result` is also known, thus `Result` is unified here with the value `[2|[1|[]]]`.

```prolog
reverse2([2|[]],[1|[]],[2|[1|[]]]) :- reverse2([],[2|[1|[]]],[2|[1|[]]]).
```

Now the left hand side is fully determined (and thus true); this means that the preceding rule in the stack is ready to be resolved.

### Step 5: General Rule Resolution is resolved

Solving the [previously-unresolved rule in Step 1](#step-1-general-rule-fires-but-is-unresolved) with the `Result` variable from [Step 2](#step-4-general-rule-resolution-is-resolved) unifies the `Result` variable on both sides with `[2|[1|[]]]`:

```prolog
reverse2([1|[2]],[],[2|[1|[]]]) :- reverse2([2],[1|[]],[2|[1|[]]]).
```

This final result `[2|[1|[]]]` is bound to our query variable and shown with syntactic sugar as `[2,1]`.

## Prolog Concepts: Backtracking

Prolog programs essentially form execution trees with nodes being either terminal leaves (facts), or dependent predicates with child predicates arranged in a logical expression comprised of AND, OR, and NOT.

Let's revisit our graph reachability example. We had written:

```prolog
can_reach(From, To) :- edge(From, To), !.
can_reach(From, To) :- edge(From, Z), can_reach(Z, To).
```

Let us ask `can_reach(b,e).`. Note that in the graph, `b` has 2 paths from `e`, via `c` and `d`.

```
?- can_reach(b,e).
true ;
true.
```

You will notice that it returns two truth values. Each of these correspond to one of these viable paths. Conceptually, Prolog constructs the following abstract execution tree.

{% mermaid %}
graph TD;
can_reach_be["can_reach(b,e)"]-->chol[AND];
chol[AND] --> edge_bZ["edge(b,Z)"];
chol[AND] --> can_reach_Ze["can_reach(Z,e)"];
style chol fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

This abstract execution tree gets concretised and constrained by the space of available solutions. In this case, the two values that `Z` can take are `c` and `d`. Thus, the execution tree runs for each of these paths.

{% mermaid %}
graph TD;
can_reach_bc["can_reach(b,e)"]-->path_Z_is_c;
can_reach_bc["can_reach(b,e)"]-->path_Z_is_d;
path_Z_is_c["Z=c"];
path_Z_is_d["Z=d"];
path_Z_is_c --> and_Zc[AND];
and_Zc[AND] --> edge_bc_Zc["edge(b,c)"];
and_Zc[AND] --> can_reach_Ze["can_reach(c,e)"];
path_Z_is_d --> and_Zd[AND];
and_Zd[AND] --> edge_bc_Zd["edge(b,c)"];
and_Zd[AND] --> can_reach_de["can_reach(d,e)"];
style and_Zc fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
style and_Zd fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

Once a solution is found, Prolog **walks back up the tree**. At each preceding level, it attempts to find more viable paths to exhaust the space of all possible solutions. In this case, both paths `Z=c` and `Z=d` are viable, `true` is returned twice, corresponding to each of those paths.

### Notes on Logical Operator Syntax

- `,` represents **AND**. All the predicates we have seen so far use this operator. `p :- a,b.` means **"`p` is true if both `a` and `b` are true"**.
- `;` represents **OR**. `p :- a;b.` means "`p` is true if either `a` or `b` is true".
- `\+` represents **NOT**. `p :- \+a.` means "`p` is true if `a` is false".

The usual operator precedence rules apply, thus you need parentheses to override precedence, for example: `p :- (a;b),c.`

Relational operators are a little different from "conventional" programming languages.

- `=` checks for **structural equality**. `f(a,b) = f(a,b)`. It also does double duty as assignment. To be honest, this is one and the same since unification applies to both sides.
- `=<` represents **"less than or equal to"**. `<` is **"less than"**.
- `>=` represents **"greater than or equal to"**. `>` is **"greater than"**.

There is a **ternary operator syntax**. We write it as `a -> b; c` which is read as **"if `a` is true, then `b`, else `c`"**. Note that since `;` is being used as a separator, any logical expressions you write must be suitably bracketed.

## My Personal Thoughts on Prolog

### Strengths
Prolog is an extremely powerful **declarative modelling** tool, in my opinion. Extremely low overhead in defining concepts and relations is one of the key draws for me, especially when **rapidly prototyping** concepts and relations. Unification plays a large part in allowing this level of brevity. In addition, being able to **run logic backwards and forwards** has given me a completely different way of looking at programming. The deductive reasoning semantics allow us to **declaratively perform pattern matching** with a naturalness I've not seen in other languages.

I will write about examples of doing static analysis on programs using Datalog, a widely-used subset of Prolog, in an upcoming post.

Many other forms of reasoning libraries (inductive, abductive, etc.) use Prolog (or a variant of it) as a base. Thus, it is emphatically a gateway to all work on **automated reasoning**, **theorem proving**, etc.

### Usage
Prolog is best used as an **embedded component** in a larger codebase written in a more conventional language, in order to play to its strengths. It has **interfaces for Java, Python**, and other more conventional programming languages.

### Performance and Scaling
Prolog's performance used to be one of the sticking points, but there are fast implementations available, and **Datalog** (a non-Turing complete subset of Prolog) comfortably scales to **millions of facts**.

## Applications

- [Datomic](https://docs.datomic.com/query/query-data-reference.html) uses Datalog as its query language (effectively, a much more expressive SQL).
- Prolog is used for pattern matching in the IBM Watson question answering system.
- [Clarissa](https://www.newscientist.com/article/dn7584-space-station-gets-hal-like-computer/), a voice operated procedure browser aboard the International Space Station uses Prolog (https://sicstus.sics.se/customers.html).
- Prolog has been used as a bytecode verifier in Java (see [here](https://openjdk.org/jeps/8267650)).

## Footnote: How I learned Prolog

This is the first language I've learned for the most part by working with an LLM to design exercises and evaluate my submissions.

## References

- [David Tonhofer's Notes on Prolog](https://github.com/dtonhofer/prolog_notes)
- [Programming Paradigms for Dummies: What Every Programmer Should Know - Peter Van Roy (2012)](https://www.researchgate.net/publication/241111987_Programming_Paradigms_for_Dummies_What_Every_Programmer_Should_Know)
- [Byrd Box Model and Ports](https://eu.swi-prolog.org/pldoc/man?section=byrd-box-model)
