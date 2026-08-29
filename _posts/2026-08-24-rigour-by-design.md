---
title: "Rigour by Design: What AI-Powered Software Construction Should Look Like"
author: avishek
usemathjax: false
mermaid: true
tags: ["Software Engineering", "GenAI", "Formal Methods", "Type Systems", "Curry-Howard", "Specifications", "Architecture", "Program Analysis"]
draft: false
published: true
---

Software is hard. Personally, I oppose the position that "writing software was never the hard part", and believe that that is what people say when they have not explored the space of writing maintainable, extensible software well enough, or have not seen the impact of decisions (both good and bad) that go into building systems that are either maintainable and understandable, or become a morass of tangled, friction-laden maintenance nightmare.

Let me clarify at length what I mean by "writing software".

**What it is not:** not merely knowing and wielding the syntax / API of the language / tech stack through and through to get the job done. This is necessary but not sufficient. If that were the case, we would all be writing in assembly language.

**What it is:** Being able to express domain intention in code, in such a way that said intention is recoverable (ideally mechanically) from the code with minimal context.

We'd like to express our intentions to the computer at a level of abstraction closer to our thought processes. This is what programming languages are for, and, more recently, why LLMs have become a viable step-up in expressing these intentions. But they are also the source of a lot of slop. Slop code is different from plain text slop though: working code is working code, however bad the quality. However, I argue that the degree and fidelity of the encoding of our mental model in software, starts not from the code, but from the specifications that we write, and in the logical extreme, **the specification is the code**.

For the longest time, we have been writing specifications in plain text, which is a poor medium for encoding a mental model. Ideally, like any programming language, how do we express our intention at a higher abstraction without sacrificing precision?

There are two scenarios where these bear fruit:

- **Reasoning about existing (possibly legacy) code:** This happens mostly during reverse engineering, recovering semantic understanding from existing code. The question then is: **what is the best way to model the existing code precisely enough to recover intent and express it unambiguously, without drowning in tech/language-specific details?**
- **Encoding domain reasoning in code that we write:** This happens when we are developing new software. The question then is: **what is the best way to minimise the delta between our mental model and its expression in code?**

Both of these are two sides of the same coin, and both lead to the same question: how do we express our intention at a higher abstraction without sacrificing precision?

We have been doing this with programming languages for the longest time, and the temptation is to treat LLMs as another jump in abstraction. This is not a tenable position, though. the sort of abstractions we have been building programming languages for, do not sacrifice precision. Regardless of the language, the computer does exactly what we tell it to do, no more, no less (sometimes to the chagrin and frustration of the humans writing this code). Plain text, and by extension, LLMs, are inherently a lossy medium, if used to express precise intent for consumption and interpretation by a machine.

This is not to dissuade people from using LLMs for programming. The central theses I present in this article are:

- Shift the programming model left, from implicit human understanding to explicit deterministic encoding at runtime, and to explicit type- and structure-level encoding of domain reasoning, such that semantic errors and ambiguity are impossible by construction.
- Just like good software engineering practices are useful for future (and current) comprehension and reasoning of code to humans and teams, they are as useful to LLM agents for tracing and reasoning through code.
- The interesting thing about GenAI is not what it makes newly possible; it is what it makes newly affordable: these are the techniques which enable the two points above; and I don't see a lot of people spending that windfall on the thing worth buying.

This is an opinionated position, and it is still evolving. I purposefully took a somewhat extreme stance when writing this, because I want to represent something of a mental north star, regardless of the level of feasibility; that depends upon several factors: the language, the tooling, the maturity of the engineering team(s), and so on.

Every technique and tool I am going to describe here predates GenAI by years. Dependent types, TLA+, Dafny, abstract interpretation, Prolog, property-based testing. None of it was ever wrong. It was expensive, needed skill, and slow work that was hard to defend to anyone holding a budget. So most of us stayed on the bottom rung of the ladder and called it pragmatism.

What got cheap in the last three years is exactly the labour that kept us down there. Spending it on generating more unverified code faster wastes the opportunity, and that is what most teams are doing.

---

## Contents

- [Curry-Howard in Ninety Seconds](#curry-howard-in-ninety-seconds)
- [The Program Is the Proof](#the-program-is-the-proof)
- [The Behaviour Ladder](#the-behaviour-ladder)
- [Correct by Construction vs. Correct by Observation](#correct-by-construction-vs.-correct-by-observation)
- [Markdown Specifications Are Bullshit](#markdown-specifications-are-bullshit)
- [The Spec Ladder](#the-spec-ladder)
- [A Specification You Can Interrogate](#a-specification-you-can-interrogate)
- [Model the Domain, Not Just the Behaviour](#model-the-domain-not-just-the-behaviour)
- [Both Ladders End in the Same Place](#both-ladders-end-in-the-same-place)
- [What AI Actually Changed](#what-ai-actually-changed)
- [What You Inherited Sets Your Floor, Not Your Ceiling](#what-you-inherited-sets-your-floor-not-your-ceiling)
- [AI Proposes; the Verifier Disposes](#ai-proposes-the-verifier-disposes)
- [The Harness Is the Trust Boundary](#the-harness-is-the-trust-boundary)
- [Where This Is Over-Engineering](#where-this-is-over-engineering)
- [The Old Fundamentals Matter More Now, Not Less](#the-old-fundamentals-matter-more-now-not-less)

---

## Curry-Howard in Ninety Seconds

The Curry-Howard Isomorphism says that logic and programming are the same structure under two different names.

| Logic | Programming |
|---|---|
| proposition | type |
| proof | program |
| A implies B | function `A -> B` |
| A and B | pair `(A, B)` |
| A or B | sum `Either A B` |
| modus ponens | calling the function |
| **checking a proof** | **type-checking** |

A claim like "every valid order yields an invoice" is a proposition. Written as a type, it is:

```haskell
invoice :: ValidOrder -> Invoice
```

The proof of that proposition is an implementation of that type:

```haskell
invoice o = ...
```

If it compiles, the claim holds. Type-checking *is* proof-checking. Curry found this in 1934, Howard found it again in 1969. It is a theorem, not a nice way of talking about types.

There is one catch, and it matters enormously in the languages most of us actually ship in. Only a *total* function is a proof. Throw an exception, return null, or loop forever, and you have proved nothing at all. The compiler will be perfectly happy anyway. Most of the industry's type systems are logics with a hole in them, and everybody has agreed not to look at the hole.

---

## The Program Is the Proof

Take Curry-Howard seriously for a moment and something uncomfortable follows.

A proposition is a type. A proof of it is a program of that type. This is exact, not a metaphor. So your running code is already the most exact statement of what your system does. A type, a model, a test, a diagram, a paragraph of English: every one of them is an *approximation* of that program, and the only reason you buy an approximation is that it is cheaper than the thing it approximates.

That gives you exactly one question to ask about every technique below. **How much of the proof does it buy, and what does it cost?**

Not "is this rigorous?", not "is this best practice?". How much of the proof, at what price.

---

## The Behaviour Ladder

Here are five ways to be sure your code does what you think, in descending order of certainty and ascending order of regret.

| Rung | Techniques | What you actually get |
|---|---|---|
| **Correct by construction** | rich domain types, dependent types (Lean, Idris), illegal states made unrepresentable | the violation cannot be written |
| **Proof without execution** | TLA+ for liveness and safety over time, Dafny for pre/postconditions and invariants | proves a model, not your code |
| **Static analysis** | control-flow graphs, data-flow analysis, def-use chains, call graphs | recovered, not encoded |
| **Symbolic execution** | abstract interpretation, reaching conditions, taint analysis | recovered, not encoded |
| **Runtime** | property tests, characterisation tests, assertions, ordinary business logic | instances of the proof, not the proof |

The direction of travel is up, and up is shift-left.

There is an asymmetry in that table that I want to talk about. **The top two rungs are forward.** You encode the property while you are writing the code, and the tool checks that you did not contradict yourself. **Static and symbolic analysis mostly run in reverse.** You are recovering properties from code that somebody else wrote, years ago, without any conscious intention of making them recoverable. That's reverse engineering.

That is why the reverse rungs show up overwhelmingly in modernisation work, and why most of the practical evidence I have sits on those two rungs rather than the two above them. Nobody is retrofitting Idris onto a Natural/ADABAS system.

---

## Correct by Construction vs. Correct by Observation

Runtime is where most teams find out they were wrong. It is the most expensive place to find out, and the latest. You are not required to work this way. It is a choice, and mostly an unexamined one.

Every rung you climb turns a runtime discovery into a compile-time refusal. The compiler says no before the test gets a chance to fail.

This is what shift-left should mean. Most people use it to mean "test earlier", which is still runtime, just sooner. Moving your integration suite from Friday to Tuesday does not change the epistemology of what you are doing: you are still observing instances and generalising. The point is to move the correctness argument out of execution altogether.

And there is a compounding effect that has nothing to do with correctness. Code that is correct by construction is also cheaper to test, cheaper to change, and cheaper for a machine to reason about. Same property, showing up in three different places on the balance sheet.

Every rung above runtime is a model of your code, not your code. Nobody gets to skip this.

TLA+ proves your model has no deadlock. Dafny proves the function you specified is correct. Neither of them has seen your production code. The proof is only ever as good as the model's resemblance to what you actually shipped, and that resemblance is maintained by hand, by humans, under deadline.

Static analysis pays the same tax in a different currency. A control-flow graph is precise right up until it hits polymorphism, at which point the call edge is a guess wearing the costume of a fact. Every soundness/precision trade-off in an analyser is a place where the model and the code diverge quietly.

The legacy world has sharper versions of this. COBOL `REDEFINES` gives you the same bytes interpreted as several incompatible types, and which one is live is decided at runtime by data. Model that wrong and your static fact is fiction.

And the bottom rung is worse than people admit. You can write tests showing that 2, 4 and 6 are even. That says precisely nothing about 8. A test is one instance of the proof, not the proof. This is not a reason to skip tests, it is a reason to be honest about what a green build is evidence *of*.

Pick your rung. Then know what you did not buy.

---

## Markdown Specifications Are Bullshit

I mean specifications written in markdown *for machine consumption*. Prose for humans is fine and always has been. Prose as the artifact you hand to a code generator and call the source of truth is not.

English is the least exact model of a program you can write. The current fashion is to hand that to a probabilistic system and call the result a methodology. You have stacked two independent sources of ambiguity on top of each other and given the stack a name and a logo.

Here is the operational problem. **Nothing in that document can be checked.** Not for internal consistency, and not against the code that supposedly implements it. So the checking falls to a human reading a diff, and to production.

"But it's readable." Readable by whom, to decide what? A specification nobody can execute cannot fail. People hear that as a feature. A specification that cannot fail, can drift merrily from intent without any alarms.
Prose sits *below* the bottom rung of the behaviour ladder. A test at least fails.

---

## The Spec Ladder

The specification side has its own ladder, with the same shape, and almost nobody climbs it.

| Rung | What it looks like | What you get |
|---|---|---|
| **Rich domain types** | the spec *is* the type; the compiler is the checker; the domain constraint is the signature | no drift is possible |
| **Declarative, machine-checkable specs** | Prolog and relatives; the specification is itself a program you can query | consistency becomes mechanical |
| **Deterministic guardrails** | lint and immutability, cyclomatic complexity, coverage and assertion quality, architectural import constraints | the floor, not the destination |
| **Prose / markdown** | verified by a human in code review, or by production, and by nothing else | nothing here can fail |

Guardrails are the cheapest rung on either ladder and there is no excuse whatsoever for skipping them. If your review process consists of senior engineers reading diffs for style and cyclomatic complexity, you are paying six figures for a linter.

---

## A Specification You Can Interrogate

The rung above guardrails is where it gets interesting, because a declarative spec is a program you can ask questions of. Here is the smallest possible example, written in Prolog.

```prolog
% the domain, stated once
parent(rajesh, meera).
parent(meera, arun).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

:- \+ grandparent(X, X).   % invariant: nobody is their own grandparent
```

```prolog
?- grandparent(rajesh, arun).  % forward
   true.

?- grandparent(X, arun).       % backward
   X = rajesh.

?- grandparent(rajesh, Z).     % backward
   Z = arun.
```

One clause, three questions. You did not write a lookup function and you did not write an inverse lookup function. Resolution ran the rule backwards from the goal, because the rule is a *relation*, not a procedure.

In markdown you would write "a grandparent is the parent of a parent", and then check every consequence of that sentence by hand, forever, every time anything else in the document changed.

Note also what the invariant is. It is not a comment. It is not a team convention that gets mentioned in onboarding. Breaking it is a mechanical failure, not a reviewer being alert on a Thursday afternoon.

The limit is worth stating plainly: **this does not stop an LLM writing a wrong spec.** You can check consistency, not intent. What it does stop is the model being wrong across six paragraphs that nobody reads carefully. And because the check is mechanical, iteration is close to free.

It is also exactly why prose specs cannot be rescued by better models. A wall of text cannot fail, so the loop never closes, so a human has to sit in the middle of it reading everything. You automated the proposing and kept the expensive half.

An experiment on how to extract a deterministic, queryable knowledge base with forward and backward inference from a text document, is documented in [A Statute as a Runnable Logic Program](/2026-08-24-a-statute-as-a-runnable-logic-program.md)

---

## Model the Domain, Not Just the Behaviour

The grandparent example is trivial on purpose. Put settlement eligibility, tenure calculation, or chain of custody in its place. The mechanism does not change.

The point is that you can ask questions of a domain, deterministically, before a single line of production code exists. Contradictions show up as failed queries. The alternative is two people discovering three sprints later that they meant different things by the same word, which is how most requirements defects actually happen.

Rich typing does the same job, in the language you actually ship. Not "amount is an `int`, validated somewhere upstream, probably", but an amount that cannot be *constructed* outside its valid range. The constructor is the specification.

Domain modelling and type design are the same discipline viewed from two ends. Treat types as annotation and you are wasting the best checker you already have installed and are already paying for.

A worked example of the declarative approach is at [github.com/avishek-sen-gupta/doc-pipeline](https://github.com/avishek-sen-gupta/doc-pipeline).

---

## Both Ladders End in the Same Place

That is not a coincidence.

```mermaid
flowchart TD
    subgraph CODE[Code ladder]
        R1[runtime tests] --> R2[symbolic execution]
        R2 --> R3[static analysis]
        R3 --> R4[proof without execution]
        R4 --> TOP1[rich domain types]
    end
    subgraph SPEC[Spec ladder]
        S1[prose / markdown] --> S2[deterministic guardrails]
        S2 --> S3[declarative machine-checkable specs]
        S3 --> TOP2[rich domain types]
    end
    TOP1 --> CONV[the type is the proposition\nthe program is the proof\nnothing left to drift]
    TOP2 --> CONV

    classDef default fill:#dde3f5,stroke:#6b7db3,color:#1a1f5e
    classDef bad fill:#fce7f3,stroke:#be185d,color:#831843
    classDef good fill:#d4edda,stroke:#388e3c,color:#1b5e20
    class R1,S1 bad
    class TOP1,TOP2,CONV good
```

The code ladder climbs to rich domain types: behaviour encoded at compile time, where the illegal program does not typecheck. The spec ladder climbs to rich domain types: the domain encoded at compile time, where the illegal state does not typecheck. At the top, the spec and the program stop being two artifacts that can drift apart. They are one artifact.

This matters specifically for AI-assisted construction. **A model's nondeterminism does not sit everywhere. It sits in the gap between the spec and the program.** That gap is the translation step, and translation is where guessing happens. Every rung you climb narrows it. At the limit there is no translation step left for anything to be wrong in.

You will not reach the limit, and I am not suggesting you try. What matters is which way you are walking.

---

## AI Proposes; the Verifier Disposes

AI did not add a rung to either ladder. It collapsed the cost of climbing them.

That distinction is the whole argument. Dependent types did not get more expressive because a model can write Lean. TLA+ did not get better at liveness. What changed is that the tedious, skilled, slow labour of *encoding* things — writing the property, writing the model, writing the characterisation test, writing the query — got an order of magnitude cheaper.

If you want one architectural principle out of all of this, it is that sentence. Here is what it decomposes into.

As a *design-time* proposer, on the other hand, it is welcome absolutely anywhere, as long as the output goes through a deterministic gate before anyone depends on it. Same model, opposite rules, depending on which side of the build it sits.

**Executable specs, graded by criticality.** The model proposes toward specs that run. Forward: FP discipline first, to get a verifiable substrate at all, then formalism where the language and the stakes both justify it. Legacy: characterisation tests as executable specs. The verifier disposes by running them: deterministic pass/fail with no human in the gate, and a postcondition violation that localises the fault to one unit rather than to a service.

**Legibility and traceability.** Code structured to be read by the next model, not just by the next human. Types encode intent as a machine-readable spec. Traceability is emitted *forward*, at the point of construction, never reverse-engineered later. The verifier disposes via the type checker, which is a verifier of intent, and via structure that makes reasoning safe without global context.

**Functional core, imperative shell.** Propose pure functions with no hidden state. Quarantine effects at the seam, so the model never reasons about or touches side effects. The verifier disposes because for a pure function the output *is* the full contract: you can regenerate the implementation freely with no side-effect risk, and safety and liveness properties become model-checkable.

**Composable systems, bounded contexts.** One bounded context fits in one context window, which makes the bounded context the unit of comprehension and means no global understanding is required. The verifier disposes at the contract-checked boundary: a proposal cannot violate a neighbour's contract without something catching it mechanically.

Every one of those four is an ordinary architectural virtue that predates all of this. What changed is that each one now also determines whether a machine can safely participate in your codebase.

---

## What You Inherited Sets Your Floor, Not Your Ceiling

Two situations, two sets of obligations.

**Inherited: you do not get to choose.** No types, no specs, no tests. The bottom rung is the only one on offer. That is a starting point, not a verdict.

Characterisation and property tests pin behaviour as an executable spec, and AI has collapsed the cost of writing them at volume. The catch is real and needs saying: **they pin the bugs exactly as written.** A characterisation test encodes what the system does, not what it should do. A human still has to say which behaviours were intended. What you have bought is a regression net, not a specification of intent.

Then give the model the rung above: call graphs, def-use chains, reaching conditions, coverage. Computed facts rather than guesses. I have written about the mechanics of this at length in [Harnessing LLMs with Deterministic Program Analysis](/2026/05/21/harnessing-llms-with-deterministic-program-analysis.html).

**Greenfield: you have no excuse.** Immutability, a pure core, explicit failure, algebraic data types, from the first commit.

You will, at some point, want to verify something formally. Maybe not on day one, maybe not with this team, but the option only exists if the code was built for it. Functional programming discipline is not a consolation prize for teams who will never write a proof. It is the *precondition*. Purity and immutability are what make a function tractable to a verifier at all. A method that mutates four fields and reads a clock is not hard to verify, it is outside the domain of the tools.

And if you never climb higher, it has already paid for itself in ordinary human reasoning. Your ceiling depends heavily on the language: Java is hostile to verification, Rust is not.

Tests are the rung you can always afford. That is not an argument for stopping there.

---

## The Harness Is the Trust Boundary

Not just scaffolding. The trust boundary.

An agent is a component. Components have typed contracts: input schema, output schema, invariants. "But it's nondeterministic" does not get you out of that; it is an argument for a *stricter* contract, not the absence of one. A chain of agent calls is a program, and every hop is a function call.

The harness is software you own and test, not glue you paste together. If you have not tested the harness, you have not tested the system, because the agent is not the system.

And record, as they happen, which call, which model version, which prompt, and which gate decision. Reconstructing that afterwards is the genuinely expensive part of debugging an agentic failure, and it is unnecessary.

```mermaid
flowchart TD
    CORE[deterministic core\ntyped, synchronous, owns no AI coupling] --> PORT[port / AI interface\ntyped contract: core knows nothing else]
    PORT --> ADAPT[adapter\ntimeout · retry · circuit breaker\nprompt construction · response parsing · gate]
    ADAPT --> Q[async queue · bulkhead\nnever block the caller · backpressure]
    Q --> API[LLM API\nhigh-latency · partially-failing · nondeterministic]

    classDef default fill:#dde3f5,stroke:#6b7db3,color:#1a1f5e
    classDef good fill:#d4edda,stroke:#388e3c,color:#1b5e20
    classDef bad fill:#fce7f3,stroke:#be185d,color:#831843
    class CORE good
    class API bad
```

An LLM is a distributed-systems dependency: high-latency, partially-failing, nondeterministic. Treat it as one and the engineering becomes completely ordinary — you already know how to build this. Pretend otherwise and you will meet all three of those properties in production, at the same time, on a Friday.

---

## Where This Is Over-Engineering

Climbing costs real engineering labour, and I am not going to pretend otherwise.

If the thing is low-volume, short-lived and not critical, this is over-engineering and you should not do it. Volume, longevity and criticality are what decide, and they decide per system, not per organisation. Anyone who tells you the ladder is always worth climbing is doing ideology, not engineering.

Rigour also looks slower up front, and sometimes it genuinely is. I am not going to claim it is secretly faster. What I will claim is that a lot of what gets built without it does not survive contact with production, and that cost lands on somebody's budget anyway — just later, and filed under a different name.

And "least power" will be heard by some people as fighting the tide. Do not soften it on that account. Reaching for the most powerful and least predictable tool when a simpler one will do is an engineering error, whatever the market happens to be wearing this season.

---

## The Old Fundamentals Matter More Now, Not Less

Immutability. Small composable units. Explicit failure. Contracts at the seams. Types that mean something.

This is not nostalgia. These are the properties that make code checkable, and checkable is what makes generated code safe to keep. Every one of them was a good idea when humans wrote all the code; every one of them is now also load-bearing for whether a machine can contribute without degrading the system.

One last thing, which is about people rather than code. What compounds is not the artifact. It is the rate at which the people building it get better. Automation captures a snapshot of capability and then sits there, depreciating. A person who builds the thing carries the learning into the next thing.

**So the job of AI here is to make rigour cheap. Not to make judgement optional.**

Every rung you climb makes the next batch of machine output cheaper to trust. That is the part that compounds.
