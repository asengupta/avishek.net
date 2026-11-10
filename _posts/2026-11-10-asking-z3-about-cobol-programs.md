---
title: "Asking Z3 About COBOL Programs"
author: avishek
usemathjax: false
mermaid: false
tags: ["Software Engineering", "Program Analysis", "Static Analysis", "COBOL", "Reverse Engineering", "Formal Methods", "Z3", "Horn Clauses"]
draft: true
published: false
---

*This post is about asking a solver questions about COBOL programs. The solver is Z3's Spacer engine, and it reads the programs as constrained Horn clauses. The questions are about CardDemo, the public CICS sample application. Examples are "can the account be rewritten without a confirmation?" and "can a card be saved without a change check?". The post follows the work from the first spike to a specification in Quint that the solver checks against the code. It includes each place where the solver was too slow, or where it gave an answer that was wrong.*

---

## Table of Contents

- [Why a solver](#why-a-solver)
- [Horn clauses in one example](#horn-clauses-in-one-example)
- [The plan: stages, each gated by real questions](#the-plan-stages-each-gated-by-real-questions)
- [Strings were too slow](#strings-were-too-slow)
- [Seven wrong proofs, and one fix for all of them](#seven-wrong-proofs-and-one-fix-for-all-of-them)
- [Copies make the state grow](#copies-make-the-state-grow)
- [A conversation is many tasks](#a-conversation-is-many-tasks)
  - [The state explodes](#the-state-explodes)
  - [One pertinent field](#one-pertinent-field)
  - [Automatic refinement did not finish](#automatic-refinement-did-not-finish)
  - [An LLM picks the fields](#an-llm-picks-the-fields)
- [Spurious runs are facts about the CFG](#spurious-runs-are-facts-about-the-cfg)
- [Fast tests that became slow](#fast-tests-that-became-slow)
- [Splitting the question across programs](#splitting-the-question-across-programs)
- [Writing the hand-over as a specification](#writing-the-hand-over-as-a-specification)
- [Toward a harness](#toward-a-harness)
- [What I took from it](#what-i-took-from-it)

---

## Why a solver

cobble is the static analyser in [red-dragon-forge](https://github.com/avishek-sen-gupta/red-dragon-forge). It has a `guards` query. The query lists the tests along each control-flow path to a statement. It does not track what the statements on a path assign, so it reports paths that cannot run.

I checked it on two statements:

- **Sign-on's transfer to the user menu.** `guards` returned five paths. Only one of them can run. A blank user id sets an error flag to `'Y'`, and the next test needs the flag off.
- **Bill pay's account `REWRITE`.** `guards` returned 64 paths, which is its cap. 55 of them contain a test together with its negation.

So the output is neither a necessary condition nor a sufficient one. A question such as "under which conditions does this statement run?" gets no answer from it. To answer it, something must track what each statement does to the data. That is the job of a solver.

## Horn clauses in one example

A constrained Horn clause (CHC) encoding writes a program once, with no particular question in mind:

- one predicate per program point
- one clause per control-flow edge
- on each clause, the constraint the statement puts on the data.

A question is one more clause, the goal. Take this fragment, simplified from sign-on:

```cobol
       MOVE 'N' TO WS-ERR-FLG.
       IF USERIDI = SPACES
           MOVE 'Y' TO WS-ERR-FLG
       END-IF.
       IF WS-ERR-FLG NOT = 'Y'
           EXEC CICS XCTL PROGRAM('COMEN01C') END-EXEC
       END-IF.
```

The clauses below are an illustration. They are not the exact encoding.

```
p1('N', uid)  <- p0(err, uid)
p2('Y', uid)  <- p1(err, uid), uid = SPACES
p2(err, uid)  <- p1(err, uid), uid != SPACES
xctl(err, uid) <- p2(err, uid), err != 'Y'
goal          <- xctl(err, uid), uid = SPACES
```

The goal asks: "can the program reach the `XCTL` with a blank user id?" Spacer answers in one of three ways:

- `unsat`: no run does it. The solver has an invariant as a certificate.
- `sat`: some run does it. The solver has a derivation as a witness.
- `unknown`: the solver stopped without an answer.

Here the answer is `unsat`. The only clause that gives a blank `uid` also sets `err` to `'Y'`, and the `XCTL` clause needs `err` to be something else.

## The plan: stages, each gated by real questions

I agreed on a design with the agent before it wrote code. The full design is in the repository. Its main decisions are:

- **Lower from cobble's own artefacts.** The input is the control-flow graph, the byte extents, the fixpoint dataflow and the cross-program links. The lowering does not read the Datalog export that cobble also writes.
- **Keep it removable.** The code lives in one package. Two import-linter contracts stop anything else from importing it or importing z3. Deleting the experiment means deleting that package.
- **Unknown means free.** Anything the encoding does not model exactly becomes an unconstrained value. A missing fact can only add runs, never remove them. So an `unsat` answer stays a proof at every stage. A `sat` answer may be spurious.
- **Gates come from the source first.** Each stage has questions whose answers I read from the CardDemo source, with file and line, before the code ran. A stage passes only if the solver agrees with every answer. `unknown` counts as a failure.

Stage 0 encoded only control flow and test atoms. Sign-on became 122 rules. The four gate questions answered in 0.006 to 0.035 seconds.

## Strings were too slow

Stage 1 added flag assignments: `SET` and `MOVE` of a literal. The state held each tracked field as a z3 string. The gate run did not finish. One query ran for 30 minutes before I stopped it.

A diagnostic run with a 60-second limit showed the problem. The four stage 0 questions still answered in under 0.1 seconds. All four questions about a blank or low-values user id or password hit the limit.

Two changes fixed it:

- **Bit-vectors in place of strings.** Each field became a bit-vector, 8 bits per byte.
- **Undeclared names are not tracked.** Four of the eleven tracked "fields" were names the program did not declare, such as `EIBAID`.

The same four questions then answered `unsat` in 0.22 to 0.28 seconds.

## Seven wrong proofs, and one fix for all of them

A fresh reviewing agent read the stage 0 and stage 1 code. It found seven cases where the encoding could answer `unsat` wrongly. That is the worst kind of error, because `unsat` is the answer that claims a proof. The cases included:

- a test on `F3(1:1)` read as a test on all of `F3`
- a sort `RETURN ... INTO` that recorded no write
- a qualified 88-level resolved to the wrong parent
- the edge past an unresolved `PERFORM` that carried values through as if nothing changed.

The agent proposed seven or eight conservative patches, one per case. I asked a different question: can the encoding follow byte extents directly? Would that remove all of them in one step?

It did. The state became the bytes that tests read. Overlapping extents merged into one bit-vector per segment. Each write came from the fixpoint's own transfer, in order. A known value sets bytes. Any other write frees the bytes it covers. A name the control-flow graph lists as written, but that the transfer missed, also frees its bytes. That fallback caught 34 missed writes in 40 online programs, such as `EXEC CICS FORMATTIME`.

Two of the cases were defects in cobble itself, not in the encoding. I paused the experiment and fixed them in cobble first.

With bytes, sign-on tracked 36 bytes in 8 segments. Bill pay tracked 50 bytes in 10 segments, with 328 rules. Its questions answered in 1.6 to 5 seconds. One answer showed a defect:

| Question at bill pay's `REWRITE` | Answer | Time |
|---|---|---|
| confirmation `'Y'`, error flag `'N'` | `unsat` | 3.7 s |
| confirmation `'Y'`, error flag `'Y'` | `sat` | 4.4 s |

A `sat` is only a run of the encoding, so I read the run in the source, `COBIL00C.cbl`:

```cobol
                   PERFORM WRITE-TRANSACT-FILE
                   COMPUTE ACCT-CURR-BAL = ACCT-CURR-BAL - TRAN-AMT
                   PERFORM UPDATE-ACCTDAT-FILE
```

When the `WRITE` fails, `WRITE-TRANSACT-FILE` sets the error flag to `'Y'` and sends the screen with an error message. It does not end the task. Control comes back to the lines above, which do not test the flag. So the program rewrites the account balance after its write to the transaction file fails.

## Copies make the state grow

Stage 2a encoded exact copies within one task. A `MOVE` whose bytes the fixpoint places sets each target byte equal to its source byte. Into a number, only the digit half of each byte is copied. The zone half, and so the sign, is left free, because compilers set zones and signs in different ways.

The tracked bytes grow with copies. If a question watches a field, the encoding must also follow the sources of every copy into it. Bill pay watching four fields grew from 50 bytes to 125. Its stage 0 and stage 1 questions became three to four times slower.

## A conversation is many tasks

A CICS online program does not run as one task. It shows a screen, ends with `RETURN TRANSID`, and starts again when the user presses a key. Programs also pass control to each other with `XCTL`. The data that survives between tasks is the commarea. Each new task starts with fresh working storage.

So stage 2b chained tasks. Each task of a program enters with the commarea bytes the previous task handed over, mapped byte for byte by cobble's link rows. A conversation starts with no commarea at all. If each task's commarea were left free, it could claim any user type. Then "can a normal user reach the admin menu?" would be `sat` for the model and false for the system.

The question for this stage was about the card-update program: can it `REWRITE` a card without a change check earlier in the conversation?

### The state explodes

Card update tracked about 1,100 bytes. The results:

| What was kept | Answer | Time |
|---|---|---|
| every test, three programs | none | stopped at 58 min |
| every test, card update's own chain | none | stopped at 25 min |
| control fields chosen by name, 112 bytes | `unsat` | 152 s |
| a control-dependence slice, 1,008 bytes | none | stopped at 10 min |

The only run that finished used fields chosen by name. That is a judgement, not an analysis. At this point the agent said the solver route was a dead end.

### One pertinent field

I asked what would happen with one pertinent field. A test that the encoding does not keep constrains nothing. So dropping a test is always sound. An `unsat` stays a proof. A `sat` may be a run that the dropped tests would have stopped.

| Fields kept | Scope | Answer | Time |
|---|---|---|---|
| change flag and `EIBCALEN`, 3 bytes | card update | `unsat` | 2.4 s |
| the same two | three programs | `sat`, spurious | 0.07 s |
| the same, and two commarea fields | three programs | `unsat` | 149.8 s |

The spurious `sat` was useful. It passed an entry test the wrong way, and that test named the two fields to add.

### Automatic refinement did not finish

That suggests a loop: ask, read the spurious run, keep the test it passed the wrong way, and ask again. This is counterexample-guided refinement (CEGAR). On card update's own chain it worked. It went from no fields to three in four rounds, in about 2.5 minutes.

Across three programs it did not finish in 40 minutes. Two problems caused this:

- **The witness costs too much.** Spacer gives a run in terms of the original rules only with its transforms off. With eager inlining off, a query that took 0.6 seconds took 316 seconds. With the transforms on, 26 of 35 steps in a witness were rewritten rules.
- **The ranking picked the wrong field.** It picked a message field, which took card update from 9 tracked bytes to 167.

### An LLM picks the fields

The loop needed judgement, not ranking. So a fresh agent read the three programs and named eight fields, with a line citation for each. The first question answered `unsat` in 159 seconds. A control question, built to be reachable, answered `sat` in 0.4 seconds.

This became part of cobble as "kept tests": a question can keep only the tests that name chosen fields. The LLM chooses. The solver proves. A wrong choice cannot produce a wrong `unsat`. It can only produce a spurious `sat`, or a slow query.

## Spurious runs are facts about the CFG

Other questions followed. Most spurious `sat` answers did not come from the encoding. They came from places where the control-flow graph is deliberately generous. Each one showed a fact the lowering had to add:

- **A `PERFORM` that never comes back.** The graph keeps an edge past a `PERFORM` whose range ends in an `XCTL`. That is a margin against its own mistakes. The solver used the edge, so a task ran on into code that only a later task reaches. The lowering now drops that edge when the program has no `ALTER` and every way out of the range ends the task.
- **A dynamic `XCTL`.** A transfer to a program named in a field has one link row for each program it can reach. Each row fired whatever the field held. The hand-over now fires only while the field holds that program's name.
- **`PGMIDERR`.** The graph gives every CICS command an edge into each armed condition handler. So a `SEND MAP` could appear to raise "program not found". The lowering now keeps that edge only from commands that the CICS reference says can raise it.

A different kind of gap appeared too. Thirteen CardDemo programs take in their commarea with `MOVE DFHCOMMAREA(1:EIBCALEN)`. The length is a variable, so the encoding freed the target, and every commarea field arrived unknown. Every question then came back `sat` in 0.07 seconds. The encoding now handles this move byte by byte: each target byte is the source byte while `EIBCALEN` exceeds its position, and a space after it.

## Fast tests that became slow

After the `EIBCALEN` change, the commit hooks ran for about two hours. A bisect found the cause. One bill-pay question went from 3 seconds to 138 seconds. The commarea copy added bytes that every rule had to carry.

Two cheaper encodings of the copy did not help. I asked how to make this reliable in general. The answer was to cut bytes, not paths. A test at a node from which no path reaches the goal cannot change the answer, whichever way it goes. Z3 already drops the rules that cannot reach the goal. But it keeps the bytes those rules read. The new cut drops those bytes too. Within one program the answers stay exact.

Bill pay went from 125 tracked bytes to 85. With a 300-second limit per question:

| Bill-pay question | Answer | Full | Cut |
|---|---|---|---|
| confirm: any | `sat` | 149.8 s | 37.6 s |
| confirm: neither `Y` nor `y` | `unsat` | 117.1 s | 43.9 s |
| confirm: blank account | `unsat` | 112.9 s | 42.5 s |
| confirm: low-values account | `unsat` | 171.8 s | 40.6 s |
| rewrite: `Y`, no error | `unsat` | 149.1 s | 60.1 s |
| rewrite: `Y`, error | `sat` | none | 37.2 s |
| written amount differs from the balance | `unsat` | 0.1 s | 0.0 s |

Across a conversation the same cut kept almost everything: 342 of 345 tests, and 1,678 of 1,750 bytes. In a conversation nearly every test leads somewhere. So I did not build it there.

## Splitting the question across programs

The three-program question was still slow, because the solver had to discover what card update can be entered with. So I tried to give it that fact and ask it to check the fact instead. This is an assume-guarantee split:

1. Write a rule P for what a card-update task can be entered with. P allows three cases:
   - no commarea
   - 160 bytes from the menu or the card list, with context `'0'`
   - 2,000 bytes from itself, where any pending change was checked
2. Ask, for each rule that enters a card-update task: can it enter with a state that breaks P?
3. Ask, for one task that starts from any state P allows: can it reach the `REWRITE` with no check?

The first version of P failed twice. Each counterexample showed a case I wrote too loosely. One had context byte `0x01`, so P now requires `'0'`. After that the split proved the gate in about 25 seconds in total. The hand-over checks took 0.1 to 23.5 seconds. The goal took 0.5 seconds.

So an LLM no longer only picks fields. It writes the hand-over description, and the solver checks it. A wrong description cannot give a wrong proof. The check fails, and the counterexample says where.

I also tried to give Spacer an invariant directly with `add_cover`. On a 16-bit counter loop it raised "Covers are incompatible with slicing". With slicing off, the process crashed with a segmentation fault. So the split checks its invariant by hand: entry implies P, P and a step imply P, and P with the goal is `unsat`.

## Writing the hand-over as a specification

Horn clauses are a poor language in which to say what a system does. They are too low-level for a person to read. I compared Lean 4, Dafny, and Quint (a modern syntax for TLA+). For a safety property over a conversation of tasks, Quint fits best. A part of the card-update specification:

```quint
action updateTask = all {
  running == CardUpdate,
  nondet ran = Set(true, false).oneOf()
  nondet saves = Set(true, false).oneOf()
  ...
  all {
    not(saves) or (calen == 2000 and (pending or ran)),
    rewroteUnchecked' = (rewroteUnchecked or (saves and not(checked or ran))),
    checked' = (checked or ran),
    ...
  },
}

val gate = not(rewroteUnchecked)
```

Apalache proved the gate and the hand-over description as an inductive invariant of the specification in 4.1 seconds. A copy of the specification with the check removed failed in the simulator in 11 milliseconds, in 3 steps.

That proves only the specification. To connect it to the code, each Quint action became a Horn check on the real programs. Each check enters a task from any state the specification allows. It then asks whether the task can end in a step that no Quint action allows. This is a refinement check. The three programs lowered to 1,991 rules in 0.9 seconds.

- All 13 start checks and all 11 hand-over checks answered `unsat`. The longest, card update returning to itself, took 22.6 seconds.
- The `REWRITE` never runs without a change check. This agrees with the earlier proof.
- The specification needed three revisions, each driven by a failing check:
  - The menu and the card list are also entered from programs outside these three.
  - The card list returns to itself with 2,000 bytes.
  - I let card update be entered with 160 bytes that name card update itself. The code does not do that. This was my error.

One check still fails. My specification says the `REWRITE` runs only on a return visit. The encoding can reach it on a first visit, when a check already ran earlier in the chain. This may be real. It may also be a run that the dropped tests would stop, because the checks keep tests on only eight fields. I have not checked which.

The mapping from Quint variables to bytes was the part that needed judgement. The specification's `pending` means that the change-action byte is `'N'`. The program's 88-level defines that value as "changes OK, not confirmed". I first assumed `'Y'`.

## Toward a harness

The translation from Quint to Horn checks was mechanical, but I wrote it by hand in a script. The probes that located each failure were also by hand. An agent can do this work. The question is what a harness outside the coding assistant adds. It adds what a session cannot enforce:

- **The harness runs the verdict.** The agent hands in the specification, the mapping and the checks. The harness runs Apalache and Spacer itself.
- **The property is locked.** The gate lives in a file the agent cannot edit.
- **The checks must fail on broken inputs.** Each round, the harness runs the checks on a specification with the gate broken, and on the code with the change check removed. If either passes, the checks prove nothing. A wrong byte offset gives a clean `unsat` that means nothing.
- **Every widening is logged.** Each fix so far made the specification allow more. The easiest way to pass a check is to allow everything.

One agent cannot both write a specification and judge its failures. In this work I did both, and one of the three revisions was my own error. So the loop needs two agents:

- **A writer** writes the specification, the mapping and the checks. It never judges a failure.
- **An interpreter** gets the code, the specification and the raw outputs. It does not get the writer's reasoning. It can run the solver itself, because a bare `sat` needs narrower probes before anyone can read it. It sorts each failure into one of four kinds:
  1. a gap in the specification
  2. an error in the specification
  3. an artefact of the encoding
  4. a real defect in the program.

The interpreter's feedback says what is wrong, with the code path as evidence. It never says how to fix it. A real defect stops the loop and goes to a person. An encoding artefact goes to whoever owns the encoding, not to the writer.

## What I took from it

- **`unsat` is the answer worth protecting.** Every design choice kept it a proof: unknown values are free, dropped tests constrain nothing, and a wrong hand-over description fails its check. The seven wrong proofs came from the first encoding, which followed names. Following bytes removed all of them.
- **Representation decides the cost first.** Strings to bit-vectors took questions from over 60 seconds to under 0.3 seconds. Cutting bytes no path needs took bill pay from about 150 seconds to about 40.
- **The state grows faster than the program.** One program was fine. Three programs in a conversation were not, until something chose what to keep.
- **Judgement and proof split cleanly.** An LLM chooses fields, or writes a hand-over description. The solver checks it. A bad choice costs time or a spurious `sat`, never a wrong proof.
- **Spurious runs are information.** Each one named a missing test, a missing CICS fact, or a generous edge in the control-flow graph.
- **A specification needs its own check against the code.** Apalache proved my Quint specification in seconds. The refinement check then found three places where the specification did not describe the code, and one of them was my mistake.

The encoding is documented in the repository's [Horn clause reference](https://github.com/avishek-sen-gupta/red-dragon-forge/blob/main/docs/reference/cobble/horn.md).
