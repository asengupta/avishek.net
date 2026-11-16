---
title: "Asking Z3 About COBOL Programs"
author: avishek
usemathjax: false
mermaid: false
tags: ["Software Engineering", "Program Analysis", "Static Analysis", "COBOL", "Reverse Engineering", "Formal Methods", "Z3", "Horn Clauses"]
draft: false
---

*A static analyser can list the tests on the way to a statement. It cannot say which of those paths a program can actually run. This post shows how a solver closes that gap for COBOL. Z3 reads each program as constrained Horn clauses and proves which runs are impossible. The examples come from CardDemo, the public CICS sample application.*

---

## Table of Contents

- [The problem](#the-problem)
- [Horn clauses in one example](#horn-clauses-in-one-example)
- [How the encoding works](#how-the-encoding-works)
- [What it found](#what-it-found)
- [Checking a specification against the code](#checking-a-specification-against-the-code)
- [Reaching conditions](#reaching-conditions)
- [Learnings](#learnings)

---

## The problem

cobble is the static analyser in [red-dragon-forge](https://github.com/avishek-sen-gupta/red-dragon-forge). Its old `guards` query listed the tests along each control-flow path to a statement. It did not track what the statements on a path assign, so most paths it listed cannot run:

- For sign-on's transfer to the user menu, it gave five paths. Only one can run.
- For bill pay's account `REWRITE`, it gave 64 paths. 55 of them contain a test and its negation.

To say which paths can run, something must track what each statement does to the data. That is a solver's job.

## Horn clauses in one example

A constrained Horn clause (CHC) encoding has one predicate per program point and one clause per control-flow edge. Each clause carries the constraint its statement puts on the data. A question is one more clause, the goal. Take this fragment of sign-on:

```cobol
       MOVE 'N' TO WS-ERR-FLG.
       IF USERIDI = SPACES
           MOVE 'Y' TO WS-ERR-FLG
       END-IF.
       IF WS-ERR-FLG NOT = 'Y'
           EXEC CICS XCTL PROGRAM('COMEN01C') END-EXEC
       END-IF.
```

In outline:

```
p1('N', uid)   <- p0(err, uid)
p2('Y', uid)   <- p1(err, uid), uid = SPACES
p2(err, uid)   <- p1(err, uid), uid != SPACES
xctl(err, uid) <- p2(err, uid), err != 'Y'
goal           <- xctl(err, uid), uid = SPACES
```

The goal asks: "can the program reach the `XCTL` with a blank user id?" Z3's Spacer engine answers in one of three ways:

- `unsat`: no run does it. This is a proof.
- `sat`: some run of the encoding does it.
- `unknown`: the solver stopped without an answer.

Here the answer is `unsat`. A blank user id sets the error flag, and the `XCTL` needs the flag off.

## How the encoding works

One rule governs the whole design: **anything the encoding does not model exactly becomes a free value.** A missing fact can only add runs, never remove them. So an `unsat` is always a proof, and a `sat` is only a candidate that someone must read.

- **The state is bytes, not names.** The state holds the bytes that the program's tests read, as bit-vectors. Each write comes from cobble's dataflow transfer for that statement. A known value sets bytes. An exact copy carries bytes from source to target. Any other write frees the bytes it covers.
- **A conversation is a chain of tasks.** A CICS program shows a screen, ends, and starts again on the next key press. Programs also hand control to each other with `XCTL`. Only the commarea survives between tasks. So each task starts with the commarea bytes the previous task handed over, mapped byte for byte from cobble's link data.
- **Facts about CICS remove impossible runs.** A `PERFORM` whose range always ends the task does not come back. A dynamic `XCTL` goes to a program only while the program-name field holds that name. Only the commands that can raise `PGMIDERR` lead into its handler.
- **A question keeps only the tests it needs.** A test the encoding drops constrains nothing, so dropping a test is always sound. Within one program, tests from which no path reaches the goal are dropped, together with the bytes they read. Across programs, an LLM reads the code and names the few fields that matter. The LLM chooses and the solver proves. A bad choice can cost time or give a spurious `sat`. It can never give a wrong proof.

## What it found

**Bill pay can rewrite a balance after a failed write.** The solver found a run that reaches the account `REWRITE` with the error flag set. The run is real. In `COBIL00C.cbl`:

```cobol
                   PERFORM WRITE-TRANSACT-FILE
                   COMPUTE ACCT-CURR-BAL = ACCT-CURR-BAL - TRAN-AMT
                   PERFORM UPDATE-ACCTDAT-FILE
```

When the `WRITE` fails, `WRITE-TRANSACT-FILE` sets the error flag and sends an error screen. It does not end the task. Control comes back here, and nothing tests the flag before the rewrite.

**Card update never saves a card without checking the changes.** Across card update, the card list and the main menu, the solver proved that no conversation reaches the card `REWRITE` without first passing the change check. The proof takes about three minutes.

## Checking a specification against the code

Horn clauses are too low-level for a person to read. So I wrote the card-update conversation as a specification in Quint, a modern syntax for TLA+:

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

This takes two checks:

1. **The specification is right about itself.** Apalache proves the gate as an inductive invariant in about four seconds.
2. **The code does what the specification says.** Each Quint action becomes a Horn check on the real programs. A task enters in any state the specification allows. The check then asks whether the task can end in a step that no action allows.

All 24 entry and hand-over checks answer `unsat`. One check is still open: the encoding can reach the `REWRITE` on a first visit, which the specification does not allow. That run may be real, or the dropped tests may stop it. I have not checked which.

## Reaching conditions

The original problem now has an answer. `cobble-experimental-reaching-conditions` replaces `guards`. It takes one statement and lists the **ways** the statement is reached:

- A way is the chain of tests the statement is control dependent on, each with the outcome it needs. Ways come from the program's structure, not from its paths, so an `IF` that rejoins adds nothing.
- Each way is one solver question. If the answer is `unsat`, the way is dropped.
- Every other test on the route is asked twice, once for each outcome. The output says whether the test can go **either way**, goes one way **if it runs**, or **never runs** on this way.

For sign-on's transfer to the user menu, five paths become one way:

```
COSGN00C READ-USER-SEC-FILE/1.when.0.0.then.5.else.0 (617): 1 way
  way 0: EIBCALEN == 0 (461) FALSE, EIBAID == DFHENTER (466) TRUE, ERR-FLG-ON (WS-ERR-FLG == 'Y') (519) FALSE, WS-RESP-CD == 0 (602) TRUE, SEC-USR-PWD == WS-USER-PWD (604) TRUE, CDEMO-USRTYP-ADMIN (CDEMO-USER-TYPE == 'A') (611) FALSE
    if it runs: USERIDI == SPACES (498) FALSE, USERIDI == LOW-VALUES (498) FALSE, PASSWDI == SPACES (498) FALSE, PASSWDI == LOW-VALUES (498) FALSE
    never runs on this way: %HANDLER[ANY] RAISED (579), %HANDLER[ANY] RAISED (583), %HANDLER[ANY] RAISED (532)
```

The blank-input tests show under "if it runs", with the outcome FALSE. A blank input sets the error flag, and this way needs the flag off.

On sign-on each question takes under a second. On the largest programs a question can hit the three-minute limit. So I am now field-testing a faster check. It writes the same question as one plain SMT formula. Each `PERFORM` is inlined at its call, and each loop is cut at its header, with every byte the loop may write freed. Freeing only adds runs, so an `unsat` from it is still a proof.

## Learnings

- **Protect `unsat`.** It is the answer that claims a proof, so every design choice must keep it sound. The first encoding followed field names, and a review found seven ways it could prove something false. Following bytes removed all seven at once. Leaving anything uncertain free keeps the proof safe as the encoding grows.
- **Representation decides the cost.** Fields as z3 strings made simple questions run past a minute. The same questions as bit-vectors took a fraction of a second.
- **State grows faster than the program.** One program was fast. Three programs in a conversation, with every test kept, had no answer after almost an hour. Keeping fewer tests is what made it work.
- **Let the LLM choose and the solver prove.** Automatic refinement from counterexamples picked the wrong fields and did not finish. An LLM that reads the code picks well. Its mistakes cost time, never correctness.
- **A spurious `sat` is information.** Each one named a missing fact: a test that should have been kept, a CICS rule, or an edge the control-flow graph keeps only to be safe.
- **Fix a fact where it is made.** An 88-level with several values reached the encoding as one string, so its tests constrained nothing. The fix belonged in the control-flow graph, where the 88-level now becomes one test per value, chained as an `OR`.
- **Check the specification against the code.** Apalache proved my specification in seconds. The refinement checks then found three places where it did not describe the code, and one of them was my own mistake.
- **The author should not judge their own failures.** I wrote the specification and also decided what each failure meant. A harness should split those jobs between two agents. One writes the specification. The other reads each failure and classifies it, without seeing the writer's reasoning.

The encoding is documented in the repository's [Horn clause reference](https://github.com/avishek-sen-gupta/red-dragon-forge/blob/main/docs/reference/cobble/horn.md).
