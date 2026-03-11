---
title: "Building Non-Trivial Systems with an AI Coding Assistant"
author: avishek
usemathjax: false
tags: ["Software Engineering", "Compilers", "Program Analysis", "AI-Assisted Development"]
draft: false
---

*Notes from building two code analysis tools — a reverse engineering toolkit and a multi-language compiler pipeline — across 600+ conversation sessions with Claude Code.*

---

## Table of Contents

- [Context](#context)
- [The Reverse Engineering Tool: Learning to Direct an AI](#the-reverse-engineering-tool-learning-to-direct-an-ai)
  - [Integration Detection and Classification](#integration-detection-and-classification)
  - [Embedding Experiments](#embedding-experiments)
  - [Takeaways from the First Project](#takeaways-from-the-first-project)
- [RedDragon: How the Architecture Emerged](#reddragon-how-the-architecture-emerged)
  - [The Initial Session (Feb 25–26)](#the-initial-session-feb-2526)
  - [The Determinism Pivot](#the-determinism-pivot)
  - [Deterministic Frontends for 15 Languages](#deterministic-frontends-for-15-languages)
  - [The Implementation Rhythm](#the-implementation-rhythm)
- [Growing the Test Suite](#growing-the-test-suite)
  - [Cross-Language Testing via Rosetta and Exercism](#cross-language-testing-via-rosetta-and-exercism)
  - [The Dispatch Audit Loop](#the-dispatch-audit-loop)
- [The Assertion Audit](#the-assertion-audit)
  - [Weak Assertion Patterns](#weak-assertion-patterns)
  - [The Audit Process](#the-audit-process)
  - [Bugs Found Behind Weak Assertions](#bugs-found-behind-weak-assertions)
  - [Assertion Audit Lessons](#assertion-audit-lessons)
- [Guardrails: CLAUDE.md](#guardrails-claudemd)
  - [Build Rules](#build-rules)
  - [Testing Rules](#testing-rules)
  - [Programming Rules](#programming-rules)
  - [The Workflow Evolution](#the-workflow-evolution)
- [Structured Agent Memory](#structured-agent-memory)
  - [The Continuity Problem](#the-continuity-problem)
  - [Issue Tracking with Beads](#issue-tracking-with-beads)
  - [Gap Analysis as Planning](#gap-analysis-as-planning)
  - [The Type System Evolution](#the-type-system-evolution)
  - [Memory Files](#memory-files)
  - [The Quick Win Trap](#the-quick-win-trap)
- [Patterns and Observations](#patterns-and-observations)
- [What I Would Change](#what-i-would-change)
- [Conclusion](#conclusion)

---

## Context

Over January–March 2026, I built two open-source code analysis tools almost entirely through conversations with Claude Code, plus a terminal UI that integrates them:

- An internal reverse engineering toolkit that detects integration points in source code using regex patterns, ML classifiers, code embeddings, and LLM classification — 250 commits across 35 days
- **[RedDragon](https://github.com/avishek-sen-gupta/red-dragon)**: A multi-language code analysis engine with a universal IR, deterministic VM, type system, and iterative dataflow analysis

The reverse engineering tool accumulated ~195 conversation sessions. RedDragon was built in an initial session, then refined across 237+ more, with 73 additional sessions on its precursor project. That's roughly 600+ human-AI conversation sessions total.

This post documents what I learned about directing an AI to build systems of this scale.

---

![Demo](/assets/red-dragon-tui.gif)

## The Reverse Engineering Tool: Learning to Direct an AI

The first project was a repo surveying toolkit — given a repository path, it would identify the technology stack, detect integration points (HTTP, SOAP, messaging, database connections), and classify them by type and direction.

The first session produced a `RepoSurveyor` class that walked directory trees and identified languages by file extension. Over the next five days, the project developed its engineering principles: every external system boundary became a protocol, enums replaced strings throughout, and integration detections were renamed from `IntegrationPoint` to `IntegrationSignal` — because detections are signals, not confirmed facts.

The structural analysis phase added call-flow extraction via LSP (using tree-sitter queries instead of regex), framework-aware integration detection (restructured into a plugin architecture with declarative configuration), and a language-independent CFG builder on tree-sitter parse trees. The CFG work started with LLM-generated role mappings for 99 languages, but the output was unreliable. The mappings were hand-authored for 14 languages instead. This became a recurring pattern: **use LLM to bootstrap, then replace with a deterministic approach once the problem is understood.**

### Integration Detection and Classification

The core challenge was that regex-based pattern matching produces many false positives. How do you classify which detected signals are genuine and which direction they face (inward vs outward)?

The approach went through several iterations:

**LLM classification** worked on small inputs, but when run on a real Java repository (2,116 signals across 1,809 groups), it needed 37+ batches. Too slow. The LLM path was removed, though AST walk-up (grouping signals by enclosing function) was retained.

**ML classifier.** A TF-IDF + logistic regression model trained on data generated via Claude's Batches API and a GitHub training-data harvester. Fast, but mediocre — confidence scores were low on framework-specific patterns.

**Code embeddings.** Testing `nomic-embed-code` and Gemini's embedding model showed that embeddings could separate integration code from non-I/O code, but directional classification was weak.

### Embedding Experiments

Five embedding backends were compared:

| Backend | Type | Result |
|---------|------|--------|
| nomic-embed-code | Cloud API | 91% on small test |
| Gemini embedding-001 | Cloud API | 91% on small test |
| CodeT5p-110m | Local, code-specific | 50% — scores too compressed |
| CodeRankEmbed | Local, code-specific | 4.5% — failure |
| **BGE-base-en-v1.5** | Local, general-purpose | **100%** on all test sets |

The finding: **general-purpose text embedding models trained for semantic similarity outperformed code-specific models** on this description-to-code classification task. MTEB leaderboard scores did not predict performance.

A sub-experiment through iterative testing against a single Java line showed that **passive-voice, subject-first descriptions** maximise cosine similarity in embedding space. Scores improved from 0.448 to 0.773. All 3,017 pattern descriptions were rewritten to this template.

The final architecture was a two-stage pipeline: an embedding gate (BGE, distance-weighted KNN) to separate signal from noise, then Gemini Flash to classify direction on survivors only. Seven distinct classification pipelines were built over the course of the project, all sharing the same detection and output phases, differing only in classification.

### Takeaways from the First Project

The human's job was strategic, not tactical. I wasn't writing code. I was evaluating approaches by running them on real data, making pivot decisions based on empirical feedback, composing architectures, and interrupting when a direction wasn't working.

My prompts got terser as trust built up. Early on: detailed specifications with context. By day 3: *"do all of them"*, *"push"*, *"run it on smojol and show me the results"*.

---

## RedDragon: How the Architecture Emerged

### The Initial Session (Feb 25–26)

On Feb 25, I opened a fresh session and described what I wanted: a universal symbolic interpreter that parses source in any language, lowers it to a flat IR, builds a CFG, and executes it symbolically, handling missing imports and unknown externals gracefully.

The first thing I asked: *"Is there an existing IR/VM that already does this?"* There wasn't a good fit for what I needed — symbolic execution of incomplete programs across 15 languages — so we proceeded.

The git log from that day shows the progression:

```
3bfbead  Initial implementation of LLM symbolic interpreter
7eb9721  Local function dispatch, builtins, and scope chain
ed4a3c8  Break up interpreter.py into modular package
b459cdc  Make VM fully deterministic: replace all LLM fallbacks
         with symbolic value creation
bd51810  Add LLM-based frontend for multi-language source-to-IR lowering
4b6f815  Add closure support
6bdd973  Add deterministic tree-sitter frontends for 14 languages
         (346 tests, 0 failures)
```

Seven commits, each one a distinct architectural decision. The ADR log (which I had Claude write retroactively) captures the reasoning behind each.

### The Determinism Pivot

The initial design had the LLM deciding state changes at each execution step. After implementing it, I asked: *"Given that the IR is always bounded, shouldn't the IR execution be deterministic?"*

This turned out to matter. We replaced all LLM calls in the VM with symbolic value creation. When the VM encountered an unresolved import, it created a `SymbolicValue` with a descriptive hint instead of asking an LLM. The value propagated through computation deterministically. The entire execution became reproducible.

With the VM deterministic, the LLM's role narrowed to one thing: translating source code to IR. And even that was constrained — the prompt provided all opcode schemas, concrete patterns, and a worked example. The LLM was acting as a mechanical translator, not a reasoning engine.

This was the decision that shaped everything else. Once execution was deterministic, everything became testable. The entire test suite runs with zero LLM calls.

### Deterministic Frontends for 15 Languages

Rather than using the LLM at runtime to lower source to IR, I asked: *"How hard is it to write deterministic logic to lower ASTs to IR for 16 languages?"*

Not that hard, with tree-sitter and a dispatch table engine. Claude generated tree-sitter-based frontends for 14 languages in a single session. Each frontend extends a `BaseFrontend` class with two dispatch tables (one for statements, one for expressions) mapping AST node types to handler methods. Common constructs (`if/else`, `while`, `for`, `return`) are handled in the base class. Language-specific constructs override or extend.

Sub-millisecond. Zero LLM calls. Fully testable. 346 tests on day one.

When the LLM frontend hit context window limits on large files, we added a chunked frontend that decomposes files into per-function chunks via tree-sitter, lowers each independently, then renumbers registers and reassembles.

### The Implementation Rhythm

The initial session followed a repeating cycle:

1. **Implement a feature** (30–60 minutes)
2. **Run it on real code** and inspect the output
3. **Identify the next gap** ("any other language features not covered?")
4. **Audit for completeness**, then batch-implement all gaps
5. **Clean up immediately**: refactor, split large files, reorganise tests

I didn't let technical debt accumulate. When `interpreter.py` hit 1,200 lines, I said *"break up interpreter.py, it's too big."* When the registry module grew three responsibilities, I split it into three files. When tests were in a flat directory, I separated them into `unit/` and `integration/`.

Filling language-specific gaps across all 15 frontends was systematic: ask Claude to audit every frontend for missing constructs, prioritise by impact, say *"implement all the critical and common ones"*, push, re-audit. This cycle repeated 4–5 times, each time catching a smaller set of remaining gaps.

---

## Growing the Test Suite

### Cross-Language Testing via Rosetta and Exercism

The test count tells the progression:

```
Initial frontends:    346 tests
After tooling:       ~700 tests
Rosetta suite:     ~1,200 tests
Exercism (final):   7,268 tests
COBOL + audit:      8,569 tests
Type system:       ~9,400 tests
Gap analysis:     10,152 tests
```

The Rosetta cross-language test suite (15 algorithms across 15 languages) and the Exercism integration suite drove most of this growth. Each exercise exposed new frontend gaps, VM limitations, and edge cases, and each fix was immediately verified across all 15 languages.

All tests run without LLM calls and are deterministic.

### The Dispatch Audit Loop

After every batch of frontend work, I ran a two-pass audit:

**Pass 1 (Dispatch Comparison):** Parse source samples in all 15 languages, collect every AST node type that appears, compare against the frontend's dispatch tables, and classify unhandled types as structural (harmless, consumed by parent handlers) or substantive (gaps that produce `SYMBOLIC`).

**Pass 2 (Runtime SYMBOLIC check):** Lower the source through each frontend, scan the resulting IR for `SYMBOLIC` instructions with `"unsupported:"` operands.

The classification heuristic itself went through three iterations:

1. **Naive:** Flag everything not in a dispatch table. Hundreds of false positives, because nodes like `parameter_list` are consumed by parent handlers.
2. **Parent heuristic:** Flag unhandled nodes only if their immediate parent isn't handled. Reduced false positives but still produced 259.
3. **Block-reachability analysis:** Walk the AST and identify which unhandled nodes are direct named children of block-iterated nodes. Only these can reach `_lower_stmt` and produce `SYMBOLIC`. This dropped substantive gaps from 259 to 1.

The audit loop ran dozens of times:

```
Audit -> 34 gaps found -> implement all 34 (57 new tests)
Re-audit -> 19 gaps found -> implement all 19 (28 new tests)
Re-audit -> 12 gaps found -> implement all 12 (18 new tests)
Re-audit -> 0 gaps, 0 SYMBOLIC
```

This pattern (audit, batch-fix, re-audit) was more effective than trying to enumerate every missing feature upfront.

---

## The Assertion Audit

By March 2026, the test suite had grown to ~8,400 tests across ~130 files. All green. The question I'd been putting off: **if every test passes, how do I know each test is actually checking what it says it's checking?**

When an AI writes your tests, you get volume and coverage breadth. What you don't get is *assertion depth*. The AI produces a test for every function, every edge case, every language, but each individual assertion may be checking the easy thing (does it not crash?) rather than the hard thing (does it produce the right output?). The AI optimises for the test *passing*, not for the test *verifying*.

### Weak Assertion Patterns

Over two days and 11 audit passes, I had Claude scan every test file, comparing each test's name against its actual assertions. The patterns that emerged:

**OR-fallback assertions.** Tests like `assert Opcode.BRANCH_IF in opcodes or Opcode.BRANCH in opcodes` — where `BRANCH` (unconditional jump) exists in virtually every program, making the assertion tautologically true. 23+ instances across Scala match/case, C# switch expressions, COBOL PERFORM ordering, and IR stats tests.

**Existence-only checks.** `assert len(writes) >= 1` on WRITE_REGION, satisfied by DATA DIVISION initial-value writes, leaving the PROCEDURE statement under test untested. The strengthened version decoded the EBCDIC bytes and checked specific values.

**Cross-product matching.** `assert any(bi < pi for bi in branch_if_indices for pi in print_indices)` — `any()` over a cross-product matches if *any* `BRANCH_IF` appears before *any* print, even from unrelated parts of the program.

**Silent parametrised passes.** Bare `return` in parametrised tests for excluded languages — 11 languages were showing as green in the closure test report with zero assertions executed. The fix was `pytest.skip()` with a reason string.

**Tautological guards.** `if "x" in result.definitions:` where `result.definitions` was a list of `Definition` objects, not a dict. The `in` check always returned `False`, so the assertion never fired.

### The Audit Process

**Phase 1: Discrepancy audits.** Tests whose names contradicted what the code did. A diamond-shape test that asserted stadium shape; `test_constructor_sets_fields` that never verified field values.

**Phase 2: Name-vs-assertion audits.** Does each test assert what its name claims? 52 violations across 22 files.

**Phase 3: Priority-based audits.** P0 (false confidence), P1 (missing key assertion), P2 (weak/generic), P3 (cosmetic). Re-scans after each fix batch drove the count down: 82 to 56 to 17.

**Phase 4: Reconciliation.** The violation list kept changing between audits. Items fixed reappeared with different wording. The fix was anchoring: starting from the previous audit's known remaining items and verifying each against the current code.

The governing principle throughout: **strengthen the assertion to match the name, never weaken the name to match the assertion.** Renaming moves the problem. Strengthening closes the gap.

### Bugs Found Behind Weak Assertions

P0 fixes exposed genuine bugs:

**Pascal bare-except.** The Pascal frontend silently dropped bare `except` blocks (without `on E: Exception do` wrapper). The test passed because it only checked that `STORE_VAR "x"` existed, satisfied by the try body alone. Strengthening the assertion exposed the bug.

**C# else-if chain lowering.** A weak assertion masked incomplete lowering of chained else-if blocks.

**`test_no_self_dependency_without_loop`.** A guard checking membership on a list of `Definition` objects always returned `False`, so the assertion never executed.

### Assertion Audit Lessons

| Metric | Value |
|--------|-------|
| Audit passes | 11 |
| Unique violations (deduplicated) | ~90 |
| Violations fixed | ~75 |
| Frontend bugs exposed | 2 |
| False-pass tests eliminated | 5 |

The test count went *up*, not down. Strengthening assertions sometimes meant splitting one weak test into multiple specific ones.

Key lessons:

- **Green tests are necessary but not sufficient.** 15 P0 violations where the test would pass even when the feature was broken.
- **OR-fallback assertions are the most dangerous pattern.** One side trivially satisfied by unrelated instructions.
- **Audit stability requires anchoring.** Fresh scans produce inconsistent results. The reconciliation approach is necessary.
- **Fixing assertions requires running the code.** Many fix attempts failed because the assertion assumed a representation that didn't match reality. The cycle (write assertion, run, discover actual representation, fix, re-run) never happened voluntarily.
- **Parametrised tests need explicit skips, not silent returns.**

---

## Guardrails: CLAUDE.md

The file that had the most impact on consistency wasn't any Python module. It was `CLAUDE.md`, the development rules file that Claude Code reads at the start of every session. The rules evolved over the project's lifetime, each one added in response to a specific failure mode.

### Build Rules

**"Before committing anything, run all tests, fixing them if necessary."** This prevented test count regression across 292 commits. If test assertions are being *removed*, ask for human review first.

**"Before committing anything, run `poetry run black` on the full codebase."** CI enforces this.

**"Before committing anything, update the README based on the diffs."** Without this, the README would have drifted within the first week.

**"For each feature, treat it as an independent commit / push, with its own testing."** Atomic, reviewable commits. Combined with "do not start a new task until the current one is committed," this prevented half-finished features from accumulating across sessions.

**"Once a design is finalised, document it as an ADR."** This produced 100+ architectural decision records that serve as the project's institutional memory.

### Testing Rules

**"When fixing tests, do not blindly change test assertions to make the test pass."** Without this, the AI's default behaviour is to modify the assertion to match whatever the code produces, regardless of whether the code is correct.

**"Make sure you are not creating any special implementation behaviour just to get the tests to pass."** Without this, the AI occasionally added if-branches in production code solely to satisfy a test expectation.

**"Do not use `unittest.mock.patch`. Use proper dependency injection."** This forced every external dependency to be injectable. The entire VM, all 15 frontends, and all analysis passes are testable in isolation.

**"For every bug you fix, make sure you have a test that fails without the bug fix."**

### Programming Rules

**"STOP USING FOR LOOPS WITH MUTATIONS IN THEM."** This forced a functional style: list comprehensions, `map`, `filter`, `reduce` instead of mutable accumulators.

**"Categorically avoid defensive programming."** Defensive code hides bugs. A `None` check that silently returns an empty list masks the fact that a value should never have been `None`. Without this rule, the AI adds defensive checks reflexively.

**"If a function has a non-None return type, never return None."** Use null object pattern instead.

**"When writing `if` conditions, prefer early return."** Without this, the AI nests the happy path inside increasingly deep conditionals.

**"Do not use static methods."** Static methods resist dependency injection and create hidden coupling.

**"Use a ports-and-adapter type architecture. Adhere to 'Functional Core, Imperative Shell'."** The VM handlers are pure functions returning `StateUpdate` data objects. The dataflow module is a pure analysis pass. I/O lives at the edges.

**"Parameters in functions, if they must have default values, must have those values as empty structures corresponding to the non-empty types."** Empty dicts, empty lists, never `None`.

### The Workflow Evolution

The workflow encoded in CLAUDE.md changed over time. Early: **Brainstorm -> Plan -> Implement -> Test.** Tests came after implementation, which led to the weak assertion patterns the audit uncovered.

The revised workflow: **Brainstorm -> Discuss trade-offs -> Plan -> Write unit tests -> Implement -> Fix tests -> Commit -> Refactor.** Tests come before implementation. The AI writes tests that encode expected behaviour, then writes code to make them pass.

Every rule was reactive. "STOP USING FOR LOOPS WITH MUTATIONS" came after mutation bugs. "Don't blindly change test assertions" came after watching the AI weaken tests to make them pass. "Categorically avoid defensive programming" came after silent `None` checks masked real bugs. Each rule represents a mistake that happened at least once.

---

## Structured Agent Memory

### The Continuity Problem

By session 200, the project had outgrown ad-hoc session management. At 8,500+ tests, 15 frontends, 66 ADRs, and a type system under active development, I'd open a new conversation and describe what I wanted — but determining *what* was getting harder. Which frontend gaps remained? Which were P0 vs. P1? What depended on what?

The deeper problem was agent amnesia. Each conversation started from zero. Claude didn't remember that the TypeExpr migration was complete, that Pascal `declLabels` was already handled, that `poetry run python -m pytest` was the correct test command. At 50 sessions, this was a minor annoyance. At 200, it was the main cost.

The solution was a **structured memory layer** — persistent artefacts that the agent reads at session start. Four components: a curated memory file, a gap analysis document, an issue tracker, and architectural decision records. Together they answer: *what's the current state?*, *what's left to do?*, *what are the rules?*, and *why were past decisions made?*

### Issue Tracking with Beads

The inflection point was a frontend lowering gap analysis: cross-referencing every frontend's dispatch table against its tree-sitter grammar. Result: 25 P0, 187 P1, and ~326 P2 gaps across 15 languages. The P0s were resolved in three commits. But 187 P1 gaps couldn't be managed as a mental list.

I chose [Beads](https://github.com/eqlabs/beads), a local-first issue tracker that stores data in a Dolt database alongside the repo. It supports hierarchical issues (epics -> stories -> tasks), dependency chains, labels, and priority classifications from the command line.

The 187 P1 gaps became a structured breakdown:

```
red-dragon-gvu [epic]: 129 P1 frontend lowering gaps
├── gvu.1 [epic]: Cross-language pattern matching (25 tasks across 6 languages)
├── gvu.2 [epic]: Class/OOP features (14 tasks)
├── gvu.3 [epic]: Type system and generics (11 tasks)
├── gvu.4 [epic]: Async/coroutine support (8 tasks)
├── gvu.5 [epic]: Destructuring and rest patterns (7 tasks)
├── gvu.6 [epic]: Metaprogramming/macros (9 tasks)
├── gvu.7 [epic]: Module system and imports (8 tasks)
└── gvu.8 [epic]: Language-specific features (remaining)
```

This changed how sessions started. Instead of describing work from memory, I could say *"what's the next open task under gvu.1?"* The issue tracker became the session boundary — work was defined before the session began, not discovered during it.

Beads data is backed up as JSONL files committed to the repository, so issue state travels with the code. On a new machine, `bd backup restore` rebuilds the full database — 168 issues, 538 events, 163 dependencies.

### Gap Analysis as Planning

The gap analysis document (`docs/frontend-lowering-gaps.md`) became a living planning artefact. Each P1 gap had a row with language, node type, description, and status. As gaps were resolved, status flipped to DONE. As new gaps were discovered, they were added.

Many P1 gaps clustered thematically. Pattern matching was a gap in 6 languages simultaneously. Class/OOP features were missing in 5. The right unit of work wasn't "fix Go's missing `iota`" — it was "implement pattern matching across all 6 languages that need it." The themed epics emerged from the data, not from upfront planning.

The analysis also forced honest assessment of what "done" meant. I'd have Claude classify some gaps as "no-ops." When I asked Claude to verify each claimed no-op against the actual tree-sitter AST, most contained meaningful content. TypeScript's `ambient_declaration` had full type signatures. C#'s `unsafe_statement` wrapped executable blocks. Out of 8 claimed no-ops, only 2 were genuine. The lesson: **verify against the AST, not against assumptions about what a node type name implies.**

### The Type System Evolution

The largest post-sprint development was the type system. RedDragon started with string-based type hints — `"Int"`, `"String"`, `"List<Int>"` — threaded through the IR as operand annotations. This worked for simple inference but couldn't represent parameterised types, union types, or subtype relationships.

The evolution happened in phases, each driven by a concrete limitation:

**Phase 1: TypeExpr ADT.** Replaced string type hints with an algebraic data type: `ScalarType("Int")`, `ParameterizedType("List", (ScalarType("Int"),))`, `UnknownType`. A `parse_type()` function handled roundtripping from legacy strings. String-compatible equality was preserved for backward compatibility during the migration.

**Phase 2: TypeGraph.** A directed acyclic graph encoding subtype relationships (`Int <: Number <: Object`, `String <: Object`). Covariant `is_subtype_expr()` for parameterised types. `common_supertype_expr()` for join operations.

**Phase 3: Interface-aware inference.** When the inference engine encountered `animal.speak()` where `animal` was typed as interface `Animal`, it couldn't resolve the return type. The fix was a chain walk: check interface method types when the class's own methods don't have type information. This required seeding `interface_implementations` from 5 frontends.

Each phase was documented as an ADR, tested with both unit and integration tests, and committed independently. The type system alone accounts for ~1,500 tests and 34 ADRs.

### Memory Files

Claude Code supports a persistent memory directory (`.claude/projects/.../memory/MEMORY.md`) loaded at session start. My memory file contains:

- **Key references**: Links to audit results, gap analyses, and their current status
- **Workflow reminders**: Commands that have been forgotten before
- **Project state**: Current test count, resolved issues, known gotchas
- **Type system state**: Which migration phases are complete
- **Future work pointers**: What's been deferred and why

The memory file is curated, not appended. Outdated information is removed. Entries are updated when facts change.

The distinction between the memory file and `CLAUDE.md`: `CLAUDE.md` encodes *rules*. The memory file encodes *state*. Together with the gap analysis (*plan*) and the issue tracker (*work queue*), they form a four-layer structured memory:

| Layer | Artefact | Purpose | Update frequency |
|-------|----------|---------|-----------------|
| Rules | `CLAUDE.md` | How to behave | When failure modes are discovered |
| State | `MEMORY.md` | What's been done | Every few sessions |
| Plan | Gap analysis doc | What's left to do | After each implementation batch |
| Work queue | Beads issues | What to do next | After each task closes |

A fresh session can orient itself in seconds rather than minutes.

### The Quick Win Trap

When managing 168 issues, there is a temptation to chase quick wins — tasks that appear trivial. During the gap analysis breakdown, Claude identified ~16 "quick wins" including 8 claimed no-ops. When I asked Claude to verify each one against the actual AST structure, the list shrank to 7, with only 2 genuine no-ops. The rest needed real handlers.

**What looks like a no-op from the node type name often isn't.** `ambient_declaration` sounds like metadata; it's actually `declare const VERSION: string`. `unsafe_statement` sounds like a compiler pragma; it's actually a block wrapper around executable code.

The AI will optimise for closing tickets, not for closing them correctly. The human's job is to challenge the classification before the work begins.

---

## Patterns and Observations

**Brainstorm, probe, crystallise.** I didn't start with fixed architectures. I started with a problem, brainstormed approaches with Claude, then implemented each and tested on real data. The deterministic VM emerged from asking "shouldn't this be deterministic?" after seeing the LLM-based approach work.

**The plan document as interface.** After brainstorming and discussing trade-offs, I'd formulate a plan document covering context, phases, file-by-file changes, and verification steps. The plan is specific enough for unambiguous execution but high-level enough to retain architectural control. This happened ~15 times.

**Breadth over depth.** Tasks like "generate frontends for 14 languages" or "audit all 130 test files for weak assertions" are where the AI works well. These breadth tasks — applying a consistent pattern across many targets — would have taken days. Where it needed more guidance was depth: closure capture semantics (snapshot vs. shared environment), when to use `SYMBOLIC` fallback vs. crash, whether an assertion is vacuous. These required me to probe with specific test cases.

**Empirical validation over specification.** I rarely specified exact behaviour upfront. I implemented a feature, ran it on real code, and judged the results. The AI made this feedback loop fast enough to be practical.

**Terse directives after trust.** Early prompts were detailed. By mid-project: *"do all of them"*, *"push"*, *"commit and push this"*. Trust built through consistent execution.

**The AI hallucinated audit findings.** During the assertion audit, the AI reported violations that didn't exist or had already been fixed. Different parallel agents flagged different things, inconsistently applied priority criteria, and re-reported fixed items with different wording. The reconciliation pass caught this. The auditor itself needs auditing.

**CLAUDE.md rules are reactive.** Every rule was added in response to a specific failure. They accumulate over time, and each one represents a mistake that happened at least once.

**Screenshot-driven debugging.** For the CFG visualisation work, I'd generate a diagram, screenshot it, paste it into the conversation, and ask "why does it look so disjointed?" Claude could see the rendering and diagnose layout issues. The visualisation went through five rounds.

---

## What I Would Change

**Start with the audit earlier.** The two-pass dispatch audit should have existed from the first batch of frontends, not after 50 sessions.

**Invest in cross-language tests from day one.** The Rosetta and Exercism suites exposed more bugs than all the language-specific unit tests combined. A single exercise tested across 15 languages covers more surface area than 50 unit tests in one language.

**Be more aggressive about the functional core.** Even with the FP rules in CLAUDE.md, some mutation crept in, especially in the VM executor. The dataflow module is almost purely functional and is the easiest module to test. The correlation is not a coincidence.

---

## Conclusion

Building non-trivial systems with an AI coding assistant is about architectural direction, not prompting. The human's role is strategic: choosing problems, evaluating approaches empirically, making pivot decisions, and encoding quality standards. The AI's role is tactical: implementing plans, auditing for completeness, applying patterns at breadth, and maintaining consistency.

The workflow I converged on (brainstorm, discuss trade-offs, plan, test-first, implement, clean up) emerged through trial and error across 400+ sessions.

What changed between the early sessions and the later ones was the emergence of structured agent memory. The early sessions were exploratory, driven by momentum. The later sessions were systematic — the agent started each conversation knowing the project state, the active work queue, and the governing rules. The human's role expanded from architect to memory curator: defining work, challenging classifications, and maintaining the persistent artefacts that make each session productive from its first message.

The limiting factor in AI-assisted development is not the AI's capability — it's the AI's memory. A capable agent with no memory rediscovers context every session. A capable agent with structured memory picks up where the last session left off.

_This post has not been written or edited by AI._
