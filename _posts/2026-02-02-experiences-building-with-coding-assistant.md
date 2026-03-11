---
title: "Architecting Non-Trivial Systems with Claude Code: A Practitioner's Account"
author: avishek
usemathjax: false
tags: ["Software Engineering", "Compilers", "Program Analysis", "Code Embeddings", "AI-Assisted Development"]
draft: false
---

*How I built three interconnected code analysis tools — spanning 15 language frontends, a deterministic VM, dataflow analysis, a full type system, 7 embedding/ML/LLM classification pipelines, and Datalog engines — across 40+ days, 10,000+ tests, and 600+ conversation sessions with an AI pair programmer.*

---

## Table of Contents

- [The Setup](#the-setup)
- [Part 1: Codescry, Learning to Steer](#part-1-codescry-learning-to-steer)
  - [Origins: Cartographer (Jan 30 – Feb 3)](#origins-cartographer-jan-30--feb-3)
  - [Structural Analysis: Call Flow and CFGs (Feb 9 – Feb 18)](#structural-analysis-call-flow-and-cfgs-feb-9--feb-18)
  - [The Concretisation Problem (Feb 18 – Feb 25)](#the-concretisation-problem-feb-18--feb-25)
  - [Engineering Maturity (Feb 20 – Mar 2)](#engineering-maturity-feb-20--mar-2)
  - [What I Learned: Steering](#what-i-learned-steering)
- [Part 2: RedDragon (Feb 25–26)](#part-2-reddragon-feb-2526)
  - [The Starting Point](#the-starting-point)
  - [How the Architecture Took Shape](#how-the-architecture-took-shape)
  - [The Implementation Rhythm](#the-implementation-rhythm)
  - [Filling Language Feature Gaps](#filling-language-feature-gaps)
  - [Screenshot-Driven Debugging](#screenshot-driven-debugging)
- [Part 3: Patterns That Emerged](#part-3-patterns-that-emerged)
- [Part 4: Where It Surprised Me](#part-4-where-it-surprised-me)
- [Part 5: The Evolution, From Monolith to 10,000+ Tests](#part-5-the-evolution-from-monolith-to-10000-tests)
- [Part 6: The Audit Loop, Systematic Completeness](#part-6-the-audit-loop-systematic-completeness)
- [Part 7: The Assertion Audit, Green Tests, False Confidence](#part-7-the-assertion-audit-green-tests-false-confidence)
  - [The Taxonomy of Weak Assertions](#the-taxonomy-of-weak-assertions)
  - [The Audit Timeline](#the-audit-timeline)
  - [The Governing Principle: Strengthen, Don't Rename](#the-governing-principle-strengthen-dont-rename)
  - [Errors During Fix Attempts](#errors-during-fix-attempts)
  - [Real Bugs Found](#real-bugs-found)
  - [The Feedback Loop Structure](#the-feedback-loop-structure)
  - [Assertion Audit Results](#assertion-audit-results)
  - [Assertion Audit Lessons](#assertion-audit-lessons)
- [Part 8: Guardrails, The CLAUDE.md as Architecture](#part-8-guardrails-the-claudemd-as-architecture)
- [Part 9: What I'd Do Differently](#part-9-what-id-do-differently)
- [Part 10: The Numbers](#part-10-the-numbers)
- [Part 11: From Sprints to Structured Agent Memory](#part-11-from-sprints-to-structured-agent-memory)
  - [The Scaling Problem](#the-scaling-problem)
  - [Beads: Local-First Issue Tracking](#beads-local-first-issue-tracking)
  - [Gap Analysis as Planning Infrastructure](#gap-analysis-as-planning-infrastructure)
  - [The Type System as Architectural Maturation](#the-type-system-as-architectural-maturation)
  - [Memory Files: The Agent's Working Memory](#memory-files-the-agents-working-memory)
  - [The Quick Win Trap](#the-quick-win-trap)
  - [The Structured Memory Lessons](#the-structured-memory-lessons)
- [Conclusion](#conclusion)

---

## The Setup

Over January–March 2026, I built three open-source projects almost entirely through conversations with Claude Code:

- **[Codescry](https://github.com/avishek-sen-gupta/codescry)** (originally "Cartographer"): A repo surveying toolkit that detects integration points in source code using regex patterns, ML classifiers, code embeddings, and LLM-based classification — 250 commits across 35 days
- **[RedDragon](https://github.com/avishek-sen-gupta/red-dragon)**: A multi-language code analysis engine with a universal IR, deterministic VM, and iterative dataflow analysis
- **[Rev-Eng TUI](https://github.com/avishek-sen-gupta/reddragon-codescry-tui)**: A terminal UI integrating the two

Codescry has ~195 conversation sessions (101 under its original name "Cartographer", the rest as Codescry). RedDragon was built in an initial session, then refined across 237+ more sessions. The llm-symbolic-interpreter (RedDragon's precursor) added another 73. That's roughly 600+ human-AI conversation sessions.

Here's what I learned about directing an AI to build non-trivial systems, and where the process surprised me.

---

![Demo](/assets/red-dragon-tui.gif)
## Part 1: Codescry, Learning to Steer

### Origins: Cartographer (Jan 30 – Feb 3)

Codescry started under a different name. On January 30, I opened a fresh session and said: *"Given the path to a repo, I want repo_surveyor to deduce the technology stack(s) used in the repository, and produce a basic report."* The project was called **Cartographer**.

From that single prompt, the first session produced a `RepoSurveyor` class — walking a directory tree, identifying languages by file extension, detecting frameworks from indicator files. CTags integration followed the next day, adding code symbol extraction across languages via Universal CTags as an external subprocess.

Then the project's engineering principles crystallised rapidly over Feb 1–2. In 26 commits across 5 days:

- `Neo4jPersistence` was refactored to accept a `Neo4jDriver` protocol, then renamed to `AnalysisGraphBuilder`. This set the pattern — every external system boundary became a protocol.
- `IntegrationDetector` appeared, finding system integration points (HTTP, SOAP, messaging, databases) via regex patterns. Over the next 15 commits, detection split into common/language-specific/framework-specific layers, with COBOL, IDMS, and PL/I patterns.
- Enums replaced strings everywhere: `IntegrationType`, `Confidence`, `EntityType`, `Language`. `IntegrationPoint` was renamed to `IntegrationSignal` — detections are signals, not confirmed facts.
- `PHILOSOPHY.md` was written: *"Code wants to be free. This project exposes mechanisms, not workflows."*

### Structural Analysis: Call Flow and CFGs (Feb 9 – Feb 18)

After a week's pause, the project expanded in two directions.

**Call-flow extraction via LSP.** A Java call-flow example used JDTLS via the [mojo-lsp](https://github.com/avishek-sen-gupta/mojo-lsp) bridge to trace method call trees. Regex-based call extraction was replaced with tree-sitter queries, the LSP bridge was extracted into a reusable module, and call-flow became a standalone `call_flow` module.

**Framework-aware integration detection.** A major expansion on Feb 12: 20 commits in a single day. Integration patterns were grouped per framework, naive substring matching was replaced with structured config file parsing (preventing `"reactive-streams"` from matching `"react"`), the codebase was restructured into a plugin architecture with a declarative `languages.json`, and support was added for Dropwizard, Vert.x, Play, Apache CXF, .NET frameworks, FILE_IO and GRPC types.

**CFG construction.** The most complex addition: a language-independent CFG builder on tree-sitter parse trees. The approach started with LLM-generated CFG role mappings for 99 languages — but the LLM output was unreliable, and the mappings were hand-authored for 14 languages instead. This was the first instance of a pattern that would recur: **use LLM to bootstrap, then replace with a deterministic approach once the problem is understood.**

**The rename.** On Feb 18, commit 120: *"Rename Cartographer to Codescry across project."* Accompanied by a topographic terrain banner, CI badges, and Graphviz export for both CFG and integration signal diagrams.

### The Concretisation Problem (Feb 18 – Feb 25)

This was the most experimental phase. The core challenge: regex-based pattern matching produces many false positives. How do you classify which detected signals are genuine (SIGNAL vs NOISE) and their direction (INWARD vs OUTWARD)?

What struck me about this phase, looking back at my prompts, is how *exploratory* it was. I didn't have a fixed architecture in mind. I had a problem and I was using Claude as a thinking partner to evaluate approaches in rapid succession:

**Attempt 1: LLM classification.** My first instinct was to throw an LLM at the problem: take each regex match, grab surrounding AST context, ask Claude or a local model whether it's a real integration point. This worked... on small inputs. When I ran it on a real Java repo (2,116 signals across 1,809 groups), it needed 37+ LLM batches. *"This is taking way too long,"* I told Claude. *"What other tricks can be used to reduce the number of signals before it's sent to the LLM?"* The LLM concretisation path was removed, though AST walk-up (grouping signals by enclosing function) was retained as infrastructure.

**Attempt 2: ML classifier.** I pivoted to training a TF-IDF + logistic regression classifier, using Claude's Batches API to generate training data at 50% cost. A GitHub training-data harvester mined real code using the pattern registry itself as search queries — each HIGH-confidence pattern became a GitHub search query, with the pattern's `SignalDirection` providing labels with no LLM required. The classifier was fast but mediocre; confidence scores were low, and it struggled with framework-specific patterns.

**Attempt 3: Code embeddings.** I tested `nomic-embed-code` to see if embeddings could separate integration code from non-I/O code. They could. Then Gemini's embedding model. Both worked, but directional classification (inward vs. outward) was weak. Then came an unexpected finding.

**The embedding model shootout.** Five embedding backends were evaluated:

| Backend | Type | Result |
|---------|------|--------|
| nomic-embed-code | Cloud API | 91% on small test |
| Gemini embedding-001 | Cloud API | 91% on small test |
| CodeT5p-110m | Local, code-specific | 50% — scores too compressed |
| CodeRankEmbed | Local, code-specific | 4.5% — catastrophic failure |
| **BGE-base-en-v1.5** | Local, general-purpose | **100%** on all test sets |

The core finding: **general-purpose text embedding models trained for semantic similarity outperform code-specific models** on this description-to-code classification task. MTEB leaderboard scores did not predict performance.

A critical sub-experiment: through iterative testing against a single Java line (`String text = new String(Files.readAllBytes(src.toPath()))`), I discovered that **passive-voice, subject-first descriptions** maximise cosine similarity in embedding space. Scores improved from 0.448 to 0.773. All 3,017 pattern descriptions were rewritten to this template.

**Attempt 4: Hybrid pipeline.** The winning architecture was a two-stage pipeline: a fast embedding gate (framework-specific pattern embedding, BGE, distance-weighted KNN) to separate signal from noise, then Gemini Flash to classify direction on only the signals that survived the gate. This achieved **84.7% exact-match accuracy** and **82.3% precision** — surpassing standalone Gemini Flash.

**The Datalog tangent.** Midway through, I had Claude build a Datalog-based structural analysis system that emits tree-sitter parse trees as a 16-relation Soufflé ontology, then write declarative queries for framework patterns — annotation-based, type-reference-based, and instantiation-based detection. Patterns impossible with line-level regex. This tangent became a real feature, though it remained a PoC not integrated into the main pipeline.

**Evidence checks.** Seven universal suppression checks (test files, vendor dirs, generated code, config dirs, string literals, log statements, constant declarations) adjusted raw embedding scores before thresholding. This addressed the dominant false positive source: 219 of 758 smojol signals were COBOL string literals in Java test code.

In total, **seven distinct classification pipelines** were built, all sharing the same detection Phase 1 and output Phase 3, differing only in Phase 2 classification: TF-IDF+LogReg, generic embedding, pattern-embedding KNN, Ollama LLM, Gemini Flash LLM, hybrid generic+Gemini, and hybrid pattern+Gemini.

### Engineering Maturity (Feb 20 – Mar 2)

The final phase focused on production-readiness: vectorized cosine similarity (741x speedup), batched AST extraction, embedding caching with SHA-256 content-hash invalidation, a Reveal.js presentation iteratively refined over 8 commits, codebase restructuring (all relative imports to absolute, `repo_surveyor` into classified subpackages, tests into functional subdirectories), and retroactive ADRs.

### What I Learned: Steering

The takeaway from Codescry was that **the human's job is strategic, not tactical**. I wasn't writing code. I was:

- Evaluating approaches by running them on real data and reading the results
- Making pivot decisions based on empirical feedback ("this is too slow", "confidence is too low")
- Composing architectures ("bolt the LLM onto the embedding gate")
- Interrupting when a direction wasn't working
- **Replacing LLM-based approaches with deterministic ones** once the problem was understood (CFG roles, signal classification, training data)

My prompts got terser as trust built up. Early on: detailed specifications with context. By day 3: *"do all of them"*, *"push"*, *"run it on smojol and show me the results"*.

The scale of experimentation was unusual: 5 embedding backends tested, 7 classification pipelines built, 3,017 pattern descriptions rewritten based on a single embedding-space experiment, three independent classifiers compared head-to-head. The AI made this breadth of exploration practical — each experiment took minutes, not days.

---

## Part 2: RedDragon (Feb 25–26)

### The Starting Point

On Feb 25, I opened a fresh session and described what I wanted: a universal symbolic interpreter that parses source in any language, lowers it to a flat IR, builds a CFG, and executes it symbolically, handling missing imports and unknown externals gracefully.

The first question I asked Claude: *"Is there an existing IR/VM that already does this?"* This is a habit I've developed: always check for prior art before building. There wasn't a good fit for what I needed (symbolic execution of incomplete programs across 15 languages), so we proceeded.

### How the Architecture Took Shape

What followed was a series of architectural decisions, each triggered by testing the previous one:

**Decision 1: Custom TAC IR.** We chose a flattened three-address code with ~19 opcodes. Simple enough to target from any language, rich enough to preserve data flow.

**Decision 2: Deterministic VM.** The initial design had the LLM deciding state changes at each step. After implementing it, I asked: *"Given that the IR is always bounded, shouldn't the IR execution be deterministic?"* This turned out to matter. We ripped out all LLM calls from the VM and replaced them with symbolic value creation. The entire execution became reproducible.

**Decision 3: LLM as compiler frontend, not runtime oracle.** With the VM deterministic, the LLM's role narrowed to one thing: translating source code to IR. And even that was constrained. We gave it all 19 opcode schemas, concrete patterns, and a worked example. The LLM was acting as a mechanical translator, not a reasoning engine.

**Decision 4: Code-generate the deterministic frontends.** Rather than using the LLM *at runtime* to lower source to IR, I asked: *"How hard is it to write deterministic logic to lower ASTs to IR for 16 languages?"* Claude generated tree-sitter-based frontends for 14 languages in a single session. Sub-millisecond, zero LLM calls, fully testable.

**Decision 5: Chunked LLM frontend for large files.** When the LLM frontend hit context window limits on large files, we added a chunked frontend that decomposes files into per-function chunks via tree-sitter, lowers each independently, then renumbers registers and reassembles.

Each decision emerged from testing the previous one on real code. I didn't plan this architecture in advance. It took shape through iterative probing.

### The Implementation Rhythm

The initial session followed a repeating cycle:

1. **Implement a feature** (30–60 minutes)
2. **Run it on real code** and inspect the output
3. **Identify the next gap** ("any other language features not covered?")
4. **Audit for completeness**, then batch-implement all gaps
5. **Clean up immediately**: refactor, split large files, reorganise tests

I never let technical debt accumulate. Every feature push was followed by cleanup. When `interpreter.py` hit 1,200 lines, I immediately said *"break up interpreter.py, it's too big."* When the registry module grew three responsibilities, I split it into three files. When tests were in a flat directory, I separated them into `unit/` and `integration/`.

### Filling Language Feature Gaps

Plugging language-specific gaps across all 15 frontends was systematic:

1. Ask Claude to audit every frontend for missing constructs
2. Prioritise by impact (high/medium/low)
3. Say *"implement all the critical and common ones"*
4. Push, then immediately re-audit

This cycle repeated 4–5 times, each time catching a smaller set of remaining gaps. The test count tracked the progression: 645 → 672 → 682 → 687 → 690 → 698 → 720 → 746 → 775 → 1053 → 1176 → 1186 → 1198 → 1203.

### Screenshot-Driven Debugging

For the Mermaid CFG visualisation work, a different mode of interaction emerged. I'd generate a diagram, screenshot it, paste it into the conversation, and ask *"why does it look so disjointed?"* Claude could see the rendering and diagnose layout issues. The CFG visualisation went through five implementation rounds before I was satisfied: subgraphs, call edges, block collapsing, shape conventions, unreachable block pruning.

---

## Part 3: Patterns That Emerged

### Pattern 1: Brainstorm → Probe → Crystallise

I never started with a fixed architecture. I started with a problem, brainstormed approaches with Claude, then *probed* each approach by implementing it and testing on real data. The architecture took shape from empirical feedback, not upfront design.

The deterministic VM wasn't planned. It emerged from asking *"shouldn't this be deterministic?"* after seeing the LLM-based approach work. The hybrid embedding pipeline wasn't planned either. It emerged from watching the LLM classifier be too slow and the embedding classifier lose directional information.

### Pattern 2: The Plan Document as Interface

An interaction pattern that worked well was the structured plan. After brainstorming and discussing trade-offs, I'd formulate a plan document covering context, phases, file-by-file changes, verification steps, and feed it to Claude as an implementation spec. This happened ~15 times across the projects.

The plan document serves as an *interface contract* between the human architect and the AI implementer. It's specific enough that Claude can execute without ambiguity, but high-level enough that the human retains architectural control.

### Pattern 3: Audit-Implement-Reaudit Loops

After every batch of features, I immediately asked *"what else is missing?"* This tightening loop drove the system toward completeness without requiring me to enumerate every gap upfront. Claude's ability to audit a large codebase for consistency (e.g., "which of the 15 frontends is missing switch statement support?") was useful here.

### Pattern 4: Immediate Cleanup as Discipline

I never said "we'll refactor later." Every implementation was immediately followed by:
- Black formatting
- Test organisation
- Module splitting if a file grew too large
- README updates
- Architectural decision records

This discipline was encoded in my `CLAUDE.md` file, which Claude followed for every commit. The CLAUDE.md itself evolved over time as I added rules as I discovered patterns that needed enforcement.

### Pattern 5: Terse Directives After Trust

Early prompts were detailed and cautious. By mid-project, I was saying *"do all of them"*, *"push"*, *"commit and push this"*. Trust built through consistent execution. When Claude produced correct, formatted, tested code for the 50th time, I stopped micromanaging the implementation details and focused on architectural direction.

### Pattern 6: Context Window as Session Boundary

The initial session exhausted the context window 5–6 times. Each continuation carried a structured summary of what was done and what remained. This forced a natural "checkpoint" discipline. I couldn't rely on Claude remembering earlier decisions, so I had to be explicit about state. In hindsight, this was healthy: it prevented architectural drift and kept each session focused.

---

## Part 4: Where It Surprised Me

### Surprise 1: The AI is better at breadth than depth

Claude was good at tasks like "generate deterministic frontends for 14 languages" or "audit all 15 frontends for missing switch support." These breadth tasks, applying a consistent pattern across many targets, would have taken me days of tedious work. Claude did them in minutes.

Where it needed more guidance was depth: subtle semantic decisions like closure capture semantics (snapshot vs. shared environment), or when to use SYMBOLIC fallback vs. crash. These required me to probe with specific test cases and reason about the implications.

### Surprise 2: The workflow matters more than the prompts

The more useful lever wasn't clever prompting. It was the workflow encoded in `CLAUDE.md`. Rules like "run all tests before committing", "use dependency injection not mock.patch", "prefer early return", "one class per file". These accumulated into a consistent codebase even across hundreds of sessions. In practice, the CLAUDE.md file acted as the architecture document.

### Surprise 3: Empirical validation beats specification

I rarely specified exact behaviour upfront. Instead, I'd implement a feature, run it on real code (usually the `smojol` Java repo or a multi-language test suite), and judge the results. *"The confidence scores seem low"* → pivot to embeddings. *"Why does the CFG look disjointed?"* → fix the visualisation. The AI made this loop fast enough to be practical. I could test an idea and get results in minutes, not hours.

### Surprise 4: The TDD workflow changed how I direct AI

Late in the project, I modified my workflow to: Brainstorm → Trade-offs → Plan → **Write unit tests** → Implement → Fix tests → Commit → Refactor. Writing tests first forced me to think about the interface before the implementation, and it gave Claude a concrete target to implement against. The tests became the specification.

---

## Part 5: The Evolution, From Monolith to 10,000+ Tests

RedDragon's evolution followed a clear pattern of phases, each triggered by testing the previous one on real code:

**Phase 1: The monolith (Hour 0 to 2).** A single `interpreter.py` with an LLM-based lowering and execution engine. ~1,200 lines. It worked, barely.

**Phase 2: The determinism pivot (Hour 2 to 4).** The key insight: execution should be deterministic. Ripped out all LLM calls from the VM. Added symbolic value creation. Once the VM was deterministic, everything became testable.

**Phase 3: Multi-language frontends (Hour 4 to 8).** Asked: *"How hard is it to write deterministic logic to lower ASTs for 15 languages?"* The answer: not that hard, with tree-sitter and a dispatch table engine. 15 frontends generated in a single session. 346 tests.

**Phase 4: Analysis and tooling (Hour 8 to 14).** Added iterative dataflow analysis, chunked LLM frontend, Mermaid CFG visualisation with subgraphs and call edges, source location traceability. Extracted CLI into composable API.

**Phase 5: Systematic hardening (Sessions 50 to 130).** This is where the test count grew rapidly. The Rosetta cross-language test suite (14 algorithms x 15 languages) and then the Exercism integration suite drove the test count from ~700 to 7,268 across 80+ sessions. Each exercise exposed new frontend gaps, VM limitations, and edge cases, and each fix was immediately verified across all 15 languages.

The test count tells the story:

```
Phase 1-3:           346 tests
Phase 4:            ~700 tests
Rosetta:           ~1,200 tests
Exercism 1:        ~2,700 tests
Exercism 2:        ~4,200 tests
Exercism 3:        ~5,150 tests
Exercism 4:        ~7,076 tests
Exercism done:      7,268 tests
COBOL + audit:      8,569 tests
Type system:       ~9,400 tests
Gap analysis:     10,152 tests
```

All tests run without LLM calls and are deterministic.

---

## Part 6: The Audit Loop, Systematic Completeness

A recurring pattern in RedDragon's development was the *audit-fix-reaudit* loop. After every batch of frontend work, I ran a comprehensive two-pass audit:

**Pass 1 (Dispatch Comparison):** Parse source samples in all 15 languages, collect every AST node type that appears, compare against the frontend's dispatch tables, and classify unhandled types as structural (harmless, consumed by parent handlers) or substantive (gaps that produce `SYMBOLIC`).

**Pass 2 (Runtime SYMBOLIC check):** Actually lower the source through each frontend, scan the resulting IR for `SYMBOLIC` instructions with `"unsupported:"` operands. This catches gaps that the static analysis might miss.

The classification heuristic itself went through three iterations:

1. **Naive:** Flag everything not in a dispatch table. This produced hundreds of false positives, because nodes like `parameter_list` and `type_annotation` are consumed by parent handlers and never reach the dispatch chain independently.

2. **Parent heuristic:** Flag unhandled nodes only if their immediate parent isn't handled. This reduced false positives but still produced 259. When the parent was also unhandled but deep structural (never block-iterated), its children got flagged incorrectly.

3. **Block-reachability analysis:** The final approach. Walk the AST and identify which unhandled nodes are *direct named children of block-iterated nodes* (the root, or nodes whose type maps to `_lower_block` in the dispatch table). Only these nodes can ever reach `_lower_stmt` and produce `SYMBOLIC`. Everything else is a deep structural node consumed by parent handlers.

The block-reachability approach dropped substantive gaps from 259 to 1 (a C `case_statement` that was a genuine gap, subsequently fixed with a defensive handler). The evolution of this heuristic is a good example of how empirical feedback drives design. Each version was tested against the full corpus, and false positives were investigated until the classification matched reality.

The audit loop ran dozens of times across the project's life:

```
Audit -> 34 gaps found -> implement all 34 (57 new tests)
Re-audit -> 19 gaps found -> implement all 19 (28 new tests)
Re-audit -> 12 gaps found -> implement all 12 (18 new tests)
Re-audit -> 0 gaps, 0 SYMBOLIC
```

This pattern (audit, batch-fix, re-audit) was more effective than trying to enumerate every missing feature upfront. The audit told me what was missing, I fixed what it found, and repeated until it found nothing.

---

## Part 7: The Assertion Audit, Green Tests, False Confidence

The dispatch audit ensured that every language construct was *handled*. But it said nothing about whether the tests *verifying* that handling were any good. By March 2026, the test suite had grown to ~8,400 tests across ~130 files. All green. The question I'd been avoiding: **if every test passes, how do I know each test is actually checking what it says it's checking?**

Not "does the code work"; 8,400 green dots covered that. But: does `test_constructor_sets_fields` actually verify that field values are set? Does `test_switch_statement` actually verify that switch/case lowering works? Does `test_perform_until_loop_ordering` actually verify that instructions appear in the right order?

When an AI writes your tests, you get volume and coverage breadth for free. What you don't get is *assertion depth*. The AI produces a test for every function, every edge case, every language, but each individual assertion may be checking the easy thing (does it not crash?) rather than the hard thing (does it produce the right output?). The AI optimises for the test *passing*, not for the test *verifying*.

Over two days and 11 audit passes, I had Claude Code scan every test file, comparing each test's name against its actual assertions. Here is what I found.

### The Taxonomy of Weak Assertions

**1. OR-fallback assertions: the most dangerous pattern.** Tests like:

```python
assert Opcode.BRANCH_IF in opcodes or Opcode.BRANCH in opcodes
```

claimed to verify conditional branching, but `BRANCH` (unconditional jump) exists in virtually *every* program. It's the basic goto. The `or` made the assertion tautologically true. The match expression lowering could be completely broken and this test would still pass. I found this in Scala match/case, C# switch expressions, COBOL PERFORM ordering, and IR stats tests: 23+ instances where the test would pass even if the feature it named was completely broken.

**2. Existence-only checks.** `assert len(writes) >= 1` on WRITE_REGION was a systematic problem in COBOL tests. It was satisfied by DATA DIVISION initial-value writes, making the PROCEDURE statement under test effectively untested. The strengthened version decoded the EBCDIC bytes and checked `_decode_zoned_unsigned(region, 0, 3) == 100`. The fix pattern across all COBOL tests was to count initial-value writes and assert the total exceeds that baseline.

**3. Cross-product matching.**

```python
assert any(bi < pi for bi in branch_if_indices for pi in print_indices)
```

This claimed to verify that `BRANCH_IF` appears before a print `CALL` in the loop body. But `any()` over a cross-product means it matches if *any* `BRANCH_IF` appears before *any* print, even if they're from completely unrelated parts of the program. The test passes even when the loop structure is wrong.

**4. Substring-by-coincidence.** `assert any("x" in inst.operands for inst in stores)` checks `STORE_VAR` for `"x"`, which passes whether the try body or the except body produced it, giving no confidence that both branches were lowered.

**5. Silent parametrised passes.** Bare `return` in parametrised tests for excluded languages:

```python
@pytest.mark.parametrize("language", ALL_15_LANGUAGES)
def test_closure_captures_variable(self, language):
    if language not in CLOSURE_LANGUAGES:
        return  # <-- silently passes
```

Eleven languages that can't express closures were showing as green in the test report. Not skipped. Not xfailed. *Passed.* With zero assertions executed. The test report said "15/15 languages pass closure tests" when only 4 actually ran. The fix was `pytest.skip()` with a reason string, making the exclusion visible in test output.

**6. Tautological guards.**

```python
if "x" in result.dependency_graph:
    assert "x" not in result.dependency_graph["x"]
```

If `x` is missing from the dependency graph entirely (the exact bug this test should catch), the `if` guard silently skips the assertion. Worse: `result.definitions` was a list of `Definition` objects, not a dict. The `in` check always returned `False`, so the guard *never* fired, and the test silently passed on every run.

### The Audit Timeline

**Phase 1: Discrepancy audits (2026-03-03).** The first pass focused on claim-vs-reality mismatches: tests whose names or docstrings contradicted what the code actually did. Three audits found 30+ discrepancies, including: a diamond-shape test that actually asserted stadium shape; a "two closures share state" test with only one closure; `test_cpp_catch_ellipsis` that set up a `stores_after_caught` list but never asserted on it; `test_constructor_sets_fields` that never verified field values; and `test_java_lambda_apply` that never verified the `FUNC_REF` dispatch mechanism. The human directive during this phase set the tone for the entire effort: **strengthen, don't delete.** Fix the test, don't remove it.

**Phase 2: Name-vs-assertion audits (2026-03-04).** The focus shifted from docstring accuracy to assertion strength: does the test assert what its name claims? The first scan found 22 violations. A broader scan found 30 more across 22 files, introducing the taxonomy: Category A (misleading name, 7), Category B (missing key assertion, 8), Category C (weak/generic assertion, 13).

**Phase 3: Priority-based audits.** A full scan of 122 files found ~82 violations. Introduced the priority classification: P0 (false confidence), P1 (missing key assertion), P2 (weak/generic), P3 (cosmetic). Re-scans after each fix batch drove the count down: 82 to 56 to 17.

**Phase 4: Reconciliation.** After several audits, I noticed the violation list kept changing. Items that were fixed reappeared with different wording. New phantom items appeared. Priority assignments shifted. Each audit launched fresh agents that re-discovered the codebase independently, producing inconsistent results. The fix was **reconciliation passes**: starting from the previous audit's known remaining items and verifying each against the current code. This produced a stable, verified list of 13 remaining items.

### The Governing Principle: Strengthen, Don't Rename

Early in the audit, Claude tried to fix a misleading test name by renaming `test_compute_expression` to `test_compute_respects_operator_precedence`. I stopped it immediately: **always strengthen the assertion to match the name, never weaken the name to match the assertion.** Renaming moves the problem. The original name described behaviour that *should* be tested. Renaming it to match the weak assertion just means that behaviour goes unverified forever. Strengthening the assertion closes the gap.

This became the governing principle for all subsequent fixes.

### Errors During Fix Attempts

Strengthening assertions is harder than writing the original test. Many fix attempts failed on the first try:

**Asserting on the wrong representation.** Go's `test_return_without_value` initially asserted `returns[0].operands == []` but bare return has `['%0']` (register pointing to default value). The Python decorator test asserted `"my_dec" in inst.operands` but `CALL_FUNCTION` uses register references (`%4`, `%3`), not names. The fix was to check `LOAD_VAR` instead. Pascal FILLER EBCDIC expected `0x40` (EBCDIC space) at FILLER offsets but got `\xe2\xd7`.

**String vs integer operands.** Pascal's `test_try_lowers_body` asserted `1 in const_vals` (integer) but IR operands are strings (`["1"]`). Fixed to `"1" in const_vals`.

**Misunderstanding the execution model.** `test_initial_value_100_in_region` expected value 100 but the test helper `_execute_straight_line` runs ALL instructions (DATA + PROCEDURE division), producing 125 (100+50-25). The test had been asserting `len(region) == 5` which was tautologically true.

**Private vs public attributes.** `test_llm_frontend_any_language` asserted `frontend.language` but `LLMFrontend` stores it as `_language`.

The fix cycle was consistently: read test, write assertion, run test, discover actual representation, fix assertion, re-run. The AI is good at this cycle once you force it through it, but it will never enter this cycle voluntarily. It will write the assertion it thinks is correct and move on.

### Real Bugs Found

P0 fixes exposed genuine bugs that had been hiding behind weak assertions:

**Pascal bare-except.** The test `test_try_lowers_body` used `try x := 1; except x := 0; end;` but the Pascal frontend silently dropped bare `except` blocks (without `on E: Exception do` wrapper). The test passed because it only checked that `STORE_VAR "x"` existed, which was satisfied by the try body alone. Strengthening the assertion to verify both the try-body constant (`"1"`) and the except-body constant (`"0"`) exposed the bug. The fix was in `_extract_pascal_try_parts` to handle `statements` nodes in the except section.

**C# else-if chain lowering.** A similar pattern where a weak assertion masked incomplete lowering of chained else-if blocks.

**`test_no_self_dependency_without_loop`.** The guard `if "x" in result.definitions:` was checking membership on a list of `Definition` objects, not a dict. The `in` check always returned `False`, so the assertion never executed. The test had been passing for weeks while the behaviour it named went completely unverified.

**`test_initial_value_100_in_region`.** Expected 100 but the helper runs all instructions sequentially, producing 125. The test's original assertion `len(region) == 5` was tautologically true and would pass regardless of the actual computed value.

### The Feedback Loop Structure

Each audit cycle followed this structure:

```
Human: "audit all tests"
  -> AI: launches parallel agents, scans ~130 files
  -> AI: produces prioritised violation list
  -> Human: reviews, selects priority tier to fix
  -> AI: reads each test, writes fix, runs test
    -> Fix fails (wrong representation, wrong attribute, wrong type)
    -> AI: investigates actual behaviour, corrects fix
    -> Fix passes
  -> AI: runs full test suite (8,457 tests)
  -> AI: runs Black formatter
  -> AI: commits and pushes
  -> Human: "audit again"
  -> [repeat]
```

The critical human interventions were:
1. **"Always strengthen assertions, do not rename"**: set the governing principle
2. **"You keep changing the list"**: identified the audit stability problem
3. **"Do a reconciliation pass"**: prescribed the fix for drift
4. **"Fix the frontend bug"**: escalated from test fix to code fix when an assertion exposed a real bug
5. **Selective priority focus**: always fixing P0 first, then P1, deferring P2

The AI's role was execution at scale (scanning 130 files, writing fixes, running 8,457 tests) while the human provided quality gates and strategic direction.

### Assertion Audit Results

| Metric | Value |
|--------|-------|
| Calendar span | 2 days |
| Audit passes | 11 (3 discrepancy + 8 name-vs-assertion) |
| Total violations identified | ~180 (with overlap across audits) |
| Unique violations (deduplicated) | ~90 |
| Violations fixed | ~75 |
| Remaining (P2 cosmetic) | ~13 |
| Frontend bugs exposed | 2 |
| False-pass tests eliminated | 5 |
| Commits for assertion audit fixes | 33 |
| Tests at end of assertion audit | 8,469 |

The test count went *up*, not down. Strengthening assertions sometimes meant splitting one weak test into multiple specific ones.

### Assertion Audit Lessons

**Green tests are necessary but not sufficient.** Every test was green before the audit. The audit found 15 P0 violations where the test would pass even when the behaviour it named was completely broken. Two of these exposed genuine frontend bugs.

**OR-fallback assertions are the most dangerous pattern.** The pattern `assert A or B` appeared in every frontend category. In every case, one side of the OR was trivially satisfied by unrelated instructions, making the assertion vacuous.

**Audit stability requires anchoring.** Fresh scans produce inconsistent results because the agents don't share context. The reconciliation approach (starting from the previous audit's known list and verifying each item) is the only way to produce a stable, trustworthy list.

**Fixing assertions requires running the code under test.** Many fix attempts failed because the assertion assumed a representation that didn't match reality. The fix cycle (write assertion, run, discover actual representation, fix, re-run) was consistently necessary and never happened voluntarily.

**Parametrised tests need explicit skips, not silent returns.** Bare `return` in excluded branches produces green dots with zero assertions. `pytest.skip()` with a reason string makes exclusions visible.

The CLAUDE.md rules were updated to encode these lessons: never modify assertions to make tests pass without verifying the actual output, and never create special implementation behaviour just to satisfy tests.

---

## Part 8: Guardrails, The CLAUDE.md as Architecture

RedDragon was built almost entirely through conversations with Claude Code, across 400+ sessions. The file that had the most impact on consistency wasn't any Python module. It was `CLAUDE.md`, which encodes the development rules. Claude Code reads this file at the start of every session, so every conversation begins with the same constraints. The rules evolved over the project's lifetime, each one added in response to a specific failure mode.

### Build Rules

The build section encodes a strict pre-commit discipline:

**"Before committing anything, run all tests, fixing them if necessary."** This prevented test count regression across 292 commits. The rule also specifies: if test assertions are being *removed*, ask for human review first. This distinction matters. Adding assertions is safe; removing them requires justification.

**"Before committing anything, run `poetry run black` on the full codebase."** The CI pipeline enforces Black formatting and will fail if this is skipped. Encoding this in CLAUDE.md means the AI formats before every commit without being asked.

**"Before committing anything, update the README based on the diffs."** This keeps the README in sync with the code. Without this rule, the README would have drifted within the first week.

**"For each feature, treat it as an independent commit / push, with its own testing."** This produced atomic, reviewable commits. Combined with "do not start a new task until the current one is committed," it prevented half-finished features from accumulating across sessions.

**"Once a design is finalised, document it as an ADR."** This produced 66 timestamped architectural decision records that serve as the project's institutional memory. Each records the context, the decision, and the consequences, including trade-offs.

### Testing Patterns

The testing rules address the specific failure modes of AI-generated tests:

**"When fixing tests, do not blindly change test assertions to make the test pass."** This is the most important testing rule. Without it, the AI's default behaviour is to modify the assertion to match whatever the code produces, regardless of whether the code is correct. The rule forces it to verify the actual output before changing assertions.

**"Make sure you are not creating any special implementation behaviour just to get the tests to pass."** The complement of the above. Without this rule, the AI occasionally added if-branches or special cases in production code solely to satisfy a test expectation, rather than fixing the underlying logic.

**"Do not use `unittest.mock.patch`. Use proper dependency injection."** This forced every external dependency (LLM clients, file I/O, clocks) to be injectable. The result: the entire VM, all 15 frontends, and all analysis passes are testable in isolation without mocking.

**"Always start from writing unit tests for the smallest feasible units of code."** The rule further specifies the directory structure: true unit tests (no I/O) go in `tests/unit/`, tests that exercise I/O (LLM calls, databases) go in `tests/integration/`. This separation means `tests/unit/` can run in CI without API keys.

**"For every bug you fix, make sure you have a test that fails without the bug fix."** This prevents fixes that are never actually verified. Without this rule, the AI occasionally produced fixes that looked plausible but didn't address the actual failure path.

### Programming Patterns

The programming rules enforce a specific coding style that reduces the surface area for bugs:

**"STOP USING FOR LOOPS WITH MUTATIONS IN THEM. JUST STOP."** This rule forced a functional programming style across the codebase. List comprehensions, `map`, `filter`, `reduce` instead of mutable accumulators. The code is denser but more predictable.

**"Categorically avoid defensive programming. This includes checking for None, and adding generic exception handling."** This is counterintuitive but deliberate. Defensive code hides bugs. A `None` check that silently returns an empty list masks the fact that a value should never have been `None` in the first place. Without this rule, the AI adds defensive checks reflexively, and each one is a potential silent failure.

**"If a function has a non-None return type, never return None."** Combined with the null object pattern enforcement ("if a function cannot return an object of that type because of some condition, use null object pattern"), this eliminated an entire class of `NoneType` errors. Functions that can't produce a value return a null object, not `None`.

**"When writing `if` conditions, prefer early return. Use `if` conditions for checking and acting on exceptional cases."** This keeps the happy path at the top level of indentation. Without it, the AI tends to nest the happy path inside increasingly deep conditionals.

**"Do not use static methods. EVER."** Static methods resist dependency injection and create hidden coupling. This rule forced all behaviour to live on instances, making every dependency explicit.

**"Use a ports-and-adapter type architecture. Adhere to the tenet of 'Functional Core, Imperative Shell'."** This principle shaped the overall architecture: the VM handlers are pure functions returning `StateUpdate` data objects, the dataflow module is a pure analysis pass, and I/O lives at the edges. The modules that follow this pattern most closely (dataflow, frontends) are the easiest to test and reason about.

**"If enums map to actual objects with behaviour, resolve them into the actual executable objects as early on in the call chain as possible."** This prevents enum values from being threaded through multiple layers as configuration tokens. The resolution happens once, at the entry point, and the resulting objects are injected as dependencies.

**"Parameters in functions, if they must have default values, must have those values as empty structures corresponding to the non-empty types."** Empty dicts, empty lists, never `None`. This eliminates the mutable default argument bug in Python and removes an entire category of `is None` checks from the codebase.

### The Workflow Contract

The workflow encoded in CLAUDE.md wasn't the original one. Early in the project, the workflow was simply **Brainstorm -> Plan -> Implement -> Test**. Tests came after implementation, which meant the AI wrote code first and tests second. This worked for the initial sprint but led to the weak assertion patterns that the later audit uncovered: when tests are written to match existing code, they tend to verify what the code *does* rather than what it *should do*.

Midway through the project, I changed the workflow to: **Brainstorm -> Discuss trade-offs -> Plan -> Write unit tests -> Implement -> Fix tests -> Commit -> Refactor.** Tests now come *before* implementation. The AI writes tests that encode the expected behaviour, then writes code to make them pass. This inversion also added two new phases: discussing trade-offs (forcing the AI to consider alternatives before committing to an approach) and an explicit refactoring step after the commit. Every session begins with these rules loaded into context. The brainstorming phase explicitly requires considering whether open source projects perform similar functionality, and balancing absolute correctness against "good enough." If in doubt, the rule says to ask for guidance rather than guessing.

---

## Part 9: What I'd Do Differently

**Start with the audit earlier.** The two-pass audit should have existed from the first batch of frontends. Instead, I relied on manual inspection for the first 50 sessions, and only built the audit when the number of frontends made manual checking impossible. In hindsight, the audit loop was what kept quality from drifting.

**Invest in cross-language tests from day one.** The Rosetta and Exercism suites exposed more bugs than all the language-specific unit tests combined. A single exercise tested across 15 languages covers more surface area than 50 unit tests in one language.

**Be more aggressive about the functional core.** Even with the FP rules in CLAUDE.md, some mutation crept in, especially in the VM executor. The dataflow module, by contrast, is almost purely functional and is by far the easiest module to reason about and test. The correlation is not a coincidence.

**The AI is better at breadth than depth.** Tasks like "generate deterministic frontends for 15 languages" or "audit all 130 test files for weak assertions" are where the AI excels. These breadth tasks, applying a consistent pattern across many targets, would have taken days of tedious work. The AI did them in minutes. Where it needed more guidance was depth: subtle semantic decisions like closure capture semantics (snapshot vs. shared environment), when to emit `SYMBOLIC` vs. crash, or whether an assertion is vacuous. These required me to probe with specific test cases and reason about the implications.

**Empirical validation beats specification.** I rarely specified exact behaviour upfront. Instead, I implemented a feature, ran it on real code, and judged the results. "The confidence scores seem low" led to pivoting from LLM classification to embeddings. "Why does the CFG look disjointed?" led to five rounds of visualisation fixes. The AI made this feedback loop fast enough to be practical. I could test an idea and get results in minutes, not hours.

**Terse directives after trust.** Early prompts were detailed and cautious: full specifications with context, constraints, and expected behaviour. By mid-project, I was saying "do all of them", "push", "commit and push this". Trust built through consistent execution. When the AI produced correct, formatted, tested code for the 50th time, I stopped micromanaging implementation details and focused on architectural direction.

**The AI hallucinated audit findings.** During the assertion audit, the AI reported violations that didn't exist or had already been fixed. Different parallel agents flagged different things based on traversal order, inconsistently applied priority criteria, and occasionally re-reported fixed violations with different wording. The reconciliation pass caught this. The lesson: the auditor itself needs auditing. Fresh scans without anchoring against previous findings produce unreliable results.

**CLAUDE.md rules are reactive, not proactive.** Every rule in CLAUDE.md was added in response to a specific failure mode. "STOP USING FOR LOOPS WITH MUTATIONS" came after seeing mutation bugs. "Don't blindly change test assertions" came after watching the AI weaken tests to make them pass. "Categorically avoid defensive programming" came after the AI added silent `None` checks that masked real bugs. They accumulate over time, and each one represents a mistake that happened at least once.

**The plan document as interface contract.** An interaction pattern that worked well was the structured plan. After brainstorming and discussing trade-offs, I'd formulate a plan document covering context, phases, file-by-file changes, and verification steps, then feed it to the AI as an implementation spec. The plan serves as a contract between the human architect and the AI implementer: specific enough for unambiguous execution, high-level enough to retain architectural control. This happened roughly 15 times across the project.

**The determinism pivot was the most impactful decision.** The original VM had the LLM deciding state changes at each execution step. When I asked "given that the IR is always bounded, shouldn't execution be deterministic?", the answer changed the project's direction. We ripped out all LLM calls from the VM and replaced them with symbolic value creation. Once execution was deterministic, everything became testable, reproducible, and fast. The entire test suite runs with zero LLM calls. This decision wasn't planned; it emerged from questioning an assumption.

---

## Part 10: The Numbers

| Metric | Codescry | RedDragon | Total |
|--------|----------|-----------|-------|
| Conversation sessions | ~195 (101 as Cartographer + 94 as Codescry) | ~237 | ~432 |
| Transcript data | ~379 MB | — | — |
| Development span | 35 days (Jan 30 – Mar 2) | 14 days (Feb 25 – Mar 11) | — |
| Git commits | 250 | 472 | 722 |
| Language frontends | N/A | 15 | 15 |
| Classification pipelines | 7 | N/A | 7 |
| Embedding backends tested | 5 | N/A | 5 |
| Pattern descriptions | 3,017 | N/A | 3,017 |
| Test count (final) | — | 10,152 | — |
| Architectural decision records | 23 | 100 | 123 |
| Architectural pivots | 7 | 5 | 12 |
| Tracked issues (Beads) | N/A | 168 | 168 |
| Audit substantive gaps (final) | N/A | 0 P0, 187 P1 | N/A |

---

## Part 11: From Sprints to Structured Agent Memory

The first ten parts of this account describe a project built in sprints: intense sessions, audit loops, and test count as the primary progress metric. By session 200, that model started to break — not because the AI got worse, but because the project outgrew what any single conversation could hold.

### The Scaling Problem

At 8,500+ tests, 15 frontends, 66 ADRs, and a type system under active development, the project had outgrown ad-hoc session management. I'd open a new conversation, describe what I wanted, and Claude would execute — but the *what* was getting harder to determine. Which frontend gaps remained? Which were P0 vs. P1? What depended on what? The answers lived in my head, in scattered ADRs, and in the git log. There was no single view of outstanding work.

The deeper problem was *agent amnesia*. Each conversation started from zero. Claude didn't remember that the TypeExpr migration was complete, that Pascal `declLabels` was already handled, that `poetry run python -m pytest` was the correct test command. Every session began with me re-establishing context. At 50 sessions, this was a minor annoyance. At 200, it was the dominant cost.

The solution wasn't any single tool. It was building a **structured memory layer** — a set of persistent artefacts that the agent reads at session start, giving it continuity across conversations. The memory layer has four components: a curated memory file, a gap analysis document, an issue tracker, and architectural decision records. Together, they answer the questions that every session needs answered: *what's the current state?*, *what's left to do?*, *what are the rules?*, and *why were past decisions made?*

The inflection point was the frontend lowering gap analysis. I had Claude cross-reference every frontend's dispatch table against its tree-sitter grammar definition and classify every unhandled node type. The result: 25 P0, 187 P1, and ~326 P2 gaps across 15 languages. The P0s took priority and were all resolved in three commits. But 187 P1 gaps couldn't be managed as a mental list. I needed an issue tracker.

### Beads: Local-First Issue Tracking

I chose [Beads](https://github.com/eqlabs/beads), a local-first issue tracker that stores its data in a Dolt database alongside the repo. The command is `bd`. It supports hierarchical issues (epics → stories → tasks), dependency chains, labels, and priority classifications — all from the command line.

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

Each sub-epic broke down further into per-language groups, and each language group into individual node-type tasks. A single `bd list` showed the full tree. `bd show red-dragon-gvu.1.2.5` gave me the details of C#'s `and_pattern` gap.

This changed how sessions started. Instead of describing work from memory, I could say *"what's the next open task under gvu.1?"* and have Claude pick it up with full context. The issue tracker became the session boundary — work was defined before the session began, not discovered during it.

Beads data is backed up as JSONL files committed to the repository, so the issue state travels with the code. On a new machine, `bd backup restore` rebuilds the full database from the committed exports — 168 issues, 538 events, 163 dependencies, all recoverable.

### Gap Analysis as Planning Infrastructure

The frontend lowering gap analysis document (`docs/frontend-lowering-gaps.md`) became a living planning artefact. It wasn't just a one-time audit; it was the source of truth that fed the issue tracker. Each P1 gap had a row in the document with language, node type, description, and status. As gaps were resolved, the status flipped to DONE. As new gaps were discovered through testing, they were added.

The gap analysis also exposed a structural insight: many P1 gaps clustered thematically. Pattern matching was a gap in 6 languages simultaneously. Class/OOP features were missing in 5. This meant the right unit of work wasn't "fix Go's missing `iota`" — it was "implement pattern matching across all 6 languages that need it." The themed epics emerged from the data, not from upfront planning.

The analysis also forced honest assessment of what "done" meant. Early on, I'd have Claude classify some gaps as "no-ops" — node types that could be safely ignored. When I challenged this, asking Claude to verify each claimed no-op against the actual tree-sitter AST, most turned out to contain meaningful content. TypeScript's `ambient_declaration` had full type signatures. C#'s `unsafe_statement` wrapped executable blocks. Pascal's `implementation` section contained procedure definitions. Out of 8 claimed no-ops, only 2 were genuine. The lesson: **verify against the AST, not against assumptions about what a node type name implies.**

### The Type System as Architectural Maturation

The largest post-sprint development was the type system. RedDragon started with string-based type hints — `"Int"`, `"String"`, `"List<Int>"` — threaded through the IR as operand annotations. This worked for simple inference but couldn't represent parameterised types, union types, or subtype relationships.

The evolution happened in phases, each driven by a concrete limitation:

**Phase 1: TypeExpr ADT.** Replaced string type hints with a proper algebraic data type: `ScalarType("Int")`, `ParameterizedType("List", (ScalarType("Int"),))`, `UnknownType`. A `parse_type()` function handled roundtripping from legacy strings. The entire type environment was migrated to store `TypeExpr` values, with string-compatible equality preserved for backward compatibility.

**Phase 2: TypeGraph.** An 11-node directed acyclic graph encoding subtype relationships (`Int <: Number <: Object`, `String <: Object`). Covariant `is_subtype_expr()` for parameterised types. `common_supertype_expr()` for join operations. This laid the groundwork for type narrowing and assignment compatibility.

**Phase 3: Interface-aware inference.** When the type inference engine encountered `animal.speak()` where `animal` was typed as interface `Animal`, it couldn't resolve the return type — the method was defined on `Dog`, not on `Animal`. The fix was a chain walk: if a class implements interfaces, check their method types when the class's own methods don't have type information. This required seeding `interface_implementations` from 5 frontends (Java, C#, Kotlin, TypeScript, Go) and walking the chain during inference.

Each phase was documented as an ADR, tested with both unit and integration tests, and committed independently. The type system alone accounts for ~1,500 tests and 34 ADRs.

### Memory Files: The Agent's Working Memory

The memory file is an easy-to-overlook component of the structured memory layer. Claude Code supports a persistent memory directory (`.claude/projects/.../memory/MEMORY.md`) that is loaded into context at the start of every session. This is the agent's working memory — curated, compact, and always current.

My memory file contains:

- **Key references**: Links to audit results, gap analyses, roadmaps, and their current status
- **Critical workflow reminders**: "ALWAYS run `poetry run python -m black .` on the entire codebase before committing" — the kind of thing that's in CLAUDE.md but worth reinforcing because it's been forgotten before
- **Project state**: Current test count, resolved issues, known gotchas (`poetry run python -m pytest`, not `poetry run pytest`)
- **Type system state**: Which migration phases are complete, what the current TypeExpr API looks like
- **Future work pointers**: What's been deferred and why, so Claude doesn't re-derive decisions that were already made

The memory file is curated, not appended. Outdated information is removed. Entries are updated when facts change. It functions as a living project brief — the minimum context needed to start any session productively. Without it, every session would spend its first 10 minutes rediscovering project state from git logs and file reads. With it, Claude starts knowing what the test count is, which phases are complete, and what the active work items are.

The distinction between the memory file and `CLAUDE.md` is important. `CLAUDE.md` encodes *rules* — how to behave, what patterns to follow, what mistakes to avoid. The memory file encodes *state* — what's been done, what's in progress, what's next. Together with the gap analysis (the *plan*) and the issue tracker (the *work queue*), they form a four-layer structured memory:

| Layer | Artefact | Purpose | Update frequency |
|-------|----------|---------|-----------------|
| Rules | `CLAUDE.md` | How to behave | When failure modes are discovered |
| State | `MEMORY.md` | What's been done | Every few sessions |
| Plan | Gap analysis doc | What's left to do | After each implementation batch |
| Work queue | Beads issues | What to do next | After each task closes |

This layered structure means a fresh session can orient itself in seconds rather than minutes. The agent reads the rules, the state, and the plan, then picks up work from the queue. No re-derivation, no context-setting preamble, no "where were we?"

### The Quick Win Trap

A recurring temptation when managing 168 issues is to chase quick wins — tasks that appear trivial and can be closed fast. During the gap analysis breakdown, Claude identified ~16 "quick wins" including 8 claimed no-ops. When I asked Claude to verify each one against the actual AST structure, the list shrank dramatically.

The pattern: **what looks like a no-op from the node type name often isn't.** `ambient_declaration` sounds like metadata. It's actually `declare const VERSION: string` — a full type signature that the inference engine needs. `unsafe_statement` sounds like a compiler pragma. It's actually a block wrapper around executable pointer operations.

The verified quick-win list went from 16 items to 7, with only 2 genuine no-ops. The rest needed real handlers. This is a general lesson for AI-assisted project management: the AI will optimise for closing tickets, not for closing them correctly. The human's job is to challenge the classification before the work begins, not after.

### The Structured Memory Lessons

**Agent memory is not a feature — it's an architecture.** No single tool solves the continuity problem. `CLAUDE.md` alone gives you rules without state. A memory file alone gives you state without a plan. An issue tracker alone gives you tasks without context. The four-layer structure (rules, state, plan, work queue) works because each layer answers a different question, and the agent needs all four answers to start a session cold.

**Issue trackers change how you direct the AI.** Before Beads, every session started with "here's what I want to do." After Beads, sessions start with "pick up the next task." The issue tracker externalises the planning, freeing the conversation for execution.

**Gap analysis documents are more valuable than roadmaps.** A roadmap says "implement pattern matching." A gap analysis says "Python needs `class_pattern`, `complex_pattern`, `union_pattern`, `keyword_pattern`, `tuple_pattern`, `pattern_list`, `as_pattern`; C# needs `recursive_pattern`, `list_pattern`, `var_pattern`, `type_pattern`, `and_pattern`, `or_pattern`, `negated_pattern`, `relational_pattern`, `parenthesized_pattern`, `tuple_pattern`." The gap analysis is actionable. The roadmap is aspirational.

**Themed epics beat per-language epics.** Implementing pattern matching across 6 languages in one themed push means each language's implementation informs the next. The pattern for `or_pattern` in C# is structurally similar to `union_pattern` in Python and `alternative_pattern` in Scala. Per-language epics miss these cross-cutting patterns.

**Curate memory aggressively.** The memory file must be edited, not appended. Stale entries are worse than no entries — they cause the agent to make decisions based on outdated state. When a migration completes, update the entry. When a test count changes, update the number. When a workaround is no longer needed, delete it. The memory file is a cache, and caches need invalidation.

**Backup your structured memory with your code.** Beads stores its data in a Dolt database next to the repo. JSONL backups are committed to git, so `bd backup restore` on a new machine reconstructs the full issue tree. The gap analysis doc is already in `docs/`. The memory file lives under `.claude/`. All of it travels with the repository. On a new machine, everything the agent needs to resume work is already checked in.

**The structured memory layer is itself a product of the AI workflow.** I didn't plan to need an issue tracker, or a four-layer memory architecture. The need emerged from the project's growth. The gap analysis, the themed epics, the priority classifications, the curated memory file — all of these were produced collaboratively with Claude, then curated by me. The AI is good at generating the raw material; the human is good at spotting when the material is wrong (the quick win trap) and structuring it into something the agent can consume reliably.

---

## Conclusion

Building non-trivial systems with an AI pair programmer is not about "prompting"; it's about *architectural direction*. The human's role is strategic: choosing problems, evaluating approaches empirically, making pivot decisions, and encoding quality standards. The AI's role is tactical: implementing plans, auditing for completeness, applying patterns at breadth, and maintaining consistency.

The workflow I converged on (brainstorm, discuss trade-offs, plan, test-first, implement, clean up) emerged through trial and error across 400+ sessions. It's not the only way to work with AI, but it produced working systems: tested, documented, formatted, and internally consistent.

What changed between session 100 and session 237 was the emergence of *structured agent memory*. The early sessions were sprints — intense, exploratory, driven by momentum. The later sessions were systematic — the agent started each conversation already knowing the project state, the active work queue, and the governing rules. The AI's role expanded from "implement this feature" to "break down 187 gaps into themed epics, then verify the quick wins aren't shortcuts." The human's role expanded from architect to memory curator: defining work, challenging classifications, and maintaining the persistent artefacts that make each session productive from its first message.

If I had to distil 600+ sessions into one lesson, it's this: **the limiting factor in AI-assisted development is not the AI's capability — it's the AI's memory.** A capable agent with no memory rediscovers context every session. A capable agent with structured memory — rules, state, plan, work queue — picks up where the last session left off. The architecture of that memory matters as much as the architecture of the code it helps build.
