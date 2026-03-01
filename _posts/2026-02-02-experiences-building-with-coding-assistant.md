---
title: "Architecting Non-Trivial Systems with Claude Code: A Practitioner's Account"
author: avishek
usemathjax: false
tags: ["Software Engineering", "Compilers", "Program Analysis", "Symbolic Execution", "AI-Assisted Development"]
draft: false
---

*How I built three interconnected code analysis tools — spanning 15 language frontends, a symbolic VM, dataflow analysis, embedding classifiers, and Datalog engines — in 9 days with an AI pair programmer.*

---

## The Setup

Over nine days in late February 2026, I built three substantial open-source projects almost entirely through conversations with Claude Code:

- **[Codescry](https://github.com/avishek-sen-gupta/codescry)** — A repo surveying toolkit that detects integration points in source code using regex patterns, ML classifiers, code embeddings, and LLM-based classification
- **[RedDragon](https://github.com/avishek-sen-gupta/red-dragon)** — A multi-language symbolic code analysis engine with a universal IR, deterministic VM, and iterative dataflow analysis
- **[Rev-Eng TUI](https://github.com/avishek-sen-gupta/reddragon-codescry-tui)** — A terminal UI integrating the two

This wasn't a weekend hack. Codescry has ~195 conversation sessions. RedDragon was built in an 18-hour marathon, then refined across 131 more sessions. The llm-symbolic-interpreter (RedDragon's precursor) added another 73. That's roughly 400 human-AI conversation sessions, producing thousands of lines of tested, formatted, documented Python with CI pipelines, Reveal.js presentations, and comprehensive READMEs.

Here's what I learned about directing an AI to build non-trivial systems — and where the process surprised me.

---

![Demo](/assets/red-dragon-tui.gif)
## Part 1: Codescry — Learning to Steer (Feb 18–23)

### The Problem

I had an existing codebase for surveying source code repositories — scanning for integration points like HTTP calls, database queries, message queue interactions. The detection was regex-based: fast but noisy. I wanted to make it smarter.

### The Exploration Phase

What struck me about this project, looking back at my prompts, is how *exploratory* it was. I didn't have a fixed architecture in mind. I had a problem and I was using Claude as a thinking partner to evaluate approaches in rapid succession:

**Attempt 1: LLM classification.** My first instinct was to throw an LLM at the problem — take each regex match, grab surrounding AST context, ask Claude or a local model whether it's a real integration point. This worked... on small inputs. When I ran it on a real Java repo (2,116 signals across 1,809 groups), it needed 37+ LLM batches. *"This is taking way too long,"* I told Claude. *"What other tricks can be used to reduce the number of signals before it's sent to the LLM?"*

**Attempt 2: ML classifier.** I pivoted to training a TF-IDF + logistic regression classifier, using Claude's Batches API to generate training data at 50% cost. The classifier was fast but mediocre — confidence scores were low, and it struggled with framework-specific patterns.

**Attempt 3: Code embeddings.** I tested `nomic-embed-code` to see if embeddings could separate integration code from non-I/O code. They could. I then tried Gemini's embedding model. Both worked, but directional classification (inward vs. outward) was weak.

**Attempt 4: Hybrid pipeline.** The winning architecture was a two-stage pipeline — a fast embedding gate to separate signal from noise, then Gemini Flash to classify direction on only the signals that survived the gate.

**The Datalog tangent.** Midway through, I had Claude build a Datalog-based structural analysis system — emit tree-sitter parse trees as Souffle facts, then write declarative queries for framework patterns. This was a fascinating tangent that became a real feature.

### What I Learned: Steering

The key insight from Codescry was that **the human's job is strategic, not tactical**. I wasn't writing code. I was:

- Evaluating approaches by running them on real data and reading the results
- Making pivot decisions based on empirical feedback ("this is too slow", "confidence is too low")
- Composing architectures ("bolt the LLM onto the embedding gate")
- Interrupting when a direction wasn't working

My prompts got terser as trust built up. Early on: detailed specifications with context. By day 3: *"do all of them"*, *"push"*, *"run it on smojol and show me the results"*.

---

## Part 2: RedDragon — The 18-Hour Marathon (Feb 25–26)

### The Vision

On Feb 25, I opened a fresh session and described what I wanted: a universal symbolic interpreter that parses source in any language, lowers it to a flat IR, builds a CFG, and executes it symbolically — handling missing imports and unknown externals gracefully.

The first question I asked Claude: *"Is there an existing IR/VM that already does this?"* This is a habit I've developed — always check for prior art before building. There wasn't a good fit for what I needed (symbolic execution of incomplete programs across 15 languages), so we proceeded.

### The Architectural Cascade

What happened next was a rapid cascade of architectural decisions, each triggered by testing the previous one:

**Decision 1: Custom TAC IR.** We chose a flattened three-address code with ~19 opcodes. Simple enough to target from any language, rich enough to preserve data flow.

**Decision 2: Deterministic VM.** The initial design had the LLM deciding state changes at each step. After implementing it, I asked: *"Given that the IR is always bounded, shouldn't the IR execution be deterministic?"* This was the key insight. We ripped out all LLM calls from the VM and replaced them with symbolic value creation. The entire execution became reproducible.

**Decision 3: LLM as compiler frontend, not runtime oracle.** With the VM deterministic, the LLM's role narrowed to one thing: translating source code to IR. And even that was constrained — we gave it all 19 opcode schemas, concrete patterns, and a worked example. The LLM was acting as a mechanical translator, not a reasoning engine.

**Decision 4: Code-generate the deterministic frontends.** This was the most surprising decision. Rather than using the LLM *at runtime* to lower source to IR, I asked: *"How hard is it to write deterministic logic to lower ASTs to IR for 16 languages?"* Claude generated tree-sitter-based frontends for 14 languages in a single session. Sub-millisecond, zero LLM calls, fully testable.

**Decision 5: Chunked LLM frontend for large files.** When the LLM frontend hit context window limits on large files, we added a chunked frontend that decomposes files into per-function chunks via tree-sitter, lowers each independently, then renumbers registers and reassembles.

Each decision emerged from testing the previous one on real code. I didn't plan this architecture in advance — it *crystallised* through iterative probing.

### The Implementation Rhythm

The 18-hour session had a distinctive rhythm:

1. **Implement a feature** (30–60 minutes)
2. **Run it on real code** and inspect the output
3. **Identify the next gap** ("any other language features not covered?")
4. **Audit for completeness**, then batch-implement all gaps
5. **Clean up immediately** — refactor, split large files, reorganise tests

I never let technical debt accumulate. Every feature push was followed by cleanup. When `interpreter.py` hit 1,200 lines, I immediately said *"break up interpreter.py, it's too big."* When the registry module grew three responsibilities, I split it into three files. When tests were in a flat directory, I separated them into `unit/` and `integration/`.

### The Language Feature Blitz

The most intense phase was plugging language-specific gaps across all 15 frontends. My approach was systematic:

1. Ask Claude to audit every frontend for missing constructs
2. Prioritise by impact (high/medium/low)
3. Say *"implement all the critical and common ones"*
4. Push, then immediately re-audit

This cycle repeated 4–5 times, each time catching a smaller set of remaining gaps. The test count tracked the progression: 645 → 672 → 682 → 687 → 690 → 698 → 720 → 746 → 775 → 1053 → 1176 → 1186 → 1198 → 1203.

### Screenshot-Driven Debugging

For the Mermaid CFG visualisation work, a different mode of interaction emerged. I'd generate a diagram, screenshot it, paste it into the conversation, and ask *"why does it look so disjointed?"* Claude could see the rendering and diagnose layout issues. The CFG visualisation went through five implementation rounds before I was satisfied — subgraphs, call edges, block collapsing, shape conventions, unreachable block pruning.

---

## Part 3: Patterns That Emerged

### Pattern 1: Brainstorm → Probe → Crystallise

I never started with a fixed architecture. I started with a problem, brainstormed approaches with Claude, then *probed* each approach by implementing it and testing on real data. The architecture crystallised from empirical feedback, not upfront design.

The deterministic VM wasn't planned — it emerged from asking *"shouldn't this be deterministic?"* after seeing the LLM-based approach work. The hybrid embedding pipeline wasn't planned — it emerged from watching the LLM classifier be too slow and the embedding classifier lose directional information.

### Pattern 2: The Plan Document as Interface

My most effective interaction pattern was the structured plan. After brainstorming and discussing trade-offs, I'd formulate a plan document — context, phases, file-by-file changes, verification steps — and feed it to Claude as an implementation spec. This happened ~15 times across the projects.

The plan document serves as an *interface contract* between the human architect and the AI implementer. It's specific enough that Claude can execute without ambiguity, but high-level enough that the human retains architectural control.

### Pattern 3: Audit-Implement-Reaudit Loops

After every batch of features, I immediately asked *"what else is missing?"* This tightening loop drove the system toward completeness without requiring me to enumerate every gap upfront. Claude's ability to audit a large codebase for consistency (e.g., "which of the 15 frontends is missing switch statement support?") was one of its highest-leverage capabilities.

### Pattern 4: Immediate Cleanup as Discipline

I never said "we'll refactor later." Every implementation was immediately followed by:
- Black formatting
- Test organisation
- Module splitting if a file grew too large
- README updates
- Architectural decision records

This discipline was encoded in my `CLAUDE.md` file, which Claude followed for every commit. The CLAUDE.md itself evolved over time — I added rules as I discovered patterns that needed enforcement.

### Pattern 5: Terse Directives After Trust

Early prompts were detailed and cautious. By mid-project, I was saying *"do all of them"*, *"push"*, *"commit and push this"*. Trust built through consistent execution. When Claude produced correct, formatted, tested code for the 50th time, I stopped micromanaging the implementation details and focused on architectural direction.

### Pattern 6: Context Window as Session Boundary

The 18-hour marathon exhausted the context window 5–6 times. Each continuation carried a structured summary of what was done and what remained. This forced a natural "checkpoint" discipline — I couldn't rely on Claude remembering earlier decisions, so I had to be explicit about state. In hindsight, this was healthy: it prevented architectural drift and kept each session focused.

---

## Part 4: Where It Surprised Me

### Surprise 1: The AI is better at breadth than depth

Claude excelled at tasks like "generate deterministic frontends for 14 languages" or "audit all 15 frontends for missing switch support." These breadth tasks — applying a consistent pattern across many targets — would have taken me days of tedious work. Claude did them in minutes.

Where it needed more guidance was depth: subtle semantic decisions like closure capture semantics (snapshot vs. shared environment), or when to use SYMBOLIC fallback vs. crash. These required me to probe with specific test cases and reason about the implications.

### Surprise 2: The workflow matters more than the prompts

My single biggest productivity lever wasn't clever prompting — it was the workflow encoded in `CLAUDE.md`. Rules like "run all tests before committing", "use dependency injection not mock.patch", "prefer early return", "one class per file" — these accumulated into a consistent codebase even across hundreds of sessions. The CLAUDE.md file was the real architecture document.

### Surprise 3: Empirical validation beats specification

I rarely specified exact behaviour upfront. Instead, I'd implement a feature, run it on real code (usually the `smojol` Java repo or a multi-language test suite), and judge the results. *"The confidence scores seem low"* → pivot to embeddings. *"Why does the CFG look disjointed?"* → fix the visualisation. The AI made this loop fast enough to be practical — I could test an idea and get results in minutes, not hours.

### Surprise 4: The TDD workflow changed how I direct AI

Late in the project, I modified my workflow to: Brainstorm → Trade-offs → Plan → **Write unit tests** → Implement → Fix tests → Commit → Refactor. Writing tests first forced me to think about the interface before the implementation, and it gave Claude a concrete target to implement against. The tests became the specification.

---

## Part 5: The Numbers

| Metric | Codescry | RedDragon | Total |
|--------|----------|-----------|-------|
| Conversation sessions | ~195 | ~204 | ~399 |
| Development days | 6 | 3 | 9 |
| Language frontends | — | 15 | 15 |
| Test count (final) | — | 1,214 | — |
| Architectural pivots | 7 | 5 | 12 |
| Lines of Python (est.) | ~5,000+ | ~8,000+ | ~13,000+ |

---

## Conclusion

Building non-trivial systems with an AI pair programmer is not about "prompting" — it's about *architectural direction*. The human's role is strategic: choosing problems, evaluating approaches empirically, making pivot decisions, and encoding quality standards. The AI's role is tactical: implementing plans, auditing for completeness, applying patterns at breadth, and maintaining consistency.

The workflow I converged on — brainstorm, discuss trade-offs, plan, test-first, implement, clean up — emerged through trial and error across 400 sessions. It's not the only way to work with AI, but it produced systems I'm genuinely proud of: tested, documented, formatted, and architecturally coherent.

The most important file in all three repositories isn't any Python module. It's `CLAUDE.md`.
