---
title: "Designing RedDragon: A Multi-Language Symbolic Code Analysis Engine"
author: avishek
usemathjax: false
tags: ["Software Engineering", "Compilers", "Program Analysis", "Symbolic Execution", "AI-Assisted Development"]
draft: false
---

*How a universal IR, 15 deterministic frontends, a symbolic VM, and an obsessive audit loop produced 7,268 tests with zero LLM calls.*

---

## The Problem

I wanted to analyse source code across many languages (trace data flow, build control flow graphs, understand how variables depend on each other) without writing a separate analyser for each language. The conventional approach is to build language-specific tooling (Roslyn for C#, javac's AST for Java, etc.), but that means duplicating every downstream analysis pass for every language. I wanted one representation, one analyser, many languages.

The twist: I also wanted to handle *incomplete* programs gracefully. Real-world code depends on imports, frameworks, and external systems that aren't available during static analysis. Most tools crash or give up when they hit an unresolved reference. I wanted mine to keep going, creating symbolic placeholders for unknowns and tracing data flow through them.

RedDragon is the result. It parses source in 15 languages, lowers it to a universal intermediate representation, builds control flow graphs, performs iterative dataflow analysis, and executes programs symbolically via a deterministic virtual machine. All with zero LLM calls for programs with concrete inputs.

This post covers how the system was designed, how it evolved, and the engineering discipline that kept it coherent across 28 architectural decisions and 400+ conversation sessions with Claude Code.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The IR: 19 Opcodes to Rule Them All](#the-ir-19-opcodes-to-rule-them-all)
3. [Frontends: Three Strategies, One Output](#frontends-three-strategies-one-output)
4. [The Dispatch Table Engine](#the-dispatch-table-engine)
5. [The Deterministic VM](#the-deterministic-vm)
6. [Dataflow Analysis](#dataflow-analysis)
7. [The Evolution: From Monolith to 7,268 Tests](#the-evolution-from-monolith-to-7268-tests)
8. [The Audit Loop: Systematic Completeness](#the-audit-loop-systematic-completeness)
9. [Cross-Language Verification via Exercism](#cross-language-verification-via-exercism)
10. [Guardrails: The CLAUDE.md as Architecture](#guardrails-the-claudemd-as-architecture)
11. [What I'd Do Differently](#what-id-do-differently)

---

## Architecture Overview

RedDragon follows a classic compiler pipeline, extended with symbolic execution:

```
Source Code (15 languages)
    │
    ▼
┌─────────────────────┐
│  Frontend           │  tree-sitter parse → AST → IR
│  (deterministic or  │  Sub-millisecond, zero LLM calls
│   LLM-based)        │
└─────────┬───────────┘
          │  list[IRInstruction]
          ▼
┌─────────────────────┐
│  CFG Builder        │  Basic blocks + edges
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐  ┌──────────────┐
│  VM    │  │  Dataflow     │
│(symex) │  │  Analysis     │
└────────┘  └──────────────┘
```

Every stage operates on the same flat IR. The VM and dataflow analysis are completely language-agnostic. They have no idea whether the instructions came from Python, Rust, or COBOL. That's the whole point.

---

## The IR: 19 Opcodes to Rule Them All

The intermediate representation is a flattened three-address code with 19 opcodes:

```
CONST, BINOP, UNOP,
STORE_VAR, LOAD_VAR,
STORE_FIELD, LOAD_FIELD,
STORE_INDEX, LOAD_INDEX,
CALL_FUNCTION, CALL_METHOD,
BRANCH, BRANCH_IF, LABEL,
RETURN, THROW,
NEW_OBJECT, NEW_ARRAY,
SYMBOLIC
```

Every instruction is a flat dataclass: an opcode, a list of operands, a destination register, and a source location tracing it back to the original code. No nested expressions. `a + b * c` decomposes into:

```
%0 = CONST b
%1 = CONST c
%2 = BINOP *, %0, %1
%3 = CONST a
%4 = BINOP +, %3, %2
```

This verbosity is the trade-off for universality. CFG construction, dataflow analysis, and VM execution all operate on the same flat list. Adding a new language means emitting these 19 opcodes; everything downstream works automatically.

The `SYMBOLIC` opcode is the escape hatch. When a frontend encounters a construct it doesn't handle, it emits `SYMBOLIC "unsupported:list_comprehension"` instead of crashing. The VM treats it as a symbolic value that propagates through execution. Over time, these emissions get replaced with real IR. The project's history is essentially the story of systematically eliminating every last `SYMBOLIC`.

---

## Frontends: Three Strategies, One Output

All three frontend strategies produce the same `list[IRInstruction]`. They differ in speed, coverage, and determinism:

**1. Deterministic frontends (15 languages):** Python, JavaScript, TypeScript, Java, Ruby, Go, PHP, C#, C, C++, Rust, Kotlin, Scala, Lua, Pascal. These use tree-sitter for parsing and a dispatch-table-based recursive descent for lowering. Sub-millisecond. Zero LLM calls. Fully testable.

**2. LLM frontend:** For languages without a deterministic frontend. The source is sent to an LLM constrained by a formal schema: all 19 opcode specs, concrete patterns, and worked examples. The LLM acts as a mechanical compiler frontend, not a reasoning engine. This distinction matters: the prompt doesn't ask *"what does this code do?"* It asks *"translate this into these specific opcodes."*

**3. Chunked LLM frontend:** For large files that overflow context windows. Tree-sitter decomposes the file into per-function chunks, each is LLM-lowered independently, registers and labels are renumbered to avoid collisions, and the chunks are reassembled into a single IR.

The key architectural decision was making the LLM path a *compiler frontend*, not a *reasoning engine*. When you constrain the LLM to pattern-matching against a formal schema, output quality jumps dramatically. It's not reasoning about semantics. It's translating syntax, which is exactly what LLMs trained on millions of code files are good at.

---

## The Dispatch Table Engine

The heart of the deterministic frontends is a `BaseFrontend` class (~950 lines) that all 15 languages inherit from. It uses two dispatch tables (one for statements, one for expressions) mapping tree-sitter AST node types to handler methods.

The lowering dispatch chain:

```
lower(root)
  → _lower_block(root)           # iterate named children
    → _lower_stmt(child)         # skip noise/comments; try STMT_DISPATCH
      → _lower_expr(child)       # fallback: try EXPR_DISPATCH
        → SYMBOLIC("unsupported:X")  # final fallback
```

Common constructs (`if/else`, `while`, `for`, `return`, `function_definition`, `class_definition`, `try/catch`) are handled in the base class. Language-specific constructs override or extend. Overridable constants handle the small but persistent differences across grammars:

```python
# Python says "True", Go says "true", Lua says "true"
TRUE_LITERAL: str = "True"    # default
FALSE_LITERAL: str = "False"
NONE_LITERAL: str = "None"

# Python puts the body in "body", Go puts it in "block"
FUNC_BODY_FIELD: str = "body"
IF_CONSEQUENCE_FIELD: str = "consequence"
```

All 15 languages canonicalise their native null/boolean forms to Python-form at lowering time. `nil`, `null`, `undefined`, `NULL` all become `"None"`. `true`, `True`, `TRUE` all become `"True"`. This means the VM only handles one set of literals, regardless of source language.

Adding support for a new AST node type is mechanical: write a handler method, register it in the dispatch table. This is what made the systematic coverage push possible. When the audit flagged 34 missing node types across 15 languages, implementing them was straightforward because each one followed the same pattern.

---

## The Deterministic VM

The most important architectural decision in RedDragon was making the VM fully deterministic.

The original design had the LLM deciding state changes at each execution step. When the VM encountered an unknown value, it asked the LLM what to do. This was slow, non-deterministic, untestable, and fragile.

The key insight came from a simple question: *"Given that the IR is always bounded, shouldn't execution be deterministic?"* Yes. If the IR has no unbounded loops (or loops are bounded by concrete values), execution is a mechanical process. Unknown values don't need to be *resolved*. They can be *created* as symbolic placeholders that propagate through computation.

So we ripped out all LLM calls from the VM. When execution hits an unresolved import or function, it creates a `SymbolicValue` with a descriptive hint:

```
sym_0 (hint: "math.sqrt(16)")
```

This symbolic value propagates through arithmetic, field access, and method calls deterministically. `sym_0 + 1` produces `sym_1` with the provenance chain intact. The entire execution trace is reproducible. Run it twice, get exactly the same result.

The trade-off is that symbolic branches always take the true path (a simplification), and symbolic values can't be resolved to concrete results without help. For the latter, a configurable `UnresolvedCallResolver` allows plugging in an LLM oracle that makes lightweight calls to get plausible concrete values. This is opt-in, not the default path.

### Closures

One subtle design iteration worth mentioning: closure capture semantics. The initial implementation captured variables by snapshot (copy at definition time). This broke counter factories:

```python
def make_counter():
    count = 0
    def inc():
        count += 1
        return count
    return inc
```

With snapshot capture, `inc()` always reads `count = 0`. The fix was shared `ClosureEnvironment` cells: all closures from the same scope share a mutable environment, matching Python/JavaScript semantics. This is the kind of deep correctness issue that only surfaces through specific test cases. It's documented as ADR-019 in the project's decision records.

---

## Dataflow Analysis

The dataflow module performs iterative intraprocedural analysis:

1. **Collect definitions**: identify every point where a variable or register is assigned
2. **Reaching definitions**: GEN/KILL worklist fixpoint iteration over the CFG
3. **Def-use chains**: link each use to the definition(s) that reach it
4. **Variable dependency graph**: trace through register chains to discover named-variable-to-named-variable dependencies, with transitive closure

The interesting part is step 4. The IR uses temporary registers (`%0`, `%1`, ...) for all intermediate values. A statement like `y = x + 1` becomes:

```
%0 = LOAD_VAR x
%1 = CONST 1
%2 = BINOP +, %0, %1
     STORE_VAR y, %2
```

The raw def-use chain says "`y` depends on `%2`". But a human wants to know "`y` depends on `x`". The dependency graph builder traces through the register chain: `%2` comes from `BINOP` on `%0` and `%1`; `%0` comes from `LOAD_VAR x`; `%1` is a constant. Therefore `y` depends on `x`. Transitive closure extends this across multi-step computations.

The dataflow module has zero dependencies on the VM, frontends, or backends. It's a pure analysis pass over the CFG. This is the ports-and-adapters architecture in practice: the functional core (analysis logic) is completely decoupled from the imperative shell (parsing, I/O, LLM calls).

---

## The Evolution: From Monolith to 7,268 Tests

RedDragon's evolution followed a clear pattern of phases, each triggered by testing the previous one on real code:

**Phase 1: The monolith (Hour 0 to 2).** A single `interpreter.py` with an LLM-based lowering and execution engine. ~1,200 lines. It worked, barely.

**Phase 2: The determinism pivot (Hour 2 to 4).** The key insight: execution should be deterministic. Ripped out all LLM calls from the VM. Added symbolic value creation. This was the decision that made everything else possible. Suddenly the system was testable.

**Phase 3: Multi-language frontends (Hour 4 to 8).** Asked: *"How hard is it to write deterministic logic to lower ASTs for 15 languages?"* The answer: not that hard, with tree-sitter and a dispatch table engine. 15 frontends generated in a single marathon session. 346 tests.

**Phase 4: Analysis and tooling (Hour 8 to 14).** Added iterative dataflow analysis, chunked LLM frontend, Mermaid CFG visualisation with subgraphs and call edges, source location traceability. Extracted CLI into composable API.

**Phase 5: Systematic hardening (Sessions 50 to 130).** This is where the test count exploded. The Rosetta cross-language test suite (8 algorithms x 15 languages) and then the Exercism integration suite drove the test count from ~700 to 7,268 across 80+ sessions. Each exercise exposed new frontend gaps, VM limitations, and edge cases, and each fix was immediately verified across all 15 languages.

The test count tells the story:

```
Phase 1–3:   346 tests
Phase 4:     ~700 tests
Rosetta:     ~1,200 tests
Exercism 1:  ~2,700 tests
Exercism 2:  ~4,200 tests
Exercism 3:  ~5,150 tests
Exercism 4:  ~7,076 tests
Final:        7,268 tests (+ 3 xfailed)
```

Every test runs with zero LLM calls. Every test is deterministic.

---

## The Audit Loop: Systematic Completeness

The most distinctive engineering pattern in RedDragon was the *audit-fix-reaudit* loop. After every batch of frontend work, I ran a comprehensive two-pass audit:

**Pass 1 (Dispatch Comparison):** Parse source samples in all 15 languages, collect every AST node type that appears, compare against the frontend's dispatch tables, and classify unhandled types as structural (harmless, consumed by parent handlers) or substantive (gaps that produce `SYMBOLIC`).

**Pass 2 (Runtime SYMBOLIC check):** Actually lower the source through each frontend, scan the resulting IR for `SYMBOLIC` instructions with `"unsupported:"` operands. This catches gaps that the static analysis might miss.

The classification heuristic itself went through three iterations:

1. **Naive:** Flag everything not in a dispatch table. This produced hundreds of false positives, because nodes like `parameter_list` and `type_annotation` are consumed by parent handlers and never reach the dispatch chain independently.

2. **Parent heuristic:** Flag unhandled nodes only if their immediate parent isn't handled. This reduced false positives but still produced 259. When the parent was also unhandled but deep structural (never block-iterated), its children got flagged incorrectly.

3. **Block-reachability analysis:** The final approach. Walk the AST and identify which unhandled nodes are *direct named children of block-iterated nodes* (the root, or nodes whose type maps to `_lower_block` in the dispatch table). Only these nodes can ever reach `_lower_stmt` and produce `SYMBOLIC`. Everything else is a deep structural node consumed by parent handlers.

The block-reachability approach dropped substantive gaps from 259 to 1 (a C `case_statement` that was a genuine gap, subsequently fixed with a defensive handler). The evolution of this heuristic is a good example of how empirical feedback drives design. Each version was tested against the full corpus, and false positives were investigated until the classification matched reality.

The audit loop ran dozens of times across the project's life:

```
Audit → 34 gaps found → implement all 34 (57 new tests)
Re-audit → 19 gaps found → implement all 19 (28 new tests)
Re-audit → 12 gaps found → implement all 12 (18 new tests)
Re-audit → 0 gaps, 0 SYMBOLIC
```

This pattern (audit, batch-fix, re-audit) was the single most effective technique for driving the system toward completeness. I didn't enumerate every missing feature upfront. I let the audit tell me what was missing, fixed everything it found, and repeated until it found nothing.

---

## Cross-Language Verification via Exercism

The most ambitious verification effort was the Exercism integration test suite. The idea: take Exercism's canonical test cases (which define expected inputs and outputs for programming exercises), write equivalent solutions in all 15 languages, and verify that RedDragon's pipeline produces the correct answer for every case in every language.

Each exercise tests a specific set of language constructs:

| Exercise | Key Constructs | Cases | Total Tests |
|----------|----------------|-------|-------------|
| leap | modulo, boolean logic, short-circuit eval | 9 | 287 |
| collatz-conjecture | while loop, conditional, integer division | 4 | 137 |
| difference-of-squares | accumulator, function composition | 9 | 287 |
| two-fer | string concatenation, string literals | 3 | 107 |
| hamming | string indexing, character comparison | 5 | 167 |
| reverse-string | backward iteration, char-by-char building | 5 | 167 |
| rna-transcription | multi-branch if, character mapping | 6 | 197 |
| perfect-numbers | divisor loop, three-way string return | 9 | 287 |
| triangle | nested ifs, validity guards, 3-arg functions | 21 | 647 |
| space-age | float division, float constants | 8 | 257 |
| grains | exponentiation, large integers | 8 | 257 |
| isogram | nested loops, continue, helper functions | 14 | 437 |
| nth-prime | trial division, primality testing | 3 | 107 |
| resistor-color | string-to-int mapping | 3 | 107 |
| pangram | string variable indexing, nested loops | 11 | 347 |
| bob | multi-branch string classification | 22 | 633 |
| luhn | two-pass validation, right-to-left traversal | 22 | 677 |
| acronym | word boundary detection, toUpperChar | 9 | 269 |

For each exercise, every canonical test case generates tests across three dimensions:

1. **Lowering quality**: does the IR contain any `unsupported:` SYMBOLIC? (15 tests per exercise)
2. **Cross-language consistency**: do all 15 languages produce structurally equivalent IR? (2 tests per exercise)
3. **VM execution correctness**: does the VM produce the expected output? (cases x languages tests)

The argument substitution mechanism deserves a mention: a `build_program()` helper finds the `answer = f(default_arg)` line in each solution and substitutes new arguments for each canonical test case. This works across languages with different assignment syntaxes (`=`, `:=`, `: type =`) via regex.

The Exercism suite was the single biggest driver of quality. Each exercise exposed new gaps: Ruby's `parenthesized_statements` vs Python's `parenthesized_expression`, Rust's expression-position loops, Pascal's single-quote string escaping, PHP's `.` concatenation operator. Every gap found was a bug fixed.

---

## Guardrails: The CLAUDE.md as Architecture

RedDragon was built almost entirely through conversations with Claude Code. Across 400+ sessions, the most important technical artifact wasn't any Python module. It was `CLAUDE.md`, the file that encodes development rules.

Some of the rules that shaped the codebase most:

**"STOP USING FOR LOOPS WITH MUTATIONS IN THEM. JUST STOP."** This rule forced a functional programming style across the codebase. List comprehensions, `map`, `filter`, `reduce` instead of mutable accumulators. The code is denser but more predictable.

**"Do not use `unittest.mock.patch`. Use proper dependency injection."** This forced every external dependency (LLM clients, file I/O, clocks) to be injectable. The result: the entire VM, all 15 frontends, and all analysis passes are testable in isolation without mocking.

**"If a function has a non-None return type, never return None."** Combined with null object pattern enforcement, this eliminated an entire class of `NoneType` errors. Functions that can't produce a value return a null object, not `None`.

**"Before committing anything, run all tests, fixing them if necessary."** This simple rule prevented test count regression across 100+ commits. The test count only ever went up.

**"Once a design is finalised, document it as an ADR."** This produced 28 timestamped architectural decision records that serve as the project's institutional memory. Each records the context, the decision, and the consequences, including trade-offs.

The workflow encoded in CLAUDE.md is: **Brainstorm → Discuss trade-offs → Plan → Write unit tests → Implement → Fix tests → Commit → Refactor.** This isn't just process documentation. It's an enforceable contract. Every session begins with these rules loaded into context.

---

## What I'd Do Differently

**Start with the audit earlier.** The two-pass audit should have existed from the first batch of frontends. Instead, I relied on manual inspection for the first 50 sessions, and only built the audit when the number of frontends made manual checking impossible. The audit loop was the highest-leverage quality tool in the project.

**Invest in cross-language tests from day one.** The Rosetta and Exercism suites exposed more bugs than all the language-specific unit tests combined. A single exercise tested across 15 languages covers more surface area than 50 unit tests in one language.

**Be more aggressive about the functional core.** Even with the FP rules in CLAUDE.md, some mutation crept in, especially in the VM executor. The dataflow module, by contrast, is almost purely functional and is by far the easiest module to reason about and test. The correlation is not a coincidence.

---

## The Numbers

| Metric | Value |
|--------|-------|
| Supported languages | 15 (deterministic) + any (LLM) |
| IR opcodes | 19 |
| Tests (all passing) | 7,268 + 3 xfailed |
| LLM calls at test time | 0 |
| Architectural decision records | 28 |
| Exercism exercises | 18 (across 15 languages) |
| Rosetta algorithms | 8 (across 15 languages) |
| Conversation sessions | ~400 |
| Git commits | 125 |
| Co-authored with Claude | 124 (99.2%) |
| Solo human commits | 1 |
| Audit substantive gaps (final) | 0 |
| Audit SYMBOLIC emissions (final) | 0 |

---

## Conclusion

RedDragon started as a question: *"Can I build a single system that analyses code in any language?"* It evolved through iterative probing into a compiler pipeline with 15 deterministic frontends, a symbolic VM, and cross-language verification that would be impractical to build by hand.

The system design isn't novel in its components. TAC IR, dispatch tables, worklist dataflow, symbolic execution are all textbook techniques. What's unusual is the *combination*: applying classical compiler techniques to build a practical multi-language analysis tool, hardened through systematic auditing and cross-language testing.

The architecture crystallised through empirical feedback, not upfront design. The deterministic VM wasn't planned. It emerged from asking *"shouldn't this be deterministic?"* The audit loop wasn't planned. It emerged from asking *"what's still missing?"* The Exercism test suite wasn't planned. It emerged from wanting more confidence than unit tests alone could provide.

Each of these decisions was triggered by testing the previous one on real code and noticing a gap. The fastest path to a good architecture isn't to design it upfront. It's to build, test, and let the gaps tell you what to fix next.

_This post has not been written or edited by AI._
