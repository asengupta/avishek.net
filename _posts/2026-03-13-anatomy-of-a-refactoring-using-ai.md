---
title: "Engineering Log: Anatomy of a Refactoring Using AI"
author: avishek
usemathjax: false
mermaid: true
tags: ["Software Engineering", "Refactoring", "AI-Assisted Development", "Type Systems"]
draft: false
---

*Tracing the full arc of a multi-phase refactoring — from "Java string concatenation crashes the VM" to "every value in the system carries its type" — done across a dozen sessions with Claude Code over two days.*

---

## Table of Contents

- [The Problem](#the-problem)
- [The Tools](#the-tools)
  - [Superpowers: Enforced Design Discipline](#superpowers-enforced-design-discipline)
  - [Beads: Local-First Task Tracking](#beads-local-first-task-tracking)
  - [Code Simplifier: Automated Cleanup After Every Change](#code-simplifier-automated-cleanup-after-every-change)
- [Phase 1: TypedValue and BinopCoercionStrategy](#phase-1-typedvalue-and-binopcoercionstrategy)
  - [The Design Decision That Shaped Everything](#the-design-decision-that-shaped-everything)
  - [The Boundary Table](#the-boundary-table)
- [Phase 2: Handler Migration](#phase-2-handler-migration)
  - [The Serialize/Deserialize Roundtrip](#the-serializedeserialize-roundtrip)
- [Phase 3: Return Values](#phase-3-return-values)
  - [The Constructor Bug](#the-constructor-bug)
- [Phases 4–6: Heap and Closures](#phases-46-heap-and-closures)
  - [The Double-Wrapping Landmine](#the-double-wrapping-landmine)
- [Phase 7: Cleaning Up After Ourselves](#phase-7-cleaning-up-after-ourselves)
- [Detour: Builtins](#detour-builtins)
  - [BuiltinResult: Side Effects Should Be Declarative](#builtinresult-side-effects-should-be-declarative)
  - [Builtin Args: The Atomic Commit Problem](#builtin-args-the-atomic-commit-problem)
- [More Detours](#more-detours)
  - [BinopCoercionStrategy Return Type](#binopcoercionstrategy-return-type)
  - [UnopCoercionStrategy](#unopcoercionstrategy)
  - [Demo Scripts and LLM Path Leaks](#demo-scripts-and-llm-path-leaks)
  - [The Question That Came After](#the-question-that-came-after)
- [The Shape of the Work](#the-shape-of-the-work)
- [Takeaways](#takeaways)

---

## The Problem

[RedDragon](https://github.com/avishek-sen-gupta/red-dragon) is a multi-language code analysis engine with a universal IR, deterministic VM, and a type system. It parses 15 languages, lowers them to IR, and executes the IR on a virtual machine.

The VM stored values as raw Python primitives — `int`, `str`, `float`, `bool`. Type information lived in a completely separate structure: a `TypeEnvironment` built by a static inference pass before execution. The operators themselves never saw types. They received raw values via `_resolve_reg` and used Python's native operators.

This created an obvious problem. When Java code did `"int:" + 42`, Python raised `TypeError` because it can't concatenate `str` and `int`. The VM caught the exception and degraded the result to a `SymbolicValue` — a placeholder meaning "I don't know what this is." The concrete information was gone. Java, C#, Kotlin, and Scala all auto-stringify non-string operands in string concatenation. The VM had no way to implement this because type information was absent at the point of operation.

The fix looked straightforward: make type information available to operators. What followed was a refactoring that touched almost every layer of the VM, exposed hidden assumptions in constructor handling, revealed that builtins were bypassing the state management contract, and prompted several side detours into coercion protocols, demo scripts, and the question of whether two separate type-tracking mechanisms were still both necessary.

This post traces that arc. It's written partly as documentation and partly because the shape of the work — the way a focused fix expanded into a system-wide migration, the side detours, the bugs that only surfaced because something else changed — is characteristic of refactoring work in general. The AI didn't change the nature of that work. It changed the speed.

All of this was done through conversations with Claude Code, using three tools that shaped the work as much as the code decisions did: [Superpowers](https://github.com/anthropics/claude-code-plugins) for enforced design discipline, [Beads](https://github.com/anthropics/beads) for local-first task tracking, and [Code Simplifier](https://github.com/anthropics/claude-code-plugins) for automated post-implementation cleanup. The next section describes all three.

The refactoring spanned about a dozen sessions over two days. I'm including specific moments from those conversations — places where I had to course-correct, where I got frustrated with the codebase or the AI's approach, where a question I asked led to discovering something unexpected — because the texture of those interactions is part of the story.

---

## The Tools

Three tools ran alongside Claude Code throughout this migration. None are part of Claude Code itself — they're open-source plugins that layer structure on top of it. I'm describing them here because the post references them repeatedly, and their constraints shaped how the work unfolded.

### Superpowers: Enforced Design Discipline

[Superpowers](https://github.com/anthropics/claude-code-plugins) is a skill system for Claude Code. Skills are structured prompts that activate automatically based on the task at hand. They don't add capabilities the AI doesn't have — they enforce workflows the AI would otherwise skip.

The skills that mattered for this migration:

**Brainstorming.** Every major phase started here. The brainstorming skill runs a structured dialogue: it asks one clarifying question at a time, proposes alternative approaches with explicit trade-offs, and refuses to produce a design spec until you've agreed on the direction. It won't let you skip to implementation. This is the skill that caught the serialize/deserialize split (Phase 2) and the `BuiltinResult` design (Detour) — cases where the obvious approach wasn't the best one, and the skill's insistence on proposing alternatives surfaced something simpler.

The brainstorming pipeline looks like this:

```mermaid
flowchart LR
    Q("🔍 Clarifying<br/>questions<br/><i>one at a time</i>"):::ask --> A("⚖️ Alternative<br/>approaches<br/><i>with trade-offs</i>"):::think
    A --> D{"🎯 Design<br/>decision<br/><i>user chooses</i>"}:::decide
    D -- "more questions" --> Q
    D -- "agreed" --> S("📋 Design spec"):::output

    classDef ask fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef think fill:#fff3e0,stroke:#e8a735,stroke-width:2px,color:#5c3a0a
    classDef decide fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#5c0a0a
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a

    linkStyle 2 stroke:#2e7d32,stroke-width:2px
```

The early phases (1–3) had longer brainstorming cycles — eight questions in Phase 1 before any code was discussed. By the later phases, the cycles were shorter because the patterns were established: the skill would propose an approach, I'd confirm it matched the established pattern, and we'd move to planning.

**Writing-plans.** Takes the design spec from brainstorming and breaks it into granular TDD steps — test first, then implementation, then verification. Each step is small enough to be a single commit. The plan for Phase 5 (heap fields) had 12 steps; the plan for the builtin args migration (Detour) explicitly mandated zero intermediate commits because the interface change was atomic.

**Subagent-driven development.** Dispatches fresh Claude Code agents per task from the plan, each with its own context window. The dispatching agent reviews each sub-agent's work before accepting it. This is where the review caught the `value is not None` guard in Phase 3 — a sub-agent had taken a shortcut that violated the design spec, and the reviewing agent flagged it. Without the two-stage review, that shortcut would have shipped.

The full pipeline:

```mermaid
flowchart LR
    B("🧠 Brainstorm"):::phase1 --> SP("📋 Spec"):::phase2
    SP --> PL("📐 Plan<br/><i>TDD steps</i>"):::phase2
    PL --> SA("🤖 Sub-agents<br/><i>one per task</i>"):::phase3
    SA --> RV("🔎 Review<br/><i>two-stage</i>"):::phase4
    RV --> CM("✅ Commit"):::phase5

    classDef phase1 fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef phase2 fill:#fff3e0,stroke:#e8a735,stroke-width:2px,color:#5c3a0a
    classDef phase3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#3a0a3a
    classDef phase4 fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#5c0a0a
    classDef phase5 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a
```

The discipline this pipeline enforces is not novel — brainstorm, spec, plan, implement, review is how careful engineering has always worked. What's different is that a tool *enforces* the steps. When you're twelve sessions into a migration and tempted to just start coding the next phase, the skill doesn't let you. It asks its questions first.

### Beads: Local-First Task Tracking

[Beads](https://github.com/anthropics/beads) is a local-first issue tracker that lives alongside the repo as flat files. No server, no web UI, no syncing. Tasks are created and queried from the command line: `bd create`, `bd ready`, `bd update <id> --status closed`.

The features that mattered:

**Dependencies.** Every Beads task can declare dependencies on other tasks. `bd ready` shows only tasks whose dependencies are all closed — the next unblocked work. This turned the dependency graph in [The Shape of the Work](#the-shape-of-the-work) from a diagram into an operational tool. After closing a phase, `bd ready` surfaced whatever was next — the next planned phase or a detour that had just become unblocked.

**Instant triage.** When a detour surfaced mid-session — the constructor bug in Phase 3, the builtin side-effect problem after Phase 7 — it got filed as a new Beads task in two seconds: `bd create "BuiltinResult: builtins bypass apply_update" --dep red-dragon-xyz`. The dependency was set, and the task appeared in `bd ready` at the right time. Without this, mid-session discoveries would have been either fixed immediately (derailing the current work) or forgotten.

**Mid-session pivots.** The clearest example: midway through brainstorming the heap fields migration (`red-dragon-f6i`), I realized handlers needed to be migrated first. I interrupted, ran `bd update red-dragon-f6i --status deferred`, created the handler migration task, and pivoted. When the handler migration was done and closed, `bd ready` surfaced the deferred heap fields task automatically. The tracker absorbed the pivot without losing the deferred work.

**Session boundaries.** Each Beads task has an ID (like `red-dragon-gsl`) that appears in commit messages and in the brainstorming/planning conversations. When a new Claude Code session starts, the first thing I do is `bd ready` — the task list is the handoff between sessions. The AI doesn't need to remember what happened last session; the tracker tells it what's next.

Over a dozen sessions and five detours, the pattern was: close a task → `bd ready` → claim the next one → brainstorm → plan → implement → commit → close. The tracker turned "I should also fix this other thing I just noticed" from a context-switching hazard into a two-second operation.

### Code Simplifier: Automated Cleanup After Every Change

[Code Simplifier](https://github.com/anthropics/claude-code-plugins) is a Claude Code plugin that runs as a dedicated review agent after implementation work completes. It focuses on code that was just modified — not the entire codebase — and refines it for clarity, consistency, and maintainability without changing behaviour.

In a migration like this, where sub-agents are churning out handler-by-handler changes across dozens of files, the code that lands is functional but not always clean. A sub-agent focused on migrating `_handle_store_field` to produce `TypedValue` will get the types right but might leave behind redundant intermediate variables, unnecessarily verbose conditionals, or naming inconsistencies with the surrounding code. Code Simplifier catches this.

What it does:

- **Reduces unnecessary complexity.** Nested ternaries become `if`/`else` chains. Three-line variable assignments that exist only to be passed once get inlined. Guard clauses replace deep nesting.
- **Eliminates redundant code.** During the transition phases, handlers accumulated isinstance checks, temporary unwrap/rewrap sequences, and defensive guards that were necessary mid-migration but dead after the phase completed. Code Simplifier flagged many of these before Phase 7's explicit cleanup pass.
- **Enforces consistency.** When 15 handler groups are migrated one at a time across multiple sessions, naming conventions drift. One handler might call the coerced value `lhs_coerced`, another `coerced_lhs`, another `left`. Code Simplifier normalises these.

The key constraint: it only touches recently modified code. It won't "improve" stable code you didn't ask about. This prevents the scope creep that happens when cleanup tools audit everything — you end up with a 200-file diff when you wanted a 3-file fix.

In practice, I invoked it after each major phase commit. The simplifier would produce a small follow-up diff — typically 10–30 lines changed — that tightened the code the sub-agents had just written. These were fast reviews because the behavioural correctness was already established by the tests; the simplifier was only adjusting form, not function.

---

## Phase 1: TypedValue and BinopCoercionStrategy

The first design session used the brainstorming skill to work through the problem space. The skill's process is structured: it asks one clarifying question at a time, proposes approaches with trade-offs, and won't let you skip to implementation until a design is approved. In this case, the dialogue went through eight questions before code was discussed.

The skill started with motivation: *"What's the primary goal — language-correct operators, eliminating the side-car type system, or both?"* I said both. Then it asked whether `TypedValue` should wrap everything (even values with unknown types) or only values with known types. It recommended wrapping everything — even with `UNKNOWN` — to eliminate all "is this typed or raw?" branching. I agreed, and this turned out to be the single most important design decision of the entire migration.

The next questions drilled into specifics. Should `TypedValue` subsume `SymbolicValue` and `Pointer`, or wrap them? The skill proposed wrapping — keeping `SymbolicValue` as a value inside `TypedValue` rather than replacing it — which preserved the existing constraint-tracking machinery without duplication. Then: how should BINOP consume types? The skill proposed two approaches: (A) unwrap, operate on raw values, rewrap with inferred type, or (B) full type-driven dispatch where operators receive `TypedValue` throughout. It recommended A as the simpler path. I agreed, but added a requirement: *"BINOP should have access to a pluggable language-specific TypeConversionStrategy."* That addition — mine, not the skill's — became `BinopCoercionStrategy`.

The skill then proposed three migration approaches:

1. **Big Bang** — change everything at once. Rejected as too risky.
2. **Incremental with accessor protocol** — wrap at `apply_update`, migrate handlers one by one. Recommended and chosen.
3. **Transparent wrapper with magic methods** — `TypedValue.__add__` delegates to the underlying value. The skill flagged this as violating the simplest-mechanism principle: `isinstance(val, int)` checks throughout the codebase would silently fail.

The last question was about scope: *"Where should language information live — on the value or on the strategy?"* The skill recommended the strategy: *"A Java Int and a C# Int are the same value with the same type; the difference is in the coercion rules applied to them."* I agreed.

That session produced two things: `TypedValue` and `BinopCoercionStrategy`.

`TypedValue` is a frozen dataclass:

```python
@dataclass(frozen=True)
class TypedValue:
    value: Any       # The raw Python value
    type: TypeExpr   # The inferred or declared type
```

`BinopCoercionStrategy` is a protocol with two methods:

```python
class BinopCoercionStrategy(Protocol):
    def coerce(self, op: str, lhs: TypedValue, rhs: TypedValue) -> tuple[TypedValue, TypedValue]:
        ...
    def result_type(self, op: str, lhs: TypedValue, rhs: TypedValue) -> TypeExpr:
        ...
```

`coerce()` transforms operands before the operator runs. `result_type()` infers the output type. The executor calls both, wraps the result in `TypedValue`, and stores it.

Here's the full round trip of a binary operation like `"int:" + 42` in Java:

```mermaid
flowchart TD
    subgraph read ["① Read"]
        R1("%r1 → TypedValue('int:', String)"):::reg --> RESOLVE1("_resolve_binop_operand"):::fn
        R2("%r2 → TypedValue(42, Int)"):::reg --> RESOLVE2("_resolve_binop_operand"):::fn
    end

    subgraph coerce ["② Coerce"]
        RESOLVE1 --> LHS("lhs: TypedValue('int:', String)"):::tv
        RESOLVE2 --> RHS("rhs: TypedValue(42, Int)"):::tv
        LHS --> COERCE("⚖️ BinopCoercionStrategy.coerce('+', lhs, rhs)<br/><i>JavaBinopCoercion: stringify rhs</i>"):::strategy
        RHS --> COERCE
        COERCE --> COERCED_L("TypedValue('int:', String)"):::tv
        COERCE --> COERCED_R("TypedValue('42', String)"):::tvchanged
    end

    subgraph compute ["③ Compute"]
        COERCED_L -- ".value" --> EVAL("Operators.eval_binop('+', 'int:', '42')"):::fn
        COERCED_R -- ".value" --> EVAL
        EVAL --> RESULT("result = 'int:42'"):::raw
    end

    subgraph typeinfer ["④ Type + Wrap"]
        LHS --> RTYPE("BinopCoercionStrategy.result_type()"):::strategy
        RHS --> RTYPE
        RTYPE --> TYPE("String"):::type
        RESULT --> WRAP("typed('int:42', String)"):::fn
        TYPE --> WRAP
    end

    subgraph store ["⑤ Store"]
        WRAP --> STORE("%r3 → TypedValue('int:42', String)"):::reg
        STORE --> APPLY("apply_update()"):::fn
    end

    classDef reg fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef fn fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121
    classDef tv fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a
    classDef tvchanged fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#5c3a0a
    classDef strategy fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#3a0a3a
    classDef raw fill:#fff3e0,stroke:#e8a735,stroke-width:1px,color:#5c3a0a
    classDef type fill:#fce4ec,stroke:#c62828,stroke-width:1px,color:#5c0a0a
```

`DefaultBinopCoercion` is a no-op — it passes operands through unchanged and infers types from operator categories (comparisons return `Bool`, arithmetic follows numeric promotion rules). `JavaBinopCoercion` overrides `coerce()` to auto-stringify non-string operands when `+` is used with a `String`.

### The Design Decision That Shaped Everything

The critical decision was: **"language on strategy, not value."** A Java `Int` and a C# `Int` are the same `TypedValue`. The difference is in the injected coercion strategy. This meant we didn't need a `JavaInt` vs. `CSharpInt` distinction. The coercion strategy is selected once at the top of the execution pipeline based on the source language, then threaded through via dependency injection.

The other critical decision: **every value is `TypedValue`, even when the type is `UNKNOWN`**. This eliminated all branching on "is this typed or raw?" throughout the codebase. It sounded like over-wrapping at first. It turned out to be the decision that kept the migration tractable.

### The Boundary Table

The spec documented five boundary crossings where values moved between storage locations and what wrapping/unwrapping happened at each. This table became the roadmap for every subsequent phase. Each phase was essentially: pick a boundary, push TypedValue one layer deeper, update the read sites, run the tests.

```mermaid
flowchart LR
    subgraph Frame ["Stack Frame"]
        direction TB
        REG("📦 Registers<br/><code>%r1 → TypedValue</code>"):::reg
        VAR("📌 Local Vars<br/><code>x → TypedValue</code>"):::var
    end

    subgraph Heap ["Heap"]
        OBJ("🗄️ HeapObject.fields<br/><code>name → TypedValue</code>"):::heap
    end

    subgraph Closure ["Closure Environment"]
        BIND("🔗 Bindings<br/><code>captured_x → TypedValue</code>"):::closure
    end

    REG -- "STORE_FIELD" --> OBJ
    OBJ -- "LOAD_FIELD" --> REG
    VAR -- "capture" --> BIND
    BIND -- "function entry" --> VAR
    REG -- "CALL_FUNCTION" --> VAR
    VAR -- "STORE_VAR" --> VAR
    REG -- "LOAD_VAR" --> REG

    classDef reg fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef var fill:#fff3e0,stroke:#e8a735,stroke-width:2px,color:#5c3a0a
    classDef heap fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a
    classDef closure fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#3a0a3a
```

Each arrow is a boundary crossing. Before the migration, values were unwrapped to raw primitives and re-wrapped at each crossing. After: `TypedValue` flows through intact.

After this phase: ~11,274 tests passing.

---

## Phase 2: Handler Migration

Phase 1 wrapped values in `TypedValue` at the `apply_update` boundary — the function that takes a `StateUpdate` and applies it to the VM state. Every handler still produced raw values. `apply_update` wrapped them.

This worked but created a pointless roundtrip. Handlers called `_serialize_value()` to flatten objects into JSON-compatible structures, then `apply_update` called `_deserialize_value()` to reconstruct them. For locally-executed instructions (the vast majority), this was a no-op. The serialization path only existed for the LLM fallback, where an LLM returns a JSON `StateUpdate` that needs deserialization.

This phase had an interesting origin. I was partway through brainstorming the heap fields migration (Phase 5) when I realized that handlers were still producing raw values — the heap fields work would be building on a half-migrated foundation. I interrupted: *"I think then we pause this plan for now, and do red-dragon-132, so that values always arrive as TypedValue."* The heap fields task was deferred in Beads (`bd update red-dragon-f6i --status deferred`), and the handler migration jumped the queue. This is one of those moments where the issue tracker earned its keep — deferring a task mid-brainstorm without losing it.

### The Serialize/Deserialize Roundtrip

The brainstorming for the `apply_update` split was another case where I had to push back on the AI's first instinct. The AI proposed a dual-path `apply_update` with isinstance branching — check if the incoming value is `TypedValue` and take one path, otherwise take the raw path. I said: *"I think `apply_update` should be split into `apply_update` (which accepts only TypedValue) and `apply_update_raw` (the path which LLMs take)."* Then, on reflection: *"On second thoughts, what should probably happen is that in the else clause (the LLM path), the raw update should be transformed into a TypedValue update... that way, the updates from both clauses are the canonical TypedValue update."* The AI adopted this — a `materialize_raw_update` function that converts LLM responses into `TypedValue` updates before they enter the standard pipeline.

The fix split `apply_update` into two paths:
- **Local path:** Handlers produce `TypedValue` directly. `apply_update` stores them with lightweight type coercion.
- **LLM path:** A new `materialize_raw_update` function takes raw values from LLM JSON responses, deserializes them, coerces them, and wraps them in `TypedValue`.

**Before:** serialize → deserialize roundtrip

```mermaid
flowchart LR
    H1("Handler"):::fn -- "_serialize_value()" --> SU1("StateUpdate<br/><i>raw/JSON</i>"):::raw
    SU1 -- "_deserialize_value()" --> AU1("apply_update()"):::fn
    AU1 -- "wrap" --> VM1("VM State"):::state

    classDef fn fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121
    classDef raw fill:#fce4ec,stroke:#c62828,stroke-width:1px,color:#5c0a0a
    classDef state fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
```

**After:** dual path — local handlers and LLM backends converge on the same `apply_update()`

```mermaid
flowchart TD
    H2("🖥️ Local Handler"):::local -- "produces TypedValue" --> SU2("StateUpdate<br/><i>TypedValue</i>"):::tv
    LLM("🌐 LLM Backend"):::llm -- "JSON response" --> RAW("Raw StateUpdate"):::raw
    RAW -- "materialize_raw_update()" --> SU2
    SU2 -- "coerce_local_update()" --> AU2("apply_update()"):::fn
    AU2 --> VM2("VM State"):::state

    classDef fn fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121
    classDef raw fill:#fce4ec,stroke:#c62828,stroke-width:1px,color:#5c0a0a
    classDef tv fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a
    classDef state fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef local fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#1a3a1a
    classDef llm fill:#fff3e0,stroke:#e8a735,stroke-width:1px,color:#5c3a0a
```

The local path — the common case — is a direct pipeline with no serialization overhead. The LLM path gets its own materialization function that handles the JSON-to-TypedValue conversion before merging into the same `StateUpdate`.

The migration touched every handler in the executor — about 15 handler groups — done one at a time in dependency order: simple value handlers first (`_handle_const`, `_handle_store_var`), then loads (`_handle_load_var`, `_handle_load_field`), then objects, then operators, then the call chain. Each group was a separate commit.

The spec documented six different serialization patterns across handlers (raw value, `_serialize_value(val)`, `sym.to_dict()`, `SymbolicValue` object directly, `Pointer` object directly, heap address string). Each had its own migration path.

After this phase: ~11,449 tests passing.

---

## Phase 3: Return Values

`_handle_return` was still serializing return values via `_serialize_value(val)`, and `_handle_return_flow` was deserializing them back. This was the same roundtrip as Phase 2, just for return values.

The migration exposed a conflation that had been hiding in the return value semantics.

### The Constructor Bug

This was the most frustrating part of the migration, and the brainstorming conversation around it was contentious.

Before TypedValue, `return_value = None` meant two different things: "this instruction doesn't have a return value" and "the function returned None/null." These were indistinguishable. For most code this didn't matter. For constructors, it did.

Constructors in RedDragon work by allocating a heap object, running the constructor body (which stores fields via STORE_FIELD), and returning `self`. The return mechanism had a guard: `if return_value is not None`. This guard prevented constructors from accidentally clobbering their result register with `None` — constructors return the `this` pointer via a different mechanism (STORE_VAR into the caller's local), not via `return_value`.

When `return_value` became a `TypedValue`, the guard broke. `TypedValue(None, Void)` is not `None` — the isinstance check passes, and the constructor's result register gets overwritten with a void value.

The brainstorming for the fix started with me pushing on the void/None distinction. The AI's initial proposal had a `value is not None` guard on the return path — essentially preserving the old ambiguity under a new name. I pushed back: *"Why is there a None check though?"* This forced an explicit discussion of void vs null semantics. Then: *"I'm not comfortable with passing a naked None back."* And finally: *"I want the 'no return value is possible because it is void' scenario to also be represented by a different TypedValue."* This led to adding `VOID` to the type system.

But the real frustration came after implementation. The implementer agent had used `value is not None` as a guard anyway — silently discarding both Void and None TypedValues. I caught this in review: *"So you completely discarded creating Void and None TypedValues?"* followed by *"Produce a plan which accommodates the proper behaviour of using TypedValue, and not your coding convenience."* The fix was redesigned from scratch.

The eventual fix came in two commits:
1. Constructor detection via scope chain inspection — if the current frame is a constructor, skip return value writes to the result register.
2. Replace the `result_reg=None` hack (constructors had been setting their result register to `None` to prevent writes) with a clean `is_ctor` flag on `StackFrame`.

The `is_ctor` idea came from me during the brainstorming: *"Tentative idea: `_try_class_constructor_call` pushes `is_ctor` onto the constructor frame... the `_handle_return_flow` guard only assigns the return value if it is not Void."* The AI refined this into the implementation pattern.

This was a classic refactoring discovery: the old code worked, but only because of an accidental coupling between "None means no value" and "constructors shouldn't write return values." TypedValue made the coupling visible by removing the ambiguity. But it was also a case where the brainstorming process — the back-and-forth about what void means, the insistence on not taking shortcuts — produced a cleaner design than either party would have reached alone.

The fix introduced a three-state return type:
- `typed(None, scalar("Void"))` — void return (no value to write)
- `typed(None, UNKNOWN)` — explicit `return None`
- `typed(42, scalar("Int"))` — concrete return value

```mermaid
flowchart TD
    RET("_handle_return()"):::fn

    RET --> CTOR{"🏗️ Constructor?<br/><i>frame.is_ctor</i>"}:::decide
    CTOR -- "yes" --> VOID("TypedValue(None, Void)"):::void
    CTOR -- "no, has operand" --> RESOLVE("_resolve_reg(operand)"):::fn
    CTOR -- "no, no operand" --> VOID

    RESOLVE --> TV("typed_from_runtime(val)"):::fn
    TV --> SU("StateUpdate<br/><i>return_value + call_pop</i>"):::state
    VOID --> SU

    SU --> POP("⬆️ Pop frame"):::fn
    POP --> FLOW{"_handle_return_flow()"}:::decide
    FLOW -- "Void" --> SKIP("🚫 Skip result_reg write"):::void
    FLOW -- "concrete" --> WRITE("✅ caller.registers[result_reg]<br/>= TypedValue"):::tv

    classDef fn fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121
    classDef decide fill:#fff3e0,stroke:#e8a735,stroke-width:2px,color:#5c3a0a
    classDef void fill:#fce4ec,stroke:#c62828,stroke-width:1px,color:#5c0a0a
    classDef state fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef tv fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a
```

After this phase: ~11,449 tests, plus the constructor fix.

---

## Phases 4–6: Heap and Closures

Three more storage locations to migrate: `HeapWrite.value` (Phase 4), `HeapObject.fields` (Phase 5), and `ClosureEnvironment.bindings` (Phase 6).

Phase 4 was modest: `HeapWrite.value` carries `TypedValue`, but `apply_update` unwraps it before storing in `HeapObject.fields`. The heap stays raw for now.

Phase 5 pushed `TypedValue` into the heap itself. `HeapObject.fields` stores `TypedValue` directly. This was the largest single phase because every read site — `_handle_load_field`, `_handle_load_index`, constructor field access, builtin method dispatch — had to stop re-wrapping values that were already wrapped.

Phase 6 was the simplest: three write sites and one read site for closure bindings.

### The Double-Wrapping Landmine

Phase 5 was where I spent the most time reading code and trying to understand the flow. The heap is read from many places — field access, index access, constructor field initialization, alias variable resolution — and each site had its own slightly different wrapping logic. I kept asking *"what does this value look like when it arrives here?"* and tracing through the call chain to find out. The codebase had grown large enough that I couldn't hold the full picture in my head, and the AI's summaries sometimes glossed over details that mattered.

The persistent source of bugs was that `typed_from_runtime()` is not idempotent. If you pass it a `TypedValue`, it wraps it inside another `TypedValue`:

```
typed_from_runtime(TypedValue(42, Int))
→ TypedValue(value=TypedValue(42, Int), type=UNKNOWN)
```

Every read site that previously called `typed_from_runtime(raw_value)` unconditionally had to get an isinstance guard to prevent double-wrapping. The plan warned about intermediate breakage: tasks 1–4 changed write sites to store `TypedValue`, but read sites still called `typed_from_runtime()` unconditionally until tasks 5–7. The test suite was broken between those groups. This was acceptable because both groups were committed atomically.

```mermaid
flowchart TD
    subgraph write ["Write Site (Phase 4–5)"]
        HANDLER("Handler"):::fn -- "typed_from_runtime(val)" --> TV1("TypedValue(42, Int)"):::tv
        TV1 -- "HeapWrite" --> HEAP("🗄️ heap.fields['x']<br/>= TypedValue(42, Int)"):::heap
    end

    HEAP --> SPLIT{"Read site calls<br/>typed_from_runtime()?"}:::decide

    subgraph bad ["❌ Before fix: double-wrapped"]
        SPLIT -- "yes" --> DOUBLE("TypedValue(<br/>  value = TypedValue(42, Int),<br/>  type = UNKNOWN<br/>)"):::danger
    end

    subgraph good ["✅ After fix: pass-through"]
        SPLIT -- "no" --> PASS("TypedValue(42, Int)<br/><i>intact</i>"):::tv
    end

    classDef fn fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121
    classDef tv fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a
    classDef heap fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef decide fill:#fff3e0,stroke:#e8a735,stroke-width:2px,color:#5c3a0a
    classDef danger fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#b71c1c
```

After Phase 6: ~11,481 tests passing.

---

## Phase 7: Cleaning Up After Ourselves

All storage locations now stored `TypedValue`. The isinstance guards added during the transition — `if isinstance(val, TypedValue)` — were dead code. Phase 7 removed them all and narrowed type annotations. This was a cleanup commit, not a behavioral change.

---

## Detour: Builtins

The TypedValue migration was "done" after Phase 7. But the work exposed two more problems in the builtin system.

### BuiltinResult: Side Effects Should Be Declarative

This detour started with me asking *"what else?"* after the main migration was done. The AI ran an audit and flagged the builtins. The fix went through another round of brainstorming — a shorter one this time, since the pattern was established. The brainstorming skill proposed three approaches: (A) builtins return `ExecutionResult` directly, (B) a lightweight `BuiltinResult` dataclass with value + side effects, or (C) split builtins into two tables (pure vs heap-mutating). I chose B.

But the interesting moment was a correction I made to the AI's initial proposal. It had suggested that only the heap-mutating builtins return `BuiltinResult`, while pure builtins continue returning raw values. I pushed back: *"Pure builtins should also return BuiltinResult."* The point was uniform interface — the caller shouldn't need isinstance branching to figure out what a builtin returned. This was the same principle that drove the "every value is TypedValue, even when the type is UNKNOWN" decision in Phase 1. A Beads task was filed (`red-dragon-vva`), and the plan was generated from the spec.

RedDragon has about 40 built-in functions (`len`, `range`, `print`, `slice`, plus 25 COBOL-specific byte manipulation builtins). Most are pure — they take arguments and return a value. Two are not: `_builtin_array_of` creates a heap object, and `_builtin_object_rest` copies fields from an existing heap object. These two wrote directly to `vm.heap` as a side effect, bypassing the `apply_update` pipeline.

This had always been the case, but it became conspicuous once every other state change flowed through `StateUpdate`. I hadn't been thinking about builtins at all during the TypedValue planning — they seemed orthogonal. They weren't. The fix introduced `BuiltinResult`:

```python
@dataclass(frozen=True)
class BuiltinResult:
    value: TypedValue
    new_objects: list[NewObject] = field(default_factory=list)
    heap_writes: list[HeapWrite] = field(default_factory=list)
```

All builtins return `BuiltinResult`. The executor unpacks it into the `StateUpdate`. No builtin directly mutates `vm.heap`.

**Before:** builtins bypass StateUpdate

```mermaid
flowchart LR
    B1("_builtin_array_of()"):::fn -- "⚡ direct write" --> HEAP1("vm.heap"):::danger
    B1 -- "raw value" --> H1("Handler"):::fn
    H1 -- "StateUpdate<br/><i>value only</i>" --> AU1("apply_update()"):::fn
    AU1 --> VM1("VM State"):::state

    classDef fn fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121
    classDef danger fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#b71c1c
    classDef state fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
```

**After:** all effects are declarative

```mermaid
flowchart LR
    B2("_builtin_array_of()"):::fn --> BR("📋 BuiltinResult<br/><i>value + new_objects<br/>+ heap_writes</i>"):::builtin
    BR --> H2("Handler unpacks"):::fn
    H2 --> AU2("apply_update()"):::fn
    AU2 --> VM2("VM State<br/><i>atomic update</i>"):::state

    classDef fn fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121
    classDef state fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef builtin fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a
```

The migration was done in eight commits, with an isinstance bridge during the transition so that old-style builtins (returning raw values) and new-style builtins (returning `BuiltinResult`) could coexist. The bridge was removed in the last commit.

### Builtin Args: The Atomic Commit Problem

After builtins returned `TypedValue` via `BuiltinResult`, they still *received* raw Python primitives. `_resolve_reg` stripped the `TypedValue` wrapper before passing arguments. This was a hole: type information was available at the call site but discarded before the builtin could see it.

The fix was conceptually simple: change all builtins from `list[Any]` to `list[TypedValue]` arguments. The implementation was not, because **every builtin and every call site had to change simultaneously**. Changing the executor to pass `TypedValue` args without changing the builtins (which do things like `args[0] + args[1]`) would break all 11,530+ tests.

The plan explicitly mandated: no intermediate commits. All production code changes happen without committing; the single commit occurs after the full test suite passes. This was the only phase where the atomic commit constraint was hard — in every other phase, at least some intermediate states were testable.

This phase touched all 40+ builtins, all method builtins, all 25 COBOL builtins, and the parameter binding code in user function and constructor calls.

After builtins: ~11,530 tests passing.

---

## More Detours

The main migration was done. But work kept surfacing.

### BinopCoercionStrategy Return Type

This one came from me staring at the code and asking: *"Why does `BinopCoercionStrategy.coerce()` return `tuple[Any, Any]`?"*

We'd just spent two days making every value in the system a `TypedValue`. The protocol that coerces binary operands — the thing that started this entire migration — was still stripping types at its return boundary. The AI had implemented it that way in Phase 1, before the rest of the migration made `TypedValue` universal. Nobody caught it because the callers immediately re-wrapped the values. But it meant type information was being discarded and re-inferred at the coercion boundary, which defeated the purpose.

It was a one-commit fix, but I insisted it be filed as its own Beads task and done separately — changing the protocol return type is a distinct concern from the builtin migration, and mixing them would have muddied the commit history. Beads made this easy: file a task, give it a dependency on the current work, and it shows up in `bd ready` at the right time.

### UnopCoercionStrategy

Phase 1 introduced `BinopCoercionStrategy` for binary operators. Unary operators (`-x`, `not x`, `~x`, `#x`) had no equivalent — `_handle_unop` still used `_resolve_reg` (raw values) and `typed_from_runtime` (runtime type inference). The fix followed the same pattern: a `UnopCoercionStrategy` protocol with `coerce()` and `result_type()`, a `DefaultUnopCoercion` implementation, and threading through the executor pipeline via kwargs.

### Demo Scripts and LLM Path Leaks

After the main migration, the AI reported "all tests pass" and I asked: *"Do any of the script files need to be updated?"* They did. Five demo scripts in the `scripts/` directory used `_format_val` to display VM state. None of them handled `TypedValue`. After the heap fields migration, the scripts showed output like `fields={'x': TypedValue(value=42, type=ScalarType(name='Int'))}` instead of `fields={'x': 42}`.

The AI's first fix added an `_unwrap()` helper with isinstance guards. I pushed back: *"Why can't `_unwrap()` unconditionally take a TypedValue and unwrap it instead of all the isinstance nonsense?"* The local variables always store `TypedValue` now — that was the whole point. The guard was leftover thinking from the transition period. The fix was to just use `.value` directly.

Then I asked the AI to *actually run* each of the scripts. Not just run tests — the scripts aren't tests, they're demos that call live LLM backends. The AI had been saying "all tests pass" as if that covered everything. It didn't. Running the scripts exposed formatting bugs that the test suite couldn't catch.

This prompted creating an external test infrastructure: a pytest marker `@pytest.mark.external` and a configuration that excludes external tests by default in both local runs and CI. The scripts now have real tests, they just don't run in the normal suite.

A separate leak was found in `LLMPlausibleResolver._parse_llm_response`, which parsed LLM JSON responses into `StateUpdate` values. The parser produced bare values — not `TypedValue` — which entered the now-TypedValue-only pipeline. I'd asked *"Are there still any bare values passed around anywhere else in the system?"* and the audit found three sites in the LLM resolver. The main migration had been so focused on the local execution path that the LLM fallback path was missed.

### The Question That Came After

After the migration was done, the documentation updated, and the detours resolved, I looked at the codebase and asked: *"We're now storing types alongside every value in TypedValue. Are the separate `register_types` and `var_types` dictionaries in `TypeEnvironment` still required?"*

The answer turned out to be yes — but for a narrower reason than before. `TypedValue.type` carries the *runtime-inferred* type (what the computation produced). `TypeEnvironment` carries the *declared* type (what the source code said). These can differ: Python's `4 / 2` produces `2.0` (a float), but if the variable was declared `int d`, the declared type is `Int`. Write-time coercion exists to reconcile the two.

But the overlap is real. If the pre-operation coercion strategies were made aware of declared types, they could produce the correct type directly, and write-time coercion would become a no-op for locally-executed instructions. It would only remain necessary for LLM-produced updates. I filed this as a future investigation task. The two-layer architecture works, but it's worth asking whether both layers are still earning their keep now that TypedValue has changed the landscape.

This is the kind of question that only becomes askable after a migration is done. Before TypedValue, the question was meaningless — types and values lived in different structures by necessity. After TypedValue, the redundancy is visible.

---

## The Shape of the Work

Here's what the migration looked like as a dependency graph:

```mermaid
flowchart TD
    P1("Phase 1<br/>TypedValue + BinopCoercion"):::main
    P2("Phase 2<br/>Handler migration"):::main
    P3("Phase 3<br/>Return values"):::main
    P4("Phase 4<br/>HeapWrite.value"):::main
    P5("Phase 5<br/>HeapObject.fields"):::main
    P6("Phase 6<br/>Closure bindings"):::main
    P7("Phase 7<br/>Guard cleanup"):::main

    B1("Detour<br/>BuiltinResult"):::detour
    B2("Detour<br/>Builtin TypedValue args"):::detour
    D1("Detour<br/>BinopCoercion return type"):::detour
    D2("Detour<br/>UnopCoercionStrategy"):::detour
    D3("Detour<br/>Demo scripts"):::detour
    D4("Detour<br/>LLM path leak"):::detour
    D5("Detour<br/>External tests"):::detour

    DOC("📄 Doc<br/>Type system update"):::doc

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7
    P5 --> B1 --> B2
    P1 --> D1 --> D2
    P5 --> D3 --> D5
    B2 --> D4
    P7 --> DOC
    D2 --> DOC

    classDef main fill:#e8f4fd,stroke:#4a90d9,stroke-width:2px,color:#1a3a5c
    classDef detour fill:#fff3e0,stroke:#e8a735,stroke-width:2px,color:#5c3a0a
    classDef doc fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a3a1a

    linkStyle 0,1,2,3,4,5 stroke:#4a90d9,stroke-width:3px
    linkStyle 6,7,8,9,10,11 stroke:#e8a735,stroke-width:2px,stroke-dasharray:5
```

The main sequence (Phases 1–7) was planned. The detours were not. Most of them started with me asking *"what next?"* or *"what else?"* after a phase completed — essentially asking the AI to audit the codebase for things I hadn't thought of. This is a pattern I use a lot: finish a unit of work, commit, then ask the AI to look for fallout. It's more effective than trying to anticipate everything upfront.

Beads kept this manageable. Each node in the graph above was a Beads task — `red-dragon-gsl` for the initial TypedValue design, `red-dragon-132` for handler migration, `red-dragon-vva` for BuiltinResult, `red-dragon-x9r` for builtin args, `red-dragon-d5c` for the BinopCoercion return type fix, and so on. When a detour surfaced — the constructor bug, the builtin side-effect problem, the PHP enum — it got filed immediately as a new task with its dependencies. Running `bd ready` after closing a task showed me the next unblocked item, which might be the next planned phase or a detour that had just become unblocked.

There was a concrete example of this working well. Midway through brainstorming the heap fields migration (`red-dragon-f6i`), I realized handlers needed to be migrated first. I interrupted the brainstorming, ran `bd update red-dragon-f6i --status deferred`, created the handler migration task, and pivoted. When the handler migration was done and closed, `bd ready` surfaced the deferred heap fields task automatically. Without the tracker, that kind of mid-session pivot would have meant losing track of the deferred work.

Each major phase also went through the brainstorming skill before implementation began. The early phases (1–3) had longer brainstorming cycles because the patterns weren't established yet. By the later phases, the brainstorming rounds were shorter — the skill would propose an approach, I'd confirm it matched the established pattern, and we'd move to planning. The skill's insistence on proposing alternatives before committing to a direction caught at least two cases where the obvious approach wasn't the best one (the serialize/deserialize split in Phase 2, and the `BuiltinResult` design over direct `StateUpdate` returns).

Each detour emerged from one of three causes:

1. **The migration exposed a pre-existing problem** (constructor bug, builtins bypassing `apply_update`, `BinopCoercionStrategy` return type).
2. **The migration broke something downstream** (demo scripts, LLM path leak).
3. **The migration created an obvious gap** (`UnopCoercionStrategy` — if binops have injectable coercion, why don't unops?).

This is, I think, the normal shape of a refactoring. The plan covers the main sequence. The detours are where the actual learning happens.

Some numbers:

| | |
|---|---|
| Duration | ~2 days |
| Sessions | ~12 |
| Phases | 9 major + 5 detours |
| Commits | ~60 |
| Test count (start) | ~11,274 |
| Test count (end) | ~11,545 |
| Files touched | ~40 |

---

## Takeaways

**A refactoring is a sequence of discoveries, not a sequence of steps.** The plan covered the main migration. The constructor bug, the builtin side-effect problem, the double-wrapping landmine, the LLM path leak — none of these were in the original plan. They emerged because changing one thing made something else visible.

**"Every value is TypedValue, even when the type is UNKNOWN" was the single most important decision.** It eliminated all branching on "is this typed or raw?" and made the migration monotonic — each phase pushed `TypedValue` one layer deeper without introducing conditional paths. Every phase that added isinstance guards for transition purposes removed them later. The guards were temporary scaffolding, not permanent complexity.

**Splitting the local and LLM execution paths was worth the upfront cost.** The serialize/deserialize roundtrip existed because one code path served both local execution and LLM fallback. Once we split them, local execution became a direct path (handler produces TypedValue, `apply_update` stores it) and the LLM path got its own `materialize_raw_update` function. This separated two concerns that had been coupled since the beginning.

**Side effects should be declarative.** The BuiltinResult migration wasn't in the original plan. It became obvious once every other state change flowed through `StateUpdate` — two builtins were writing directly to `vm.heap`, and that was suddenly conspicuous. Making heap mutations declarative via `BuiltinResult(new_objects=..., heap_writes=...)` was the natural conclusion. The refactoring didn't create this problem; it made it visible.

**Non-idempotent wrapping functions are dangerous.** `typed_from_runtime(TypedValue(...))` produces a double-wrapped value. This was the single most error-prone aspect of the migration. In hindsight, making `typed_from_runtime` idempotent (detecting and passing through already-wrapped values) would have prevented an entire class of bugs.

**Atomic commits are sometimes unavoidable.** Most phases allowed incremental commits. The builtin args migration did not — changing the interface without changing all callers simultaneously would break 11,530+ tests. The plan explicitly mandated no intermediate commits. This is a real constraint when doing interface-level changes in a system with high test coverage.

**The brainstorm → spec → plan → implement pipeline prevented false starts.** Every phase that went through the brainstorming skill produced a spec before any code was written. The spec forced me to articulate what was changing, what the boundary conditions were, and what the migration path looked like. Twice, the brainstorming skill proposed an approach I hadn't considered that turned out to be simpler (the serialize/deserialize split, the `BuiltinResult` design). The discipline of writing down the design before implementing it is not new — but having a tool that *enforces* the step, asks the right questions, and proposes alternatives makes it harder to skip when you're tempted to just start coding.

**A local issue tracker changes how you handle surprises.** Detours are the normal shape of a refactoring. The question is whether they derail you or get absorbed into the work. Beads made it trivial to file a new task the moment a detour surfaced, set its dependencies, and continue with the current work. When the current task was done, `bd ready` surfaced whatever was next — planned phase or newly-filed detour. The tracker turned "I should also fix this other thing I just noticed" from a context-switching hazard into a two-second operation. Over a dozen sessions and five detours, that added up.

**The AI didn't change the nature of the work. It changed the throughput.** The design decisions, the boundary table, the phase ordering, the detour triaging — all of that is the same work a human would do. The AI handled the mechanical parts: updating 40+ builtins to accept `TypedValue` args, threading kwargs through five layers of function signatures, updating test assertions across eight test files. The refactoring took two days. Without the AI, the same refactoring would have taken longer, but the intellectual structure would have been identical. The bottleneck was never typing speed. It was understanding what needed to change and in what order.

---

*The code is at [avishek-sen-gupta/red-dragon](https://github.com/avishek-sen-gupta/red-dragon). The type system documentation, updated after this migration, is at `docs/type-system.md`.*
