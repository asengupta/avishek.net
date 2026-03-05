---
title: "Designing RedDragon: A Multi-Language Symbolic Code Analysis Engine"
author: avishek
usemathjax: false
mermaid: true
tags: ["Software Engineering", "Compilers", "Program Analysis", "Symbolic Execution", "AI-Assisted Development"]
draft: false
---

*A universal IR, 15 deterministic frontends, a symbolic VM, and an iterative audit loop.*

![RedDragon TUI Demo](/assets/red-dragon-tui.gif)

---

## The Problem

I wanted to analyse source code across many languages (trace data flow, build control flow graphs, understand how variables depend on each other) without writing a separate analyser for each language. The conventional approach is to build language-specific tooling (Roslyn for C#, javac's AST for Java, etc.), but that means duplicating every downstream analysis pass for every language. I wanted one representation, one analyser, many languages.

The twist: I also wanted to handle *incomplete* programs gracefully. Real-world code depends on imports, frameworks, and external systems that aren't available during static analysis. Most tools crash or give up when they hit an unresolved reference. I wanted mine to keep going, creating symbolic placeholders for unknowns and tracing data flow through them.

[RedDragon](https://github.com/avishek-sen-gupta/red-dragon) is the result. It parses source in 15 languages, lowers it to a universal intermediate representation, builds control flow graphs, performs iterative dataflow analysis, and executes programs symbolically via a deterministic virtual machine. All with zero LLM calls for programs with concrete inputs.

RedDragon is part of a family of three tools: [Codescry](https://github.com/avishek-sen-gupta/codescry) (a repo surveying toolkit that detects integration points using regex, ML classifiers, code embeddings, and LLM classification) and [RedDragon-Codescry TUI](https://github.com/avishek-sen-gupta/reddragon-codescry-tui) (a terminal UI integrating the two). The TUI demo is shown above.

This post covers how the system was designed, how it evolved, and the engineering discipline that kept it coherent across 66 architectural decisions and 400+ conversation sessions with Claude Code.

### Core Theses

RedDragon explores three ideas about analysing frequently-incomplete code, the kind found in legacy migrations, decompiled binaries, partial extracts, and codebases with missing dependencies:

1. **Deterministic language frontends with LLM-assisted repair.** Tree-sitter frontends (15 languages) and a ProLeap bridge (COBOL) handle well-formed source deterministically. When tree-sitter hits malformed syntax, an optional LLM repair loop fixes only the broken fragments and re-parses, maximising deterministic coverage for real-world incomplete code. All paths produce the same universal IR.

2. **Full LLM frontends for unsupported languages.** For languages without a tree-sitter frontend, an LLM lowers source to IR entirely, supporting any language without new parser code. A chunked variant splits large files into per-function chunks via tree-sitter, lowering each independently. The LLM acts as a *compiler frontend*, constrained by a formal IR schema with concrete patterns. It's translating syntax, not reasoning about semantics.

3. **A VM that integrates LLMs only at the boundaries where information is genuinely missing.** When execution hits missing dependencies, unresolved imports, or unknown externals, a configurable resolver can invoke an LLM to produce plausible state changes, keeping execution moving through incomplete programs instead of halting at the first unknown. When source is complete and all dependencies are present, the entire pipeline (parse → lower → execute) is deterministic with zero LLM calls.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [A Worked Example: Source to Execution](#a-worked-example-source-to-execution)
3. [The IR: 27 Opcodes to Rule Them All](#the-ir-27-opcodes-to-rule-them-all)
4. [Frontends: Four Strategies, One Output](#frontends-four-strategies-one-output)
5. [The Dispatch Table Engine](#the-dispatch-table-engine)
6. [The Deterministic VM](#the-deterministic-vm)
7. [Dataflow Analysis](#dataflow-analysis)
8. [The Evolution: From Monolith to 7,268 Tests](#the-evolution-from-monolith-to-7268-tests)
9. [The Audit Loop: Systematic Completeness](#the-audit-loop-systematic-completeness)
10. [The Assertion Audit: Green Tests, False Confidence](#the-assertion-audit-green-tests-false-confidence)
11. [Cross-Language Verification via Exercism](#cross-language-verification-via-exercism)
12. [Guardrails: The CLAUDE.md as Architecture](#guardrails-the-claudemd-as-architecture)
13. [What I'd Do Differently](#what-id-do-differently)

---

## Architecture Overview

RedDragon follows a classic compiler pipeline, extended with symbolic execution:

```mermaid
%%{ init: { "flowchart": { "curve": "stepBefore" } } }%%
flowchart TD
src["Source Code (15 languages)"]-->frontend["Frontend<br/>(deterministic or LLM-based)"]
frontend-->|"list[IRInstruction]"|cfg["CFG Builder"]
cfg-->vm["VM (symex)"]
cfg-->dataflow["Dataflow Analysis"]
style src fill:#4a90d9,stroke:#000,stroke-width:2px,color:#fff
style frontend fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
style cfg fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
style vm fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
style dataflow fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
```

Every stage operates on the same flat IR. The VM and dataflow analysis are language-agnostic. They don't know whether the instructions came from Python, Rust, or COBOL.

---

## A Worked Example: Source to Execution

To make the pipeline concrete, here's a complete trace of a simple program through every stage. This is the same pipeline that runs for all 15 languages.

### Source (Python)

```python
def classify(x):
    if x > 0:
        label = "positive"
    else:
        label = "negative"
    return label

result = classify(5)
```

### Stage 1: Lowering to IR

The Python tree-sitter frontend parses this and emits flat three-address code. The function body is wrapped in a skip-over pattern (a `BRANCH` jumps past it so it's not executed at definition time):

```
branch end_classify_0                    # skip over body
func_classify_0:                         # entry point
  %0 = symbolic param:x                  # parameter binding
  store_var x %0
  %1 = load_var x                        # if x > 0
  %2 = const 0
  %3 = binop > %1 %2
  branch_if %3 if_true_0,if_false_0
if_true_0:
  %4 = const "positive"
  store_var label %4
  branch if_end_0
if_false_0:
  %5 = const "negative"
  store_var label %5
  branch if_end_0
if_end_0:
  %6 = load_var label
  return %6
end_classify_0:
  %7 = const <function:classify@func_classify_0>
  store_var classify %7
  %8 = const 5
  store_var x %8
  %9 = call_function classify %8         # classify(5)
  store_var result %9
```

Every instruction is a flat dataclass with an opcode, operands, a destination register, and a source location tracing it back to the original line and column. No nested expressions. `x > 0` decomposes into `LOAD_VAR`, `CONST`, `BINOP`.

### Stage 2: CFG Construction

The CFG builder splits the IR at every `LABEL` and after every `BRANCH`/`BRANCH_IF`/`RETURN`/`THROW`, then wires edges based on branch targets:

```mermaid
flowchart TD
    entry(["<b>entry</b><br>BRANCH end_classify_0"])
    func["<b>func_classify_0</b><br>SYMBOLIC param:x · STORE x<br>LOAD x · CONST 0 · BINOP ><br>BRANCH_IF"]
    if_true["<b>if_true_0</b><br>CONST &quot;positive&quot;<br>STORE label · BRANCH"]
    if_false["<b>if_false_0</b><br>CONST &quot;negative&quot;<br>STORE label · BRANCH"]
    if_end(["<b>if_end_0</b><br>LOAD label<br>RETURN"])
    end_classify["<b>end_classify_0</b><br>CONST function · STORE classify<br>CONST 5 · STORE x<br>CALL classify · STORE result"]

    entry --> end_classify
    entry -.->|"skip"| func
    func -- T --> if_true
    func -- F --> if_false
    if_true --> if_end
    if_false --> if_end
    end_classify -.->|"call"| func

    style entry fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
    style func fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
    style if_true fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
    style if_false fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
    style if_end fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
    style end_classify fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
```

### Stage 3: VM Execution (0 LLM calls)

The deterministic VM executes step by step. When it hits `CALL_FUNCTION classify`, it pushes a new stack frame, binds the parameter `x = 5`, and jumps to `func_classify_0`:

```
step  1: branch end_classify_0          → skip to end_classify_0
step  2: const <function:classify>       → %7 = <function:classify@func_classify_0>
step  3: store_var classify %7           → classify = <function>
step  4: const 5                         → %8 = 5
step  5: store_var x %8                  → x = 5
step  6: call_function classify %8       → push frame, jump to func_classify_0
step  7: symbolic param:x               → %0 = 5 (bound from caller)
step  8: store_var x %0                  → x = 5
step  9: load_var x                      → %1 = 5
step 10: const 0                         → %2 = 0
step 11: binop > %1 %2                   → %3 = True (5 > 0)
step 12: branch_if %3 if_true,if_false   → True, jump to if_true_0
step 13: const "positive"                → %4 = "positive"
step 14: store_var label %4              → label = "positive"
step 15: branch if_end_0                 → jump to if_end_0
step 16: load_var label                  → %6 = "positive"
step 17: return %6                       → pop frame, return "positive"
step 18: store_var result %9             → result = "positive"

Final state: result = "positive"  (18 steps, 0 LLM calls)
```

### Stage 4: Dataflow Analysis

The reaching definitions analysis traces through the register chain. The raw def-use chain says "`result` depends on `%9`". But tracing through: `%9` comes from `CALL_FUNCTION` on `classify` with argument `%8`; inside the call, `label` is set to `"positive"` (the branch taken); `label` is loaded into `%6` and returned. The dependency graph says: `result` depends on `classify` and `x`.

### What Changes With Incomplete Code

Now consider what happens when the source has a missing dependency:

```python
import math
result = math.sqrt(16) + 1
```

The frontend doesn't know what `math.sqrt` returns. Instead of crashing, the VM creates a symbolic value:

```
step 1: call_function math.sqrt 16    → sym_0 (hint: "math.sqrt(16)")
step 2: const 1                       → %1 = 1
step 3: binop + sym_0 %1              → sym_1 (constraint: "sym_0 + 1")
step 4: store_var result sym_1        → result = sym_1

Final state: result = sym_1 [sym_0 + 1, where sym_0 = math.sqrt(16)]
```

The dataflow analysis still works: `result` depends on `math.sqrt` and the constant `1`. The symbolic value propagates deterministically. If you opt into the LLM resolver (`UnresolvedCallStrategy.LLM`), the VM would instead resolve `math.sqrt(16)` to `4.0`, and the final result would be `5.0`.

This is the core idea: deterministic by default, LLM-assisted only at the boundaries where information is genuinely missing.

---

## The IR: 27 Opcodes to Rule Them All

The intermediate representation is a flattened three-address code with 27 opcodes, grouped by role:

```
Value producers:   CONST, LOAD_VAR, LOAD_FIELD, LOAD_INDEX,
                   NEW_OBJECT, NEW_ARRAY, BINOP, UNOP,
                   CALL_FUNCTION, CALL_METHOD, CALL_UNKNOWN

Value consumers:   STORE_VAR, STORE_FIELD, STORE_INDEX

Control flow:      BRANCH, BRANCH_IF, LABEL, RETURN, THROW,
                   TRY_PUSH, TRY_POP

Regions:           ALLOC_REGION, WRITE_REGION, LOAD_REGION

Continuations:     SET_CONTINUATION, RESUME_CONTINUATION

Escape hatch:      SYMBOLIC
```

The first 19 opcodes handle all general-purpose lowering across 15 languages. `TRY_PUSH` and `TRY_POP` model structured exception handling (pushing/popping handler labels onto the VM's exception stack). The three region opcodes (`ALLOC_REGION`, `WRITE_REGION`, `LOAD_REGION`) provide byte-addressed memory for COBOL-style overlays, REDEFINES, and packed data layouts. The two continuation opcodes (`SET_CONTINUATION`, `RESUME_CONTINUATION`) model COBOL's PERFORM return semantics, where control transfers to a named paragraph and returns to the caller on completion. All eight extended opcodes are language-agnostic in the IR and VM; they happen to be emitted by the COBOL frontend but could serve C struct layouts or binary protocol parsing.

Every instruction is a flat dataclass: an opcode, a list of operands, a destination register, and a source location tracing it back to the original code. No nested expressions. `a + b * c` decomposes into:

```
%0 = const b
%1 = const c
%2 = binop * %0 %1
%3 = const a
%4 = binop + %3 %2
```

This verbosity is the trade-off for universality. CFG construction, dataflow analysis, and VM execution all operate on the same flat list. Adding a new language means emitting these opcodes; everything downstream works automatically.

### Source Location Traceability

Every instruction carries a `SourceLocation` with start/end line and column, captured from the tree-sitter AST node that generated it. The IR's string representation appends this:

```
%0 = const 10  # 1:4-1:6
```

This means any IR instruction, any VM execution step, any dataflow dependency can be traced back to the exact span of source code that produced it. When a symbolic value appears in the output, its provenance chain leads back to specific source lines.

### Control Flow in the IR

All control flow is explicit. There are no structured `if`/`while`/`for` constructs in the IR. A simple `if/else` lowers to labels, conditional branches, and unconditional jumps:

```
%0 = binop > x 5
branch_if %0 if_true_0,if_false_0
if_true_0:
  %1 = const 1
  store_var y %1
  branch if_end_0
if_false_0:
  %2 = const 0
  store_var y %2
  branch if_end_0
if_end_0:
  ...
```

`BRANCH_IF` encodes both targets in its label field (comma-separated). The CFG builder splits the IR into basic blocks at every `LABEL` and after every `BRANCH`/`BRANCH_IF`/`RETURN`/`THROW`, then wires edges based on the branch targets. Loops become back-edges: a `while` loop's `BRANCH` at the end of the body points back to the condition's label.

### Functions as IR Patterns

Function definitions are lowered as *skip-over* patterns. The body is emitted inline in the IR, bracketed by a `BRANCH` that jumps past it (so the body isn't executed at definition time) and a `LABEL` marking the entry point:

```
branch end_add_0              # skip over body
func_add_0:                   # entry point
  %0 = symbolic param:a       # parameter binding
  store_var a %0
  %1 = symbolic param:b
  store_var b %1
  %2 = load_var a
  %3 = load_var b
  %4 = binop + %2 %3
  return %4
end_add_0:
  %5 = const <function:add@func_add_0>
  store_var add %5
```

Parameters are emitted as `SYMBOLIC` instructions with a `param:` prefix. A `FunctionRegistry` scans the IR to extract parameter names from these markers and maps class names to method labels. This metadata drives call resolution at execution time.

### Three Call Variants

The IR distinguishes three kinds of calls by their operand layout:

- **`CALL_FUNCTION`**: static calls where the target is a known name. Operands: `[func_name, arg0, arg1, ...]`
- **`CALL_METHOD`**: method calls on objects. Operands: `[obj_reg, method_name, arg0, arg1, ...]`
- **`CALL_UNKNOWN`**: dynamic calls where the target is a computed expression (a variable holding a function reference, or a closure). Operands: `[target_reg, arg0, arg1, ...]`

The frontend decides which to emit based on the AST: `foo(x)` emits `CALL_FUNCTION`, `obj.foo(x)` emits `CALL_METHOD`, and `some_var(x)` where `some_var` isn't a known function emits `CALL_UNKNOWN`.

### Object and Array Construction

Objects and arrays are created via `NEW_OBJECT`/`NEW_ARRAY` followed by `STORE_FIELD`/`STORE_INDEX` for each member. An array literal `[1, 2, 3]` lowers to:

```
%0 = const 3
%1 = new_array list %0
%2 = const 0
%3 = const 1
store_index %1 %2 %3         # array[0] = 1
%4 = const 1
%5 = const 2
store_index %1 %4 %5         # array[1] = 2
%6 = const 2
%7 = const 3
store_index %1 %6 %7         # array[2] = 3
```

This verbose expansion means the VM and dataflow analysis see every individual element assignment, which matters for tracking which values flow into which positions.

### The SYMBOLIC Escape Hatch

`SYMBOLIC` is the escape hatch. When a frontend encounters a construct it doesn't handle, it emits `SYMBOLIC "unsupported:list_comprehension"` instead of crashing. The VM treats it as a symbolic value that propagates through execution. Parameters use it too (`SYMBOLIC "param:x"`), as do caught exceptions (`SYMBOLIC "caught_exception:ValueError"`).

Over time, `unsupported:` emissions get replaced with real IR as frontends gain coverage. The project's history is essentially the story of systematically eliminating every last `SYMBOLIC`.

---

## Frontends: Four Strategies, One Output

All four frontend strategies produce the same `list[IRInstruction]`. They differ in speed, coverage, and determinism:

**1. Deterministic frontends (15 languages):** Python, JavaScript, TypeScript, Java, Ruby, Go, PHP, C#, C, C++, Rust, Kotlin, Scala, Lua, Pascal. These use tree-sitter for parsing and a dispatch-table-based recursive descent for lowering. Sub-millisecond. Zero LLM calls. Fully testable. Each frontend is modularised into separate files for expressions, control flow, and declarations, inheriting from a shared `BaseFrontend`. An optional **AST repair decorator** wraps any deterministic frontend with an LLM-assisted error-fixing loop: when tree-sitter produces ERROR or MISSING nodes, the repair loop extracts the error spans, asks an LLM to fix only the broken fragments, patches the source, and re-parses. This maximises deterministic coverage for real-world malformed code with zero overhead when the source is clean.

**2. COBOL frontend (ProLeap bridge):** COBOL source is parsed by the ProLeap COBOL parser (a Java-based parser producing an Abstract Syntax Graph), bridged to Python via a shaded JAR that emits JSON ASGs. The frontend includes a complete type system: PIC clause parsing (zoned decimal, COMP/COMP-1/COMP-2, packed decimal, alphanumeric, EBCDIC), REDEFINES overlays with byte-addressed memory regions, OCCURS arrays with subscript resolution, level-88 condition names with value ranges, and paragraph-based control flow via named continuations. COBOL-specific IR is emitted using the region and continuation opcodes.

**3. LLM frontend:** For languages without a deterministic frontend. The source is sent to an LLM constrained by a formal schema: all 27 opcode specs, concrete patterns, and worked examples. The LLM acts as a mechanical compiler frontend, not a reasoning engine. This distinction matters: the prompt doesn't ask *"what does this code do?"* It asks *"translate this into these specific opcodes."*

**4. Chunked LLM frontend:** For large files that overflow context windows. Tree-sitter decomposes the file into per-function chunks, each is LLM-lowered independently, registers and labels are renumbered to avoid collisions, and the chunks are reassembled into a single IR.

The key architectural decision was making the LLM path a *compiler frontend*, not a *reasoning engine*. When you constrain the LLM to pattern-matching against a formal schema, output quality improves. It's translating syntax, not reasoning about semantics.

---

## The Dispatch Table Engine

The heart of the deterministic frontends is a `BaseFrontend` class (~950 lines) that all 15 languages inherit from. It uses two dispatch tables (one for statements, one for expressions) mapping tree-sitter AST node types to handler methods.

The lowering dispatch chain:

```mermaid
%%{ init: { "flowchart": { "curve": "stepBefore" } } }%%
flowchart TD
lower["lower(root)"]-->block["_lower_block(root)<br/><i>iterate named children</i>"]
block-->stmt["_lower_stmt(child)<br/><i>skip noise/comments; try STMT_DISPATCH</i>"]
stmt-->expr["_lower_expr(child)<br/><i>fallback: try EXPR_DISPATCH</i>"]
expr-->sym["SYMBOLIC('unsupported:X')<br/><i>final fallback</i>"]
style lower fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
style block fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
style stmt fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
style expr fill:#2c3e50,stroke:#000,stroke-width:2px,color:#fff
style sym fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
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

One decision shaped the rest of the project more than any other: making the VM fully deterministic.

The original design had the LLM deciding state changes at each execution step. When the VM encountered an unknown value, it asked the LLM what to do. This was slow, non-deterministic, untestable, and fragile.

The key insight came from a simple question: *"Given that the IR is always bounded, shouldn't execution be deterministic?"* Yes. If the IR has no unbounded loops (or loops are bounded by concrete values), execution is a mechanical process. Unknown values don't need to be *resolved*. They can be *created* as symbolic placeholders that propagate through computation.

So we ripped out all LLM calls from the VM. The entire execution engine became reproducible across runs.

### VM State: Frames, Heap, and Closures

The VM's state is held in a single `VMState` dataclass:

```python
@dataclass
class VMState:
    heap: dict[str, HeapObject]          # flat object store
    call_stack: list[StackFrame]         # LIFO execution frames
    path_conditions: list[str]           # branch assumptions
    symbolic_counter: int = 0            # fresh-name generator
    closures: dict[str, ClosureEnvironment]  # shared mutable cells
```

Each `StackFrame` has a **two-level namespace**: `registers` for IR temporaries (`%0`, `%1`, ...) and `local_vars` for source-level named variables. This separation keeps the three-address code machinery invisible to the analysis layer, which only cares about named variables.

The heap is a flat dictionary mapping addresses (`"obj_0"`, `"arr_1"`) to `HeapObject` instances. Each `HeapObject` stores a `type_hint` and a `fields` dictionary. Arrays use stringified indices as field keys. This uniform representation means the VM doesn't distinguish between object field access and array indexing at the storage level.

### Opcode Dispatch

The `LocalExecutor` maps each of the 27 `Opcode` enum values to a handler function via a static dispatch table:

```python
DISPATCH: dict[Opcode, Any] = {
    Opcode.CONST: _handle_const,
    Opcode.BINOP: _handle_binop,
    Opcode.CALL_FUNCTION: _handle_call_function,
    Opcode.LOAD_FIELD: _handle_load_field,
    # ... all 27 opcodes
}
```

Every handler receives the instruction, the VM state, the CFG, and a function registry, and returns an `ExecutionResult`. No handler mutates the VM directly. Instead, each constructs a `StateUpdate` describing the desired mutations.

### StateUpdate: The Communication Contract

`StateUpdate` is the universal contract between handlers and the state engine. It's a pure data object listing all effects:

```python
class StateUpdate:
    register_writes: dict[str, Any]      # %0 = value
    var_writes: dict[str, Any]           # x = value
    heap_writes: list[HeapWrite]         # obj.field = value
    new_objects: list[NewObject]         # allocate on heap
    call_push: StackFramePush | None     # push new frame
    call_pop: bool                       # pop frame on return
    path_condition: str | None           # branch assumption
    next_label: str | None               # jump target
```

This separation of *computation* (handlers) from *mutation* (`apply_update`) is a deliberate functional core / imperative shell split. The handlers are pure functions that return data. The mutation is centralised in one place.

### `apply_update()`: The Single Mutator

All state changes flow through a single function, `apply_update()`, which applies a `StateUpdate` to the VM in a strict order:

1. **New objects**: allocate heap entries
2. **Register writes**: write to the current frame's registers
3. **Heap writes**: update object fields (materialising synthetic entries if needed)
4. **Path condition**: record branch assumptions
5. **Call push**: push a new `StackFrame` onto the call stack
6. **Variable writes**: write to the *current* frame's `local_vars` (which is the new frame if step 5 fired)
7. **Call pop**: pop the call stack on return

The ordering of steps 5 and 6 is the subtle part. When calling a function, parameters need to land in the *new* frame, not the caller's. By pushing the frame first (step 5) and writing variables second (step 6), parameter bindings automatically go to the right place without any special-casing.

Step 6 also handles closure synchronisation: if a written variable is in the frame's `captured_var_names`, the write is mirrored to the shared `ClosureEnvironment`, ensuring that mutations inside closures are visible to other closures sharing the same environment.

### Symbolic Value Propagation

When execution hits an unresolved import or function, the VM creates a `SymbolicValue` with a descriptive hint:

```
sym_0 (hint: "math.sqrt(16)")
```

This symbolic value propagates through computation deterministically. Each handler checks whether its operands are symbolic. If either operand of a `BINOP` is symbolic, the result is a fresh symbolic with a constraint recording the expression:

```
sym_0 + 1  →  sym_1 (constraint: "sym_0 + 1")
```

Field access on a symbolic object creates a symbolic field with lazy heap materialisation: the first access to `sym_0.x` allocates a synthetic heap entry and caches a symbolic value for `x`, so subsequent accesses to the same field return the same symbolic. This deduplication is important for dataflow analysis, where repeated reads of the same field should trace back to the same definition.

Concrete operations that fail (division by zero, unsupported operator) produce an `UNCOMPUTABLE` sentinel, which triggers symbolic fallback rather than crashing.

The trade-off is that symbolic branches always take the true path (a simplification), and symbolic values can't be resolved to concrete results without help.

### The UnresolvedCallResolver

For the latter trade-off, a configurable `UnresolvedCallResolver` uses the Strategy pattern:

**SymbolicResolver** (default): creates a fresh symbolic for any unknown call. Zero LLM calls, fully deterministic.

**LLMPlausibleResolver** (opt-in): sends a structured prompt to an LLM with the function name, arguments, and current VM state, then parses the response into a `StateUpdate`. Falls back to `SymbolicResolver` on failure.

This separation keeps the default execution path fast and reproducible while allowing users to opt into richer (but slower, non-deterministic) analysis when needed.

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

With snapshot capture, `inc()` always reads `count = 0`. The fix was shared `ClosureEnvironment` cells: all closures from the same scope share a mutable environment, matching Python/JavaScript semantics. When a nested function is created, the enclosing frame's variables are copied into a `ClosureEnvironment`. On each call, captured variables are injected into the new frame, and `apply_update()` mirrors writes back to the shared environment. This is the kind of deep correctness issue that only surfaces through specific test cases. It's documented as ADR-019 in the project's decision records.

### Built-in Functions

The VM includes a small table of built-in functions (`len`, `range`, `print`, `int`, `float`, `str`, `bool`, `abs`, `max`, `min`, plus array constructors) that are resolved before falling through to the `UnresolvedCallResolver`. Each built-in handles symbolic arguments gracefully: `len` of a symbolic list returns a symbolic, `range` with symbolic bounds returns `UNCOMPUTABLE`, and so on. This keeps common operations concrete without requiring language-specific runtime support.

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

The dataflow module has no dependencies on the VM, frontends, or backends. It's a pure analysis pass over the CFG, decoupled from the imperative shell (parsing, I/O, LLM calls).

---

## The Evolution: From Monolith to 7,268 Tests

RedDragon's evolution followed a clear pattern of phases, each triggered by testing the previous one on real code:

**Phase 1: The monolith (Hour 0 to 2).** A single `interpreter.py` with an LLM-based lowering and execution engine. ~1,200 lines. It worked, barely.

**Phase 2: The determinism pivot (Hour 2 to 4).** The key insight: execution should be deterministic. Ripped out all LLM calls from the VM. Added symbolic value creation. Once the VM was deterministic, everything became testable.

**Phase 3: Multi-language frontends (Hour 4 to 8).** Asked: *"How hard is it to write deterministic logic to lower ASTs for 15 languages?"* The answer: not that hard, with tree-sitter and a dispatch table engine. 15 frontends generated in a single marathon session. 346 tests.

**Phase 4: Analysis and tooling (Hour 8 to 14).** Added iterative dataflow analysis, chunked LLM frontend, Mermaid CFG visualisation with subgraphs and call edges, source location traceability. Extracted CLI into composable API.

**Phase 5: Systematic hardening (Sessions 50 to 130).** This is where the test count exploded. The Rosetta cross-language test suite (14 algorithms x 15 languages) and then the Exercism integration suite drove the test count from ~700 to 7,268 across 80+ sessions. Each exercise exposed new frontend gaps, VM limitations, and edge cases, and each fix was immediately verified across all 15 languages.

The test count tells the story:

```
Phase 1–3:        346 tests
Phase 4:         ~700 tests
Rosetta:        ~1,200 tests
Exercism 1:     ~2,700 tests
Exercism 2:     ~4,200 tests
Exercism 3:     ~5,150 tests
Exercism 4:     ~7,076 tests
Exercism done:   7,268 tests
COBOL + audit:   8,569 tests
```

All tests run without LLM calls and are deterministic.

---

## The Audit Loop: Systematic Completeness

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
Audit → 34 gaps found → implement all 34 (57 new tests)
Re-audit → 19 gaps found → implement all 19 (28 new tests)
Re-audit → 12 gaps found → implement all 12 (18 new tests)
Re-audit → 0 gaps, 0 SYMBOLIC
```

This pattern (audit, batch-fix, re-audit) was more effective than trying to enumerate every missing feature upfront. The audit told me what was missing, I fixed what it found, and repeated until it found nothing.

---

## The Assertion Audit: Green Tests, False Confidence

The dispatch audit ensured that every language construct was *handled*. But it said nothing about whether the tests *verifying* that handling were any good. By March 2026, the test suite had grown to ~8,400 tests across ~130 files. All green. The question I'd been avoiding: **if every test passes, how do I know each test is actually checking what it says it's checking?**

Not "does the code work"; 8,400 green dots covered that. But: does `test_constructor_sets_fields` actually verify that field values are set? Does `test_switch_statement` actually verify that switch/case lowering works? Does `test_perform_until_loop_ordering` actually verify that instructions appear in the right order?

When an AI writes your tests, you get volume and coverage breadth for free. What you don't get is *assertion depth*. The AI produces a test for every function, every edge case, every language, but each individual assertion may be checking the easy thing (does it not crash?) rather than the hard thing (does it produce the right output?). The AI optimises for the test *passing*, not for the test *verifying*.

Over two days and 11 audit passes, I had Claude Code scan every test file, comparing each test's name against its actual assertions. The results were sobering.

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

**Phase 3: Priority-based audits.** A full scan of 122 files found ~82 violations. Introduced the priority classification: P0 (false confidence), P1 (missing key assertion), P2 (weak/generic), P3 (cosmetic). Re-scans after each fix batch drove the count down: 82 → 56 → 17.

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
  → AI: launches parallel agents, scans ~130 files
  → AI: produces prioritised violation list
  → Human: reviews, selects priority tier to fix
  → AI: reads each test, writes fix, runs test
    → Fix fails (wrong representation, wrong attribute, wrong type)
    → AI: investigates actual behaviour, corrects fix
    → Fix passes
  → AI: runs full test suite (8,457 tests)
  → AI: runs Black formatter
  → AI: commits and pushes
  → Human: "audit again"
  → [repeat]
```

The critical human interventions were:
1. **"Always strengthen assertions, do not rename"**: set the governing principle
2. **"You keep changing the list"**: identified the audit stability problem
3. **"Do a reconciliation pass"**: prescribed the fix for drift
4. **"Fix the frontend bug"**: escalated from test fix to code fix when an assertion exposed a real bug
5. **Selective priority focus**: always fixing P0 first, then P1, deferring P2

The AI's role was execution at scale (scanning 130 files, writing fixes, running 8,457 tests) while the human provided quality gates and strategic direction.

### Results

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

### Lessons

**Green tests are necessary but not sufficient.** Every test was green before the audit. The audit found 15 P0 violations where the test would pass even when the behaviour it named was completely broken. Two of these exposed genuine frontend bugs that had been silently lurking.

**OR-fallback assertions are the most dangerous pattern.** The pattern `assert A or B` appeared in every frontend category. In every case, one side of the OR was trivially satisfied by unrelated instructions, making the assertion vacuous.

**Audit stability requires anchoring.** Fresh scans produce inconsistent results because the agents don't share context. The reconciliation approach (starting from the previous audit's known list and verifying each item) is the only way to produce a stable, trustworthy list.

**Fixing assertions requires running the code under test.** Many fix attempts failed because the assertion assumed a representation that didn't match reality. The fix cycle (write assertion → run → discover actual representation → fix → re-run) was consistently necessary and never happened voluntarily.

**Parametrised tests need explicit skips, not silent returns.** Bare `return` in excluded branches produces green dots with zero assertions. `pytest.skip()` with a reason string makes exclusions visible.

The CLAUDE.md rules were updated to encode these lessons: never modify assertions to make tests pass without verifying the actual output, and never create special implementation behaviour just to satisfy tests.

---

## Cross-Language Verification via Exercism

The broadest verification effort was the Exercism integration test suite. The idea: take Exercism's canonical test cases (which define expected inputs and outputs for programming exercises), write equivalent solutions in all 15 languages, and verify that RedDragon's pipeline produces the correct answer for every case in every language.

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

The Exercism suite surfaced more bugs than any other test approach. Each exercise exposed new gaps: Ruby's `parenthesized_statements` vs Python's `parenthesized_expression`, Rust's expression-position loops, Pascal's single-quote string escaping, PHP's `.` concatenation operator. Every gap found was a bug fixed.

---

## Guardrails: The CLAUDE.md as Architecture

RedDragon was built almost entirely through conversations with Claude Code, across 400+ sessions. The file that had the most impact on consistency wasn't any Python module. It was `CLAUDE.md`, which encodes the development rules.

Some of the rules that shaped the codebase most:

**"STOP USING FOR LOOPS WITH MUTATIONS IN THEM. JUST STOP."** This rule forced a functional programming style across the codebase. List comprehensions, `map`, `filter`, `reduce` instead of mutable accumulators. The code is denser but more predictable.

**"Do not use `unittest.mock.patch`. Use proper dependency injection."** This forced every external dependency (LLM clients, file I/O, clocks) to be injectable. The result: the entire VM, all 15 frontends, and all analysis passes are testable in isolation without mocking.

**"If a function has a non-None return type, never return None."** Combined with null object pattern enforcement, this eliminated an entire class of `NoneType` errors. Functions that can't produce a value return a null object, not `None`.

**"Before committing anything, run all tests, fixing them if necessary."** This prevented test count regression across 100+ commits.

**"Once a design is finalised, document it as an ADR."** This produced 66 timestamped architectural decision records that serve as the project's institutional memory. Each records the context, the decision, and the consequences, including trade-offs.

The workflow encoded in CLAUDE.md is: **Brainstorm → Discuss trade-offs → Plan → Write unit tests → Implement → Fix tests → Commit → Refactor.** This isn't just process documentation. It's an enforceable contract. Every session begins with these rules loaded into context.

---

## What I'd Do Differently

**Start with the audit earlier.** The two-pass audit should have existed from the first batch of frontends. Instead, I relied on manual inspection for the first 50 sessions, and only built the audit when the number of frontends made manual checking impossible. In hindsight, the audit loop was what kept quality from drifting.

**Invest in cross-language tests from day one.** The Rosetta and Exercism suites exposed more bugs than all the language-specific unit tests combined. A single exercise tested across 15 languages covers more surface area than 50 unit tests in one language.

**Be more aggressive about the functional core.** Even with the FP rules in CLAUDE.md, some mutation crept in, especially in the VM executor. The dataflow module, by contrast, is almost purely functional and is by far the easiest module to reason about and test. The correlation is not a coincidence.

---

## The Numbers

| Metric | Value |
|--------|-------|
| Supported languages | 15 (deterministic) + COBOL (ProLeap) + any (LLM) |
| IR opcodes | 27 |
| Tests (all passing) | 8,569 |
| LLM calls at test time | 0 |
| Architectural decision records | 66 |
| Exercism exercises | 18 (across 15 languages) |
| Rosetta algorithms | 14 (across 15 languages) |
| Conversation sessions | ~400 |
| Git commits | 292 |
| Audit substantive gaps (final) | 0 |
| Audit SYMBOLIC emissions (final) | 0 |

---

## Conclusion

RedDragon started as a question: *"Can I build a single system that analyses code in any language?"* It evolved into a compiler pipeline with 15 deterministic frontends, a COBOL frontend via ProLeap, LLM-assisted AST repair, a symbolic VM with byte-addressed memory regions and named continuations, and cross-language verification.

None of the individual components are novel. TAC IR, dispatch tables, worklist dataflow, symbolic execution are all textbook techniques. The value, if any, is in applying them together to a practical multi-language analysis tool and then hardening the result through systematic auditing.

The architecture wasn't planned upfront. The deterministic VM emerged from asking *"shouldn't this be deterministic?"* The audit loop emerged from asking *"what's still missing?"* The Exercism test suite emerged from wanting more confidence than unit tests alone could provide. Each decision was triggered by testing the previous one on real code and noticing a gap.

All three projects are open source: [RedDragon](https://github.com/avishek-sen-gupta/red-dragon), [Codescry](https://github.com/avishek-sen-gupta/codescry), and [RedDragon-Codescry TUI](https://github.com/avishek-sen-gupta/reddragon-codescry-tui).

_This post has not been written or edited by AI._
