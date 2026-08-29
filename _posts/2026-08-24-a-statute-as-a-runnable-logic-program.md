---
title: "A Statute as a Runnable Logic Program: Field Notes from Climbing the Spec Ladder"
author: avishek
usemathjax: false
mermaid: true
tags: ["Software Engineering", "GenAI", "Prolog", "Formal Methods", "Specifications", "Knowledge Representation", "Legal Informatics", "Program Analysis"]
draft: false
published: true
---

In [Rigour by Design](/2026/08/24/rigour-by-design.html) I argued that almost everyone is parked on the bottom rung of the spec ladder, and that what GenAI actually made cheap is the labour of climbing. That post ends with a link to a repository and one sentence of description. This post is what is actually in it.

It is a working log, not a product announcement: some of it worked, a good deal was wrong until execution proved it wrong, and one whole architecture was deleted. What the thing buys is reproducibility — the same question returning the same answer, with the provision that decided it attached — and that comes free with the representation. What it does not buy, and what nothing in here checks, is that the encoding reads the statute correctly in the first place. The failure modes were more instructive than the successes, so I am going to be precise about which is which.

The short version: I took the Singapore Land Titles Act and tried to express it as a SWI-Prolog program you can *run*, where every rule cites the provision it encodes and every answer reads back to the source — then tried to get a language model to write that program, behind deterministic gates that decide whether its output is admissible.

One thing changes how everything below should be read: **none of the Prolog was written by a human.** Not the libraries the authoring loop produced, and not the ones I will call *reference* libraries — those were authored interactively, with me directing and reviewing rather than typing. So the axis that does real work here is not human versus model. It is **interactive, under review, unbounded rounds, a human in every one** versus **one call per domain, checked by a gate, no human in the loop at all**. What a human contributed is the representation, the gates, and the judgment about which encodings were wrong.

---

## Contents

1. [What the Experiment Was](#what-the-experiment-was)
2. [Version One, and Why It Was Deleted](#version-one-and-why-it-was-deleted)
3. [The Representation: Nine Predicates](#the-representation-nine-predicates)
4. [Where the Model Sits, and the Gates That Contain It](#where-the-model-sits-and-the-gates-that-contain-it)
5. [What the Gates Actually Caught](#what-the-gates-actually-caught)
6. [Asking Questions of the Program](#asking-questions-of-the-program)
7. [Compiling the Spec Down to Readable Python](#compiling-the-spec-down-to-readable-python)
8. [The Tradeoffs, Stated Precisely](#the-tradeoffs-stated-precisely)
9. [What This Does Not Establish](#what-this-does-not-establish)
10. [The Scale Ceiling](#the-scale-ceiling)
11. [What Is Actually Novel Here](#what-is-actually-novel-here)
12. [When to Do This, and When Not To](#when-to-do-this-and-when-not-to)
13. [Back to the Ladder](#back-to-the-ladder)

---

## What the Experiment Was

The thesis, in the form I wrote it down mid-session:

> Instead of giving a pile of prose to an LLM and having it answer questions or convert it into code — with the prose forever dependent on a human or a model to interpret it as a "spec" — build a formal logical specification which is almost an intermediate language: portable, queryable, and convertible into executable code with a high degree of mechanisability.

That is the [spec ladder](/2026/08/24/rigour-by-design.html#flavours-of-specifications) argument with a concrete artifact attached. Domain: land administration — titles, instruments, caveats, mortgages. Source: imperfect markdown. Output: a logic program plus provenance, where every asserted term traces back to the character span that justifies it.

One constraint shaped everything else: **all inference in the hot loop was meant to be strictly local** — Ollama, a 7–8B model, on hardware you own. That is not asceticism, it is what you are forced into when the documents are a land registry's and cannot leave the building. And a 7–8B model cannot be trusted to author a formal specification. So the whole architecture answers one question: *what can you get out of a weak model, if a deterministic checker gets the final say?*

---

## Version One, and Why It Was Deleted

The first architecture was the obvious one: bootstrap an ontology from the document, extract facts per segment, extract rules by having the model fill slots in fixed templates, compile to Prolog, validate with a meta-program, feed the findings back into a refinement loop.

It ran end to end on real statute text with `qwen2.5:14b` locally, and the headline result is why I still believe the guardrail principle: every output compiled to valid Prolog and loaded clean in `swipl`, normalisation produced valid atoms for all 33 proposed entity types and 56 relations, closed-world extraction *skipped* 38 fact candidates using unknown relations or the wrong arity rather than admitting them, and template-only rules made all 69 constraints well-formed by construction.

The form was perfect. The **meaning** was bad, and that is the finding. The model turned s.119(4) — a *directional* prohibition, "the Registrar shall not register a dealing prohibited by the caveat" — into a symmetric mutual exclusion: structurally impeccable, legally inverted. It converted nearly every relation into "every caveat must have X". It filled typed argument slots with the literal word `"string"`, and these *promoted to accepted*, because the type checker validated literal **positions** and not literal **values**. And given a bare citation line with no normative content at all, it hallucinated five rules with **fabricated quotations**.

That last one is the most useful, because the fix was deterministic and worked completely: require every proposed rule to carry an evidence quote, and reject it unless the quote is a literal substring of the segment. All five were rejected. Paraphrased evidence correlates with bogus rules, and a substring check suppresses them with no model in the loop.

But no amount of guardrailing fixed the central problem: the representation was too weak to carry the meaning.

- **Time was a string.** `"2019-03-03"`, or worse, `"the third business day after filing"`. Prolog cannot order strings, so "what held before X", "is this within 21 days", "which registration came first" were unaskable — and statutes are made almost entirely of those questions.
- **Identity was a name.** Ids hashed `type:name`, which over-merges (every "mortgage" collapses to one) and under-merges ("the mortgage", "first mortgage", "second mortgage" fracture into three) *simultaneously*.
- **A type was one opaque atom,** so every distinction got crammed into the label: `registered_mortgage` bakes an event outcome into a name where nothing can reason about it.
- **Facts had no context.** `register(registrar, mortgage)` asserted flatly is implicitly true *everywhere*, which is false. Torrens land law turns precisely on the register differing from the world, and on what a party knew.

So it went: around 11,800 lines deleted in two commits, with the modules serving the old representation going rather than being carried forward broken. The temptation is always to keep the code and patch the model underneath it. But the stages *encoded* the bad representation, and maintaining a stage that emits a vocabulary nothing downstream reads is worse than having no stage.

---

## The Representation: Nine Predicates

What replaced it is a coordinate model. The domain-neutral core is nine names in 97 lines, mentioning no dimension, no sort, no relation and no verb:

```prolog
ist(Ctx, Statement).                    % Statement holds in Ctx
asserts(Ctx, Record).                   % INDEX -- whose account. Not a dimension.
mark(Ctx, Dim, at|lower|upper, Value).  % where Ctx sits on a dimension

dimension(Dim).                         % declared per domain
within(Dim, Point, Extent).
precedes(Dim, P1, P2).                  % OPTIONAL, strict; its presence enables bounds
offset(Dim, Point, Quantity, Point2).   % OPTIONAL -- point construction

at_or_before(Dim, A, B).                % derived
holds_in(Statement, QueryContext).      % derived -- the only query form
```

**`ist` is McCarthy's "is true in context"**, lifted whole from *Notes on Formalizing Context* (IJCAI-93). The generating rule is one sentence:

> Anything with its own lifetime gets an id, and its value or extent attaches to that id. Anything that varies gets a dimension. Whether something "holds" is membership of a query position inside an extent.

Three consequences, each of which dissolved a problem I expected to need machinery for.

**There is no Event Calculus.** An event is a context positioned at a *point*; a state is a context with a *lower* bound and possibly an *upper* one; persistence is extent-completion by derived `mark/4` rules. So inertia is not a property of time — it follows from any dimension carrying a directed order, which means the `version` dimension has it too. "In force from the 1993 Act until repealed" is structurally the same shape as "effective from registration until discharge", and the same machinery answers *"was this a caveat under the Act as it stood in 1990?"* and *"was it effective in mid-2020?"*

**Types are patterns asked at a point,** not stored labels. `caveat(Cav, Q)` is a rule, not an atom, which kills the `registered_mortgage` problem at the root.

**Closing conditions need no ordering logic.** An extent is clipped if *any* upper mark falls in the gap, so "earliest terminator wins" fell out of the definition rather than being implemented. A caveat that can lapse at five years, be withdrawn expressly, or be deemed withdrawn fourteen days after an unrectified deficiency notice just has three `upper` mark rules, each citing its own provision, and they compose:

```prolog
% s.121(1)(b) "at the expiration of 5 years from the date of the lodgment"
%  -- an UPPER bound, and the reason offset/4 exists: the point is
%  CONSTRUCTED, not compared.
mark(ctx(effective(Cav)), time, upper, T2) :-
    ist(CL, lodges(_, Cav)), mark(CL, time, at, T1),
    offset(time, T1, years(5), T2).

% s.117(3) "if, within a period of not less than 14 days ... the caveat is
% not rectified, it shall be deemed to have been withdrawn".
mark(ctx(effective(Cav)), time, upper, T2) :-
    ist(CN, notice_of_deficiency(Cav)), mark(CN, time, at, T1),
    offset(time, T1, days(14), T2),
    \+ ( ist(CR, rectified(Cav)), mark(CR, time, at, TR),
         at_or_before(time, TR, T2) ).
```

One rule, one provision, one citation. That is [Bench-Capon and Coenen's isomorphism principle](https://link.springer.com/article/10.1007/BF00871902) from 1992, and it is what makes an amendment a mechanical edit rather than an audit. Fourteen provisions in the caveat library point at character spans into a content-addressed document version, and a gate checks that every declared provision actually cites text. On its first run that gate found a *dead* declaration — a provision nothing called, left over from an abandoned modelling attempt. It found dead vocabulary, not a missing comment.

Three sorts (`entity`, `quantity`, `text`) and four dimensions (`time`, `version`, `space`, `money`). "Dimension" turns out to mean *ordered value domain*, and positioning a context in one is optional: nothing is positioned in `money`, but its `precedes/3` and `offset/4` are exactly what "exceeds $500" and "the residue after costs" needed. Quantity arithmetic looked like a gap in the vocabulary and was actually a value domain nobody had declared. Four clauses closed it.

---

## Where the Model Sits, and the Gates That Contain It

The founding tenet: **the LLM never writes Prolog and never invents structure.** It fills schema-constrained slots; deterministic code compiles terms.

That survives where it matters — the hot loop, thousands of calls per corpus, a weak local model — and was **deliberately broken in one place.** The authoring tier is one call per domain, and its output is small, static, and mechanically checked before anything depends on it. A Pydantic schema plus a JSON→Prolog compiler would buy back one frontier call per domain, in exchange for several hundred lines and a **second copy of the grammar** to keep in step with the checker, and it would only ever fix violations the shape gate already catches. Every defect that actually produced a *wrong answer* cleared the shape gate and needed execution to find. The question was asked twice and answered no twice, with revisit conditions written down: *if authoring starts running per-document; if the repair loop stops converging in about two rounds; or if something other than `swipl` consumes the output.* A tenet with stated conditions for its own reversal is worth more than a tenet.

```mermaid
flowchart TD
    subgraph OFF[Offline -- once per domain]
        DOC[statute / policy wording] --> FM[frontier model\nproposes a pattern library]
        FM --> G1{Gate 1 + 1.5\nshape · content}
        G1 -->|machine-readable violations| FM
        G1 -->|accepted| LIB[pattern library\nrelations + cited rules]
    end
    subgraph HOT[Hot loop -- per segment, NOT BUILT]
        SEG[document segments] --> LOCAL[local 7-8B\nentities · relations · dates only\nnever Prolog, never a type name]
        LOCAL --> DET[deterministic compile\nto ist / mark]
    end
    LIB --> G2{Gate 2\nrun against an\nINDEPENDENT corpus}
    DET --> G0{Gate 0\ncorpus vs declarations}
    G0 --> G2
    G2 --> RUN([runnable program\n+ provenance])

    classDef default fill:#dde3f5,stroke:#6b7db3,color:#1a1f5e
    classDef llm fill:#ede0f8,stroke:#7b2d9b,color:#3b0764
    classDef gate fill:#d4edda,stroke:#388e3c,color:#1b5e20
    classDef missing fill:#fce7f3,stroke:#be185d,color:#831843
    class FM,LOCAL llm
    class G0,G1,G2,RUN gate
    class SEG,DET missing
```

The dashed-in-spirit box is real: **the hot loop does not exist yet.** More on that below.

There are four gates, and none subsumes another.

| | Reads | Catches | In the repair loop? |
|---|---|---|---|
| **Gate 0** `fact_check.pl` | a *corpus*, against a library's declarations | undeclared relations, arity mismatches, a mark kind contradicting the relation's declared class | yes |
| **Gate 1** `shape_check.pl` | the library *file* | grammar: legal goals, sorts, exactly one cited provision per rule | yes |
| **Gate 1.5** `content_check.pl` | the library *file* | a library that is perfectly on-grammar and authors nothing | yes |
| **Gate 2** `acceptance.pl` | the library **run** against a corpus | rules that derive nothing, rules that derive everything, two rules with identical extensions | **no, deliberately** |

Gate 2 sits outside the loop on purpose: it needs a corpus, and **a corpus authored to satisfy the gate compromises the gate.** A model that writes its own fixtures writes them to agree with itself.

The repair contract is blunt. A checker returning a negative verdict wants a repair, and the violation list goes back as the next prompt; a checker that *raises* is saying no repair exists, and the loop propagates rather than retries. Exhausting the round budget is a **result**, not an exception. And one detail decides whether any of it works: **the gate must return specific, machine-readable violations.** `undeclared_relation(effective/1)` is actionable. "Try again" is not. An early run reported "0 violations" four times in a row and gave up, which is the degenerate case of the same principle — a rejection carrying nothing the model can act on is not a rejection, it is a refusal.

---

## What the Gates Actually Caught

**Land Titles Act, ss.115–121.** `claude-sonnet-4-5` via Bedrock at temperature 0 — chosen over the strongest available model deliberately, because a weaker one tests the gate harder. The first one-shot experiment came back `SHAPE_FAIL` with **13 violations**, and the split is the finding rather than the count. Everything the prompt warned about, it got right: event/state classes assigned correctly, rule-precedence machinery left untouched, no context variable shared across conjuncts, a core derived type all but identical to the reference library's, plus 28 derived rules and 14 provisions cited. Everything it got wrong was **bookkeeping** — not the silent semantic kind that took three attempts to get right in the reference libraries, with a human reviewing each one. Wired into a repair loop, the same task converged in **two rounds**: two violations, then `SHAPE_OK` with 17 relations, 16 provisions, 25 derived rules and 9 derived mark rules. A second domain — the NY standard fire policy, chosen because an insurer's wording is copyright and this one is a statutory form — behaved identically: 10 violations in round one, `SHAPE_OK` in round two, and again everything caught was vocabulary discipline.

And the accepted library was **better than the reference library in one place**, which is worth recording rather than hiding: for s.119(1) it declared acceptance as an *event* and being-in-order as a *state* whose lower bound derives from that event, where the reference library treats the latter as the event itself. One call plus a gate applied the discipline the prompt teaches at a point the reviewed interactive sessions had let slide. More rounds with a human in each of them is not uniformly better than one round and a checker.

Here is every defect worth reporting, and which gate saw it:

| Defect | Caught by | Consequence |
|---|---|---|
| 6 relations used, never declared | Gate 1 | 8 rules written over a term absent from its own vocabulary |
| a context term naming the wrong statement | Gate 1 | bookkeeping; repaired in one round |
| a fluent context passed where a query context belongs | **nothing** | fails *silently* — the gate hole a model found |
| a condition that constrained nothing | Gate 2 | "Alice may lodge lot 5 as a caveat" was derivable |
| an *acceptance* (an event) read with `holds_in` | Gate 2 | the duty to notify existed for one day, then evaporated |
| two predicates never once run | Gate 2 | no fixture exercised the provision they encoded |
| fixtures `discontiguous` but not `multifile` | Gate 2 | clauses silently discarded; false vacuity, no error |
| a citation carried by all 63 rules, defined by none | Gate 2 | every rule failed on an unknown procedure |
| a library on-grammar, fully cited, authoring nothing | Gate 1.5 | 12, 8 and 30 authoring rules on the shipped libraries, against **0** |
| `c1` appearing in 36 of 40 scenario files | Gate 0 | 29 systematic mark-kind errors, down to 6 after namespacing |

Five things follow.

**Gate 1 is structurally blind to meaning.** Every Gate 2 row above is grammatically perfect. A gate that reads the file can only check the file; only running it against a corpus finds a rule that derives nothing, derives everything, or derives the wrong thing. The citation row is the sharpest case — the shape gate asks whether a rule *carries* a citation, never whether one *resolves*.

**The silent row is the one worth studying.** The model wrote `\+ holds_in(extended(Cav), ctx(effective(Cav)))`: `holds_in` reads `at` marks off its second argument, a fluent context carries `lower`/`upper` instead, so the goal fails rather than errors, and no gate said anything because that argument position was unconstrained. It is now checked, and the check exists because a model did something nobody had thought to forbid. That is adversarial coverage of your own grammar, which I did not anticipate getting.

**The recurring error got made syntactic.** The event/state conflation appeared three times and survived human review each time, so relations now declare a class — `relation(lodges, [entity,entity], event)` — on the empirical basis that the class is a *function of the relation*: across the whole caveat corpus, every extracted relation had exactly one mark kind. The gate can now reject reading an event as a state, both historical instances are replanted as tests and caught, and the mark kind is assigned **deterministically from the declaration**, taking the hardest judgment in the representation out of the extraction model's hands. It emits a relation, its arguments and a date; deterministic code decides whether that is a point or a bound. *Find the error you keep making; find a declaration that makes it syntactic.* That is the transferable technique here, and it is not specific to Prolog or to law.

**An inert library is a real failure mode, and the gap is total rather than marginal.** The screen-validation library asserted "this notice has not expired" as a *corpus fact* where the source document gave the expiry **date**. The document licensed a rule and the library declined to write one. Two other candidate smells were tried and **discarded for firing on the reference libraries** — one found 24 instances in one authored library and 0 in another doing the identical thing under different names, so it measured naming, not modelling. Negative results in heuristic design almost never get written down.

**The residual 6 corpus errors are a finding, not noise.** A classification — "this was a fire", "this was lightning" — is neither an event nor a bounded state. The encoder dates it because the narrative dates it; the library declares it a state. That is a genuine modelling question the representation has not settled, and it surfaced only because a gate counted.

---

## Asking Questions of the Program

A specification you cannot interrogate is barely a specification. So there is a query tier: a question in, a Prolog transcript and an answer out. Four steps, and only the first and last involve a model:

```
question -> a CHECKED query structure           (model, gated)
         -> Prolog goals at one or more points  (deterministic)
         -> answer + evidence read off the marks (Prolog)
         -> prose, printed BESIDE the transcript (model)
```

The model never writes Prolog text. It picks a predicate from the introspected vocabulary, its arguments, and the points to ask at; deterministic code renders goals and checks them against the loaded program *first*. Free text cannot be gated — a goal string can name a predicate that does not exist, and the only way to find out is to run it — whereas a structure is checkable in a few lines. The gate is strict about arity for one reason: **a goal with the wrong arity fails silently, and a silent failure reads exactly like a well-formed "no."**

Asking at *several* points is the normal case, not an extra. Most real questions about a statute are comparative, and the interesting answer is the one that changes:

> *Alice lodged a caveat over lot 5 in March 2019, and a buyer now wants dealing1 registered against that lot. Is the Registrar barred from registering it in mid-2020, and would that still be so in mid-2025?*

```prolog
?- must_not_register(dealing1, q_mid_2020).    % yes
?- must_not_register(dealing1, q_mid_2025).    % no

evidence (marks bounding the relevant contexts):
   effective(cav1): time lower = d(2019,3,3)
   effective(cav1): time upper = d(2024,3,3)
   lodges(alice,cav1): time at  = d(2019,3,3)
```

The date that decides this — 3 March 2024 — appears nowhere in the source documents. It is constructed by the five-year rule in s.121(1)(b), and the explanation cites it because the engine gathers evidence **mechanically**: the marks bounding every context whose statement mentions an entity the question is about. The dates in the explanation are read off the program rather than recalled. The prose step is the weakest link and is treated as such — given the transcript and nothing else, told not to introduce facts, and always printed *beside* the run rather than instead of it. **An explanation nobody can check is worse than no explanation.**

Two results from that tier are worth reporting honestly.

**The near-miss the gate cannot catch.** Asked whether Alice's underlying claim was *justified*, the model composed `claimant(alice, Q)` — valid vocabulary, real entity, and **a different question**, since being a claimant is not having a good claim. The gate checks that a predicate exists; it never checks that the predicate answers what was asked. **Vocabulary checking is not semantic checking**, and no amount of grammar work makes it so. Both fixes are outside the gate: an explicit route to decline, because a system that cannot say "I can't" will always say *something*, and human-readable glosses so near-neighbours can be distinguished in prose where nothing mechanical can separate them. Declining is never retried, since retrying only pressures the model into naming an adjacent predicate.

Which is the point of the pair, because next door sits a question that is *correctly* declined. Whether Alice's claim is *good* needs contract and property law, evidence, and a court — that is the law, not a gap in the library. Whether the Registrar **must not consider** it is squarely answerable, because s.117(5) says he must not. A statute's refusal to decide something is itself a provision with content, and the system knows which side of that line it is on.

**The lossy rendering.** In three recorded examples the query enumerated solutions through an unbound variable and the transcript surfaced only the first. The prose tier then said, correctly, *"the transcript does not settle the question."* The Prolog was right, the presentation layer was lossy, and the interpretation step refused to fill the gap by invention — the design working and the implementation not. I would rather report a tier that declines than one that confabulates the other three answers.

---

## Compiling the Spec Down to Readable Python

If the specification is the artifact of record, you should be able to project it into the language you actually ship. So there is a generator: a pattern library in, a self-contained Python module out — one dataclass per entity kind, one method per derived predicate, one period method per fluent.

**Attempt one was rejected, and it passed every correctness test.** It was a Prolog-to-Python transpiler: nested closures over a store, generated names like `_derived_mark_1` and `_Anon29`, `continue` loops. It agreed with Prolog on every enumerated case and it was unreadable, which defeats the entire purpose. Attempt two is expression-first — a conjunction is `and`, an existential is `any(...)`, a negation is `not`. Same correctness bar, output a human can review.

The constraint that decided everything: **readability information lives in the spec, never in the generator.** A generator that guessed `lodges(Party, Caveat)` should render as "lodged_by" would be land law in disguise, custom to one statute. Every readable name comes from a declaration the library author wrote — and the mortgages library declares *none* of them and still compiles end to end against the raw statement store, which is what degrading gracefully has to mean, and it is a test.

Here is what comes out for the caveat lapse rules quoted earlier:

```python
def effective_period(self, cav: str) -> Extent:
    """When effective(cav) holds, on the time axis.

    s.121(1)(b) — "at the expiration of 5 years from the date of the
    lodgment of the caveat, ..."
    s.117(3) — "if, within a period of not less than 14 days ... the caveat
    is not rectified, it shall be deemed to have been withdrawn."
    """
    caveat = self.registry.caveats.get(cav)
    if caveat is None or caveat.lodged_by is None:
        return Extent(dim="time", stated=False)
    # without the statement there is no context to hang marks on, and an
    # unstated fluent covers nothing -- which is not the same as an
    # unconstrained one, which covers everything.
    ...
    if caveat.lodged_by is not None and caveat.lodged_on is not None:
        ends.append(add_years(caveat.lodged_on, 5))              # s.121(1)(b)
    if caveat.deficiency_notice_on is not None and not (
        caveat.rectified_on is not None
        and caveat.rectified_on <= caveat.deficiency_notice_on + timedelta(days=14)
    ):
        ends.append(caveat.deficiency_notice_on + timedelta(days=14))  # s.117(3)
```

The statutory limits are on the page as ordinary date arithmetic, the citation is in the docstring quoting the provision, and there is no Prolog runtime hiding behind it. An equivalence suite runs every predicate through both engines and diffs. Two details from building it generalise.

**Never emit a `...` stub.** The first cut emitted `...` for every derived predicate. `...` evaluates to `None`, which is falsy — so each stubbed rule silently answered **"no"**, indistinguishably from a rule that means it. Anything the compiler cannot render faithfully now *raises*, naming the clause, the goal and the reason, and the module exports an `UNTRANSLATED` list so which methods are trustworthy is visible without running anything.

**The equivalence suite caught four defect classes, each giving a plausible wrong answer:**

| Defect | What it did |
|---|---|
| an index running opposite to a dimension | silence about a dimension is *unconstrained*, but a context asserting no record answers *no* record-scoped question; the reversed polarity made two copies of the data disagree |
| an empty extent read as unconstrained | a loader bug dropped a field, the date came out `None`, empty read as "no bounds", and **every caveat was effective forever**. `stated=False` (covers nothing) versus no marks (covers everything) is the asymmetry to watch |
| a dropped conjunct | a derived call with an unbound argument emitted the scan and discarded the call, so it answered yes for any entity at all |
| a rebound parameter | a store pattern named its loop targets after already-bound variables, widening "this mortgage" to "any mortgage" |

Every one produces confident, well-formed, wrong answers, and none is visible by reading the output. That is the concrete argument for keeping two engines and diffing them, rather than trusting a translation.

---

## The Tradeoffs, Stated Precisely

| Decision | What it buys | What it costs |
|---|---|---|
| Nine-predicate domain-neutral core | Two Parts of one Act plus an insurance policy form fit with **no core changes**. Rule validity over legislative versions uses the same machinery as facts over time. | Sort conformance became nearly vacuous: collapsing to `entity` + `quantity` leaves quantity as the only checkable sort. Real cost, knowingly paid. |
| Types derived, not stored | Kills `registered_mortgage`-style label pollution outright. | Every type question costs a resolution rather than a lookup. Irrelevant at this scale, unknown at corpus scale. |
| Frontier model writes Prolog offline; local model never does | One call per domain; the artifact is small, static, checked. No second copy of the grammar. | Breaks the founding tenet in one place. And the local tier it protects **does not exist yet**, so the split is architecturally justified and empirically untested. |
| Gate 2 outside the repair loop | The acceptance corpus stays independent of the model that wrote the rules. | A whole class of defect is only found *after* the loop reports success, and every serious defect so far was in that class. The independence is also only as real as the sourcing. |
| "It checks; it does not compute" | Aggregates need no arithmetic if their value is *stated*, and for document extraction it usually is. Priority needs no sorting if rank is a declared position. Several problems that looked like they needed a computation layer evaporated. | Wrong side of the line for any engine that must **perform** a distribution rather than test one. |
| One unstratified rule shape accepted | `holds_in` inside a `mark` body is genuinely useful when deriving new bounds. | It can fail to terminate. Accepted deliberately, reproduced in a fixture, and the failure is **loud** — a stack trace naming the cycle, not a wrong answer. |
| Provenance as positioned statements | A provision's span **varies by version**: s.115 as enacted and as amended are different words in different documents. An amendment adds a citation with its own version bounds and the machinery picks the right one. | More indirection than a citation string on the rule. |

Two things I expected to be hard were not. **Partial holding** — "according to the extent of his interest", "freed from the mortgage absolutely or to any lesser extent" — looks like it needs graded truth and does not, and the commercial legal DSLs agree: eFLINT parameterises acts, Symboleo parameterises obligations, Catala computes an amount, and **none of them makes "holds" partial**. A quantity that *changes* is a reified term whose value varies by context; an extent that *limits* is an ordinary comparison on the money dimension. **Deeming** is likewise just a fact whose bound the statute supplies.

*Harder than expected:* **rule precedence.** An `overrides/2` relation was declared for it, is used in exactly **zero** rules across two libraries, and both of the Act's precedence forms defeated it. A conditional carve-out ("except as otherwise provided in s.129(1)") encoded as an override took the whole section **dark** — no dealing was ever prohibited — because the carve-out displaces the section only for the dealings the other section reaches: it is a condition on the dealing, not a precedence between rules. And the case the mechanism was *designed* for, "despite any other provision of this Act", is worse: overriding the general priority rule kills the registration ordering the overriding provision itself presupposes. **The override eats its own premise.** Both encode correctly as an exception clause inside the general rule, which is Catala's prioritised-default shape. A reader who sees "except" and reaches for a precedence relation will silently disable a section, and nothing will error.

---

## What This Does Not Establish

This is the section I most want people to read.

**Nothing extracts anything.** Every fact in every fixture was put into the repository by a model — interactively, or by a generation script — rather than extracted from a document. There is no seam from a document to `ist`/`mark`. The founding premise is extracting a model from imperfect markdown with a local 7–8B model, and in the current architecture that is **entirely untested**. The demo answers questions about a knowledge base assembled in directed sessions, which is exactly the part a viewer will assume was automated. Building it needs *instrument* documents — a folio, a lodged caveat, a judgment — that this corpus does not contain: the Act gives rules, not records.

**There is no baseline, and the baseline is the wrong comparison anyway.** I never measured whether a strong model, handed the 6KB statute extract and the same eleven questions, answers them all correctly with no machinery at all. It very likely does. But answer quality was never the axis this buys on, and running the race on that axis concedes the wrong thing.

What the apparatus establishes *automatically* is that the answer is **reproducible**, and that is a property of the artifact rather than a result about it. The same question asked at the same points returns the same answer every time. The date that decides it — 3 March 2024 — is constructed by a cited rule rather than recalled, so it is reconstructible by anyone with the program and the statute. The evidence is gathered mechanically off the marks rather than composed. Two independent engines agree or the diff fails. None of that needs an experiment to establish, because it follows from what the thing *is*: a deterministic program with citations. A model asked the same question twice is not obliged to answer the same way, and cannot show you the provision that made the difference. That is the comparison worth making, and it does not turn on which one is smarter.

The load-bearing word is *automatically*, and its price is the conditional attached to it: reproducibility holds **assuming the encoding is a correct reading of the source**, and that assumption is precisely the one nothing here checks. Of the four properties the case rests on, two — reproducibility and reconstructibility — come free with the representation. As-at-date reasoning comes free with dimensions, though it is untested at any real scale. Consistency across thousands of documents is genuinely unmeasured, and cannot be measured until the hot loop exists. Reproducing an answer faithfully is not the same as producing the right one, and this project buys the first cheaply and the second not at all.

**The gates check hygiene, not fidelity.** No gate establishes that a library is a correct *reading* of its source. A library can be well-formed, non-vacuous, non-degenerate, fully cited, and still misread the provision it points at. **Every serious defect found so far was of that kind.** A human reading rule against provision remains irreducible — made cheap by one-rule-one-provision, but not eliminated.

**The rules and the fixtures they are checked against share an author.** The sharpest consequence of everything being model-written. The land corpus and the land libraries came out of the same sessions, so a shared misreading would produce a library and a fixture that agree with each other and both diverge from the statute, and every gate would report success. The insurance experiment was designed specifically to break that circularity, and it is the experiment that did not finish. So the independence the architecture depends on is currently a design commitment rather than a demonstrated one.

**Sample size is small and the prompt was loaded.** Two Parts of one Act plus one policy form, and the "avoided all four failure modes" result is a handful of runs with those four failure modes spelled out in the prompt. Two further runs did not complete and are not counted.

**The second-domain experiment is incomplete.** The design was a genuine blind test: one call writes forty scenarios with expected verdicts from the policy alone, seeing no grammar and no hint that a logic model exists; that oracle is committed to git **before** the library is authored, so commit order is the evidence it was blind; a third call encodes the scenarios into facts *without ever seeing the verdicts*. The library was authored, the corpus encoded, the corpus gate built and run. **The final diff — derived answers against withheld verdicts — was never run.** So the strongest evidence this experiment was designed to collect remains uncollected, and I would rather say that than quote the parts that did finish as though they were the result.

**Coreference is untouched, and it is the sharpest live risk.** A wrong entity id does not produce an error, it produces a well-formed *false statement*: a caveat whose lodgement and acceptance get different ids ends up with no lower bound and reads as effective since the dawn of time. The negation ambiguity compounds it — `\+ rectified(Cav)` reads "not rectified", and in a lossy extraction it *also* reads "we did not extract the rectification". Those give opposite legal outcomes and the program cannot tell them apart.

---

## The Scale Ceiling

The authoring loop handles a policy wording and a dozen sections of a statute. It does not reach the whole Act, and the reason is **not** the context window, which is the interesting part.

The caveat library covers 7 real sections in 652 lines with 37 rules — roughly 93 lines and 5 rules per section. The Land Titles Act has 184 real sections, which extrapolates to something like 17,000 lines and 900 rules. No single call produces that, however large its input window, because the binding constraint is the **output** budget — and the repair contract makes it worse by demanding the entire file back every round. The second constraint is attention dilution: the four failure modes the prompt warns about are exactly the discipline that decays across a long generation, and the shape gate cannot catch a misreading, which is what long-form drift produces.

There is a design for this — decompose the document into a *structure map* of citable spans, cluster, author per cluster, reconcile — and its central commitment is to **assume nothing about document structure**, because the real target is not statutes. It is validation-rule tables, standard operating procedures, and design documents intermingled with architecture docs. Any design keyed on sections and numbered clauses degrades *silently* on most of that: a document with no citations yields a reference graph with no edges, a perfectly valid graph that produces one cluster per unit and no error anywhere. The one guard in it worth stealing regardless: a span's recorded text must be a **verbatim substring** of the source at the recorded offsets. Total, free, mechanical, and the same move as the evidence gate that killed five hallucinated rules in the very first shakedown.

That design is written down. It is not built.

---

## What Is Actually Novel Here

Very little, and I want to be blunt about it, because a project like this generates a lot of prose that reads as claims.

`ist(c, p)` is McCarthy, 1993, lifted whole. Contexts carrying coordinates is Guha's 1991 thesis and Cyc's microtheories. Statute-as-logic-program is Sergot, Sadri and Kowalski's British Nationality Act paper in *CACM*, 1986 — forty years old. One-rule-one-provision is Bench-Capon and Coenen, 1992, cited by name in the code. Derived types as patterns is ordinary Prolog. Fluents and inertia are Event Calculus, same 1986 vintage. Storing extents as endpoint facts rather than interval terms is how temporal databases have stored valid-time for decades. "LLM writes a DSL, a checker rejects it, repair" is a standard pattern by now, and "natural language to structured query to execution to natural language" is text-to-SQL with a validator. The reified-value pattern I made the biggest fuss about is McCarthy and Buvač's `value(c, term)`, and it was **already in the design document's own bibliography** — the project rediscovered a citation it had in front of it. Catala, meanwhile, already does the defeasibility part properly, with prioritised defaults as a first-class construct rather than exception-clauses-by-convention.

Three things might be worth something, hedged:

- **Turning your most frequent modelling error into a grammar violation.** I conflated an event with the fluent it starts three separate times. Declaring the relation's class made the error *syntactically* checkable, and both historical instances are now caught as tests. The general move transfers.
- **"It checks; it does not compute"** as an explicit scoping decision. Several hard problems evaporated, and for document extraction it is the right side of the line — though the mechanism underneath, uninterpreted function symbols, is entirely standard.
- **Dimensions versus indices.** "No order at all means it is not a dimension" killed a coordinate that was wrong and replaced it with a plain index. A clarification, not a discovery.

A careful assembly of forty-year-old ideas, with one or two sharp engineering moves and a scoping decision that does real work. That is the honest summary, and it is also the point of the [previous post](/2026/08/24/rigour-by-design.html#what-ai-actually-changed): **none of this was ever wrong, it was expensive.** What changed is that encoding it got cheap enough to try on a weekend.

---

## When to Do This, and When Not To

**Not worth it** when the document is short-lived, low-volume, or low-stakes; when one competent person can hold the whole rule set in their head; when nobody will need to ask "what was the answer as at a date two years ago"; or when there is no second document to be consistent with. Handing the statute to a good model and asking is faster, and until somebody runs the baseline nobody can honestly tell you it is worse.

**Worth considering** when the answer must be *reconstructible*, because a regulator, an auditor or a court will ask why and "the model said so" ends the conversation badly; when the same rules are applied across thousands of documents and consistency between them is the actual product; when the source **amends** and you need to know what changed and what it broke, which is exactly what one-rule-one-provision buys; when the rules must be queried forwards and backwards, since a relation runs in both directions and a function does not; or when the data cannot leave the building, which forces you into a weak model and therefore into a strong checker.

**The domain matters more than the technique.** This works on governance documents because they already *are* rule systems: citable units, defined terms, explicit temporal operators, and a tradition of being read literally. A marketing brief has none of that, and no amount of Prolog will give it any.

---

## Back to the Ladder

Everything above is one long attempt to occupy the [declarative, machine-checkable rung](/2026/08/24/rigour-by-design.html#flavours-of-specifications) of the spec ladder, and to find out what it costs in practice rather than in principle. Four things from the earlier post now have receipts, and one has a correction.

**"AI proposes; the verifier disposes"** is the load-bearing claim, and it held — more completely than I expected, because *every* proposal in the chain turned out to be a model's: the libraries, the corpora, the query compositions, the projected Python, the blind oracle. The model contributed candidate generation, which is what it is good at, and contributed nothing to any decision. The check is what converged, not the model.

The uncomfortable corollary is in the comparison the two authoring paths make possible. One had a human in every round, unlimited rounds, and full review. The other had one call and a gate. The gated path produced a *better* encoding of s.119(1), and the reviewed path is where all three instances of the event/fluent conflation came from. Human review caught the errors it was looking for and waved through the ones it was not. A mechanical check does not get tired at round nine.

**"Only a machine-checkable artifact makes an LLM worth looping"** held for a sharper reason than I gave it credit for: it is not just that a wall of prose cannot fail, it is that the rejection has to be *specific*. `undeclared_relation(effective/1)` converges; "your spec seems inconsistent" is not a check at all.

**"Least power"** decided the local tier. The way to make a 7–8B model safe is not a better prompt, it is to leave it less to decide — down to a relation, its arguments and a date, with deterministic code taking even the mark kind.

**The correction, and it is the one that matters.** In the earlier post I wrote that a model's nondeterminism sits in the gap between the spec and the program. That is true and incomplete. Here the nondeterminism moved *up*, into the gap between the **source document** and the spec — and no gate I built can close it. Every gate reads the library, or runs it. **None of them reads the statute.** Well-formed, non-vacuous, fully cited, perfectly reproducible, and a misreading, is a state this system can reach and cannot detect. The gates made the cheap errors free to find, which is genuinely worth having, and they left the expensive ones exactly where they were: with a human, reading a rule against the provision it cites.

That is a smaller claim than "verified". It is also the honest one, and it is where the work is.

The repository is at [github.com/avishek-sen-gupta/doc-pipeline](https://github.com/avishek-sen-gupta/doc-pipeline), including the recorded model runs, the round-by-round violation journals, the fixtures that are *expected* to fail, and the evaluation document recording which encodings were wrong until running them proved it.
