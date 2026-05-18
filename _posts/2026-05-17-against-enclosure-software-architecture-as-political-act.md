---
title: "Against Enclosure: Software Architecture as a Political Act"
author: avishek
usemathjax: false
mermaid: false
tags: ["Software Engineering", "Architecture", "Open Source", "Philosophy", "Unix", "Composability"]
draft: true
published: false
---

*Software architecture is not politically neutral. The choice between a plugin-based platform and a set of composable small tools is a choice about who can enclose the result. This essay argues for composability as a structural act of resistance, not because it produces better software in every case, but because it produces software that cannot be captured.*

---

## Table of Contents

- [The Chain: Plugin Architecture → Platform → Enclosure](#the-chain-plugin-architecture--platform--enclosure)
- [What Plugin Architectures Actually Do](#what-plugin-architectures-actually-do)
- [The Political Argument: Enclosure, IP, and the Escape Hatch](#the-political-argument-enclosure-ip-and-the-escape-hatch)
  - [Rent-Seeking](#rent-seeking)
  - [IP as the Mechanism of Enclosure](#ip-as-the-mechanism-of-enclosure)
  - [Licensing Theatre](#licensing-theatre)
  - [The Machinery Analogy: The Escape Hatch](#the-machinery-analogy-the-escape-hatch)
- [The Composable Alternative: Unbundling the Platform](#the-composable-alternative-unbundling-the-platform)
- [IPC as Substrate: All the Patterns Already Exist](#ipc-as-substrate-all-the-patterns-already-exist)
- [Living Proof: Five Projects, Five Enclosure Vectors Eliminated](#living-proof-five-projects-five-enclosure-vectors-eliminated)
  - [LSP: Eliminating Vendor Bundling](#lsp-eliminating-vendor-bundling)
  - [SQLite: Eliminating Server and Copyright Control](#sqlite-eliminating-server-and-copyright-control)
  - [Nix: Eliminating Registry Control](#nix-eliminating-registry-control)
  - [SourceHut: Eliminating Platform Network Effects](#sourcehut-eliminating-platform-network-effects)
  - [Gemini: Eliminating the Monetisation Surface](#gemini-eliminating-the-monetisation-surface)
  - [The Pattern](#the-pattern)
- [Intellectual Lineage](#intellectual-lineage)
- [Honest Limits](#honest-limits)

---

## The Chain: Plugin Architecture → Platform → Enclosure

I have built a number of systems using plugin architectures. I built them in good faith, as the technically correct answer to extensibility. You define a host, you define a contract, and anyone who conforms to the contract can extend the system. It is clean. It is modular. It feels like the right shape for software that is meant to grow.

What I eventually understood is that this shape has a structural logic that extends well beyond the technical. A plugin architecture, by design, produces a platform. And platforms are what get enclosed.

Enclosure is a term borrowed from history. In sixteenth- to eighteenth-century England, land that had been held in common, accessible to all, owned by no one, was progressively fenced off, privatised, and made subject to rent. The commons became property. Those who had farmed it freely became tenants who paid to continue. The mechanism of enclosure is always the same: take something that was shared, draw a boundary around it, and charge for crossing the boundary. In software, the thing being enclosed is not land but capability: the ability to extend a system, to interoperate with it, to compose it with other tools. The enclosed resource is the interface, the ecosystem, the protocol, and the boundary is drawn by whoever controls the contract. Aaron Swartz, in his *Guerilla Open Access Manifesto*, called this what it is: theft. Not of a thing that was ever anyone's private property, but of something that belonged to everyone, and was taken.

The logic runs like this. A plugin host defines discovery (how extensions are found), lifecycle (how they are loaded and unloaded), interface (how they talk to the host), and composition (what gets called when). These four concerns are bundled under a single contract that the host owns. The host becomes structurally necessary, not because it does anything especially sophisticated, but because it is the only thing that speaks the contract. Extensions cannot interoperate without it. Users cannot compose without it. The entire capability surface of the system flows through it.

That is a platform. And platforms are attractive to enclosure because they concentrate what is valuable: the contract, the ecosystem, the user base. A company can draw a boundary around a platform in ways it cannot draw around a set of independent tools. The plugin contract becomes IP. The ecosystem becomes captive. The host becomes a toll gate. The plugins, and the people who built them, become tenants.

**This is not malice. It is the structural logic of how platforms work.** The architecture produces the incentive; the incentive produces the enclosure.

The question I have been trying to answer is: what is the architectural opposite of a platform?

---

## What Plugin Architectures Actually Do

Before proposing a remedy, it is worth being precise about the diagnosis.

A plugin architecture is doing four things simultaneously:

**Discovery**: the host defines how extensions are found. It may scan a directory, query a registry, or read a manifest it controls. Extensions do not find each other; they are found by the host.

**Lifecycle**: the host calls `init()`, `register()`, `shutdown()` on each extension's behalf. It owns the process. Extensions exist at the host's pleasure.

**Interface**: the host defines the vocabulary. The plugin API, the callback table, the event types: these are specified by the host, and every extension must speak them. An extension written for host A cannot run under host B because the vocabulary is different.

**Composition logic**: the host owns the dispatch loop. It decides what fires when, what data flows where, what is in scope. Extensions react to host-defined events.

Bundling all four under one host-owned contract is what makes a plugin host a platform. **It is not one of these concerns that creates leverage: it is all four combined.** Remove any one, and the others may remain capturable. When all four sit on a public substrate, there is no host to enclose.

This is not a bug in any particular plugin architecture. It is the shape of the pattern itself. Every plugin architecture I have seen, regardless of how well-intentioned, how open-source, how liberally licensed, has this shape. The question is what to put in its place.

---

## The Political Argument: Enclosure, IP, and the Escape Hatch

### Rent-Seeking

There is a name for what platforms do: rent-seeking. They create a toll gate and extract value from traffic they did not generate. The plugin ecosystem creates the value; the platform captures it. This is not unique to software: it is the same dynamic as landlordism, as patent trolls, as financial intermediaries who insert themselves between producer and consumer without adding value to either.

Software architecture can be anti-rent-seeking by design. The Unix philosophy did this for decades without naming it explicitly. The reason Unix tools resisted enclosure is structural: **stdin and stdout are too generic to own, and each tool is too thin to gate.** The interface is text on a stream. Nobody holds a copyright on a newline. Composition happens in the user's shell, in the user's head, and those things cannot be enclosed.

### IP as the Mechanism of Enclosure

A plugin contract is not merely an API: it is a copyrightable artifact. The *Oracle v. Google* litigation made this explicit at the highest level: the question of whether an API can be owned was contested all the way to the Supreme Court, which found in Google's favour on fair use grounds but did not deny that APIs were copyrightable subject matter. The terrain is contested. Plugin interfaces, event models, callback signatures: these can be asserted as intellectual property.

**The ecosystem built on top of them compounds the lock-in.** Network effects accumulate on the platform, not on the tools. The more extensions exist, the more valuable the platform; the more valuable the platform, the harder it is to leave. This is the enclosure flywheel.

Note that open-source implementations do not resolve this. A platform whose host is MIT-licensed but whose plugin contract is controlled by a single foundation has not escaped the enclosure logic. **The contract is the thing every extension must conform to.** Whoever controls the contract controls the ecosystem, regardless of what license governs the host's implementation.

### Licensing Theatre

A second-order move has emerged in the last several years: platforms pre-emptively defending against the composable alternative through licensing. SSPL (Server Side Public License), BUSL (Business Source License), and a growing family of "source-available" licenses permit reading the code but not running it as a competing service. The architecture and the license work together: the plugin contract ties the ecosystem to the host; the license prevents anyone from forking the host itself. Enclosure by legal instrument, enabled by architectural centralisation.

These licenses are sometimes presented as protecting open-source contributors from exploitation by cloud vendors. That framing is not entirely dishonest; the exploitation is real. But the structural effect is the same: **the composition of architecture and license produces a captive ecosystem.** The parts are visible but not free.

### The Machinery Analogy: The Escape Hatch

Consider a fully assembled machine: an industrial lathe, a press, a CNC router. The assembled machine can be branded, priced, access-gated (software license, subscription, activation code), and obsoleted (vendor ends support, announces end-of-life, forces a costly upgrade). You are a customer of the machine, on the machine's terms.

The individual parts, bearings, spindles, fasteners and motors, are too generic to own. Nobody holds a meaningful patent on a bolt. These parts are commodities because they exist at a level of abstraction below any specific application. The fastener standard (M6 thread, ISO 3506) belongs to everyone, which is why a bolt from one manufacturer fits a nut from another.

**If you have the expertise to assemble, you build something tailored, not captive.** The vendor has no leverage because there is no vendor, only parts and your own labour.

In software: each small tool is a part, too thin to gate and too generic to brand. The interface conventions, text streams, public schemas, open protocols, are the fastener standard. Nobody holds a copyright on JSON. The composition is the machine you build, and it lives in your shell history, your Makefile, your Datalog rules. It is yours.

The escape hatch is: acquire generic parts, compose them yourself, own the composition.

---

## The Composable Alternative: Unbundling the Platform

If the problem is bundling four concerns under one host-owned contract, the solution is to place each concern on a substrate that nobody owns.

| Concern | Plugin approach | Composable analog |
|---|---|---|
| Discovery | Host-owned registry or manifest | `$PATH` convention, prefix naming, content-addressed hashes |
| Lifecycle | Host calls init/shutdown | `fork/exec` (the OS handles it) |
| Interface | Host-defined API or callback table | stdin/stdout JSONL, public schema, open protocol |
| Composition | Host dispatch loop | Shell pipeline, Makefile, Datalog rules (user owns it) |

When each concern sits on a public substrate, there is no host to enclose. The ecosystem does not exist as a capturable thing; there are just tools that speak a common language, and the common language belongs to everyone.

**Discovery** by `$PATH` convention means anyone can ship a tool that participates in the system by naming it appropriately. The `git-*` subcommand pattern is the canonical example: `git foo` finds `git-foo` on `$PATH`. Git does not own the namespace; it just looks. A prefix naming convention achieves the same thing: discovery without a registry owner.

**Lifecycle** by `fork/exec` means the OS is the runtime. There is no `register_plugin()` ceremony, no `init()` callback, no host-controlled process table. The tool runs, produces output, and exits. This is what CGI was, and it is what LSP is, and it handles every lifecycle concern that actually matters.

**Interface** on a public substrate means the contract is a format or a protocol, not an API. JSON on stdin/stdout is a contract that nobody owns. A published schema, such as JSON Schema, Protobuf with public `.proto` files, or Datalog facts with a public spec, is a contract that can be independently implemented by any producer or consumer. Formally publishing an interface (with a specification, a reference implementation, and an open license) collapses the IP surface. **This is the architectural equivalent of defensive publication in patent law.** Once the interface is public and documented, asserting ownership over conformance to it becomes untenable.

**Composition logic** in a shell pipeline or a Makefile is composition logic that lives in the user's hands, version-controlled in the user's repository. The user decides what fires when. The user owns the assembly.

### A Concrete Illustration

`java-bytecode-tools` is a static analysis and visualisation toolkit for Java bytecode. Its `PHILOSOPHY.md` states directly: "toolkit of mechanisms, not workflows." The architecture follows the pattern above:

- **Discovery**: prefix naming convention: `ftrace-*` tools for CFG trace analysis, `ddg-*` for data dependency graphs, `calltree-*` for call tree rendering. No registry. No manifest. A tool exists if it is on the path and speaks the right format.
- **Lifecycle**: every tool is a standalone Unix filter. `fork/exec`. The OS is the runtime.
- **Interface**: JSON on stdin/stdout throughout. The Java CLI writes JSON to stdout; the Python tools read from stdin or a file argument and write to stdout. The schema is documented. Nothing is proprietary.
- **Composition**: the pipeline stages are explicit: `xtrace → ftrace-expand-refs → ftrace-inter-slice → ftrace-semantic → ftrace-semantic-to-dot`. This lives in shell scripts and task runner configurations. Any stage can be substituted, any intermediate artifact can be cached, the pipeline can be branched, without asking permission from anyone.

There is no plugin market. There is no extension store. There is no host. There is a public schema and a naming convention, and both are older than any company that might want to enclose them.

---

## IPC as Substrate: All the Patterns Already Exist

A common objection to the composable approach is that real systems require communication patterns beyond simple pipelines: pub/sub, event-driven dispatch, fan-out, shared state. The implicit claim is that these patterns require a framework, and a framework requires an owner.

This is false. Every communication pattern that appears in a plugin architecture is available in raw POSIX primitives:

- **Request/response**: subprocess + stdio. One process writes a request; the other reads it and writes a response. This is what CGI was in 1993; it is what LSP is today.
- **Streaming pipelines**: anonymous pipes with OS-managed backpressure. Writes block when the pipe buffer fills. The OS is the flow controller.
- **Pub/sub**: append-only files with `inotify`, or a thin socket broker that fans out. A `tail -f` on a log file is pub/sub, with the file as the topic and the filesystem as the broker.
- **Fan-out/fan-in**: `xargs -P`, `make -j`, or `select`/`epoll` on multiple sockets. The scatter-gather pattern is available in every Unix shell.
- **Event sourcing**: an append-only log. Consumers read at their own pace and can rewind. This is Git's object store, syslog, and your shell history; all event sourcing without a framework.
- **Coordination**: `flock` for advisory locking, atomic `rename` for atomic publish, `mkdir` as a mutex (atomic on POSIX filesystems).
- **Capability passing**: Unix domain sockets permit passing file descriptors between processes. A process can open a resource and hand the open handle to another process, without the receiver having ambient permission to open it themselves. This is object-capability security on Unix.
- **Shared state**: SQLite or LMDB accessed via memory-mapped files, a file format nobody owns, with bindings in every language.

**The kernel is the only framework you cannot avoid, and it ships every pattern you need.** Everything above it is opinion. And opinion is where vendor contracts live.

---

## Living Proof: Five Projects, Five Enclosure Vectors Eliminated

The argument above is structural. The following five projects are empirical evidence that the structure holds in practice. Each eliminates a different enclosure vector.

### LSP: Eliminating Vendor Bundling

Before the Language Server Protocol, the problem of adding language support to editors was an N×M problem: N editors multiplied by M languages, each integration proprietary to the editor vendor. Microsoft's Visual Studio had Roslyn. JetBrains had PSI. Sublime had its own plugin model. Switching editors meant losing all language tooling.

LSP (first shipped with VS Code in 2016) defines a JSON-RPC protocol between a text editor (the client) and a language server (a separate process). The server is separate from the editor: `fork/exec`, stdio transport by default, public JSON-RPC wire format. The spec is published under Creative Commons.

The consequence: rust-analyzer works equally well in Neovim, Emacs, Helix, and VS Code. Microsoft has no technical mechanism to degrade it. JetBrains, historically the strongest proprietary language tooling vendor, was forced to implement LSP support because developers demanded it. The Helix editor has no plugin system at all; LSP is its only extensibility surface. **The N×M proprietary market collapsed because a public protocol replaced the proprietary contracts.**

### SQLite: Eliminating Server and Copyright Control

SQLite is public domain, not MIT, not Apache 2.0, but actual public domain. There is no copyright holder to threaten, no CLA to revoke, no license to change. It is serverless: a library, not a daemon. The database is a file. You link SQLite into your application; it reads and writes bytes. The single-file format is fully documented in a published specification, and that specification is stable by guarantee.

Richard Hipp: "SQLite is designed to replace `fopen()`, not Oracle."

This framing is architecturally significant. A file format that nobody owns replaces a file format (raw bytes) that nobody owns. There is no "managed SQLite" service to sell because there is no server to manage. Oracle, Microsoft, and Amazon all ship products that sit beside or above SQLite without being able to enclose it. The three standard enclosure vectors, copyright, server, proprietary wire protocol, are absent simultaneously.

### Nix: Eliminating Registry Control

Every package in Nix is identified by a cryptographic hash of its inputs: source code, build flags, dependencies, build script. The store path `/nix/store/abc123...-openssl-3.0.7` encodes the entire causal history of the artifact. There is no mutable global namespace to control, no registry whose owner can revoke or alter a package identity.

When the Nix Foundation went through governance conflicts in 2023–2024, contributors who disagreed forked to form the Lix project. The fork produced a fully compatible implementation in weeks. **This is only possible because the format, derivation files and store paths, is public data, not an API controlled by a foundation.** You cannot hold a governance controversy over a hash.

### SourceHut: Eliminating Platform Network Effects

SourceHut's code review and patch submission workflow is built on plain-text email and `git format-patch`. A patch is a file sent to a mailing list. Review is a reply. There is no proprietary API, no OAuth token to revoke, no webhook to disable, no social graph to create switching costs.

Drew DeVault, its founder: "GitHub is a platform; email is a protocol. If GitHub disappears, your project's history disappears with it. If the mailing list host disappears, you point your MX record somewhere else and nothing is lost."

The Linux kernel, Git itself, and many major open source projects have always used the email-patch model. **These workflows have never been captured by any platform because they predate and bypass platforms entirely.**

### Gemini: Eliminating the Monetisation Surface

The Gemini protocol specification is approximately three thousand words and explicitly frozen at that scope. It excludes JavaScript, cookies, inline images beyond a specific mechanism, forms, user tracking, and any execution surface on the client. These exclusions are not gaps; they are design decisions.

The specification states directly: "Gemini is deliberately designed so that it's impossible for a party with a financial interest in user attention to build the kind of engagement-maximising dark patterns that have come to dominate the web."

**Gemini is not resistant to monetisation by policy. It is structurally incapable of it.** There is no execution surface. There is no state mechanism. There is no way to run ad-tech on a Gemini capsule because the protocol provides no substrate for it to run on.

### The Pattern

| Project | Enclosure vector eliminated | How |
|---|---|---|
| LSP | Vendor bundling of language tooling | Separate process + public JSON-RPC protocol breaks N×M |
| SQLite | Server + copyright control | Public domain + fully documented single-file format |
| Nix | Registry/namespace control | Content-addressing replaces mutable names |
| SourceHut | Platform network effects | Email/protocol replaces API/social graph |
| Gemini | Monetisation surface | Spec permanently excludes execution + state |

The common structural move: **remove the centralisation point that enclosure requires.** You cannot enclose a hash. You cannot enclose SMTP. You cannot enclose a protocol that structurally cannot run your code. You cannot enclose a separate process speaking a public schema. You cannot enclose a file format you do not own.

---

## Intellectual Lineage

This argument has predecessors, and it is worth naming them.

The Unix tradition is the oldest technical articulation. Eric Raymond's *The Art of Unix Programming* states the design principles clearly, though the political content, that composition is anti-enclosure, is rarely named explicitly. The reason Unix tools resisted capture for decades was structural: the interface was too generic to own, and each tool was too thin to gate. The composition lived in the user's shell history, which has never been anyone's IP.

Ivan Illich's *Tools for Conviviality* (1973) provides the more general vocabulary. Illich argued that tools have a threshold beyond which they stop serving users and start requiring users to serve them. A hammer is convivial: it does exactly what the user makes it do, and the user's skill determines the outcome. An industrial lathe, past a certain scale, requires a trained operator who serves the machine's requirements. **Plugin architectures cross Illich's threshold.** The user serves the platform: they learn its event model, conform to its API, adapt their extension to its lifecycle. The platform does not adapt to them.

Contemporary framings are more pointed. Cory Doctorow's concept of adversarial interoperability, building tools that work with existing platforms without permission, is the legal-strategy version of this argument. Mike Masnick's *Protocols, Not Platforms* is its most precise short statement. The Ink & Switch local-first essay (Kleppmann et al., 2019) makes the technical case for architectures where computation and data live on the user's machine, collapsing the metering surface that SaaS enclosure requires. The permacomputing and small-web communities are building practice around the same ideas, under different names.

None of these sources constitute a single movement. The people writing about this rarely identify with one another. But the structural insight recurs across them: **the architecture is the politics.** You cannot separate the technical choice from its political consequence.

---

## Honest Limits

This argument has real limits and it is worth being honest about them.

Composition-heavy architecture puts a tax on non-practitioners. The machinery analogy holds here too: assembling from parts requires knowing what the parts are, how they fit together, and what the result should look like. Most people do not want to be mechanics. They want a car that starts when they turn the key.

The market answer to that tax has historically been "let a company hide the composition behind a product", which is exactly the enclosure this stance opposes. **"Hostile to monetisation" and "hostile to non-practitioners" overlap heavily.** The composable approach is appropriate for practitioners doing serious work with serious tools. It is not a universal answer to the problem of software distribution.

This is not a flaw to be engineered away. It is a genuine tradeoff, and accepting it honestly is part of what makes the argument credible. The composable stance cedes the mass market by design. The question is whether, for the work that matters to you, that is a price worth paying.

I think it is. The alternative is to build something that can be enclosed, and then watch it be enclosed.
