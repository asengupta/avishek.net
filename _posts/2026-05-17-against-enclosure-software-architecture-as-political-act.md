---
title: "Against Enclosure: Software Architecture as a Political Act"
author: avishek
usemathjax: false
mermaid: false
tags: ["Software Engineering", "Architecture", "Open Source", "Philosophy", "Unix", "Composability"]
draft: false
published: true
---

*Software architecture has politics baked into it. Choosing between a plugin-based platform and a pile of small composable tools is, among other things, choosing whether somebody can fence off the result later. This post makes the case for composability on those grounds: not because it always makes better software, but because what it makes is hard to capture.*

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

I've built a fair number of systems around plugin architectures, and I built them because they seemed like the obvious answer to extensibility. You define a host, you define a contract, and anyone who follows the contract can extend the system. It's clean and modular, and it feels like the right shape for software that's supposed to grow.

What took me longer to notice is that this shape has consequences outside the code. A plugin architecture tends to turn into a platform, and platforms are the things that get enclosed.

"Enclosure" comes from English history. Between roughly the sixteenth and eighteenth centuries, land that had been held in common (used by everyone, owned by nobody) was fenced off, privatised, and rented out. People who had farmed it for free became tenants paying to keep doing the same thing. The recipe is simple: take something shared, put a boundary around it, and charge people to cross. In software the thing being fenced isn't land, it's capability: being able to extend a system, interoperate with it, wire it up to other tools. What gets fenced is the interface, the ecosystem, the protocol, and whoever controls the contract gets to put up the fence. Aaron Swartz, in the *Guerilla Open Access Manifesto*, didn't bother with polite language for this kind of thing: he called it theft, not of anyone's private property, but of something that had belonged to everybody.

Here's how it plays out. A plugin host handles discovery (how extensions get found), lifecycle (how they're loaded and unloaded), interface (how they talk to the host), and composition (what gets called when). All four live under one contract, and the host owns it. So the host becomes indispensable, not because it does anything clever, but because it's the only thing that speaks the contract. Extensions can't talk to each other without it, users can't combine things without it, and everything the system can do goes through it.

That's a platform. Platforms make attractive targets because everything valuable is in one place: the contract, the ecosystem, the users. A company can fence off a platform in a way it can't fence off a bunch of independent tools. The plugin contract turns into IP, the ecosystem gets locked in, the host becomes a toll booth, and the plugin authors end up as tenants.

Nobody has to be acting in bad faith for this to happen. The architecture creates the incentive, and the incentive does the rest.

So what's the opposite of a platform, architecturally? That's what I've been trying to work out.

---

## What Plugin Architectures Actually Do

It helps to pull apart what a plugin architecture is doing. There are four things going on at once:

**Discovery**: the host decides how extensions are found. It might scan a directory, query a registry, or read a manifest it controls. Extensions don't find each other; the host finds them.

**Lifecycle**: the host calls `init()`, `register()`, `shutdown()` on each extension's behalf. It owns the process. Extensions exist at the host's pleasure.

**Interface**: the host defines the vocabulary. The plugin API, the callback table, the event types are all specified by the host, and every extension has to speak them. An extension written for host A won't run under host B because the vocabulary is different.

**Composition logic**: the host owns the dispatch loop. It decides what fires when, what data goes where, and what's in scope. Extensions react to events the host defines.

Bundling all four under one host-owned contract is what turns a plugin host into a platform. No single one of these gives the host leverage; it's the combination. Pull one out and the rest can still be captured. Put all four on public ground and there's no host left to capture.

This isn't a flaw in some particular plugin system; it comes with the pattern. Every plugin architecture I've worked with has looked like this, however well-meaning, open-source or liberally licensed it was. So the real question is what to use instead.

---

## The Political Argument: Enclosure, IP, and the Escape Hatch

### Rent-Seeking

Economists have a word for what platforms do: rent-seeking. They set up a toll booth and charge for traffic they didn't create. The plugin authors make the value and the platform collects it. Software didn't invent this. Landlords, patent trolls, and plenty of financial middlemen do the same thing, putting themselves between producer and consumer without adding much for either.

You can design architecture to make rent-seeking hard. Unix did this for decades without anyone really saying so out loud. Unix tools were hard to capture because stdin and stdout are too generic to own and each tool is too small to put a gate in front of. The interface is text on a stream, and nobody holds a copyright on a newline. The composition happens in the user's shell (and the user's head), and you can't fence those off either.

### IP as the Mechanism of Enclosure

A plugin contract isn't just an API; it's something you can try to copyright. *Oracle v. Google* went all the way to the Supreme Court over whether APIs can be owned. Google won on fair use, but the court sidestepped whether APIs are copyrightable in the first place, so that question is still open. Plugin interfaces, event models, callback signatures: any of these can be claimed as IP.

The ecosystem on top makes the lock-in worse. Network effects pile up on the platform, not on the individual tools. More extensions make the platform more valuable, and a more valuable platform is harder to leave. It feeds itself.

Open-source implementations don't fix this. If the host is MIT-licensed but a single foundation controls the plugin contract, you're in the same position as before. Every extension has to conform to the contract, so whoever controls the contract controls the ecosystem, whatever license the host's code happens to be under.

### Licensing Theatre

Over the last few years, platforms have also started defending themselves against the composable alternative with licenses. SSPL (Server Side Public License), BUSL (Business Source License), and a growing set of "source-available" licenses let you read the code but not run it as a competing service. The architecture and the license prop each other up: the plugin contract ties the ecosystem to the host, and the license stops anyone forking the host. It's enclosure by legal means, made possible by a centralised architecture.

These licenses are often pitched as protecting open-source maintainers from cloud vendors, and that's not entirely spin; the cloud vendors really do take advantage. But the end result is the same either way: architecture plus license gives you a captive ecosystem. You can see all the parts; you just aren't free to do much with them.

### The Machinery Analogy: The Escape Hatch

Think about a fully assembled machine: an industrial lathe, a press, a CNC router. The vendor can brand it, price it, lock it behind a license or subscription or activation code, and eventually kill it by ending support and pushing you towards an expensive upgrade. You're a customer, on their terms.

The parts (bearings, spindles, fasteners, motors) are too generic to own. Nobody holds a useful patent on a bolt. They're commodities because they sit below any particular application. Fastener standards like the M6 thread belong to everyone, which is why one company's bolt fits another company's nut.

If you know how to put the parts together, you can build something that fits what you need and that nobody else controls. No vendor has leverage over you, because there is no vendor, just parts and your own work.

Software works the same way. Each small tool is a part, too thin to gate and too generic to brand. The conventions (text streams, public schemas, open protocols) are the thread standard. Nobody holds a copyright on JSON. The composition is the machine you build, and it lives in your shell history, your Makefile, your Datalog rules. It's yours.

So the way out is: get generic parts, put them together yourself, and keep what you build.

---

## The Composable Alternative: Unbundling the Platform

If the problem is four concerns bundled under one owned contract, the fix is to put each of them somewhere nobody owns.

| Concern | Plugin approach | Composable analog |
|---|---|---|
| Discovery | Host-owned registry or manifest | `$PATH` convention, prefix naming, content-addressed hashes |
| Lifecycle | Host calls init/shutdown | `fork/exec` (the OS handles it) |
| Interface | Host-defined API or callback table | stdin/stdout JSONL, public schema, open protocol |
| Composition | Host dispatch loop | Shell pipeline, Makefile, Datalog rules (user owns it) |

Once each concern is on public ground, there's no host to capture. There's no "ecosystem" as a single thing someone could take over, just tools that speak a shared language, and the language doesn't belong to anyone.

**Discovery** by `$PATH` convention means anyone can ship a tool that joins in just by naming it right. The `git-*` subcommand pattern is the best-known example: `git foo` finds `git-foo` on `$PATH`. Git doesn't own the namespace; it just looks. A prefix naming convention does the same job: discovery without anyone owning a registry.

**Lifecycle** by `fork/exec` means the OS is the runtime. There's no `register_plugin()` ritual, no `init()` callback, no host-controlled process table. The tool runs, produces output, and exits. That's how CGI worked, it's how LSP works, and it covers every lifecycle concern that matters in practice.

**Interface** on public ground means the contract is a format or a protocol, not an API. JSON on stdin/stdout is a contract nobody owns. A published schema (JSON Schema, Protobuf with public `.proto` files, Datalog facts with a public spec) can be implemented independently by anyone producing or consuming it. Publishing an interface properly, with a spec, a reference implementation and an open license, leaves very little to claim as IP. It's a bit like defensive publication in patent law: once the interface is public and documented, it's very hard for anyone to say they own conformance to it.

**Composition logic** in a shell pipeline or a Makefile lives with the user, version-controlled in the user's own repository. The user decides what runs when. The user owns the assembly.

### A Concrete Illustration

`java-bytecode-tools` is a static analysis and visualisation toolkit for Java bytecode. Its `PHILOSOPHY.md` puts it plainly: "toolkit of mechanisms, not workflows." The architecture follows the pattern above:

- **Discovery**: a prefix naming convention: `ftrace-*` tools for CFG trace analysis, `ddg-*` for data dependency graphs, `calltree-*` for call tree rendering. No registry, no manifest. A tool exists if it's on the path and speaks the right format.
- **Lifecycle**: every tool is a standalone Unix filter. `fork/exec`. The OS is the runtime.
- **Interface**: JSON on stdin/stdout throughout. The Java CLI writes JSON to stdout; the Python tools read from stdin or a file argument and write to stdout. The schema is documented, and nothing about it is proprietary.
- **Composition**: the pipeline stages are spelled out: `xtrace → ftrace-expand-refs → ftrace-inter-slice → ftrace-semantic → ftrace-semantic-to-dot`. This lives in shell scripts and task runner configs. You can swap out any stage, cache any intermediate output, or branch the pipeline, without asking anyone's permission.

There's no plugin marketplace, no extension store, no host. There's a documented schema and a naming convention, and neither gives anybody anything to take over.

---

## IPC as Substrate: All the Patterns Already Exist

A common objection is that real systems need more than simple pipelines: pub/sub, event-driven dispatch, fan-out, shared state. The assumption is that those need a framework, and a framework needs an owner.

They don't. Every communication pattern you'd find in a plugin architecture is already there in plain POSIX:

- **Request/response**: subprocess + stdio. One process writes a request; the other reads it and writes a response. That's what CGI was in 1993, and it's what LSP is today.
- **Streaming pipelines**: anonymous pipes with backpressure handled by the OS. Writes block when the pipe buffer fills, so the OS does the flow control.
- **Pub/sub**: append-only files with `inotify`, or a thin socket broker that fans messages out. `tail -f` on a log file is pub/sub, with the file as the topic and the filesystem as the broker.
- **Fan-out/fan-in**: `xargs -P`, `make -j`, or `select`/`epoll` on several sockets. Scatter-gather is available in every Unix shell.
- **Event sourcing**: an append-only log. Consumers read at their own pace and can rewind. Git's object store, syslog, and your shell history are all event sourcing without a framework.
- **Coordination**: `flock` for advisory locking, atomic `rename` for atomic publish, `mkdir` as a mutex (atomic on POSIX filesystems).
- **Capability passing**: Unix domain sockets can pass file descriptors between processes. One process opens a resource and hands the open handle to another, which never needed permission to open it itself. That's object-capability security on Unix.
- **Shared state**: SQLite or LMDB over memory-mapped files: a file format nobody owns, with bindings in every language.

The kernel is the one framework you can't avoid, and it already ships all of this. Everything above it is somebody's opinion, and opinions are where vendor contracts tend to creep in.

---

## Living Proof: Five Projects, Five Enclosure Vectors Eliminated

So far this has been fairly abstract. Here are five real projects where it works in practice, each one closing off a different route to enclosure.

### LSP: Eliminating Vendor Bundling

Before the Language Server Protocol, adding language support to editors was an N×M problem: N editors times M languages, with every integration tied to the editor vendor. Microsoft's Visual Studio had Roslyn. JetBrains had PSI. Sublime had its own plugin model. Switching editors meant losing all your language tooling.

LSP (first shipped with VS Code in 2016) is a JSON-RPC protocol between a text editor (the client) and a language server (a separate process). The server runs separately from the editor: `fork/exec`, stdio transport by default, a public JSON-RPC wire format. The spec is published under Creative Commons.

The upshot: rust-analyzer works just as well in Neovim, Emacs, Helix, and VS Code, and Microsoft has no technical way to make it worse elsewhere. JetBrains, historically the strongest proprietary language-tooling vendor, ended up adding LSP support because developers wanted it. The Helix editor has no plugin system at all; LSP is the only way to extend it. A public protocol replaced the proprietary integrations, and the N×M market fell apart.

### SQLite: Eliminating Server and Copyright Control

SQLite is in the public domain. Not MIT, not Apache 2.0: actually public domain. There's no copyright holder to make threats, no CLA to revoke, no license to change. It has no server either; it's a library, not a daemon. The database is a file. You link SQLite into your application and it reads and writes bytes. The file format is fully documented in a published spec that's guaranteed to stay stable.

SQLite's own [When To Use](https://www.sqlite.org/whentouse.html) page puts it like this: "SQLite does not compete with client/server databases. SQLite competes with `fopen()`."

SQLite replaces raw bytes on disk, which nobody owns, with a file format that nobody owns either. There's no "managed SQLite" to sell, because there's no server to manage. Oracle, Microsoft, and Amazon all ship products that sit next to or on top of SQLite without being able to take it over. The usual three routes to enclosure (copyright, a server, a proprietary wire protocol) are all missing at once.

### Nix: Eliminating Registry Control

Every package in Nix is identified by a cryptographic hash of its inputs: source code, build flags, dependencies, build script. A store path like `/nix/store/abc123...-openssl-3.0.7` encodes the artifact's whole build history. There's no mutable global namespace to control, and no registry whose owner can revoke a package or change what a name points to.

When the Nix Foundation went through its governance fights in 2023–2024, contributors who disagreed forked off to form the Lix project, and had a fully compatible implementation within weeks. That only worked because the formats (derivation files, store paths) are public data rather than an API a foundation controls. You can't have a governance fight over a hash.

### SourceHut: Eliminating Platform Network Effects

SourceHut's code review and patch workflow runs on plain-text email and `git format-patch`. A patch is a file sent to a mailing list, and a review is a reply. There's no proprietary API, no OAuth token to revoke, no webhook to switch off, no social graph that makes it painful to leave.

Drew DeVault, its founder, makes the case in [Git is already federated & decentralized](https://drewdevault.com/blog/Git-is-already-distributed/) (2018): email is "already standardized", "decentralized and federated, and it's *already* integrated with git." And if the host goes away, nothing is lost: "If everything is being conducted over email instead, all of the data already lives in the user's inbox. Freely available tools can take their mail spool and publish a new archive if our services go down."

The Linux kernel, Git itself, and plenty of other big open source projects have always worked this way. No platform has ever captured these workflows, because they're older than platforms and go around them.

### Gemini: Eliminating the Monetisation Surface

The Gemini protocol spec is about three thousand words, and it's deliberately frozen at that size. It leaves out JavaScript, cookies, inline images (beyond one specific mechanism), forms, user tracking, and any way of running code on the client. These are deliberate omissions, not missing features.

The [Gemini FAQ](https://geminiprotocol.net/docs/faq.gmi) is upfront about the consequence: "Because by default it only presents users with text, and because authors have no way to control the font or size or colour of that text, Gemini is naturally resistant to advertising and marketing."

So Gemini doesn't avoid monetisation as a matter of policy; it just doesn't give monetisation anything to work with. There's no code execution and no state, so there's nowhere for ad-tech to run.

### The Pattern

| Project | Enclosure vector eliminated | How |
|---|---|---|
| LSP | Vendor bundling of language tooling | Separate process + public JSON-RPC protocol breaks N×M |
| SQLite | Server + copyright control | Public domain + fully documented single-file format |
| Nix | Registry/namespace control | Content-addressing replaces mutable names |
| SourceHut | Platform network effects | Email/protocol replaces API/social graph |
| Gemini | Monetisation surface | Spec permanently excludes execution + state |

What these have in common is that each one gets rid of the central point that enclosure depends on. You can't fence off a hash, or SMTP, or a protocol that won't run your code, or a separate process speaking a public schema, or a file format nobody owns.

---

## Intellectual Lineage

None of this is new; plenty of people have made versions of this argument before me.

The Unix tradition got there first. Eric Raymond's *The Art of Unix Programming* lays out the design principles well, though it mostly doesn't spell out the political side: that composition works against enclosure. As I said earlier, Unix tools stayed hard to capture because the interface was too generic to own and each tool was too small to gate. The composition lived in people's shell histories, which have never been anybody's IP.

Ivan Illich's *Tools for Conviviality* (1973) gives a more general way of talking about it. Illich argued that past a certain point, tools stop serving the people using them and start making those people serve the tool. A hammer is "convivial": it does what you make it do, and how well things go depends on your skill. An industrial lathe, past a certain scale, needs a trained operator who works to the machine's requirements. I'd put plugin architectures on the wrong side of that line. You end up working for the platform: learning its event model, conforming to its API, fitting your extension to its lifecycle. It doesn't adapt to you.

Cory Doctorow's "adversarial interoperability" (building things that work with existing platforms without asking their permission) is roughly the legal-strategy version of this argument. Mike Masnick's *Protocols, Not Platforms* is probably the best short version. Ink & Switch's local-first essay (Kleppmann et al., 2019) makes the technical case for keeping data and computation on the user's machine, which takes away the metering point SaaS relies on. The permacomputing and small-web people are putting a lot of the same ideas into practice, under other names.

These people aren't one movement, and most of them wouldn't think of themselves as being on the same side. But the same idea keeps turning up across all of them: the architecture is the politics. You can't really pull the technical choice apart from what it does politically.

---

## Honest Limits

This approach has real costs, and most of them land on people who aren't practitioners. Back to the machinery analogy: building from parts means knowing what the parts are, how they fit, and what you're trying to end up with. Most people don't want to be mechanics. They want a car that starts when they turn the key.

Historically, the market's answer has been to let a company hide the composition behind a product, which is exactly the enclosure I've been arguing against. "Hard to monetise" and "hard for non-experts to use" overlap a lot. Composable tools suit people doing serious work who are willing to learn their tools. They aren't a general answer to how software should reach everyone.

I don't think you can engineer that away; it's a real tradeoff. Going composable means giving up the mass market. Whether that's worth it depends on what work you care about.

For me it is. The alternative is building something that can be enclosed, and then watching it happen.
