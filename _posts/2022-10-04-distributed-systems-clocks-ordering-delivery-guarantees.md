---
title: "Ordering, Clocks, and Delivery Guarantees"
author: avishek
usemathjax: false
tags: ["Distributed Systems"]
draft: true
---

We discuss the **message-passing model** of computation, which will be used to reason about distributed computing algorithms. The mathematical formalism will be introduced in stages, as it is useful to gain an intuitive understanding of the model first.

Bolting on the mathematical notation becomes easier after this understanding.

The message-passing model is usually represented using spacetime diagrams or Lamport diagrams, and take the form of processes evolving over the time axis over a series of events, represented as fixed points in time.

- Lamport diagrams
  - Internal Events
  - Send and Receive Events
  - Happened-Before Relation and Concurrent Events
  - Review of Set Theory
    - Relations
    - Reflexive and Irreflexive Relations
    - Symmetric, Antisymmetric, and Assymmetric Relations
    - Transitive Relations
    - Reflexive Partial Order
    - Irreflexive Partial Order (Posets)
    - Happened-Before Relation as an Irreflexive Partial Order
  - Ordering Events: Lamport Clocks and Limitations
  - Improving Ordering: Vector Clocks
  - Delivery Events
  - Types of Delivery
    - FIFO Delivery (with Violations)
    - Causal Delivery (with Violations)
    - Totally Order Delivery (with Violations)
