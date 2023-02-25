---
title: "Economic Factors in Software Architectural Decisions"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Software Engineering Economics"]
draft: false
---

This article continues from where [Every Software Engineer is an Accountant]({% post_url 2023-02-04-every-engineer-is-an-accountant %}) left off. I have had feedback that I need to make my posts on these a little more explainable; I will attempt to do that here.

The posts in this series of **Software Engineering Economics** are, in order:

- [Every Software Engineer is an Economist]({% post_url 2023-01-22-every-engineer-is-an-economist %})
- [Every Software Engineer is an Accountant]({% post_url 2023-02-04-every-engineer-is-an-accountant %})
- [Economic Factors in Software Architectural Decisions]({% post_url 2023-02-20-economic-factors-in-software-architectural-decisions %}) (this one)

## Introduction

In previous articles, we have spoken of examples of doing **NPV analysis for architectural and technical decisions**, to determine viability and bubble up the tangible value of these intangible decisions to senior stakeholders. However, apart from examples, we have mostly glossed over what sort of economic factors should be considered when assigning value to these decisions. As it turns out, this is not hard: **these economic factors are very closely tied to the factors we use to judge the technical benefits and costs of these decisions**. We mostly need to tie them to actual financial value, in terms of hours, and ultimately, money. Thus, we list [tables](#catalogue-of-economic-factors) containing economic factors to consider for common architectural decisions.

In parallel, we need to measure these **costs and benefits relative to some baseline**. Thus, we propose certain [baselines](#baselines) to judge common architectural decisions against.

There is also an important point we implicitly assumed: that implementing these decisions in code will automatically give us this value. However, there needs to be some arbiter of whether this value was *actually* delivered or not. Architectural decisions require effort, and the decision of whether that concrete effort achieved everything we set out to do, must be supported by something similarly concrete. We argue that [**feature-level tests as arbiters of value**](#tests-as-markers-of-economic-value); given the fact that almost all teams use feature tests to verify that the software is fit for purpose, this seems to be a natural place to assign economic value to. We use the term **"feature tests"** rather loosely; these could be testing functionality, as well as verifying performance of these features. Any test that can demonstrate an aspect of the solution to which the business has assigned explicit value to, falls into this category of "feature test".

## Baselines

Decisions need to be taken at multiple levels of abstraction of a codebase. Some examples are:

- Should I **rename** this variable or not?
- Should I **refactor** this piece of code into its own function or not?
- Should I apply this **design pattern** (e.g., factory) or not?
- Should I implement **tracing** or not? (You really should :-)
- Should I use a **plugin architecture** or not?
- Should I break this out into its own **microservice** or not?
- Should I use **Kafka** or **Google PubSub**?
- ...and so on

**To enumerate the costs and benefits of these decisions, we need to calculate them relative to some baseline implementation.** This baseline implementation may exist already or not, but it serves as a useful yardstick to drive out all the benefits that would occur if the decision was taken, or all the future problems which would occur if the decision was not taken (which would ultimately translate to financial losses), or the costs involved in implementing this decision.

As programmers, we make several lower level decisions over the course of a programming session with an intuitive understanding of the benefits of taking a particular decision (renaming a variable to be more descriptive, ultimately helps in readability for others -- current and future -- working on the codebase). This is fine; we don't really need to evaluate the economic value for every small decision where the cost to make the change is vanishingly small, thanks to modern refactoring tools.

**The decisions start to matter at higher levels of abstraction: at the architecture level, at the service level, and so on.** Changes at those macro levels occur relatively less often, and corresponding changes require greater effort; new deployments, additional dependency fixups, etc. Decisions at these levels thus benefit the most from explicit economic evaluations. These are the places where a baseline would help.

We thus propose the following baselines for some frequently occurring decisions:

- **Monoliths** as baseline when considering **microservices**
- **Hardcoded plugins** as baseline when considering **microkernel architectures**
- **Peer-to-peer invocations** as baseline when considering **event-driven architectures** / **event buses**
- **Hardcoded components** as baseline when considering **pipe and filter patterns**
- **RDBMS** (something like PostgreSQL) as baseline when considering **NoSQL databases**

Each of the above decisions has one or more expansion factors: these are the factors that make taking the decision potentially worthwhile. For example, if there was no need for future plugins to extend or add new functionality, there would be no need for a microkernel architecture; the number of future extensions is thus a expansion factor for this decision. If the list of components in a processing pipeline did not change at all, there would be no need of a pipe and filter pattern; the future configurability of components is the expansion factor for this decision.

It is also important to note that the above decisions are not exclusive. A microservice may encapsulate a microkernel, parts of a pipe-and-filter architecture might involve invoking microservices, and so on.

## Catalogue of Economic Factors

In this section, we present a set of tables summarising sets of economic factors to consider when making some common architectural decisions. **The lists of factors are not complete: expect changes as we add more over time.** Nevertheless, these should get you started on making your decisions.

**Notation Alert:** The **'+'** symbols represent potential economic benefits; the **'-'** symbols represent potential economic downsides.

| Dimension      | Microservices with Monolith Baseline                                                                                                                                                                                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Deployment** | - What are the savings in development/deployment time when services are deployed independently?<br/>- What is the effort in building pipelines for separate deployments?<br/>- What is the cost of building reusable provisioning scripts?                                                                              |
| **Monitoring** | - What are the costs of setting up dashboards, alerts, and monitors for one microservice? For N microservices?                                                                                                                                                                                                          |
| **Tracing**    | - What are the costs of setting up standard tracing integrations across microservices?<br/>- What are the costs of maintaining traceability across a heterogenous chain, part of which might be legacy?<br/>- What time losses could occur when tracing issues across services if tracing is not uniformly implemented? |
| **Resources**  | - What is the cost of additional cloud compute and DB resources will be needed if each microservice needs to deploy and potentially scale independently?<br/>- Which services need to reserve capacity vs. which services have predictable load?                                                                        |
| **Downtime**   | - What is the cost of building circuit breaker/throttling infrastructure for multiple services?<br/>- What is the cost of building caching layers across services if services need to be available?<br/>+ What are the benefits in terms of uptime when failures are localised to specific microservices?               |
| **Latency**    | - What is the loss in profits (if applicable) if a certain latency threshold is not met?<br/>- What is the cost of reducing latency to acceptable levels so (caching, duplication of data, etc.) that latency is below this threshold?                                                                                  |
| **Scaling**    | + What is the expected opportunity loss if the monolith cannot be scaled beyond a certain point?<br/>- What is the cost of having to scale X microservice along with corresponding components like databases, downstream microservices, etc.?                                                                           |
| **Option Premium** | + What is the cost of building a modular monolith to take advantage of migrating to microservices later?                                                                                                                                                                                                                |

| Dimension | Microkernel with Hardcoded Components Baseline                                                                                            |
|--|-------------------------------------------------------------------------------------------------------------------------------------------|
| **Future Functionality** | + What is the cost savings of adding substitute/added functionality with standard plugin interfaces?         |
| **Error Handling / Failure Scenarios** | + What is the cost savings of not having to rewrite common/standard error handling scenarios?  |
| **Static/Dynamic Binding** | + What are the cost savings of being able to swap out plugin implementations at compile time/runtime?      |
| **Plugin Testing** | + What are the cost savings of being able to test plugins independently?  |

| Dimension         | Event-Driven with Peer-to-Peer Baseline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Future Consumers**  | + What are the cost savings of being able to add additional consumers without rewiring direct invocation?<br/>+ What are the cost savings of being able to test future consumers independently using synthetic events?<br/>- What is the cost of having to maintain and evolve backward-compatible event schemas?                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Architecture**      | - What is the cost of having to build orchestrators or choreographing facilities?<br/>- What is the cost (if any) of having to deal with potential incoming out-of-order events?<br/>- What is the cost of using a product to facilitate these interactions?<br/>- What is the cost of having to build facilities to persist states in case multiple events need to be received to reconstruct a domain entity?<br/>- What is the cost of building caching to rebuild your store if this is an event-sourced system?<br/>- What is the cost of setting up periodic compaction of historical events, if this is an event-sourced system?<br/>What is the cost of separating and maintaining read and write schemas, if this is a CQRS system? |
| **Tracing**           | - What is the cost of having to reconstruct fault trees from event traces?<br/>- What is the cost of building infrastructure to propagate tracing information across separate processes (if applicable)?<br/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Failure Scenarios** | - What is the cost of setting up additional infrastructure to handle / retry in the case of failure scenarios?<br/>- What is the cost of performing event replays in the middle of a event chain?<br/>- What is the cost of building in explicit event flows for rollbacks in an event chain?<br/>What is the additional cost of building detection of events lost in transit and possibly compensating for incomplete event chains?                                                                                                                                                                                                                                                                                                         |
| **Evolution**         | + What are the cost advantages in terms of adding/removing consumers without modifying sourcing events? + What are the potential future cost savings gained by allowing replacement of the system by strangulation?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Performance**       | - What is the potential opportunity loss of higher latencies of certain performance-sensitive operations exceeding acceptable SLAs?<br/>- What is the cost of any architectural changes to optimise reads and writes (e.g., CQRS)?<br/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

| Dimension                    | Pipe and Filter with Hardcoded Components Baseline                                                                                                                                                                                                                                            |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Future Reconfiguration**       | + What are the cost savings of being able to add/modify/remove components to the pipeline without having to modify the underlying infrastructure?                                                                                                                                             |
| **Monitoring**                   | - What is the cost of having to set up monitoring for each individual data processing step?<br/>- What is the cost of having to aggregate this at an enterprise level (like federated Prometheus, for example)?                                                                               |
| **Tracing**                      | - What is the cost of having to set up extra tracing to trace data flow in error/diagnosis scenarios?                                                                                                                                                                                         |
| **Stream Processing complexity** | - What is the cost of configuring the system to handle complex dependencies between streaming data events (things like streaming joins, out of order events, etc.)?                                                                                                                           |
| **Failure Scenarios**            | - What is the cost of setting up additional infrastructure to handle / retry in the case of failure scenarios?<br/>- What is the cost of performing event replays in the middle of a event chain?<br/>- What is the cost of building in explicit event flows for rollbacks in an event chain? |

| Dimension                                                                 | NoSQL with RDBMS Baseline                                                                                                                                                                                                                                                                                                |
|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Constraints and References**                                                | - What is the cost of having to define software-level constraints and reference integrity checks?<br/>+ What are the cost savings in speedups achieved because of lack of constraints?<br/>                                                                                                                              |
| **Data Schema**                                                               | - What are the costs in maintaining backward-compatible schemas?<br/>+ What are the cost savings of not having to do schema migrations with data model changes?                                                                                                                                                          |
| **[PACELC](http://www.cs.umd.edu/~abadi/papers/abadi-pacelc.pdf) guarantees** | - Are there any potential cost implications of inconsistent or slow-to-retrieve data (like time-sensitive data in financial markets) even when the system is not partitioned? If so, what is this cost?<br/>+ If there is partitioning, what are the cost benefits of having the system available (if the system is AP)? |
| **Scaling**                                                                   | + What are the cost savings of not having to scale vertically, or introduce other techniques like partitioning to keep the database performant?                                                                                                                                                                          |
| **Redundancy and Replication**                                                | + What are the cost savings of building replicas and failovers for disaster recovery over their RDBMS counterparts?<br/>+ What are the cost benefits of being able to tap into the database's event stream for change data capture?                                                                                      |

## Tests as Markers of Economic Value

We have spoken about how value can be measured, uaing the income approach, the market approach, etc. However, the question still remains: how do we connect the decisions we make (at the code level, at the architecture level, etc.) to the actual economic value.

**At the business level, the closest connection to economic value is the feature of an application.** Features are more or less atomic units of user-facing functionality (the user can be a human or another system) which can be (hopefully) deployed, enabled/disabled, and monetised independently.

Using **features as units of economic value** therefore seems plausible. The next question then arises: how do we verify that these features satisfy all the criteria to deliver this value? We propose a simple and natural answer: tests. Developers already use tests to validate every part of the system, at multiple levels of abstration, ranging from unit tests to integration tests to regression tests.

**We propose that economic value be attached to the tests which verify that features function properly.** Different aspects of the feature can be validated by different sorts of tests.

{% mermaid %}
graph TD
subgraph features[Features]
feature1[Feature]
feature2[Feature]
feature3[Feature]
end
subgraph patterns[Patterns]
pattern1[Pattern 1]
pattern2[Pattern 2]
pattern3[Pattern 3]
end
subgraph architecture[Architecture]
adr1[Architecture Decision 1]
adr2[Architecture Decision 2]
adr3[Architecture Decision 3]
end
code[Code]-->patterns
code-->architecture
patterns-->features
architecture-->features
feature1-->test1[Tests]
feature2-->test2[Tests]
feature3-->test3[Tests]
test1-->economic_value[Economic Value]
test2-->economic_value
test3-->economic_value
{% endmermaid %}

Code may be refactored into patterns; more macro-level organisational units are generally represented as architectural elements. For this discussion, patterns are treated as lower level abstractions than architectures, even though they appear at the same level in the fiagram above. Thus, patterns are largely independent of the architectures they are applied in. For example, whether you are using a microservice architecture or not does not constrain you from either using or not using a factory pattern in any of those microservices.

As an example of how value flows through this chart, consider an e-commerce payment integration system: it could have requirements which deliver value. We'd like to derive these concrete, qualitative values from these features. A sampling of these features is listed below:

- It should be able to process Visa and Mastercard credit cards.
- It should be able to process at least 100 transactions per second.
- It should be able to cancel an amount which has already been authorised if indicated.

Each of the above requirements can be verified to a certain degree of rigour through tests. What would be the economic contribution of the above requirements?

- For the **requirement of processing Visa/Mastercard credit cards**, the income streams arising from the expected number of users with these kinds of credit cards (based on demographic analysis) making purchases of amounts (determined from historical data) over some period could be a straightforward derivation of the financial value of this feature. If we expect a median of 100,000 users/month with Visa/Mastercard credit cards to buy things at the site for a median amount of $50, the projected value of this feature over 3 months would be: $$ $5000000 + \displaystyle\frac{$5000000}{(1+1.1)} + \frac{$5000000}{ {(1+1.1)}^2 } \approx $13677685 $$ (given the hypothetical discount rate is 10% per month).
- For the **requirement of processing at least 1000 transactions per second**, if the processing capability is already at or above the 1000 TPS number, the value is already counted as part of the transaction processing feature (i.e., no extra work needs to be done). If the capability is less than 1000 TPS, say 800 TPS, then the value of the feature is the opportunity loss because of not processing those extra 1000-800=200 transactions per second. The income streams arising from those 200 transactions per second performing financial transactions of some median amount over a sustained period of time could be a straightforward way to quantify the financial value of this feature.
- For the **requirement to cancel an already-authorised amount**, the cost of having support staff available to respond to customer calls for cancellation, and perform this action manually, could be one way to quantify the value of this feature. If 10 support staff personnel are paid about $4000/month, and deploying this feature could halve the support staff needs, then the value of this feature over 2 months would be $$ 5\times $4000 + \displaystyle\frac{5 \times $4000}{1+1.1} \approx $38182 $$ (obviously, we are simplifying this for the purposes of illustration).

## References
- Papers and Reports
    - [Consistency Tradeoffs in Modern Distributed Database System Design](http://www.cs.umd.edu/~abadi/papers/abadi-pacelc.pdf)
    - [Making Sense of Stream Processing](https://assets.confluent.io/m/2a60fabedb2dfbb1/original/20190307-EB-Making_Sense_of_Stream_Processing_Confluent.pdf)
