---
title: "Every Software Engineer is a Strategist"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Software Engineering Economics"]
draft: false
---

This article continues from where [Every Software Engineer is an Accountant]({% post_url 2023-02-04-every-engineer-is-an-accountant %}) left off. I have had feedback that I need to make my posts on these a little more explainable; I will attempt to do that here.

Previous posts in this series are:

- [Every Software Engineer is an Economist]({% post_url 2023-01-22-every-engineer-is-an-economist %})
- [Every Software Engineer is an Accountant]({% post_url 2023-02-04-every-engineer-is-an-accountant %})

## Introduction

Specifically, we cover the following:

- [Articulating Value: Baselines and Markers](#articulating-value-baselines-and-markers)
- [Articulating Value: Pair Programming](#articulating-value-pair-programming)
- [Articulating Value: Information Security](#articulating-value-information-security)
- Value Articulation Example

## Articulating Value: Baselines and Markers

Decisions need to be taken at multiple levels of abstraction of a codebase. Some examples are:

- Should I rename this variable or not?
- Should I refactor this piece of code into its own function or not?
- Should I apply this design pattern (e.g., factory) or not?
- Should I implement tracing or not? (You really should :-)
- Should I use a plugin architecture or not?
- Should I break this out into its own microservice or not?
- Should I use Kafka or Google PubSub?
- ...and so on

To enumerate the costs and benefits of these decisions, we need to calculate them relative to some baseline implementation. This baseline implementation may exist already or not, but it serves as a useful yardstick to drive out all the benefits that would occur if the decision was taken, or all the future problems which would occur if the decision was not taken (which would ultimately translate to financial losses), or the costs involved in implementing this decision.

As programmers, we make several lower level decisions over the course of a programming session with an intuitive understanding of the benefits of taking a particular decision (renaming a variable to be more descriptive, ultimately helps in readability for others --current and future -- working on the codebase). This is fine; we don't really need to evaluate the economic value for every small decision where the cost to make the change is vanishingly small, thanks to modern refactoring tools.

The decisions start to matter at higher levels of abstraction: at the architecture level, at the service level, and so on. Changes at those macro levels do not occur that often, and corresponding changes require greater effort; new deployments, additional dependency fixups, etc. Decisions at these levels thus benefit the most from explicit economic evaluations. These are the places where a baseline would help.

We thus propose the following baselines for some frequently occurring decisions:

- Monoliths as baseline when considering microservices
- Hardcoded plugins as baseline when considering microkernel architectures
- Peer-to-peer invocations as baseline when considering event-driven architectures
- Hardcoded components as baseline when considering pipe and filter patterns
- Peer-to-Peer invocations as baseline when considering event buses
- RDBMS (something like PostgreSQL) as baseline when considering NoSQL databases

Each of the above decisions has one or more expansion factors: these are the factors that make taking the decision potentially worthwhile. For example, if there was no need for future plugins to extend or add new functionality, there would be no need for a microkernel architecture; the number of future extensions is thus a expansion factor for this decision. If the list of components in a processing pipeline did not change at all, there would be no need of a pipe and filter pattern; the future configurability of components is the expansion factor for this decision.

We have spoken about how value can be measured, uaing the income approach, the market approach, etc. However, the question still remains: how do we connect the decisions we make (at the code level, at the architecture level, etc.) to the actual economic value.

At the business level, the closest connection to economic value is the feature of an application. Features are more or less atomic units of user-facing functionality (the user can be a human or another system) which can be (hopefully) deployed, enabled/disabled, and monetised independently.

Using features as units of economic value therefore seems plausible. The next question then arises: how do we verify that these features satisfy all the criteria to deliver this value? We propose a simple and natural answer: tests. Developers already use tests to validate every part of the system, at multiple levels of abstration, ranging from unit tests to integration tests to regression tests.

We propose that economic value be attached to the tests which verify features function properly. Different aspects of the feature can be validated by different sorts of tests. For example, an e-commerce payment integration system could have the following requirements to actually deliver value:

- It should be able to process Visa and Mastercard credit cards.
- It should be able to process at least 100 transactions per second.
- It should be able to cancel an amount which has already been authorised if indicated.

Each of the above requirements can be verified to a certain degree of rigour through tests. What would be the economic contribution of the above requirements?

- For the **requirement of processing Visa/Mastercard credit cards**, the income streams arising from the expected number of users with these kinds of credit cards (based on demographic analysis) making purchases of amounts (determined from historical data) over some period could be a straightforward derivation of the financial value of this feature.
- For the **requirement of processing at least 1000 transactions per second**, The income streams arising from 100 users per second performing financial transactions of some median amount over a sustained period of time could be a straightforward way to quantify the financial value of this feature.
- For the **requirement to cancel an already-authorised amount**, the cost of having support staff available to respond to customer calls for cancellation, and perform this action manually, could be one way to quantify the value of this feature.

{% mermaid %}
graph TD
subgraph features[Features]
    feature1[Feature]
    feature2[Feature]
    feature3[Feature]
end
subgraph designs[Designs]
    patterns
    architecture
end
code[Code]-->patterns[Patterns]
code-->architecture[Architecture]
patterns-->features
architecture-->features
feature1-->test1[Tests]
feature2-->test2[Tests]
feature3-->test3[Tests]
test1-->economic_value[Economic Value]
test2-->economic_value
test3-->economic_value
{% endmermaid %}


## Articulating Value: Pair Programming

Pair programming effectiveness seems to be a mixed bag, based on a survey of multiple studies in the paper [The effectiveness of pair programming: A meta-analysis](https://www.researchgate.net/publication/222408325_The_effectiveness_of_pair_programming_A_meta-analysis).

The key takeaway is this:

>  If you do not know the seniority or skill levels of your programmers, but do have a feeling for task complexity, then employ pair programming either when task complexity is low and time is of the essence, or when task complexity is high and correctness is important.

## Articulating Value: Information Security

Stuff Stuff Stuff

## References
- Books
- Papers
    - [The effectiveness of pair programming: A meta-analysis](https://www.researchgate.net/publication/222408325_The_effectiveness_of_pair_programming_A_meta-analysis)
    - [The Economic Impacts of Inadequate Infrastructure for Software Testing](https://www.nist.gov/system/files/documents/director/planning/report02-3.pdf)
- Web
