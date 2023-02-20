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

- **Monoliths** as baseline when considering **microservices**
- **Hardcoded plugins** as baseline when considering **microkernel architectures**
- **Peer-to-peer invocations** as baseline when considering **event-driven architectures**
- **Hardcoded components** as baseline when considering **pipe and filter patterns**
- **Peer-to-Peer invocations** as baseline when considering **event buses**
- **RDBMS** (something like PostgreSQL) as baseline when considering **NoSQL databases**

Each of the above decisions has one or more expansion factors: these are the factors that make taking the decision potentially worthwhile. For example, if there was no need for future plugins to extend or add new functionality, there would be no need for a microkernel architecture; the number of future extensions is thus a expansion factor for this decision. If the list of components in a processing pipeline did not change at all, there would be no need of a pipe and filter pattern; the future configurability of components is the expansion factor for this decision.

It is also important to note that the above decisions are not exclusive. A microservice may encapsulate a microkernel, parts of a pipe-and-filter architecture might involve invoking microservices, and so on.

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
