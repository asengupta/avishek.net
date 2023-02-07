---
title: "Every Software Engineer is an Accountant"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Software Engineering Economics"]
draft: false
---

This article continues from where [Every Software Engineer is an Economist]({% post_url 2023-01-22-every-engineer-is-an-economist %}) left off, and delves slightly deeper into some of the topics already introduced there, as well as several new ones. Specifically, we cover the following:

- [Waterfall Accounting: Capitalisable vs. Non-Capitalisable Costs](#waterfall-accounting-capitalisable-vs-non-capitalisable-costs)
- [How much are you willing to pay to reduce uncertainty? (the Expected Value of Perfect Information)](#how-much-are-you-willing-to-pay-to-reduce-uncertainty)
- [Articulating Value: The Cost of Unreleased Software](#articulating-value-the-cost-of-unreleased-software)
- [Articulating Value: NPV Analysis Example](#articulating-value-compound-options)
- [Articulating Value: Pair Programming](#articulating-value-pair-programming)
- Value trees and Probabilistic Graphical Models
- Value Tree Repository

## Waterfall Accounting: Capitalisable vs. Non-Capitalisable Costs

Capitalizable is an accounting term that refers to costs that can be recorded on the balance sheet, as opposed to being expensed immediately. These costs are viewed more favorably as they are spread out over the useful life of the asset, reducing the impact on net income. The accounting standards outline specific criteria for determining which costs are capitalizable. One criterion is the extent to which they provide a long-term benefit to the organization.

Accounting plays a significant role in software development processes. There are specific guidelines which state rules about what costs can be capitalised, and what costs should be accounted as expenses incurred. **Unfortunately, the accounting world lags behind the agile development model; GAAP guidelines have been established based on the waterfall model of software development.**

![Waterfall Accounting](/assets/images/waterfall-accounting.png)

Costs can be capitalised once "technological feasibility" has been achieved. Topic 985 says that:
> "the technological feasibility of a computer software product is established when the entity has completed all planning, designing, coding, and testing activities that are necessary to establish that the product can be produced to meet its design specifications including functions, features, and technical performance requirements."

Agile doesn't work that way. Agile does not have "one-and-done" stages of development since it is iterative; there is not necessarily a clear point at which "technological feasibility" is achieved; therefore **the criteria for "technological feasibility" may be an important point to agree upon between client and vendor**.

The problem is this: the guidelines state that the costs that should not be capitalized include the work that needs to be done to understand the product’s desired features and feasibility; **these costs should be expensed as incurred costs**.

For example, using development of external software (software developed for purchase or lease by external customers) as an example, the following activities cannot be capitalised:

- Upfront analysis
- Knowledge acquisition
- Initial project planning
- Prototyping
- Comparable design work

The above points apply even during iterations/sprints.
If we wanted to be really pedantic, during development, the following activities cannot be capitalised either, but must be expensed:

- Troubleshooting
- Discovery

**This may be an underlying reason why companies are leery of workshops and inceptions, because these probably end up as costs incurred instead of capitalised expenses.** ([Source](https://www.journalofaccountancy.com/news/2018/mar/accounting-for-external-use-software-development-costs-201818259.html))

**Value Proposition:** We should aim to optimise workshops and inceptions.

### Capitalisable and Non-Capitalisable Costs for Cloud

For Cloud Costing, we have the following categories from an accounting perspective:

- Capitalizable Costs
    - External direct costs of materials
    - Third-party service fees to develop the software
    - Costs to obtain software from third-parties
    - Coding and testing fees directly related to software product
- Non-capitalisable Costs
    - Costs for data conversion activities
    - Costs for training activities
    - Software maintenance costs

[This link](https://leasequery.com/blog/asc-350-internal-use-software-accounting-fasb/) and [Accounting for Cloud Development Costs](https://www.pwc.com/us/en/services/consulting/cloud-digital/cloud-transformation/cloud-computing.html) are readable treatments of the subject.

Also see [this](https://dart.deloitte.com/USDART/home/publications/deloitte/accounting-spotlight/cloud-computing-arrangements).

## How much are you willing to pay to reduce uncertainty?

We will use [this spreadsheet](https://docs.google.com/spreadsheets/d/1jBHwntpPI3QK5rM5yw5m2Gge9otgDf7pddNZs1sBZlw/edit?usp=sharing) again for our calculations. We spoke of the risk curve, which is the expected loss if the actual effort exceeds 310. Let us assume that the customer is adamant that we put in extra effort in narrowing our estimates so that we know whether we are over or below 310.

The question we'd like to answer is: **how much are we willing to pay to reduce the uncertainty of this loss to zero?** In other words, what is the maximum effort we are willing to spend to reduce the uncertainty of this estimate?

For this, we create a **Loss Function**, and this loss is simply calculated as $$L_i=P_i.E_i$$ for every estimate $$i \geq 310$$. Not too unsurprisingly, this is not the only choice for a loss function.

The answer is the area under the loss curve. This would usually done by integration, and is easily achieved if you are using a normal distribution, but is usually done through numerical integration for other arbitrary distributions. In this case, we can very roughly numerically integrate as shown in the diagram below, to get the maximum effort we are willing to invest.

![EVPI Example](/assets/images/evpi-example.png)

In our example, this comes out to 1.89. We can say that we are willing to make a maximum investment of 1.89 points of effort for the reduction in uncertainty to make economic sense. This value is termed the **Expected Value of Information** and is broadly defined as the amount someone is willing to pay for information that will reduce uncertainty about an estimate, or the information about a forecase. This technique is usually used to calculate the maximum amount of money you'd be willing to pay for a forecast about a business metric that affects your profits, but the same principle applies to estimates as well.

**Usually, the actual effort to reduce the uncertainty takes far longer, and hopefully an example like this can convince you that refining estimates is not necessarily a productive exercise.**

## Articulating Value: The Cost of Unreleased Software

## Statistics (aka Lies)

## Articulating Value: Circuit Breaker and Microservice Template example

We show an example of articulating value for a simple (or not-sp-simple case), where multiple factors can be at play.

We are building a platform on Google Cloud Platform, consisting of a bunch of microservices. Many of these microservices are projected to call external APIs. Some of these APIs are prone to failure or extended downtimes; we need to be able to implement the circuit breaker pattern. We assume that one new microservice will be built per month for the next 6 months.

- The development cost of these microservices is $2000.
- The rate of return (hurdle rate) is 10%. This will be used to calculate the Net Present Value of future costs and benefits.
- These microservices also require ground-up work when creating a new one. A microservice template or starter pack would reduce work required to deploy future microservices as well.

Unfortunately, Istio is currently not being used. Istio is an open source service mesh that layers transparently onto existing distributed applications. If Istio was being used, we could have leveraged its circuit breaker pattern pretty easily. We need to advocate for using Istio in our ecosystem. Let us assume that currently we have no circuit breaker patterns implemented at all. How can we build a business case around this?

There are a couple of considerations:

- The deployment of the service mesh may be an expensive process.
- The microservice template could also encapsulate a library-level circuit breaker implementation.
- The microservice template would have other benefits that are not articulated in this example.

1. Articulate Tech Debt due to No Circuit Breaker
2. Articulate Library-level Circuit Breaker Option
3. Articulate Microservice Starter Pack-level Circuit Breaker Option
4. Articulate Service Mesh Circuit Breaker Option
5. Explore combinations of these options

All the calculations are shown in [this spreadsheet](https://docs.google.com/spreadsheets/d/1jBHwntpPI3QK5rM5yw5m2Gge9otgDf7pddNZs1sBZlw/edit?usp=sharing).

### 1. Articulate Tech Debt due to No Circuit Breaker

Suppose we analyse the downtime suffered by our platform per month because of requests piling up because of slow, or unresponsive third party APIs. We assume that this number is around $10000. This cost and that of new microservice development, are shown below.

![Original Tech Debt Outflow](/assets/images/tech-debt.png)

The current cash outflow projected over 6 months, discounted to today, comes out to -$58733. This is the first step towards convincing stakeholders that they are losing money. Of course, we can project further out into the future, but the uncertainty of calculations obviously grows the more you go out.

We'd like to propose a set of options 

### 2. Articulate Immediate Library-level Circuit Breaker Option

![Only Library Option](/assets/images/circuit-breaker-library.png)

### 3. Articulate Immediate Service Mesh Option

![Only Service Mesh](/assets/images/service-mesh.png)

### 4. Articulate Immediate Starter Pack Option

![Only Starter Pack](/assets/images/starter-pack.png)

### 5. Articulate Immediate Library + Delayed Starter Pack Option

![Library and Delayed Starter Pack](/assets/images/library-plus-delayed-starter-pack.png)

### 6. Articulate Immediate Library + Delayed Starter Pack Option + Delayed Service Mesh Option

![Library and Delayed Starter Pack and Service Mesh](/assets/images/library-plus-delayed-starter-pack-and-service-mesh.png)

### 7. Review and Rank all Options

![All Options Returns](/assets/images/value-realisation-of-all-options.png)

## Articulating Value: Pair Programming

## Footnotes

- The [IEEE-CS/ACM Software Engineering Code of Ethics and Professional Practices](https://ethics.acm.org/code-of-ethics/software-engineering-code/) requires software professionals to quote uncertainties along with their estimates.

## References
- Papers
  - [Information Technology Investment: In Search of The Closest Accurate Method](https://www.sciencedirect.com/science/article/pii/S187705091931837X/pdf?md5=8ef46147c1296b09b1a4945fe12a8db1&pid=1-s2.0-S187705091931837X-main.pdf)
  - [The Business Value of IT; A Conceptual Model for Selecting Valuation Methods](https://www.researchgate.net/publication/239776307_The_Business_Value_of_IT_A_Conceptual_Model_for_Selecting_Valuation_Methods)
- Web
  - [Overview of Software Capitalisation Rules](https://leasequery.com/blog/software-capitalization-us-gaap-gasb/)
  - [Accounting for external-use software development costs in an agile environment](https://www.journalofaccountancy.com/news/2018/mar/accounting-for-external-use-software-development-costs-201818259.html)
  - External Use Software guidelines - FASB Accounting Standards Codification (ASC) Topic 985, Software
  - Internal Use Software guidelines - FASB Accounting Standards Codification (ASC) Topic 350, Intangibles — Goodwill and Other
  - [Accounting for internal-use software using Cloud Computing development costs](https://leasequery.com/blog/asc-350-internal-use-software-accounting-fasb/)
  - [Accounting for Cloud Development Costs](https://www.pwc.com/us/en/services/consulting/cloud-digital/cloud-transformation/cloud-computing.html) are covered under FASB Subtopic ASC 350-40 (Customer’s Accounting for Implementation Costs Incurred in a Cloud Computing Arrangement That Is a Service Contact (ASC 350-40)).
  - [Financial Reporting Developments: Intangibles - goodwill and other](https://assets.ey.com/content/dam/ey-sites/ey-com/en_us/topics/assurance/accountinglink/ey-frdbb1499-05-09-2022.pdf?download). The actual formal document is [here](https://fasb.org/document/blob?fileName=ASU%202021-03.pdf)
