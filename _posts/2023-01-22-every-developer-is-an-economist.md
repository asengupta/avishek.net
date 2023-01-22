---
title: "Every Software Engineer is also an Economist"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Economics"]
draft: false
---

**Every software engineer is also an economist; an architect, even more so.** There is a wealth of literature around articulating value of software development, and in fact, several agile development principles embody some of these, but I see two issues in my day-to-day interactions with software engineers and architects.

- Folks are reluctant to quantify things they build, beyond the standard practices they have been brought up on (like estimation exercises, test coverage). Some of this can be attributed to their prior bad experiences of being micromanaged via largely meaningless metrics.
- Folks struggle to articulate value beyond a certain point to stakeholders who demand a certain measure of rigour and/or quantifiability. The DORA metrics are good starter indicators, but I contend that they are not enough.
- Thinking about value and deciding tradeoffs based on economic factors is not something that is done enough, if at all, at the level of engineering teams. For example, questions like "Should I do this refactoring?" and "Why should we repay this tech debt?", or "How are we better at this versus our competitor?" are usually framed in terms of statements which stop before traversing the full utility tree of value.

Thinking in these terms, and projecting these decisions in these terms to micromanagers, heads/directors of engineering -- but most importantly, to execs -- is key to engineers getting the necessary clout in higher-strategic decisions. It is also an instrumental value engineers should acquire to break several firms' perceptions that "engineers are here to do what we say".

This is easier said than done, because of several factors:

- The data to apply these frameworks is not always easily available.
- Engineers are usually emotionally invested in decisions that they think are their "pet" ideas.
- It can be hard to inculcate this mindset en masse among engineers if they do not have a clear perception of the value of adopting this mindset. Engineers don't want theory, they want tools they can apply quickly and easily. Hence the burden is on us to propose advances to the state of the art in a way that is actionable.

Important Concepts that every software developer should know:

- Decision-Making Processes
- Utility-based Architecture Decision Making: CBAM
  - [The CBAM: A Quantitative Approach to Architecture Design Decision Making](https://people.ece.ubc.ca/matei/EECE417/BASS/ch12.html)
  - [Making Architecture Design Decisions: An Economic Approach](https://apps.dtic.mil/sti/pdfs/ADA408740.pdf)
  - The above goes some way towards assigning utility to architectural decisions. These are at the level of Cross-Functional Requirements, and DORA metrics, but do not point the way towards calculating financial implications.
- Cost-Benefit Analysis
- Net Present Value

### Cash Flow Sources

| Factor                                         | Type    | Uncertainty                                                 | Influencer    | Notes |
|------------------------------------------------|---------|-------------------------------------------------------------|---------------|--|
| Time to Market Leader Revenue                  | Inflow  | High                                                        | Modernisation | Closely related to Change Lead Time and Deployment Frequency, but projected into the future |
| Number of successful transactions              | Inflow  | Low                                                         | Cloud Migration, Performance, Modernisation, Architecture Refactoring (eg, CQRS) | Useful when transactions are financial. Unrealised inflow can be measured by number of failed transactions |
| Software Licenses / Contracts                  | Inflow  | Variable                                                    | Legacy Modernisation | This is at the application level, and cannot always be traced to a single architectural or design decision, unless explicitly modelled as potential new licenses because of a new capability/feature. |
| Support / On Call costs                         | Outflow | Low                                                         | Observability, Process Automation (eg, Refunds, Self Service, etc.) |  |
| Hardware / Cloud Costs                         | Outflow | Low                                                         | Legacy Modernisation |  |
| Software Licenses                              | Outflow | Low                                                         | Build vs Buy Recommendations |  |
| Future Effort of Adding New Features           | Outflow | Variable (Depends upon profitability of the feature)        | Refactoring, Tech Debt Repayment | Closely related to Change Lead Time and Deployment Frequency, but projected into the future |
| Cost of fixing potential bugs                  | Outflow | Depends upon rate of modification of possibly untested code | Testing, Refactoring/Rewriting, Tech Debt Repayment | Not necessarily related to Change Failure Rate |
| Cost of Recovering System in Production        | Outflow | Measurable from dashboard statistics                        | Observability, Cloud Migration, Traceability, Microservices | Closely related to Time to Restore Service (DORA), projected into the future |
| Internal client waste (Manual workflows, etc.) | Outflow | Low                                                         |               | This is frequently what the software might be targeted to reduce. It shows up as "money saved", even though it might not strictly be a **Cash Inflow**. |


### Real Options

We will discuss the Options Thinking approach here. See - [Software Design Decisions as Real Options](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=24f7bdda5f3721faa2da58719ae72432f782312f) and, more recently, [The Software Architect Elevator](https://www.amazon.com/Software-Architect-Elevator-Redefining-Architects-ebook/dp/B086WQ9XL1).

- Disadvantages of the NPV approach
- Opportunity Cost
- Examples
  - One example where we could have applied: The team had built a data engineering pipeline using Spark and Scala. The stakeholder felt that hiring developers with the requisite skillsets would be hard, and wanted to move to plain Java-based processing. A combination of cash flow modeling and buying the option of redesign would have probably made for a compelling case.
- Agile in SEE
  - Delayability

## Incorporating economics into daily thinking

Here are some generic tips.

- Practise drawing causal graphs. Complete the trace all the way up to where the perceived benefit is (money) is. It may be tempting to stop if you reach a DORA metric. Don't; get to the money.
- If you are already measuring DORA metrics, relentlessly ask what each DORA metric translates to in terms of money.
- Along the way of the graph, list out other incidental cash outflows.
- Remember that story points must always be converted into hours to actually be incorporated into economic estimates.
- CALCULATE NPV!!! HOW?
- Build Options tree. Deduce whether it is better to defer execution, or do it right now.

Here are some tips for specific but standard cases.

#### 1. The Economics of Microservices

If you are suggesting a new microservice for processing payments, these might be the new cash flows:
  - Recurring Cash Flows
    - Transactions: New cash inflow
    - Cost of recovering the whole system back from failure: Reduced cash outflow
    - Cost of cloud resources to scale the new microservice: New cash outflow
    - Cost of higher latency leading to lower service capacity (if the microservice is part of a workflow): Decreased cash inflow, depending upon if you ever reach the load limits of the service before other parts of the system start to fail
    - Cost of fixing bugs: New cash outflow, depending upon complexity of the microservice
- Single or Few-Time Cash Flows
  - Cost of development: New cash outflow
  - Cost of deployment setup: New cash outflow (ideally should be as low as possible)

**Causal Graph**

#### 1. The Economics of Technical Debt repayment
#### 1. The Economics of New Features

### References

- [Economics-Driven Software Architecture](https://www.amazon.in/Economics-Driven-Software-Architecture-Ivan-Mistrik/dp/0124104649)
- [Software Design Decisions as Real Options](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=24f7bdda5f3721faa2da58719ae72432f782312f)
- 
