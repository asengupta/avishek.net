---
title: "Every Software Engineer is an Economist"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Software Engineering Economics"]
draft: false
---

**Background**: This post took me a while to write: much of this is motivated by problems that I've noticed teams facing day-to-day at work. To be clear, this post does not offer a solution; only some thoughts, and maybe a path forward in aligning developers' and architects' thinking more closely with the frameworks used by people controlling the purse-strings of software development projects.

Here is a [presentation version](/assets/presentations/value-articulation-guide-ppt.html) of this article.

The other caveat is that even though this article touches the topic of estimation, it is to talk about building uncertainty into estimates as a way to communicate risk and uncertainties with stakeholders, and not to *refine* estimates. I won't be extolling the virtues or limitations of **#NoEstimates**, for example (sidebar: the smoothest teams I've worked with essentially dispensed with estimation, but they also had excellent stakeholders).

> "All models are wrong, but some are useful." - George Box

**Every software engineer is an economist; an architect, even more so.** There is a wealth of literature around articulating value of software development, and in fact, several agile development principles embody some of these, but I see two issues in my day-to-day interactions with software engineers and architects.

- Folks are reluctant to quantify things they build, beyond the standard practices they have been brought up on (like estimation exercises, test coverage). Some of this can be attributed to their prior bad experiences of being micromanaged via largely meaningless metrics.
- Folks struggle to articulate value beyond a certain point to stakeholders who demand a certain measure of rigour and/or quantifiability. Similarly, engineers fail to communicate risk to decision-makers. The problem is then that The DORA metrics are good starter indicators, but I contend that they are not enough.
- There is a reluctance to rely too much on metrics because people think metrics are easily gamed. This can be avoided if we use econometric methods, because 1) falsified data is immediately apparent 2) showing the work steps, assumptions and risks aids in this transparency because they are in the language of economics which is much more easily understandable to business stakeholders.
- Thinking about value and deciding tradeoffs based on economic factors is not something that is done enough, if at all, at the level of engineering teams. For example, questions like "Should I do this refactoring?" and "Why should we repay this tech debt?", or "How are we better at this versus our competitor?" are usually framed in terms of statements which stop before traversing the full utility tree of value.

Thinking in these terms, and projecting these decisions in these terms to micromanagers, heads/directors of engineering -- but most importantly, to execs -- is key to engineers articulating value in a manner which is compelling, and eases friction between engineering and executive management. It is also a value engineers should acquire to break several firms' perceptions that "engineers are here to do what we say".

This is easier said than done, because of several factors:

- The data to apply these frameworks is not always easily available, or people are not ready to gather that data.
- Engineers are usually emotionally invested in decisions that they think are their "pet" ideas.
- It can be hard to inculcate this mindset en masse among engineers if they do not have a clear perception of the value of adopting this mindset. Engineers don't want theory, they want tools they can apply quickly and easily. Hence the burden is on us to propose advances to the state of the art in a way that is actionable.

Important Concepts that every software developer should know:

- Decision-Making Processes
  - Analytic Hierarchy Process
- Utility-based Architecture Decision Making: CBAM
  - [The CBAM: A Quantitative Approach to Architecture Design Decision Making](https://people.ece.ubc.ca/matei/EECE417/BASS/ch12.html)
  - [Making Architecture Design Decisions: An Economic Approach](https://apps.dtic.mil/sti/pdfs/ADA408740.pdf)
  - The above goes some way towards assigning utility to architectural decisions. These are at the level of Cross-Functional Requirements, and DORA metrics, but do not point the way towards calculating financial implications.
- Net Present Value and Discounted Cash Flows

## 1. Articulating Value: Communicate Uncertainty in Estimation Models

$$
\text{Confidence Interval } = \hat{X} \pm Z.\frac{\sigma}{\sqrt{n} }
$$

- Calculate $$\sigma$$ given confidence interval of 0.9 (Z-score is correspondingly 1.65).
- Do this for each story.
- Calculate the joint probability distribution of all the random variables (one per story). This is easy if we assume all the estimate distributions are Gaussian. If not, perform Monte Carlo simulations.
- Choose acceptable confidence level, and pick that estimate. Alternatively, pick an acceptable estimate, and record confidence level, and acknowledge the risk involved.
- Negotiation should happen around acceptable levels of uncertainty levels, not on modifying story estimates to fit a particular target.

See [this spreadsheet](https://docs.google.com/spreadsheets/d/1jBHwntpPI3QK5rM5yw5m2Gge9otgDf7pddNZs1sBZlw/edit?usp=sharing) for a sample calculation.

## 2. Articulating Value: Economics and Risks of Tech Debt and Architectural Decisions
Here is some research relating **Development Metrics to Wasted Development Time**:

- [Code Red: The Business Impact of Code Quality - A Quantitative Study of 39 Proprietary Production Codebases](https://arxiv.org/abs/2203.04374)
- [The financial aspect of managing technical debt: A systematic literature review](https://www.semanticscholar.org/paper/The-financial-aspect-of-managing-technical-debt%3A-A-Ampatzoglou-Ampatzoglou/de5db6c07899c1d90b4ff4428e68b2dd799b9d6e)
- [The Pricey Bill of Technical Debt: When and by Whom will it be Paid?](https://www.researchgate.net/publication/320057934_The_Pricey_Bill_of_Technical_Debt_When_and_by_Whom_will_it_be_Paid)

ATD must have cost=principal (amount to pay to implement) + interest (continuing incurred costs of not implementing ATD)

{% mermaid %}
graph LR;
architecture_decision[Architecture Decision]-->atd_principal[Cost of Architectural Decision: Principal];
architecture_decision-->recurring_atd_interest[Recurring Cost: Interest];
architecture_decision-->recurring_atd_interest[Recurring Development Savings];
architecture_decision-->atd_option_premium[Architecture Option Premium];

style architecture_decision fill:#006fff,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

{% include_chart atd_cbam!500!500!{
    type: 'bar',
    data: {
            labels: ['Jan', 'Feb', 'March', 'April', 'May', 'June'],
            datasets: [
                {
                    label: 'Recurring Cost (Interest)',
                    data: [-5000, -10000, -13000, -5000, -1000, -1000],
                },
                {
                    label: 'Recurring Development Savings',
                    data: [0, 0, 0, 0, 5000, 6000],
                },
                {
                    label: 'Implement Architectural Decision (Principal)',
                    data: [0, 0, -20000, -5000, 0, 0],
                }]
            },
            options: {
            responsive: false,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                },
                y: {
                    stacked: true,
                    ticks: {
                        // Include a dollar sign in the ticks
                        callback: function(value, index, ticks) {
                        return '$' + value;
                    }
                }
            }
        }
    }
} %}

### Incorporating economics into daily architectural thinking

Here are some generic tips.

- Practise drawing causal graphs. Complete the trace all the way up to where the perceived benefit is (money) is. It may be tempting to stop if you reach a DORA metric. Don't; get to the money.
- If you are already measuring DORA metrics, relentlessly ask what each DORA metric translates to in terms of money.
- Along the way of the graph, list out other incidental cash outflows.
- Remember that story points must always be converted into hours to actually be incorporated into economic estimates.
- Build Options tree. Deduce whether it is better to defer execution, or do it right now.

- Non-technical things can also be calculated, i.e., the need for training.
- These metrics must be measured as part of standardised project protocols.

Here are some tips for specific but standard cases.

#### 1. The Economics of Microservices

If you are suggesting a new microservice for processing payments, these might be the new cash flows:
- **Recurring Cash Flows**
    - Transactions: New cash inflow
    - Cost of recovering the whole system back from failure: Reduced cash outflow
    - Cost of cloud resources to scale the new microservice: New cash outflow
    - Cost of higher latency leading to lower service capacity (if the microservice is part of a workflow): Decreased cash inflow, depending upon if you ever reach the load limits of the service before other parts of the system start to fail
    - Cost of fixing bugs: New cash outflow, depending upon complexity of the microservice
    - Cost of Integrations:
- **Single or Few-Time Cash Flows**
    - Cost of development: New cash outflow
    - Cost of deployment setup: New cash outflow (ideally should be as low as possible)
- **Option Premium**
    - Architecture Seam

{% mermaid %}
graph LR;
microservice-->database[Cloud DB Resources];
microservice-->development_cost[Development Cost];
microservice-->latency[Latency];
microservice-->bugs[Fixing bugs]-->bugfix_time[Wasted Bugfix Time Costs];
microservice-->downtime[Downtime]-->lost_transactions[Lost Transaction Costs];
microservice-->microservice_option_premium[Architecture Seam: Option Premium];

style microservice fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

#### 2. The Economics of Technical Debt repayment

- **Recurring Cash Flows**
    - Cost of Manual Troubleshooting and Resolution
    - Cost of recurring change to a specific module
- **Single or Few-Time Cash Flows**
    - Cost of repaying tech debt
- **Option Premium**
  - The cost of isolating the effect of the technical debt from affecting other code

{% mermaid %}
graph LR;
debt[Tech Debt]-->principal[Cost of Fixing Debt: Principal];
debt-->interest[Recurring Cost: Interest];
debt-->td_option_premium[Tech Debt Option Premium];
debt-->risk[Risk-Related Cost, eg, Security Breach];

style debt fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

**Example Tech Debt Cash Flow**

{% include_chart tech_debt_cbam!500!500!{
    type: 'bar',
    data: {
            labels: ['Jan', 'Feb', 'March', 'April', 'May', 'June'],
            datasets: [
                {
                    label: 'Downtime Costs (Interest)',
                    data: [-5000, -10000, -13000, -5000, -2000, -2000],
                },
                {
                    label: 'Bug Costs (Interest)',
                    data: [-5000, -12000, -13000, -5000, -2000, -1000],
                },
                {
                    label: 'Repay Tech Debt (Principal)',
                    data: [0, 0, -20000, -5000, 0, 0],
                }]
            },
            options: {
            responsive: false,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                },
                y: {
                    stacked: true,
                    ticks: {
                        // Include a dollar sign in the ticks
                        callback: function(value, index, ticks) {
                        return '$' + value;
                    }
                }
            }
        }
    }
} %}


## 3. Articulating Value: Deriving Value in Legacy Modernisation

$$C_{HW}$$ = Cost of Hardware / Hosting \\
$$C_{HUF}$$ = Cost of manual work equivalent of feature (if completely new feature or if feature has manual interventions) \\
$$C_{RED}$$ = Cost of recovery, including human investments (related to MTTR) \\
$$C_{LBD}$$ = Cost of lost business / productivity during downtime (related to MTTR) \\
$$C_{ENF}$$ = Cost of development of an enhancement to a feature (related to DORA Lead Time) \\
$$C_{NUF}$$ = Cost of development of a new feature (related to DORA Lead Time) \\
$$C_{BUG}$$ = Cost of bug fixes for feature \\
$$n_D$$ = Number of downtime incidents per year \\
$$n_E$$ = Number of enhancements to feature per year \\
$$n_B$$ = Number of bugs in feature per year

The cost of a feature is then denoted by $$V$$, and the total value of the feature is $$V_{total}$$. These are given by:

$$
V=C_{HUF} + n_D.(C_{RED} + C_{LBD}) + n_E.C_{ENF} + n_B.C_{BUG} \\
V_{total} = \sum_{i} V_i + C_{HW} + n_F.C_{NUF}
$$

Retention of customer base is also a valid use case. Not sure how to quantify this...

## 4. Articulating Value: The Value of Timing (aka, Real Options)

### Real Options

We will not discuss the Options Thinking approach from scratch here; rather we will delve into some of its possible applications in architectural decision-making and technical debt repayment. See the following for excellent discussions on the topic:

- [Software Design Decisions as Real Options](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=24f7bdda5f3721faa2da58719ae72432f782312f)
- [The Software Architect Elevator](https://www.amazon.com/Software-Architect-Elevator-Redefining-Architects-ebook/dp/B086WQ9XL1)
- Chapter 4 of [Extreme Programming Perspectives](https://www.amazon.com/Extreme-Programming-Perspectives-Michele-Marchesi/dp/0201770059)
- Chapter 3 of [Value-Based Software Engineering](https://link.springer.com/book/10.1007/3-540-29263-2)

- Disadvantages of the NPV approach
- Opportunity Cost
- Examples
    - One example where we could have applied: The team had built a data engineering pipeline using Spark and Scala. The stakeholder felt that hiring developers with the requisite skillsets would be hard, and wanted to move to plain Java-based processing. A combination of cash flow modeling and buying the option of redesign would have probably made for a compelling case.
- **Competent Architects and Engineers identify Real Options. Good Architects and Engineers create Real Options.**

**Valuing Real Options using [Datar-Matthews](https://www.researchgate.net/publication/227374121_A_Practical_Method_for_Valuing_Real_Options_The_Boeing_Approach)**

## 5. Articulating Value: The Value of Measurement (aka, the Cost of Information)

For a measurement to have economic value, it must support a decision. Examples of decisions are:

- The investment will either be made or not. Alternatively, the amount of investment will be more or less.
- Teams will be restructured or not.
- A feature will go live or not.
- A system (or subsystem) will be modernised or not.
- A system will be either bought or built in-house.

### Characteristics of a Decision

- Must have 2 or more realistic alternatives. These alternatives cannot be recursive, i.e., the decision based on a certain measurement should not be to take action to modify that measurement.
- A decision has uncertainty.
- A decision has potentially negative consequences.
- A decision must have a decision maker.

Quantify the **Decision Model**. The Decision Model will probably have multiple variables.

We need to decide what is the importance of these variables in making the decision. If a measurement has zero information value, then it is not worth measuring. When multiple variables are involved, use the EVPI metric coupled with Monte Carlo simulations (assuming the decision model has been quantified) to decide on the most important metrics.

- The Expected Value of Perfect Information

## References

- Books
    - [Economics-Driven Software Architecture](https://www.amazon.in/Economics-Driven-Software-Architecture-Ivan-Mistrik/dp/0124104649)
    - [How to Measure Anything](https://www.amazon.in/How-Measure-Anything-Intangibles-Business/dp/1118539273)
    - [Value-Based Software Engineering](https://link.springer.com/book/10.1007/3-540-29263-2)
    - [The Software Architect Elevator](https://www.amazon.com/Software-Architect-Elevator-Redefining-Architects-ebook/dp/B086WQ9XL1)
    - [Extreme Programming Perspectives](https://www.amazon.com/Extreme-Programming-Perspectives-Michele-Marchesi/dp/0201770059)
- Papers
    - [Software Design Decisions as Real Options](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=24f7bdda5f3721faa2da58719ae72432f782312f)
    - [A Practical Method for Valuing Real Options: The Boeing Approach](https://www.researchgate.net/publication/227374121_A_Practical_Method_for_Valuing_Real_Options_The_Boeing_Approach)
    - [Code Red: The Business Impact of Code Quality - A Quantitative Study of 39 Proprietary Production Codebases](https://arxiv.org/abs/2203.04374)
    - [The financial aspect of managing technical debt: A systematic literature review](https://www.semanticscholar.org/paper/The-financial-aspect-of-managing-technical-debt%3A-A-Ampatzoglou-Ampatzoglou/de5db6c07899c1d90b4ff4428e68b2dd799b9d6e)
    - [The Pricey Bill of Technical Debt: When and by Whom will it be Paid?](https://www.researchgate.net/publication/320057934_The_Pricey_Bill_of_Technical_Debt_When_and_by_Whom_will_it_be_Paid)
- Videos
    - [Excellent Video on Real Options ECO423: IIT Kanpur](https://www.youtube.com/watch?v=lwoCGAqv5RU)
