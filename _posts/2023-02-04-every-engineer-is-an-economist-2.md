---
title: "Every Software Engineer is an Economist 2"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Software Engineering Economics"]
draft: false
---

- The Cost of Unreleased Software
- Marrying MECE trees with Probabilistic Graphical Models
- Expected Value of Perfect Information
- Value Tree Repository

There are a lot more concepts that I'd like to cover, including:

- Possible procedures for determining the value of a metric
- When is a metric's performance good enough?

I will continue adding more information on the topic of the value of metrics going forward. Stay tuned.

## New References

- [Accounting for external-use software development costs in an agile environment](https://www.journalofaccountancy.com/news/2018/mar/accounting-for-external-use-software-development-costs-201818259.html)
- External Use Software guidelines - FASB Accounting Standards Codification (ASC) Topic 985, Software
- Internal Use Software guidelines - FASB Accounting Standards Codification (ASC) Topic 350, Intangibles — Goodwill and Other
- [Cloud Accounting Guidelines](https://leasequery.com/blog/asc-350-internal-use-software-accounting-fasb/) are covered under FASB Subtopic ASC 350-40 (Customer’s Accounting for Implementation Costs Incurred in a Cloud Computing Arrangement That Is a Service Contact (ASC 350-40)).

Accounting plays a significant role in software development processes. There are specific guidelines which state rules about what costs can be capitalised, and what costs should be accounted as expenses incurred. Unfortunately, the accounting world lags behind the agile development model; GAAP guidelines have been established based on the waterfall model of software development.

![Waterfall Accounting](/assets/images/waterfall-accounting.png)

Costs can be capitalised once "technological feasibility" has been achieved. Topic 985 says that:
> "the technological feasibility of a computer software product is established when the entity has completed all planning, designing, coding, and testing activities that are necessary to establish that the product can be produced to meet its design specifications including functions, features, and technical performance requirements."

Agile doesn't work that way. Agile does not always have a clear point at which "technological feasibility" is achieved; therefore the criteria for "technological feasibility" may be an important point to agree upon between client and vendor.

The problem is this: the guidelines state that the costs that should not be capitalized include the work that needs to be done to understand the product’s desired features and feasibility; these costs should be expensed as incurred costs. They include:

- Upfront analysis
- Knowledge acquisition
- Initial project planning
- Prototyping
- Comparable design work

The above points apply even during iterations/sprints.
If we wanted to be really pedantic, during development, the following activities cannot be capitalised either, but must be expensed:

- Troubleshooting
- Discovery

**This may be an underlying reason why companies are leery of workshops and inceptions, because these probably end up as costs incurred instead of capitalised expenses.**


## References

- Books
    - [Economics-Driven Software Architecture](https://www.amazon.in/Economics-Driven-Software-Architecture-Ivan-Mistrik/dp/0124104649)
    - [How to Measure Anything](https://www.amazon.in/How-Measure-Anything-Intangibles-Business/dp/1118539273)
    - [Value-Based Software Engineering](https://link.springer.com/book/10.1007/3-540-29263-2)
    - [The Software Architect Elevator](https://www.amazon.com/Software-Architect-Elevator-Redefining-Architects-ebook/dp/B086WQ9XL1)
    - [Extreme Programming Perspectives](https://www.amazon.com/Extreme-Programming-Perspectives-Michele-Marchesi/dp/0201770059)
    - [The Value of Custom Software as an Asset](https://paper-leaf.com/insights/custom-software-value-useful-life/)
- Papers
    - [Making Architecture Design Decisions: An Economic Approach](https://apps.dtic.mil/sti/pdfs/ADA408740.pdf) describes a pilot study of a modified CBAM approach applied at NASA.
    - [Software Design Decisions as Real Options](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=24f7bdda5f3721faa2da58719ae72432f782312f)
    - [A Practical Method for Valuing Real Options: The Boeing Approach](https://www.researchgate.net/publication/227374121_A_Practical_Method_for_Valuing_Real_Options_The_Boeing_Approach) describes the Datar-Matthews approach used in the real options example in this article.
    - [Code Red: The Business Impact of Code Quality - A Quantitative Study of 39 Proprietary Production Codebases](https://arxiv.org/abs/2203.04374)
    - [The financial aspect of managing technical debt: A systematic literature review](https://www.semanticscholar.org/paper/The-financial-aspect-of-managing-technical-debt%3A-A-Ampatzoglou-Ampatzoglou/de5db6c07899c1d90b4ff4428e68b2dd799b9d6e)
    - [The Pricey Bill of Technical Debt: When and by Whom will it be Paid?](https://www.researchgate.net/publication/320057934_The_Pricey_Bill_of_Technical_Debt_When_and_by_Whom_will_it_be_Paid)
    - [Software Risk Management: Principles and Practices](https://www.cs.virginia.edu/~sherriff/papers/Boehm%20-%201991.pdf)
    - [Generalization of an integrated cost model and extensions to COTS, PLE and TTM](https://researchrepository.wvu.edu/cgi/viewcontent.cgi?article=3261&context=etd)
- Web
    - [Excellent Video on Real Options ECO423: IIT Kanpur](https://www.youtube.com/watch?v=lwoCGAqv5RU)
    - [Risk Exposure](https://www.wallstreetmojo.com/risk-exposure/)
    - [The CBAM: A Quantitative Approach to Architecture Design Decision Making](https://people.ece.ubc.ca/matei/EECE417/BASS/ch12.html)
