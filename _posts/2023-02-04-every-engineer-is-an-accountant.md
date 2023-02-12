---
title: "Every Software Engineer is an Accountant"
author: avishek
usemathjax: true
tags: ["Software Engineering", "Software Engineering Economics"]
draft: false
---

This article continues from where [Every Software Engineer is an Economist]({% post_url 2023-01-22-every-engineer-is-an-economist %}) left off, and delves slightly deeper into some of the topics already introduced there, as well as several new ones. Specifically, we cover the following:

- [Waterfall Accounting: Capitalisable vs. Non-Capitalisable Costs](#waterfall-accounting-capitalisable-vs-non-capitalisable-costs)
- [Articulating Value: The Cost of Reducing Uncertainty](#articulating-value-the-cost-of-reducing-uncertainty)
- [Articulating Value: The Cost of Expert but Imperfect Knowledge](#articulating-value-the-cost-of-expert-but-imperfect-knowledge)
- [Articulating Value: The Cost of Unreleased Software](#articulating-value-the-cost-of-unreleased-software)
- [Static NPV Analysis Example: Circuit Breaker and Microservice Template](#static-npv-analysis-example-circuit-breaker-and-microservice-template)
- [Articulating Value: The Value of a Software System](#articulating-value-the-value-of-a-software-system)
- [Articulating Value: Pair Programming](#articulating-value-pair-programming)
- There is not Enough Data

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

The above points apply even during iterations/sprints. If we wanted to be really pedantic, during development, the following activities cannot be capitalised either, but must be expensed:

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

## Articulating Value: The Cost of Reducing Uncertainty

We will use [this spreadsheet](https://docs.google.com/spreadsheets/d/1jBHwntpPI3QK5rM5yw5m2Gge9otgDf7pddNZs1sBZlw/edit?usp=sharing) again for our calculations. We spoke of the risk curve, which is the expected loss if the actual effort exceeds 310. Let us assume that the customer is adamant that we put in extra effort in narrowing our estimates so that we know whether we are over or below 310.

The question we'd like to answer is: **how much are we willing to pay to reduce the uncertainty of this loss to zero?** In other words, what is the maximum effort we are willing to spend to reduce the uncertainty of this estimate?

For this, we create a **Loss Function**, and this loss is simply calculated as $$L_i=P_i.E_i$$ for every estimate $$i \geq 310$$. Not too unsurprisingly, this is not the only choice for a loss function.

The answer is the area under the loss curve. This would usually done by integration, and is easily achieved if you are using a normal distribution, but is usually done through numerical integration for other arbitrary distributions. In this case, we can very roughly numerically integrate as shown in the diagram below, to get the maximum effort we are willing to invest.

![EVPI Example](/assets/images/evpi-example.png)

In our example, this comes out to 1.89. We can say that we are willing to make a maximum investment of 1.89 points of effort for the reduction in uncertainty to make economic sense. This value is termed the **Expected Value of Information** and is broadly defined as the amount someone is willing to pay for information that will reduce uncertainty about an estimate, or the information about a forecase. This technique is usually used to calculate the maximum amount of money you'd be willing to pay for a forecast about a business metric that affects your profits, but the same principle applies to estimates as well.

**Usually, the actual effort to reduce the uncertainty takes far longer, and hopefully an example like this can convince you that refining estimates is not necessarily a productive exercise.**

## Articulating Value: The Cost of Expert but Imperfect Knowledge

Suppose you, the tech lead or architect, wants to make a decision around some architecture or tech stack. You've heard about it, and you think it would be a good fit for your current project scenario. But you are not *completely* sure, so in the worst case, there would be no benefit and just the cost sunk into the investment of implementing this decision. The two questions you'd like to ask are:

- **What is the maximum I'm willing to pay to reduce the uncertainty of this decision completely?** This question is exactly the same as the one in the previous section, so is not in itself that novel, but it is a stepping stone to the next question.
- **What is the maximum I'm willing to pay to bring in an expert who can help me reduce this uncertainty to a lower value, but probably not to zero?** In this case, the expert will not be able to provide you perfect information, and we must incorporate our confidence in the expert into our economics calculations.

**We can use Decision Theory to quantify these costs.** The technique we'll be using involves **Probabilistic Graphical Models**, and all of this can be easily automated: this step-by-step example is for comprehension.

Suppose we have the situation above where a decision needs to be made. There is 30% possibility that the decision will result in a savings of $100000 going forward, and 70% possibility that there won't be any benefit at all.

Let X be the event that there will be a savings of $20000. Then $$P(X)=0.3$$. We can represent all the possibilities using a Decision Tree, like below.

{% mermaid %}
graph LR
A ==>|Cost=5000| implement["Implement"]
A -->|Cost=0| dont_implement["Do Not Implement"]
implement ==>|"P(X)=0.3"| savings_1[Savings=20000-5000=15000]
implement ==>|"1-P(X)=0.7"| no_savings_1[Savings=0-5000=-5000]
dont_implement -->|"P(X)=0.3"| savings_2[Savings=0]
dont_implement -->|"1-P(X)=0.7"| no_savings_2[Savings=0]
{% endmermaid %}

Now, if we did not have any information beyond these probabilities, we'd pick the decision which maximises the expected payoff. The payoff from this decision is called the Expected Monetary Value, and is defined as:

$$
EMV=\text{max}_i \sum_i P_i.R_{ij}
$$

This is simply the maximum expected value of all the expected values arising from all the choices $$j\in J$$. The monetary value for the "Implement" decision is $$0.3 \times 15000 + 0.7 \times (-5000)=$1000$$, whereas that of the "Do Not Implement" decision is zero. Thus, we pick the monetary value of the former, and our EMV is $1000.

Now assume we had a perfect expert who knew whether the decision is going to actually result in savings or not. If they told us the answer, we could effectively know whether to implement the decision or not with complete certainty.

The payoff then would be calculated using the following graph. The graph switches the chance nodes and the decision nodes, and for each chance node, picks the decision node which maximises the payoff.

{% mermaid %}
graph LR
A ==> savings["Savings<br>P(X)=0.3"]
A ==> no_savings["No Savings<br>1-P(X)=0.7"]
savings ==>|Cost=5000| implement_1[Implement<br>Savings=20000-5000=15000]
savings -->|Cost=0| dont_implement_1[Don't Implement<br>Savings=0]
no_savings -->|Cost=5000| implement_2[Implement<br>Savings=0-5000=-5000]
no_savings ==>|Cost=0| dont_implement_2[Don't Implement<br>Savings=0]
{% endmermaid %}

We can then calculate expected payoff given perfect information (denoted as EV\|PI) as:

$$
EV|PI = \sum_i P_{j}.\text{max}_i R_{ij}
$$

In our case, this comes out to: $$0.3 \times 15000 + 0.7 \times 0=$4500$$.  
Thus the Expected Value of Perfect Information is defined as the additional amount we are willing to pay to get to EV|PI:

$$
EVPI=EV|PI-EMV=4500-1000=$3500
$$

Thus, we are willing to pay a maximum of $3500 to fully resolve the uncertainty of whether our decision will yield the expected savings or not.

But the example we have described is not a real-world example. In the real world, even if we pay an expert to help us resolve this, they are not infallible. They might increase the odds in our favour, but there is always a possibility that they are wrong. We assume that we get an expert to consult for us. **They want to be paid $3400. Are they overpriced or not?**

We'd like to know what is the maximum we are willing to pay an expert if they can give us imperfect information about our situation. To do this, we will need to quantify our confidence in the expert.

Assume that if there are savings to be made, the expert says "Good" 80% of the time. If there are no savings to be made, the expert says "Bad" 90% of the time. This quantifies our confidence in the expert, and can be written as a table like so:

| Savings (S) / Expert (E) | Good | Bad |
|--------------------------|------|-----|
| Savings                  | 0.8  | 0.1 |
| No Savings               | 0.2  | 0.9 |

In the above table, E is the random variable representing the opinion of the expert, and S is the random variable representing the realisation of savings. We can again represent all possibilities via a probability tree, like so:

{% mermaid %}
graph LR
A ==> savings["Savings<br>P(X)=0.3"]
A ==> no_savings["No Savings<br>1-P(X)=0.7"]
savings --> expert_good_1["Good<br>P(R)=0.8"]
savings --> expert_bad_1["Bad<br>1-P(R)=0.2"]
no_savings --> expert_good_2["Good<br>P(R)=0.1"]
no_savings --> expert_bad_2["Bad<br>1-P(R)=0.9"]
expert_good_1 --> p_1["P(Good,Savings)=0.3 x 0.8 = 0.24"]
expert_bad_1 --> p_2["P(Bad,Savings)=0.3 x 0.2 = 0.06"]
expert_good_2 --> p_3["P(Good,No Savings)=0.7 x 0.1 = 0.07"]
expert_bad_2 --> p_4["P(Bad,No Savings)=0.7 x 0.9 = 0.63"]
p_1-->p_good["P(Good)=0.24+0.07=0.31"]
p_3-->p_good
p_2-->p_bad["P(Bad)=0.06+0.63=0.69"]
p_4-->p_bad
{% endmermaid %}

We now have our joint probabilities $$P(S,E)$$. What we really want to find is $$P(S \vert E)$$. By Bayes' Rule, we can write:

$$
P(S|E)=\frac{P(S,E)}{P(E)}
$$

We can thus calculate the conditional probabilities of the payoff given the expert's prediction with the following graph.

{% mermaid %}
graph LR
p_1["P(Good,Savings)=0.3 x 0.8 = 0.24"]-->p_good["P(Good)=0.24+0.07=0.31"]
p_2["P(Bad,Savings)=0.3 x 0.2 = 0.06"]-->p_bad["P(Bad)=0.06+0.63=0.69"]
p_3["P(Good,No Savings)=0.7 x 0.1 = 0.07"]-->p_good
p_4["P(Bad,No Savings)=0.7 x 0.9 = 0.63"]-->p_bad
p_1 --> p_savings_good["P(Savings | Good)=0.24/0.31=0.774"]
p_good --> p_savings_good
p_2 --> p_savings_bad["P(Savings | Bad)=0.06/0.69=0.087"]
p_bad --> p_savings_bad
p_3 --> p_no_savings_good["P(No Savings | Good)=0.07/0.31=0.226"]
p_good --> p_no_savings_good
p_4 --> p_no_savings_bad["P(No Savings | Bad)=0.63/0.69=0.913"]
p_bad --> p_no_savings_bad
{% endmermaid %}

Now we go back and calculate EMV again in the light of these new probabilities. The difference in this new tree is that in addition to the probability branches of our original uncertainty, we also need to add the branches for the expert's predictions, whose conditional probabilities we have just deduced.

{% mermaid %}
graph LR
A ==>|0.31| p_good[Good]
A ==>|0.69| p_bad[Bad]
p_good ==> p_implement_good[Implement]
p_good --> p_dont_implement_good[Do Not Implement]
p_bad --> p_implement_bad[Implement]
p_bad ==> p_dont_implement_bad[Do Not Implement]

p_implement_good ==>|-5000| implement_savings_given_good["Savings=20000<br>P(Savings|Good)=0.774"]
p_implement_good ==>|-5000| implement_no_savings_given_good["Savings=0<br>P(No Savings|Good)=0.226"]
p_dont_implement_good -->|0| dont_implement_savings_given_good["Savings=0<br>P(Savings|Good)=0.774"]
p_dont_implement_good -->|0| dont_implement_no_savings_given_good["Savings=0<br>P(No Savings|Good)=0.226"]

p_implement_bad -->|-5000| implement_savings_given_bad["Savings=20000<br>P(Savings|Bad)=0.087"]
p_implement_bad -->|-5000| implement_no_savings_given_bad["Savings=0<br>P(No Savings|Bad)=0.913"]
p_dont_implement_bad ==>|0| dont_implement_savings_given_bad["Savings=0<br>P(Savings|Bad)=0.087"]
p_dont_implement_bad ==>|0| dont_implement_no_savings_given_bad["Savings=0<br>P(No Savings|Bad)=0.913"]

implement_savings_given_good ==> implement_savings_given_good_payoff["0.774 x (20000-5000)=11610"]
implement_no_savings_given_good ==> implement_no_savings_given_good_payoff["0.226 x (0-5000)=-1130"]
dont_implement_savings_given_good --> dont_implement_savings_given_good_payoff["0.774 x 0=0"]
dont_implement_no_savings_given_good --> dont_implement_no_savings_given_good_payoff["0.226 x 0=0"]

implement_savings_given_bad --> implement_savings_given_bad_payoff["0.087 x (20000-5000)=1305"]
implement_no_savings_given_bad --> implement_no_savings_given_bad_payoff["0.913 x (0-5000)=-4565"]
dont_implement_savings_given_bad ==> dont_implement_savings_given_bad_payoff["0.087 x 0=0"]
dont_implement_no_savings_given_bad ==> dont_implement_no_savings_given_bad_payoff["0.913 x 0=0"]

implement_savings_given_good_payoff ==> plus(("+"))
implement_no_savings_given_good_payoff ==> plus
plus ==> max_payoff_given_good[10480] ==> max_payoff[10480 X 0.31=3249]
{% endmermaid %}

Thus, $3249 is the maximum amount we'd be willing to pay this expert given the level of our confidence in them. This number is the **Expected Value of Imperfect Information**. Remember that the EVPI was $3500, so EVII <= EVPI. If you remember, the expert's fee was $3400. This means that we would be overpaying the expert by $3400-$3249=$151.

## Articulating Value: The Cost of Unreleased Software

[This spreadsheet](https://docs.google.com/spreadsheets/d/1jBHwntpPI3QK5rM5yw5m2Gge9otgDf7pddNZs1sBZlw/edit?usp=sharing) contains all the calculations.

![Incremental Releases Calculations](/assets/images/incremental-releases-calculations.png)

![Incremental Releases Graph](/assets/images/incremental-releases-graph.png)

## Static NPV Analysis Example: Circuit Breaker and Microservice Template

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

The current cash outflow projected over 10 months, discounted to today, comes out to -$87785. This is the first step towards convincing stakeholders that they are losing money. Of course, we can project further out into the future, but the uncertainty of calculations obviously grows the more you go out.

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

## Articulating Value: The Value of a Software System

**There is no consensus on how value of engineering practices should be articulated.** Metrics like DORA metrics can quantify the speed at which features are released, but the ultimate consequences - savings in effort, eventual profits, for example -- are seldom quantified. It is not that estimates of these numbers are not available; it is discussed when making a business case for the investment into a project, but those numbers are almost never encountered or leveraged by engineering terms to articulate how they are progressing towards their goal. The measure of progress across iterations is story points, which is useful, but that is just quantifying the run cost, instead of the actual final value that this investment will deliver.

How do we then articulate this value?

**Economics and current accounting practices can show one way forward.**

One straightforward way to quantify software value is to turn to **Financial Valuation** techniques. **Ultimately, the value of any asset is determined by the amount of money that the market wants to pay for it.** Software is an **intangible asset**. Let's take a simple example: suppose the company which owns/builds a piece of software is being **acquired**. This software could be for its internal use, e.g., accounting, order management, etc., or it could be a product that is sold or licensed to the company's clients. This software needs to be valued as part of the acquisition valuation.

The question then becomes: **how is the valuation of this software done?**

There are several ways in which valuation firms estimate the value of software.

### 1. Cost Approach

This approach is usually used for valuing internal-use software. The cost approach, based on the principle of replacement, determines the value of software by considering the expected cost of replacing it with a similar one. There are two types of costs involved: reproduction costs and replacement costs. **Reproduction Costs** evaluate the cost of creating an exact copy of the software. **Replacement Costs** measure the cost of recreating the software's functionality.

- **Trended Historical Cost Method:** The trended historical cost method calculates the actual historical development costs, such as programmer personnel costs and associated expenses, such as payroll taxes, overhead, and profit. These costs are then adjusted for inflation to reflect the current valuation date. However, implementing this method can be challenging, as historical records of development costs may be missing or mixed with those of operations and maintenance.

- **Software engineering model method:** This method uses specific metrics from the software system, like size/complexity, and feeds this information to some empirical software development models like COCOMO (Constructive Cost Model and its sequels) and SLIM (Software LIfecycle Management) to get estimated costs. The formulae in these models are derived from analyses of historical databases of actual software projects.

See [Application of the Cost Approach to Value Internally Developed Computer Software: Williamette Management Associates](https://willamette.com/insights_journal/18/summer_2018_4.pdf) for some comprehensive examples of this approach.

Obviously, this approach completely ignores the actual value that the software has brought to the organisation, whether it is in the form of reduced Operational Expenses, or otherwise.

### Market Approach
The market approach values software by comparing it to similar packages and taking into account any variations. One issue with this method is the lack of comparable transactions, especially when dealing with internal-use software designed to specific standards. More data is available for transactions related to software development companies' shares compared to software. This method could be potentially applicable to internal-use systems which are being developed even though there are commercial off the shelf solutions available; this could be because the COTS solutions are not exact fits to the problem at hand, or lack some specific features that the company could really do with.

### Income Approach
The Income Approach values software based on its future earnings, cash flows, or cost savings. The discounted cash flow method calculates the worth of software as the present value of its future net cash flows, taking into account expected revenues and expenses. The cash flows are estimated for the remaining life of the software, and a discount rate that considers general economic, product, and industry risks is calculated. If the software had to be licensed from a third party, its value is determined based on published license prices for similar software found in intellectual property databases and other sources.

The Income Approach is usually the one used most often by corporate valuation companies when valuing intangible assets like software during acquisition. However, this software is usually assumed to be complete, and serving its purpose, and not necessarily software which is still in development (or not providing cash flows right now).

- **Discounted cash flow method:** This is the usual method where an NPV analysis is done on projected future cash flows arising from the product.
- **Relief from Royalty Method:** This method is used to determine the value of intangible assets by taking into account the hypothetical royalty payments that would be avoided by owning the asset instead of licensing it. The idea behind the RRM is straightforward: owning an intangible asset eliminates the need to pay for the right to use that asset. The RRM is commonly applied in the valuation of domain names, trademarks, licensed computer software, and ongoing research and development projects that can be associated with a particular revenue stream, and where market data on royalty and license fees from previous transactions is available. One possible example is if a company is building its own private cloud as an alternative to AWS; the value that the project provides could be calculated from the fees that are projected to be saved if the company did not use AWS for hosting its services.

### Real Options Valuation

This is used when the asset (software) is not currently producing cash flows, but has the potential to generate cash flows in the future, incorporating the idea of the uncertain nature of these cash flows. We will look at this in more detail. Specifically, we will look at the most commonly used technique for valuing real options. The paper [Modeling Choices in the Valuation of Real Options: Reflections on Existing Models and Some New Ideas](https://realoptions.org/openconf2011/data/papers/24.pdf) discusses classic and recent advances in the valuation of real options. Specifically surveyed are:

- **Black-Scholes Option Pricing formula**: The original, rigid assumptions on underlying model, not originally intended for pricing real options
- **Binomial Option Pricing Model:** Discrete time approximation model of Black-Scholes; not originally intended for pricing real options
- **Datar-Matthews Method:** Simulation-based model with cash flows as expert inputs; no rigid assumptions around cash flow models
- **Fuzzy Pay-off Method:** Payoff treated as a fuzzy number with cash flows as expert input; no rigid assumptions

I admit that I'm partial to the Binomial Option Pricing Model, because the binomial lattice graphic is very explainable; we'll cover the **Binomial Option Pricing Model** and the **Datar-Matthews Method** in a sequel.

### What approach do we pick?

**1. Platform**  
**Use: Real Option Valuation**  
A platform by itself does not provide value; it is the opportunities that it creates to rapidly build and offer new products to the market that is its chief attraction.

**2. Specific Products providing External Transactional Value**  
**Use: Income, Market**

**3. Internal-Use products**  
**Use: OpEx NPV Analysis, Relief from Royalty**

**4. Enterprise Modernisation initiatives**  
**Use: Real Option Valuation, OpEx NPV Analysis**

Enterprise Modernisation can certainly benefit from an NPV analysis of Operational Expenses, but the main reason for undertaking modernisation is usually creating options for greater traffic, a more diverse product portfolio, etc.

**5. Maintenance**  
**Use: OpEx NPV Analysis**


## Articulating Value: Pair Programming

Pair programming effectiveness seems to be a mixed bag, based on a survey of multiple studies in the paper [The effectiveness of pair programming: A meta-analysis](https://www.researchgate.net/publication/222408325_The_effectiveness_of_pair_programming_A_meta-analysis).

The key takeaway is this:

>  If you do not know the seniority or skill levels of your programmers, but do have a feeling for task complexity, then employ pair programming either when task complexity is low and time is of the essence, or when task complexity is high and correctness is important.

## There is not Enough Data

The title of this topic is somewhat misleading, in that there is enough data, but that it is ignored or not synthesized, beyond qualitative platitudes of success.

## References
- Books
  - Real Options Analysis
- Papers
  - Real Options
    - [How Do Real Options Concepts Fit in Agile Requirements Engineering?](https://www.researchgate.net/publication/221541824_How_Do_Real_Options_Concepts_Fit_in_Agile_Requirements_Engineering)
    - [Decision Analysis and Real Options: A Discrete Time Approach to Real Option Valuation](https://www.researchgate.net/publication/220461843_Decision_Analysis_and_Real_Options_A_Discrete_Time_Approach_to_Real_Option_Valuation)
    - [Modeling Choices in the Valuation of Real Options: Reflections on Existing Models and Some New Ideas](https://realoptions.org/openconf2011/data/papers/24.pdf)
  - Valuation
    - [Illustrative Example of Intangible Asset Valuation: Shockwave Corporation](https://www.oecd.org/tax/transfer-pricing/47426115.pdf)
    - [The Valuation of Modern Software Investment in the US](https://www.researchgate.net/publication/351840180_THE_VALUATION_OF_MODERN_SOFTWARE_INVESTMENT_IN_THE_US)
    - [Information Technology Investment: In Search of The Closest Accurate Method](https://www.sciencedirect.com/science/article/pii/S187705091931837X/pdf?md5=8ef46147c1296b09b1a4945fe12a8db1&pid=1-s2.0-S187705091931837X-main.pdf)
    - [The Business Value of IT; A Conceptual Model for Selecting Valuation Methods](https://www.researchgate.net/publication/239776307_The_Business_Value_of_IT_A_Conceptual_Model_for_Selecting_Valuation_Methods)
    - [The effectiveness of pair programming: A meta-analysis](https://www.researchgate.net/publication/222408325_The_effectiveness_of_pair_programming_A_meta-analysis)
    - [Software Economics: A Roadmap](https://www.researchgate.net/publication/2411293_Software_Economics_A_Roadmap)
- Web
  - Decision Theory
    - [Video on Expected Value of Perfect and Imperfect Information](https://www.youtube.com/watch?v=jOafCEFZ1_8)
  - Software Valuation
    - [Application of the Cost Approach to Value Internally Developed Computer Software: Williamette Management Associates](https://willamette.com/insights_journal/18/summer_2018_4.pdf)
    - [Valuation of Software Intangible Assets by Willamette Management Associates](https://immagic.com/eLibrary/ARCHIVES/GENERAL/WMA_US/W020828T.pdf)
    - [Parameters of Software Valuation: Finantis Value](https://www.finantisvalue.com/en/2018/03/21/how-to-estimate-the-value-of-your-software-or-digital-products/)
    - [Valuing Software Assets from an Accounting Perspective: EqVista](https://eqvista.com/business-assets/value-software-asset/)
  - Software Accounting
    - [Overview of Software Capitalisation Rules](https://leasequery.com/blog/software-capitalization-us-gaap-gasb/)
    - [Accounting for external-use software development costs in an agile environment](https://www.journalofaccountancy.com/news/2018/mar/accounting-for-external-use-software-development-costs-201818259.html)
    - External Use Software guidelines - FASB Accounting Standards Codification (ASC) Topic 985, Software
    - Internal Use Software guidelines - FASB Accounting Standards Codification (ASC) Topic 350, Intangibles — Goodwill and Other
    - [Accounting for internal-use software using Cloud Computing development costs](https://leasequery.com/blog/asc-350-internal-use-software-accounting-fasb/)
    - [Accounting for Cloud Development Costs](https://www.pwc.com/us/en/services/consulting/cloud-digital/cloud-transformation/cloud-computing.html) are covered under FASB Subtopic ASC 350-40 (Customer’s Accounting for Implementation Costs Incurred in a Cloud Computing Arrangement That Is a Service Contact (ASC 350-40)).
    - [Financial Reporting Developments: Intangibles - goodwill and other](https://assets.ey.com/content/dam/ey-sites/ey-com/en_us/topics/assurance/accountinglink/ey-frdbb1499-05-09-2022.pdf?download). The actual formal document is [here](https://fasb.org/document/blob?fileName=ASU%202021-03.pdf)
