---
title: "Fundamentals, not Buzzwords"
author: avishek
usemathjax: true
tags: ["Technology"]
draft: false
---
## A Tirade against Flavour of the Month

> "Well, in our country," said Alice, still panting a little, you'd generally get to somewhere else — if you ran very fast for a long time, as we’ve been doing."
>  
> "A slow sort of country!" said the Queen. "Now, here, you see, it takes all the running you can do, to keep in the same place.
> 
> If you want to get somewhere else, you must run at least twice as fast as that!" - Lewis Carroll

## Durable Foundations in a Shifting Landscape
React. Data Mesh. Keras. By the time, I finish writing this, probably half of these technologies will have been superseded by a new set of challengers. You'd be forgiven for thinking that the race to stay relevant is specifically dictated by the (often breakneck) speed at which you can assimilate the latest framework, or the latest "enterprise innovation".

More importantly, if you are a senior technologist, you are expected to have fluent opinions about the latest and greatest "in-thing"; and that's fine, since that becomes increasingly a major part of how you add value to clients. It might end up seeming like a race to merely maintain the position you are in.

This is obviously encouraged -- at times even mandated -- by the sometimes-stringent requirements of knowledge in a particular technology or framework. *Must know Kafka.* *Must know Data Engineering*. Our way or the highway. It also encourages specialisation: *I'm only comfortable doing .NET projects, so that's the only kind of work I'd like to pick up*.

Sure, I am not underestimating specialisation: indeed, specialisation results in efficiency of a sort, but only in that particular context. Take someone hyper-specialised in technology X, and drop them into an area where they will spend mostly working on that self-same technology, and you have created a dependency, a bottleneck, and a recipe for planned obsolescence, because sooner or later, that tech is consigned to the dust of technology history. There are hyperspecialisations which are fruitful, because the work in those areas have much larger ramifications for technology as a whole; pure mathematics, for example: we'd be nowhere as advanced without it (and by extension, translating its results via engineering disciplines), or medical science: I'd rather hope that a neurosurgeon operating on my brain has studied specifically about the brain extensively (with the accompanying practice).

But software technology (at least at the commercial applications level) is not that. Sure, you'd need good specialised knowledge if you are designing or programming an embedded system, or if you are writing code to send humans into space. But nooo, we are building a hyper-customisable self-service decentralised platform for accelerating analytics capabilities across the enterprise to provide unparalleled insights into the end-to-end value chain of the organisation.

Now, to be certain, I am not knocking the intent behind the buzzwords which have been strung together to create this description. But what I am at pains to point out, is that the future is always ready to eat your hyperspecialty for breakfast. What was amazing technology at the time I began my employment in 2004, is now commonplace, unremarkable.

There is no escaping learning new technology. However, I hypothesise can contribute to the longevity of your knowledge base without it being buffeted every so often by the winds of change. And that, in my opinion, is a firm focus on fundamentals.

This is not a new concept, but it is an often deprioritised one. What do I mean by "fundamentals"? Nothing mysterious, just more or less what it says on the tin.

> fun·da·men·tal /ˌfʌndəˈmentl/ serving as a basis supporting existence or determining essential structure or function - Merriam-Webster

The idea is that the fundamentals constitute a concept (or concepts) which underlie an observation or a practice or another group of concepts. Usually, the fundamentals end up being a sort of a unifying layer underlying multiple (potentially disparate) ideas.

## Examples

Let's take a look at some examples of what I consider as **'fundamental'**. Your corresponding definitions for those examples may differ, and that's quite alright; we'll address that in a moment.

Here's the lens applied to only a single aspect of the design of a transactional system, maybe a Payments integrator.

{% mermaid %}
graph TD;
payments[Payments System]
system_of_record[System of Record]
tradeoffs[Tradeoffs]
types_of_consistency_models[Eventual / Causal / etc. Consistency]
consistency_models[Consistency Models]
properties[Liveness / Safety]
model_checking[Model Checking]

system_of_record --> payments
tradeoffs --> system_of_record
types_of_consistency_models --> tradeoffs
consistency_models --> types_of_consistency_models
consistency_models --> model_checking
properties --> consistency_models

style payments fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
style system_of_record fill:#006fff,stroke:#000,stroke-width:2px,color:#fff
style tradeoffs fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff

{% endmermaid %}

A few notes before introducing the main lesson. We need to work with the actual system of record (possibly the payment service provider), and thus need to ensure consistency of data between the our system and the service provider. There is not a single way of ensuring consistency, and we would want to make tradeoffs depending upon the operational constraints that are present to us. But given knowledge of those consistency models, we are in a position to precisely state how the system will behave under unexpected conditions. Thus, we are able to enumerate failure scenarios and edge cases in a much more principled and disciplined fashion.

Taking it a step further leads us to a somewhat more formal (read mathematical) treatment of the properties of such a system. In a distributed system, those are usually **Safety** and **Liveness**. The next logical step would be an investigation of the techniques which can assure the existence of certain desirable properties (or the absence of undesirable ones). Model Checking is one such technique.

Of course, this is highly simplified, and the system design of such a component would cover a raft of other considerations. However, the point I'd like to make is this: 
Let's look at another, more detailed map, this time of a certain technique in Machine Learning, called Gaussian Processes.

{% mermaid %}
graph TD;
quad[Quadratic Form of Matrix]
chol[Cholesky Factorisation]
tri[Triangular Matrices]
det[Determinants]
cov[Covariance Matrix]
mvn[Multivariate Gaussian]
mvnlin[MVN as Linearly Transformed Sums of Uncorrelated Random Variables]
crv[Change of Random Variable]
gp[Gaussian Processes]
kernels[Kernels]
rkhs[Reproducing Hilbert Kernel Space]
mercers_theorem[Mercer's Theorem]

quad-->chol;
tri-->chol;
det-->chol;
cov-->mvn
chol-->mvn
mvn-->mvnlin
crv-->mvnlin
mvnlin-->Conditioning
mvnlin-->Marginalisation
Conditioning-->gp
Marginalisation-->gp
kernels--> gp
rkhs-->kernels
mercers_theorem-->kernels

style chol fill:#006f00,stroke:#000,stroke-width:2px,color:#fff
style mvn fill:#006fff,stroke:#000,stroke-width:2px,color:#fff
style gp fill:#8f0f00,stroke:#000,stroke-width:2px,color:#fff
{% endmermaid %}

The exact details of the graph are not very important; what is salient to note is that the nodes closer to the sources are progressively more generally applicable to a problem statement in the domain. As simple examples, understanding kernels, and by extension Mercer's Theorem and/or Reproducing Hilbert Kernel Spaces, equips us equally well to understand how coefficients are computed in a different ML technique called Support Vector Machines. As another, more trivial, example, understanding the concept of a covariance matrix and general matrix theory, equips us very well to understand the tools used in a lot of other ML techniques, ranging from Linear Regression to Principal Components Analysis.

Now, everyone will have a different concept of what "fundamental" is, for a different domain, and this is going to be subjective. For example, you can always go deeper than matrix theory and delve into Linear Algebra (and hence to Real Analysis, Abstract Algebra, etc.), or you might choose to stay in the realm of matrix algebra and be satisfied with its results. That is perfectly fine: everyone has their own comfort level, and it would be unreasonable to ask your vanilla developer to start studying Real Analysis just so they can use TensorFlow.

But the point is this: regardless of which level of fundamental detail you are comfortable with, this knowledge base is determinedly more resilient to change than the latest version of your favourite machine learning library and its APIs, or the latest buzzword that you may have been asked to get behind. Five years ago, it was Data Lake, now people speak of Data Mesh; several years later, there will be a new term for a new something. For sure, you will need to learn that specialised knowledge to some degree to be functional, but the foundations will always equip you well to work in a different environment, e.g., explaining the fundamentals to a curious client, improving existing implementations, making informed technology decisions based on your knowledge of the fundamentals, or even rolling your own implementations, if it comes to that. At its heart, you start to think more critically; you do not see technology as a writhing mass of disparate elements that you are forever racing to catch up to, but more of different aspects of a few underlying principles.

I thoroughly and strenuously object to the "Lost without APIs" attitude, which is fine when you have implementations to use in some prescribed fashion, within a limited context. But you do end up doing a disservice to yourself by stunting your growth as a technologist, if this is the stage at which you continue to work in.

### Some Suggestions

Far be it from me to claim any authority on how we can get from here to there: I can only offer a few suggestions based on what has worked for me.

#### Seek new ways of thinking about the same thing
#### Read some Theory
#### Put Theory to Practice
#### Don't Mind the Gap
#### Persevere

- Go a few knowledge levels deeper than what your current technology competency requires.
- 
To end this essay, I think I should let Heinlein have the last word.

> "A human being should be able to change a diaper, plan an invasion, butcher a hog, conn a ship, design a building, write a sonnet, balance accounts, build a wall, set a bone, comfort the dying, take orders, give orders, cooperate, act alone, solve equations, analyze a new problem, pitch manure, program a computer, cook a tasty meal, fight efficiently, die gallantly.
>
> Specialization is for insects." - Robert Anson Heinlein

Well, obviously, not everyone can do all of those things, so let us restrict ourselves for the moment to the world of 
