---
title: "Reverse Engineering with LLMs"
author: avishek
usemathjax: false
mermaid: true
tags: ["Software Engineering", "GenAI", Reverse Engineering", "Program Analysis"]
draft: false
published: true
---

A model that proposes unstructured facts is something you have to trust. A model that proposes a thousand candidates against a deterministic check is a *search*. Only the second one is worth building on.

![Reverse Engineering Loop](/assets/reverse-engineering-llm-loop.svg)

**The check is what converges here, not the model.** Coverage ranks the next target. The consistency check throws out the bad spec. The typechecker throws out the bad program. The model contributes candidate generation, which is the part it is genuinely good at, and contributes nothing to the decision, which is the part it is bad at.

Which is why "the LLM might write a bad spec" is not much of an objection. Of course it will. Iterating against a mechanical check costs almost nothing, and the bad ones never leave the loop.

## Spec to Code:

And not in bloody plaintext.

Today's AI-authored code is tomorrow's legacy code. That part is not new. The new part is the *rate*. It piles up faster than any team could previously manage, and nobody in the building feels they wrote it, which removes the last informal mechanism by which intent used to survive.

Recovering intent from code is the single most expensive thing our industry does, and in the general case it is undecidable. There is an entire consulting sector doing it badly and expensively, and I have spent a good deal of time in it.

So the direction is not a stylistic preference. **The spec goes forward into code. You do not reconstruct it afterwards.**

But note the trap, because it is currently the most popular thing in the field: a prose spec pointed forward is still unfalsifiable. The direction on its own buys you nothing. You have automated the drift instead of removing it. "Spec" here means whichever rung you actually paid for — guardrails, a declarative model, or types.

If your artifact of record is generated prose describing code that was generated from prose, there is no source of truth anywhere in that loop. You have a fast way to add to a system nobody understands.

---
