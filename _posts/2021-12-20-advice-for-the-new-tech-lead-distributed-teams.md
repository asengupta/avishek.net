---
title: "Advice for the New Tech Lead: The Realities of Distributed Development"
author: avishek
usemathjax: false
tags: ["Technology", "Leadership"]
draft: true
---

Test
**Note: I resurrected this post from 2013-2014 on my old site. Most of the things I've written about here, I still see as relevant even at the end of 2021. I've revised and added more information based on my own learnings since I originally wrote this.**

This is part of an ongoing blog post series “Advice for the new Tech Lead”.

I want to start by clarifying what I mean by Distributed Development, or at least the flavour that I’m accustomed to. It basically boils down to two teams, internally co-located, but separated from each other in time and space (a bit melodramatic, yes?) by a nontrivial quantity. My experience has mostly involved teams between India and the US/UK, but it’s not hard to substitute other relatively distributed locales.

Let’s face it; not everyone has been there and done that, when it has come to Distributed Development. And if you have, there is a high probability that you were probably in a distributed team, you mostly worked with one group or the other, but not both. Sure, there might have been trips across these groups for short or extended periods of time; but now you’re the Technical Lead, there are some things that you need to consciously start acknowledging. In some cases, embracing some of those discoveries. In many cases, combating their (ugly) implications.

These words I will probably keep repeating in the future, but I’m not apologising for them: Never be Complacent.

![Never Be Complacent](/assets/images/behold-the-field-in-which-i-grow.png)

The inherent characteristic of distributed teams is that there are certain constraints that you, as the tech lead, need to deal with. “Constraints” might seem like a negative term; however, as we’ll see, some constraints can actually be liberating, while others are…well…constraining. So, yeah, let’s get to it.

### Time Zones and Empowerment

A team fractures along the weakest seams. Really, almost anything does. In the case of a distributed team, that seam — I mean, those seams — tend to be geography and the circadian cycle. This is nothing new. What you, as the tech lead need to start realising, is that these put certain constraints on what you can and cannot do, during a 24-hour cycle.

If all your business decisions/prioritisation activities can mostly be done on the client side, having a remote analyst is doing a disservice to both the analyst and the remote team. The analyst, because he/she is not empowered to take decisions on the spot if needed. The team, because they are forced to subscribe to the decision cycle of the onshore clients.
Rotation between onshore and remote teams is a powerful tool in fostering a sense of empowerment within members of the remote members. Seeing, working and interacting with clients, closer to business realities, might just be the thing that makes everyone comfortable with the idea that the remote team “gets” it, and can function more or less autonomously.

### Geography and The Broken Telephone

Geography hurts. Geography distorts. Especially if most (or all) of your project stakeholders are located in one country, and most of your development team is somewhere else. So, your development team is great. It has brilliant developers who can get stuff done. None of that means much if your remote team is unable to make decisions for itself, and operate with some degree of autonomy.

This autonomy comes with getting more information about what is happening on the ground. Frequently, this information filters down in a very sanitised, watered-down fashion. “Bob and Greg had a screaming match over the business value of so-and-so feature.” becomes “There is still ongoing discussion about whether to work on so-and-so feature.” See, this is a problem. By abstracting away details of a situation, the reality that the onshore team/client is not a single-minded amorphous blob is obscured from the remote team. The remote team is already disadvantaged in terms of the “handedness” of the information that they receive (first-hand, second-hand). Do not work to further this Broken Telephone Effect.

Of course, you don’t want to turn all your communications with the remote team into a gossip channel. However, you will want to have more informal channels of discussion with the remote teams where some minutiae of the situation can be discussed without needing to be politically correct.

### Perceptions (Hint: Use Stoicism)

This is not the only problem your remote team faces. Geography distorts perception in subtle ways. You’ll inevitably encounter situations where people from the onshore team — your co-located non-client colleagues — will sometimes form opinions about the remote team that are based on very limited — or even no — interactions with that team. The flavour of these opinions is very wide-ranging, but the ones you’ll want to actively combat are the ill-informed ones. This is the most insidious form of chaos that seeps in, almost unnoticed, and then informs a wide range of discussions/choices that ultimately cause frustration on both sides.

Do not let such perceptions linger if you see them.

We are not done with the problems yet. Consulting is always a challenging task, in the best of circumstances. Sometimes, it is a matter of patience. Sometimes, it means talking to a few people, planting the germ of an idea in their heads, letting them ruminate on it, and hopefully take it forward. This is very evident if you are working with very closely with your client. See, the thing is, the remote team is not seeing this. Some of them are probably thinking “What do these guys onshore really do half of the time?” This can potentially breed a skewed opinion of the onshore team. Your job, as usual, is to combat this type of thinking.

In both these cases, the ounce of prevention, as well as the pound of cure, are multiple channels of communication. You will want to keep most of these informal; whether they are scheduled or unscheduled, is up to you. But, do not let the talking die down. Silence in distributed teams can be downright poisonous.

- [ ] Be the Driver (or, your team is in control of its destiny)
- [ ] Read, Borrow, or Steal (read how others have done it)
- [ ] Make Foundational Bets (or, don't chase the latest shiny)
- [ ] Call out Bullshit (nicely)
- [ ] Hold Others to a High Standard (and stick to it yourself)
- [ ] No One is an Island (or, leverage your network)
