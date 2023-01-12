---
title: "A Tale of Unintentional Learning"
author: avishek
usemathjax: true
tags: ["Learning", "Vim"]
draft: false 
---

**TL;DR:** I went to using Vim once in a year to using it everyday by accident when I got into a flow mindset after the effort of understanding a Machine Learning paper. It feels like a miracle.

I never intended on learning **Vim**. I was doing fine without it. I don't even think it was anywhere near the top of my list of things to learn this year. Nor next year. Nor...well, you get the point. Truth is: I've only had to use Vim in some exceptional circumstances; that too not because it was the only editor available, but because I was too lazy to change my editor to something else when amending Git commits. Thus, I knew the bare minimum. In this context, "bare minimum" equals knowing how to exit Vim (yes, I know it's ```:wq```; get off my back!). The rest was about as uncivilised as you could expect: going into **Insert** mode and navigating using the arrow keys (horror of horrors!). It was slow, felt like pulling teeth, and I was just not motivated to learn it well enough to use it half-decently.

Well, that's not *quite* true. I had tried giving Vim a *serious* try over the last decade or so. Every so often, I'd fire up Vim and think to myself: "Today's the day!". I'd open up [OpenVim](https://www.openvim.com) or **vimtutor** and start doing a few exercises. This lasted about...10 minutes tops, before I either got distracted by the next shiny thing, and would abandon the endeavour. In fact, in retrospect, I am pretty sure that at the back of my mind, I was looking for *any* excuse to abandon my efforts. It didn't necessarily feel like an uphill battle; the value system in my brain simply complained that I could be learning other things, more *useful* things, and at some point, it won out.

Truth be told, I am not a fast typist. I cannot touch-type, and while programming, I'd be lost without my IntelliJ IDEA shortcuts. I suppose that is a saving grace; I avoid using the mouse as much as possible when coding, thanks to some good habits drilled into me by an amazing mentor at the start of my career (thanks [Fred](https://twitter.com/fgeorge52)!). But still, everywhere else, it's been a combination of arrow keys, Page Up/Down, and a lot of eye-rolling.

Now comes the good bit, or at least what I consider the interesting part.

I had been working on implementing a Machine Learning paper for the past couple of weeks ([this one](https://arxiv.org/abs/2112.05131), in case anyone is interested). As part of my learning process, I was documenting the different concepts I was learning, under the firm belief that trying to explain it to an audience would expose holes in my understanding (the full series starts from [here]({% post_url 2021-07-19-functional-analysis-results-for-operators %})). The details are not too important, except that I was nearing the end of the series of blog posts I was writing. It was 3 am and I was about to write the last few paragraphs before heading to bed. Which was when a particular thought occurred.

The thought was: **"Wouldn't it be cool if I could write the rest of this in Vim?"**

To this day, I will not be able to fathom why this particular thought occurred to me right there, right then, right in that mental state. There was no particular rhyme or reason why it came to me; I just needed to finish the post, proof-read it, and then publish it to my blog, and be done for the day (night?). Maybe all those failed attempts at learning Vim over the years had bred a sort of fierce combination of regret and longing; maybe it was something else. I will never know.

I launched Vim, and started editing the file, resolving to not touch the arrow keys, no matter how long it took me to know the Vim way of navigating. Alright, maybe I wasn't *that* harsh on myself; I did use the arrow keys but not as a tool for navigating wide swathes of text.

**Thought process follows:**

- *I need to get to the end of the file.*
- Google "go to end of file in vim". *Ahh, now how do I start editing?* Well, I know that one, it's ```i```.
- But *no*, it brings the cursor just before the last character.
- Google "edit after last character". Try it out a few times. *Hey, this is pretty cool.*
- ...

You know where this is going. The above sequence repeated itself many times during the course of the next hour or so. Now, the point I wanted to highlight is not the above sequence of events that I went through. No.

**It is the fact that picking up and reusing the commands felt effortless.**

Now, I will be the first to admit that there was no Matrix-like "I know kung-fu" moment. But, if I had to analyse my mental state at that point, in retrospect, there were two things I recall about it.

- **I was exhausted.** Not exhausted in the sense of drowsy and drained; more along the lines of feeling like a rubber band which had stayed stretched for too long because of all the learning I'd been doing recently, including that night.
- **I was still in a state of learning.** I was in a state of flow. It was very probably a side effect of all the effort I'd expended in understanding new concepts in the paper I was implementing.

Putting the two things together, the best analysis I can come up with is this: **I was in the state where anything that I came across, I could start picking it up without too much of kickstarting my brain to motivate itself to learn.**

Am I a Vim ninja now? *Heck, no.* Do I still use arrow keys sometimes? *Oh yes.* Am I using Vim a lot? *Yes and no.* I'm still using **IntelliJ IDEA** or my work, but I've enabled Vim mode in its editor windows. For notes and stuff at work, I've switched to NeoVim full-time, having abandoned **Sublime Text**.

But I do try to put the Vim philosphy to use every day as part of my editing...whether it's text or code.

I will not add more about how enjoyable it is, and how I feel like going back to using other text editors is like devolving from using metal tools to using sticks and stones, except to say that it really is as good as enthusiasts and fanatics say it is.

I'm reasonably sure this could constitute a learning hack for me: **to learn something you aren't necessarily motivated to learn, learn something that you are naturally interested in. Once you are in that learning mode, learning the "less palatable" thing becomes just another thing to pick up.**

Maybe, this was a one-off, a rare moment of inspiration. Who knows?

But I sure intend to try it again, and see how it works :-)

