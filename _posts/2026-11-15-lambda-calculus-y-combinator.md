---
title: "Y Combinators for Fun and (No) Profit"
author: avishek
usemathjax: false
mermaid: false
tags: ["Y Combinator", "Lambda Calculus", "Recursion", "Scheme"]
draft: false
---

*In this post, we discuss the Y Combinator (no, not the company!) and attempt to gently introduce you to the derivation of the construct through small bite-sized refactoring steps done on a simple recursive function. The example is in Racket (and can run in Scheme if you replace the `λ` with `lambda`). It is simple enough to be ported without effort into most languages, functional-first or not.*

## Contents

- [Why the Y Combinator?](#why-the-y-combinator)
- [A Tentative Stab](#a-tentative-stab)
- [Interlude: Why does it recurse?](#interlude-why-does-it-recurse)
- [Delayed Evaluations vs. Infinity](#delayed-evaluations-vs-infinity)
- [Interlude: Why does this not recurse infinitely?](#interlude-why-does-this-not-recurse-infinitely)
- [Expansion: The Attack of the Parentheses](#expansion-the-attack-of-the-parentheses)
- [An Alternative Path to the Y Combinator](#an-alternative-path-to-the-y-combinator)
- [An infinite loop](#an-infinite-loop)
- [Making the infinite loop stop](#making-the-infinite-loop-stop)
- [Making the infinite loop do something useful](#making-the-infinite-loop-do-something-useful)
- [References](#references)

I have been revisiting **"The Little Schemer"** recently, and I keep continually getting reminded what a gem of a language Lisp is (any variant). Though I have arrived at Functional Programming through various angles, it was never a straight line; instead being a series of progressive dead ends, half-completed exercises, re-reading things from the start, applying some concepts over and over again in languages not originally meant to be functional, going over-zealous about it and burning myself...oh so many detours.

I still consider myself mostly a novice in this area.
So, I started re-reading the part about the Y Combinator, followed through the "length" exercises, and then promptly lost the plot after a little while. What followed after that was watching a bunch of videos, reading multiple well-intentioned articles and Github gists, using [Jane Street's Socratic Tutor skill](https://github.com/janestreet/agent_skills) - it's not that great, btw - and of course, the mandatory shouting at Claude Code in order to cover up my own frustrations of not being able to build up a mental model that I was comfortable with.

This post is an attempt to distill all of those hours of banging my head to come up with a coherent model of how the Y combinator works. Of course, none of this is new: it has been around since Haskell Curry discovered it, and there are tons of good explainers of the idea out there. This is just my spin on it.

I will be using **Scheme** for this post. I actually ran all of it on Racket, and the original Scheme interpreter works just as well. I will not deny that I find walls of parentheses rather unreadable beyond a certain threshold, but that probably speaks to how code should be structured in the first place, rather than how readable a Lisp is. In any case, as we will see, the constraints of the problem that leads us to the Y Combinator, somewhat force us to deal with more parentheses than I would normally find comfortable.

## Why the Y Combinator?

Most material about the Y-Combinator treats its rationale from a couple of different angles.

- It proves that **you do not need unbound names** to do recursion.
- It lets you build **recursion in languages which do not support recursion**, i.e., do not allow reference to functions by names.
- It lets you **separate out the actual step computation from control flow** (in this case, recursion).

All three appeal to me: the first two, from a pure mind-expanding, curiosity-whetting angle, and the third one from an engineering perspective. As programmers, we deal with a lot of abstractions, and useful abstractions make code easier to maintain, modify, and reuse.

The motivation behind Y Combinator may be best understood when we start with a vanilla example, and start putting some stringent constraints. Since, learning by doing is easily the best way to retain this material (together with writing about it), I not only wrote the code, but more importantly, did a dry run of the code by writing out the execution trace of the code for the simplest non-trivial case. I firmly believe that this builds intuition the best; sometimes even better than just writing the code and reasoning about it.

We start with the example of a recursive function which calculates the sum of numbers up to `n`. No special tricks here, just basic recursion.

```scheme
(define sum-of-n (λ(n)
                  (cond
                    ((zero? n) 0)
                    (else (+ n (sum-of-n (sub1 n))))
                    )
                   )
  )
```

At an abstract level, a function like this is doing two things:

- Deciding what logic to execute at every step; here it decides that getting a zero as its argument means it should return zero, and for other numbers, it should add the current argument to the rest of the computation (with the recursive part)
- Recursing, i.e., the actual logic of continuing the computation. This part is independent of what logic the function encapsulates.

The reason I separate out the two aspects is this: let's look at another function, the factorial:

```scheme
(define factorial (λ(n)
                    (cond ((zero? n) 1)
                          (else (* n (factorial (sub1 n))))
                      )
                    ))
```

If you examine this and the `sum-of-n` function, you will see that the recursion mechanic in both cases is the same, i.e., a call to itself with some new updated value. To drive the point home, look at the Fibonacci function, which has differently-shaped logic:

```scheme
(define fib (λ(n)
          (cond
            ((or (eq? n 1) (eq? n 2)) 1)
            (else (+ (fib (sub1 n)) (fib (- n 2)))
            )
            )
           ))
```

The logic here is quite different: there are two recursive calls, and they are joined by addition. However, the mechanism for the recursive call stays the same: a call to the function itself, together with some (hopefully updated) value.
The question that we'd like to pose is: Is there a way to decouple the mechanism of recursion from the logic of the recursive function? More concretely, we'd like to have the logic-specific function stick to deciding what calculations (loosely speaking) to run, and when it's time to do the next step, it should not have hardcoded a call to itself (which is the recursion bit).

This is a slightly different way of imposing our first constraint, i.e., perform recursion assuming you are not allowed to refer to function names at all; more strictly, perform recursion on a function which does not have a name to begin with.

We will explore the solution by performing small refactoring steps, so that the path from named functions to the final result is traceable.

## A Tentative Stab

One of the first things we usually reach for, when attempting to hide implementation details from a function, is to have it simply call another function that has been passed in. This by itself, does not solve the problem, but it is a start. Accordingly, we pass in a lambda `foo` that `sum-of-n` calls instead of invoking itself for recursion. But let's take a smaller step: pass in a `foo` which is not used at all, and thus, is effectively an unused argument, like so:

```scheme
(define sum-of-n-taking-foo (λ(foo n)
                  (cond
                    ((zero? n) 0)
                    (else (+ n (sum-of-n-taking-foo foo (sub1 n))))
                    )
                   )
  )
```

We can call this function like `(sum-of-n-taking-foo (λ() '()) 3)`, and this will still work. Any `foo` we pass in would be fine, because we have not even attempted to use it in any capacity (yet).

Let's try using `foo`, for recursion. For that we need to invoke `foo` as the recursive lambda instead of directly calling `sum-of-n-taking-foo`.

```scheme
(define sum-of-n-taking-foo-no-ref (λ(foo n)
                  (cond
                    ((zero? n) 0)
                    (else (+ n (foo foo (sub1 n))))
                    )
                   )
  )
```

This makes sense, but we need to talk about how we invoke such a method:

```scheme
(sum-of-n-taking-foo-no-ref sum-of-n-taking-foo-no-ref 10)
```

This should have the same desired effect because the `foo` that we are passing in, is the original lambda itself (in this case, `sum-of-n-taking-foo-no-ref`).

For compactness, let's put the body in a single line. This ends up being a lot more parentheses on a single line, but we will not need to parse all of the details on every line.

```scheme
(define sum-of-n-explicit-1
    (λ(foo n) (cond ((zero? n) 0) (else (+ n (foo foo (sub1 n)))))))
```

The next small refactoring: let's break up the outer lambda into two separate curried functions.

```scheme
(define sum-of-n-explicit-2
(λ(foo) (λ(n) (cond ((zero? n) 0) (else (+ n ((foo foo) (sub1 n))))))))
```

We can call this using `((sum-of-n-explicit-2 sum-of-n-explicit-2) 10)`. No real difference, it's just that the call is explicitly sequentialised.
One of the things we are aiming for is for the `sum-of-n` function to not worry about the mechanics of recursion. In the above code, we see that evaluating `(foo foo)` and then applying it to `(sub1 n)` makes the whole thing look ugly. Therefore, we wish to inject `(foo foo)` into the `λ(n)`. This means we "create" `λ(n)` by calling an intervening factory lambda which takes in a `foo-recur` which is bound to a `(foo foo)` passed into the `λ(n)` factory lambda. The factory lambda returns the `λ(n)` function.

Loosely speaking, we are aiming for something like below:

```scheme
sum-of-n-logic-factory = (λ(foo-recur) (λ(n) ...(else (+ n (foo-recur (sub1 n))))))
(sum-of-n-logic-factory (foo foo))
```
Written out properly without the `sum-of-n-logic-factory` name, we might write something like:

```scheme
(define sum-of-n-explicit-3
(λ(foo) ((λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n)))))))(foo foo))))
```

**However, the refactoring, as it stands, won't work because it recurses forever**, owing to eager evaluation of `(foo foo)`.

## Interlude: Why does it recurse?

Let's evaluate the following to understand why it recurses forever:

```scheme
((sum-of-n-explicit-3 sum-of-n-explicit-3) 1)
```

Let's put in some shorthand, so that our eyes don't bleed looking at all the parentheses. Assume that:

```scheme
LADDER = (λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n)))))))
```

so that we can write:

```scheme
E = (λ(foo) (LADDER (foo foo)))
```

**STEP 1:** We will now evaluate `((E E) 1)`.

**STEP 2:** So we need to evaluate `(E E)` first, let's substitute the second `E` which is the parameter to `foo`, so: `foo = E`.

We unwrap one level of λ, and put `E` wherever `foo` is, to get:

```scheme
(E E) = (LADDER (E E))
```

**STEP 3:** To evaluate the above, we need to evaluate `(E E)`. This is the same exact step as STEP 2, so we are now in an infinite recursion.

## Delayed Evaluations vs. Infinity

One fix is to delay the evaluation of `(foo foo)`, by wrapping it in a λ, evaluate it inside, like `(foo-recur)`, and then apply it to `n-1`. This works, but the sum-of-n logic has to remember to evaluate the λ, then apply it.

```scheme
(define sum-of-n-explicit-no-arg-recur
(λ(foo) ((λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n ((foo-recur) (sub1 n)))))))(λ() (foo foo)))))
```

To make the function oblivious of worrying about evaluating `foo-recur` and then applying it to `n-1`, we can enhance the `(λ() (foo foo))` lambda to also take in the `n-1` parameter and evaluate it. Now the `sum-of-n` logic just assumes the recursion lambda is a plain function it can just apply on `n-1`.

```scheme
(define sum-of-n-explicit-1-arg-recur
(λ(foo) ((λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n)))))))(λ(n) ((foo foo) n)))))
```

Note that the extra brackets around `foo-recur` are no longer present because we do not need to evaluate + apply inline: the evaluation happens inside `foo-recur` itself. Loosely speaking, we now have (in some Scheme-like pseudocode):

```scheme
sum-of-n-logic-factory = (λ(foo-recur) (λ(n) ...(else (+ n (foo-recur (sub1 n))))))
(sum-of-n-logic-factory (λ(n)(foo foo) n))
```

## Interlude: Why does this not recurse infinitely?

Let's verify that this does NOT recurse. The simple, if somewhat blase, explanation is given first. After that, we trace through a simple but nontrivial execution to solidify our intuition.

Let's evaluate `(E E)` again.

```scheme
E=(λ(foo) (LADDER (λ(n) ((foo foo) n))))
```

`LADDER` is the same, but we get a different result, on substitution:

```scheme
(E E) = (LADDER (λ(n) ((E E) n)))
```

The expression `(λ(n) ((E E) n))` does not get evaluated right then, since it is a lambda expression, in contrast to `(E E)`, which Scheme would have attempted to evaluate straight away. For completeness' sake, note that the original version would have worked in Haskell, because Haskell has lazy evaluation (see [References](#references)).

What is more instructive is to see the trace of what happens through a complete execution. We will use 2 as the input parameter, because that exercises both halves of the `sum-of-n` logic.

Let `E = sum-of-n-explicit-1-arg-recur`, then evaluate `((E E) 2)`. Then, substituting `E` everywhere `foo` is, we get:

```scheme
((λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n))))))) (λ(n) ((E E) n) ))
=> P = (E E) = (λ(n) (cond ((zero? n) 0) (else (+ n ( (λ(n) ((E E) n) ) (sub1 n))))))
```

Let's call `P(2)`:

```scheme
=> P(2) = (cond ((zero? 2) 0) (else (+ 2 ( (λ(n) ((E E) n)  ) (sub1 2)))))
=> P(2) = (+ 2 ( (λ(n) ((E E) n)  ) 1))
=> P(2) = (+ 2 ( (E E) 1))
=> P(2) = (+ 2 ( 
            (λ(n) (cond ((zero? n) 0) (else (+ n ( (λ(n) ((E E) n) ) (sub1 n))))))
                                              1))

=> P(2) = (+ 2
  (cond ((zero? 1) 0) (else (+ 1 ((λ(n) ((E E) n)) (sub1 1)))))
                          )

=> P(2) = (+ 2
  (cond ((zero? 1) 0) (else (+ 1 ((λ(n) ((E E) n)) 0))))
                          )

=> P(2) = (+ 2 
  (+ 1 ((λ(n) ((E E) n)) 0))
          )

=> P(2) = (+ 2 
  (+ 1 ((E E) 0))
          )


=> P(2) = (+ 2 
  (+ 1 (
            (λ(n) (cond ((zero? n) 0) (else (+ n ( (λ(n) ((E E) n) ) (sub1 n))))))
              0))
          )

=> P(2) = (+ 2 
  (+ 1
            (cond ((zero? 0) 0) (else (+ 0 ( (λ(n) ((E E) n) ) (sub1 0)))))
              )
          )

=> P(2) = (+ 2 (+ 1 0)) = 3
```

## Expansion: The Attack of the Parentheses

You can run `((sum-of-n-explicit-1-arg-recur sum-of-n-explicit-1-arg-recur) 10)` for the above. Now that we have (hopefully) convinced ourselves that this does work, let's take stock of where we are.
We have a function called `sum-of-n-explicit-1-arg-recur` which looks like:

```scheme
(define sum-of-n-explicit-1-arg-recur
(λ(foo) ((λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n)))))))(λ(n) ((foo foo) n)))))
```

We can call it like:

```scheme
((sum-of-n-explicit-1-arg-recur sum-of-n-explicit-1-arg-recur) 10)
```

As you can see, we are still using names. As the next step, if we go in and blindly expand `sum-of-n-explicit-1-arg-recur` into its body, we get the following:

```scheme
(
 (λ(foo) ((λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n)))))))(λ(n) ((foo foo) n))))
  (λ(foo) ((λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n)))))))(λ(n) ((foo foo) n))))
)
```

We are actually very close to the end, because, if you see the above definition, you will see that there are no references to any unbound names. The only references present are those of the bound arguments.

Let's refactor by pulling out the `sum-of-n` logic, which is the thing which differs in recursive walks; the recursion itself is always a standard pattern.

```scheme
(define sum-of-n-step-factory
  (λ(foo-recur) (λ(n) (cond ((zero? n) 0) (else (+ n (foo-recur (sub1 n)))))))
)

(define sum-of-n-explicit-5
(
 (λ(foo) (sum-of-n-step-factory (λ(n) ((foo foo) n))))
  (λ(foo) (sum-of-n-step-factory (λ(n) ((foo foo) n))))
)
)
```

`sum-of-n-step-factory` is a hardcoded step function in `sum-of-n-explicit-5`; it is called a step function because it returns a function which tells us how to take the next step; this can be either termination or further recursion. We can make the above more general by passing a (generic) step function into a factory function. **This factory function is the Y Combinator.** 

```scheme
(define y-combinator (λ(step-fn-factory)
(
 (λ(foo) (step-fn-factory (λ(n) ((foo foo) n))))
  (λ(foo) (step-fn-factory (λ(n) ((foo foo) n))))
)
))
```

We can call this like:

```scheme
((y-combinator sum-of-n-step-factory) 10)
```
Note what we have done: the `sum-of-n-step-factory` knows nothing about recursion. All it gets is a `foo-recur` lambda which it invokes when it needs to run the next step; from that angle, it contains only the program logic, not control flow logic. The **Y Combinator** knows nothing about the program logic, whether it be factorial, Fibonacci, or something completely different; all it knows is to take the program logic lambda, and convert it into a recursive lambda which fuses the program logic and control flow (in this case, recursion).

## An Alternative Path to the Y Combinator

We started with the `sum-of-n` function, and then tried to extract the recursion aspect out of it. We can arrive at the same definition, in a more piecemeal version. This path, in my opinion, starts much more simply, and is how I got to the Y Combinator *myself* without constantly getting lost; thus mentally it is a superior path for me to remember the basis for the Y Combinator. Remember though, that this is how my own journey of understanding went: yours may very well differ, and I urge you to build your own mental model to understanding.

## An infinite loop

First, we start with the infinite loop: it does nothing, and it runs forever.

```scheme
(define infinite-loop (λ(step) (step step)))

(infinite-loop infinite-loop)
```

How does this work? Let's trace it:

```scheme
((λ(step) (step step)) infinite-loop)
=> (infinite-loop infinite-loop)
...
```
The second step ends up being the same as the original invocation, so it repeats ad infinitum.

Our task is two-fold:

- Make this infinite loop stop at some point
- Make this infinite loop do something useful

## Making the infinite loop stop

Let's start with the first task: making this infinite loop stop. We can simplify this task further, by making it stop immediately.

```scheme
(define immediately-stopping-infinite (λ(step) 
    (cond
      ((eq? #t #t) 0)
      (else (step step))
    )
  )
)

(immediately-stopping-infinite immediately-stopping-infinite)
```

The above stops immediately because it checks whether `#t == #t` (which is obviously true) and returns 0. The `else` branch, which would have taken the infinite loop path, is never invoked.

## Making the infinite loop do something useful

We are now in a good position to build out our `sum-of-n` functionality straight away; the pieces are all there. The `(eq? #t #t)` is the placeholder for the recursion termination condition, and the `else` branch does the recursive part, in addition to adding to the current value. We need to add the `n` parameter to this function to carry the decrementing counter.

```scheme
(define sum-of-n-1 (λ(step n)
    (cond
      ((zero? n) 0)
      (else (+ n (step step (sub1 n))))
    )
  )
)

(sum-of-n-1 sum-of-n-1 5)
```

We are almost at the point at which we can extract out the Y Combinator; technically, you can inline the invocation right here, and you'd have achieved your goal, but the Y Combinator requires some more mechanical refactoring. The next step is very mechanical: break up `λ(step n)` into two separate curried functions.

```scheme
(define sum-of-n-curried
  (λ(next)
    (λ(n)
      (cond 
        ((zero? n) 0)
        (else (+ n ((next next) (sub1 n))))
      )
    )
  )
)

((sum-of-n-curried sum-of-n-curried) 5)
```
I won't get into the rest of the derivation here, since it is the same from here on out.

## References

- [Y Combinator in Haskell](https://gist.github.com/lukechampine/a3956a840c603878fd9f)
- [The Little Schemer](https://amazon.in/Little-Schemer-MIT-Press/dp/0262560992)
- [Essentials: Functional Programming's Y Combinator - Computerphile](https://www.youtube.com/watch?v=9T8A89jgeTI)

