---
title: "Lisp Continuations"
author: avishek
usemathjax: false
mermaid: false
tags: ["Continuations", "Lambda Calculus", "Lisp"]
draft: false
---

*Some Stuff*

- A simple addition, and chaining without return values
- A simple case of recursion - just doubling
- Not allowed to use return values
- Parameters are the results
- Lambdas as boxes: a useful mental model
- Boxes within boxes: nesting boxes through recursive calls
- Boxes within boxes: triggering the outer box, and the progressive box unlocking as execution flow
- A basic nested list operation
- Nest sibling boxes also
- Why continuations?

# A simple addition, and chaining without return values

Let's start with something simple: addition. We can write an expression like so:

```racket
(+ 1 (+ 2 3))
```

This gives us 6. What is the sequence of function calls? Let's break it down, assigning intermediate names to the results:

```racket
(define 1plus2 (+ 1 2)
(define 1plus2plus3 (+ 3 1plus2))
```

The function `+` is evaluated first, its *result* assigned to `1plus2`, and then subsequently to add 2 to it.

Let's set ourselves a little challenge: we want the same result, but we are not allowed to use the results of functions, i.e., both inputs and outputs can only be passed through the function arguments ONLY. This means that `1plus2` cannot return 9, it needs some way to pass that 9 to the next operation, which is `(+ 2)`. How can we do that?

The obvious solution is to pass `(+ 2)` as a lambda; but obviously, we cannot use the `+` function, since it only takes in two arguments. So, let us define our own version of addition, which does this. Call it `add-cc`:

```racket
(define add-cc
    (λ(lhs rhs next)
        (next (+ lhs rhs))
    )
)
```

With this, we can rewrite our original addition as follows:

```racket
(add-cc 1 2 (λ(r) (+ 3 r)))
```

But why stop there? even the passed-in lambda could be written using `add-cc`, like so:

```racket
(add-cc 1 2 (λ(r1) (add-cc 3 r1 (λ(r2) ???? ))))
```

Well, what do we pass in as the last lambda? Why, the next operation of course. In this case, there is no next operation, so for convenience, we can pass in an identity function `(λ(r2) r2)`, so that we get:

```racket
(add-cc 1 2 (λ(r1) (add-cc 3 r1 (λ(r2) r2))))
```

We get the same result 6. This post concerns different ways to put together the above sequence of operations in an automated fashion. We automate sequences of operations all the time, for example when we perform recursion. Take the example of summing the first n natural numbers. Let's first define a curried `+.` first, because `+` does not have a curried version in Racket, and I need that version to illustrate a point (a single `(+ 1)` in Racket does not return a curried function).

```racket
(define +.
    (λ(lhs)
        (λ(rhs)
            (+ lhs rhs)
        )
    )
)
```

```racket
(define sum-n 
    (λ(n) 
        (cond 
            ((zero? n) 0) 
            (else ((+. n) (sum-n (sub1 n))))
        )
    )
)
```

For an example run of `(sum-n 3)`, the structure ends up being:

```racket
(+. 3 (+. 2 (+. 1 0)))
```

Let's write `sum-n` such that it does not use its return values for any computations. We will build it step by step. To start off, let us defined a curried version of `add-cc`, decomposing all the arguments, like so:

```racket
(define add-n-cc
  (λ(lhs)
    (λ(next)
      (λ(rhs)
        (next (+ lhs rhs))
      )
    )
  )
)

(define add-1-cc (add-n-cc 1))
(define add-2-cc (add-n-cc 2))
(define add-3-cc (add-n-cc 3))
```

Each of the above `add-1-cc`, `add-2-cc`, and `add-3-cc` define a function which takes in a `next` and the right hand side of the addition which is needed to complete the computation. When we use these, it will become clearer why the order of the arguments is the way it is.

What is the following expression equivalent to?

```racket
(add-1-cc (add-2-cc (add-3-cc (lambda(x) x))))
```
This fills up the `next`-shaped holes in each of the three `add-n-cc` variants, leaving only the innermost `λ(rhs)` functions of each nested inside each other. Performing substitution, gives us the following:

```racket
    (λ(rhs-1)
        ((λ(rhs-2)
            ((λ(rhs-3)
                ((λ(x) x)
                (+ 3 rhs-3))
            ) (+ 2 rhs-2))
        ) (+ 1 rhs-1))
    )
```

Note that this does not actually give you the result immediately, the above is still a lambda, which has been built and primed, but not triggered. To run this, we need to invoke this with 0. Let's perform the substitution step-by-step to see what happens:


```racket
    ((λ(rhs-1)
        ((λ(rhs-2)
            ((λ(rhs-3)
                ((λ(x) x)
                (+ 3 rhs-3))
            ) (+ 2 rhs-2))
        ) (+ 1 rhs-1))
    ) 0)

=       ((λ(rhs-2)
            ((λ(rhs-3)
                ((λ(x) x)
                (+ 3 rhs-3))
            ) (+ 2 rhs-2))
        ) (+ 1 0))

=       ((λ(rhs-2)
            ((λ(rhs-3)
                ((λ(x) x)
                (+ 3 rhs-3))
            ) (+ 2 rhs-2))
        ) 1)

=       ((λ(rhs-3)
            ((λ(x) x)
            (+ 3 rhs-3))
        ) (+ 2 1))

=       ((λ(rhs-3)
            ((λ(x) x)
            (+ 3 rhs-3))
        ) 3)

=       ((λ(x) x)
        (+ 3 3))

=       ((λ(x) x) 6)
=       6
```

A few things to note:

- From the structural point of view, the last calculation is the innermost lambda, and it is the last to execute. The ordinary return-version of the recursive `sum-n` is structurally inverted (`((+. 3) (+ 2 ((+. 1) 0)))`), but correspondingly, it is the innermost calculations run first.
- There is no way to delay the ordinary return version of the computation; you must put the 0 at the time of writing the expression. Writing `((+. 3) ((+. 2) (+. 1)))` is an error; whereas the expression `(add-1-cc (add-2-cc (add-3-cc (lambda(x) x))))` can be delayed as long as we like until we invoke it with 0.

