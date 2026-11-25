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

This manual construction of the sum of 1, 2, 3, is the first step towards building the recursive definition in the Continuation Passing Style. But before moving to building the recursive version, I want to share the mental model which helps me visualise the structure that builds the core intuition about this way of doing things.

# Boxes within boxes

Picture the lambda as a box. If you are from an OO background, think of it as an object with a method and some state in it. The state is what is already in the box, i.e., the box does not depend upon anything outside to know the values of its state. This state is put in the box the moment it is constructed.
This box is inert, and does nothing unless the method in it is invoked. The computation the method does depends upon both the state of the box and the arguments to the method. Thus, the computation is dependent upon some things specified at construction time, and some things specified at execution time.`add-1-cc` is one such a box. It has some state which is specified at construction time, the `lhs` 1 when it is defined. Seen this way, the construction `(add-1-cc (add-2-cc (add-3-cc (lambda(x) x))))` is effectively a set of boxes within boxes. Each box has two items as its state: the `lhs` value and a smaller box.
To make this more explicit, this is what the boxes contain:

- Innermost box (identity): `(lambda(x) x)`
- Outer box 3 (`(add-3-cc (lambda(x) x))`): `lhs=3` and `next`= `(lambda(x) x)`
- Outer box 2 (`(add-2-cc (add-3-cc (lambda(x) x)))`): `lhs=2` and `next`= `(add-3-cc (lambda(x) x))`
- Outer box 1 (`(add-1-cc (add-2-cc (add-3-cc (lambda(x) x))))`): `lhs=1` and `next`= `(add-2-cc (add-3-cc (lambda(x) x)))`

This is the execution trace in words:

- When the outer box 1 is triggered with 0, it sums 0 and `lhs`=1, and calls its `next` (outer box 2) with 1.
- Outer box 2 is triggered with 1, it sums 1 and its `lhs`=2, and calls its `next` (outer box 1) with 3.
- Outer box 3 is triggered with 3, it sums 3 and its `lhs`=3, and calls its `next` (innermost box) with 6.
- The innermost box receives 6, and simply returns 6.

Another intuitive way of thinking of this construction is as a set of nested decorators. Each decorator calls its wrapped box with values: these values could be part of the decorator data itself as well as the parameters the outer decorator receives.

# Recursive construction

Having seen how the box unlocking works when constructed manually, we now have all the pieces to recursively construct `sum-n-cc` such that it gives us the same "boxes-within-boxes" artifact as above.

```racket
(define sum-n-cc
  (λ(n next)
    (cond
      ((zero? n) (next 0))
      (else
        (sum-n-cc (sub1 n)
              (λ(running-total)
                (next (+ n running-total))
                )
              )
      )
    )
  )
)
```

Every call to `sum-n-cc` with a value `n` wraps the current `next` lambda into a new lambda (box, decorator, etc.), and recursively calls `sum-n-cc` with this new wrapping lambda and `n - 1`, until `n` reaches 0; at which point the current lambda is triggered with a 0. Recognise that the unnamed lambda in `sum-n-cc` is the same as the innermost lambda in `add-n-cc`. In the manual recursion example, the values for `lhs` and `next` came from the wiring we did by hand; here, they are both wired in by the parameters of the recursive function, `n` and `next`.


