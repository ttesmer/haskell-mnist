# Haskell AI (HAI)
Minimalistic implementation of a [basic neural network](https://iamtrask.github.io/2015/07/12/basic-python-network/) with [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in Haskell using only the Prelude, [Data.Matrix](https://hackage.haskell.org/package/matrix-0.3.6.1) and [System.Random](https://hackage.haskell.org/package/random).

## Run
Code runs using [Stack](https://docs.haskellstack.org/en/stable/README/) to manage imports. First clone the repository, then run

```zsh
stack build
stack exec hai-exe
```
or alternatively just,

```
stack run
```

## Code
The code can be found in [app/Main.hs](app/Main.hs) and [src/Lib.hs](src/Lib.hs). The latter includes the **train** function whereas the former has the **main** monad which initializes the matrices and calls the **train** function.
The **x** matrix in (app/Main.hs)[app/Main.hs] can be changed (to experiment) and the amount of epochs in the **train** function can be changed (base is 60k but it converges way lower than that (~50 epochs)).

## Results
```zsh
Training Data X:
┌             ┐
│ 1.0 0.0 1.0 │
│ 1.0 1.0 1.0 │
│ 0.0 0.0 1.0 │
└             ┘
Desired Ouput Y:
┌     ┐
│ 1.0 │
│ 1.0 │
│ 0.0 │
└     ┘
Result After Training:
┌                      ┐
│   0.9968757999762028 │
│   0.9993985458752835 │
│ 3.899624248201951e-3 │
└                      ┘
```
The last matrix is the final approximation of the "goal matrix" above it after 100.000 epochs of training. As seen above, it converged to ~0.99 for the points where 1.0 was correct and to ~0.00389 where 0.0 was correct (The 3.89e-3 is scientific notation).

## Possible Additions
- MNIST Classifier
- [Repa arrays](https://hackage.haskell.org/package/repa) and [Repa algorithms](https://hackage.haskell.org/package/repa-algorithms-3.4.1.3)
- [Good read on randomness in Haskell](https://jtobin.io/randomness-in-haskell)
- [OG Tutorial "Future Work" section](https://iamtrask.github.io/2015/07/12/basic-python-network/)
