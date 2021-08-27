# Haskell AI
Implementation of a [basic neural network](https://iamtrask.github.io/2015/07/12/basic-python-network/) in Haskell using only the Prelude, [Data.Matrix](https://hackage.haskell.org/package/matrix-0.3.6.1) and [System.Random](https://hackage.haskell.org/package/random).
Written in less than 30 lines of code (could be even less without syntax sugar, print statements etc.).

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
The code can be found in *app/Main.hs* and *src/Lib.hs*. The latter includes the *train* function whereas the former has the *main* monad which initializes the matrices and calls the *train* function.
The 10th and 11th line of *app/Main.hs* can be changed to experiment and in line 21 the amount of epochs can be changed (base is 100k but it converges way lower than that (~50 epochs)).
If the size of the network is changed then the size of the weight matrix (line 18) needs to be changed as well.

## Results
```zsh
user :: ~/hai ‹master› » stack build
user :: ~/hai ‹master› » stack exec hai-exe
Base Matrix (X):
┌             ┐
│ 1.0 0.0 1.0 │
│ 1.0 1.0 1.0 │
│ 0.0 0.0 1.0 │
└             ┘
Goal Matrix (Y):
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
