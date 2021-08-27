# Haskell AI
Implementation of a [basic neural network](https://iamtrask.github.io/2015/07/12/basic-python-network/) in Haskell using only the Prelude, [Data.Matrix](https://hackage.haskell.org/package/matrix-0.3.6.1) and [System.Random](https://hackage.haskell.org/package/random).
Written in less than 30 lines of code (could be even less without syntax sugar, print statements etc.).

## Run
Code runs using [Stack](https://docs.haskellstack.org/en/stable/README/) to manage imports. First clone the repository, then run

```bash
stack build
stack exec hai-exe
```
or alternatively just,

```
stack run
```

## Code
The code can be found in *app/Main.hs* and *src/Lib.hs*. The latter includes the *train* function whereas the former has the *main* monad which initializes the matrices and calls the *train* function.
The 10th and 11th line of *app/Main.hs* can be changed to experiment and in line 21 the amount of epochs can be changed (base is 100k but it converges way lower than that).
If the size of the network is changed then the size of the weight matrix (line 18) needs to be changed as well.
