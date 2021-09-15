# MNIST Classifier From Scratch in Haskell
MNIST classifier using only the Prelude and [Data.Matrix](https://hackage.haskell.org/package/matrix-0.3.6.1).

## Run
Code runs using [Stack](https://docs.haskellstack.org/en/stable/README/) to manage imports. First clone the repository, then run

```zsh
stack build
stack exec hmnist-exe
```
or alternatively just,

```zsh
stack run
```

## Code
Everything interesting is in the [src/Network.hs](src/Network.hs) file. 

## Results

## Possible Additions/Changes
- [Repa arrays](https://hackage.haskell.org/package/repa) and [Repa algorithms](https://hackage.haskell.org/package/repa-algorithms-3.4.1.3)
