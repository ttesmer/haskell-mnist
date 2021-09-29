# MNIST Classifier From Scratch in Haskell
Handwritten digit classifier trained on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Written using one [linear algebra package](https://hackage.haskell.org/package/hmatrix) from scratch in Haskell.
Architecture of the network is implemented according to the first two chapters of the [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) book. Specifically, the [quadratic loss function](https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function) and mini-batch [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) are the heart of the implementation.

## Prerequisites
The repo uses [BLAS](http://www.netlib.org/blas/) and [LAPACK](http://www.netlib.org/lapack/) through [hmatrix](https://hackage.haskell.org/package/hmatrix) for basic linear algebra operations. You will need to install both of them:

### On Linux:
```zsh
sudo apt-get install libblas-dev liblapack-dev
```

### On MacOS:
This has not been tested. MacOS apparently ships with BLAS and LAPACK out of the box, but I am not sure if hmatrix is aware of them out of the box. You might need to do the following:
```zsh
brew install openblas
brew install lapack
```
And execute the program (see [Run](#Run)) with the `openblas` flag:
```zsh
stack exec -- hmnist-exe -f openblas
```

### On Windows:
See [here.](https://icl.cs.utk.edu/lapack-for-windows/lapack/)

## Run
Code runs using [Stack](https://docs.haskellstack.org/en/stable/README/) to manage imports. First clone the repository, then run
```zsh
$ stack build
$ stack exec hmnist-exe
```
Note that you will have to run it from /path/to/repo/ because it gets the current directory and looks for the MNIST data in data/mnist_dataset/.

## Code
Everything interesting is in the [src/Network.hs](src/Network.hs) file. The order of functions in the Network.hs file is (sort of) from most to least important; the functions are organized into blocks of *arithmetic functions*, *helper functions* (which should probably be renamed to *utility functions*, though this may be misleading) and *data processing functions*. Above them are the actually important functions that execute the code. In the future I will refactor a bit more, especially the *backprop* function. Currently it is very difficult to understand the code, even if you fundamentally understand how the mini-batch SGD works in languages like Python. This is because I tried to write it by *folding* and *scanning* over lists to do the computations. Because of this, it is difficult to read the code. Additionally, the abstractions of the functions are a bit odd. The *updateBatches*, *train* and *test* functions should be reduced in the size of number of arguments they take since this unnecessarily obfuscates the code.

## Performance
At first it took 20 minutes to train a single epoch, compared to ~20 seconds in the referenced book, but this number was reduced after I learned about [profiling](https://www.tweag.io/blog/2020-01-30-haskell-profiling/) in Haskell and was able to find the bottlenecks. Most bottlenecks were eliminated once their [laziness](https://github.com/hasura/graphql-engine/pull/2933) was changed to strict evaluation. Optimizing the code and switching from the matrix package to hmatrix to enable BLAS usage, reduced the time per epoch to ~10 seconds. The entire thing now takes about 5 minutes to train on my machine.

However, looking at [htop](https://en.wikipedia.org/wiki/Htop), you can see that the parallelization is not working as well as it is in NumPy. In the future it might be interesting to look into how NumPy works with multithreading. Most people say that it is all thanks to BLAS, but you can see that if you run `python` and open `htop`, there's only one thread running. Now, if you import NumPy, instantly all threads are used. Haskell allows threading with the `-threading` [GHC option](https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/phases.html#ghc-flag--threaded), but it uses multithreading in a [rather odd way](https://stackoverflow.com/questions/5847642/haskell-lightweight-threads-overhead-and-use-on-multicores/5849482#5849482) and I do not know how this multithreading works with that of BLAS.

Though there are a lot of things I have not figured out and the code is suboptimal in multiple ways, these problems have been great sources of learning about memory usage, multithreading, concurrency, parallelism etc.

## Results
```zsh
$ stack exec hmnist-exe
```
After 30 epochs of training (5 minutes) with the base settings, the network classifies about 95% of the 10000 testing images correct. There are many more things that could be done to improve these results (E.g. learning rate annealing/scheduling, different cost functions; cross entropy, other tuning of hyperparameters), and I will keep adding those adjustments to this repository as I go through more chapters of the referenced [book](http://neuralnetworksanddeeplearning.com/).

## Possible Additions/Changes
- [Accelerate package](https://hackage.haskell.org/package/accelerate)
- [Repa arrays](https://hackage.haskell.org/package/repa) and [Repa algorithms](https://hackage.haskell.org/package/repa-algorithms-3.4.1.3)
- See [RESOURCES](RESOURCES.md)
