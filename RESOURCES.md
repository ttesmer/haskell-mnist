# Resources
Collection of links I found while googling about backpropagation, (stochastic) gradient descent, Haskell and CLI tools such as htop.

## General ML and other things I discovered:
- [Knuth-Fisher-Yates Random Shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)
- [Numpy Config](https://github.com/numpy/numpy/blob/main/site.cfg.example)
- [Understand htop once and for all](https://peteris.rocks/blog/htop/)
- [Blog about ML](http://www.wildml.com/) 
- [Blog of Director of AI @ Tesla](https://karpathy.medium.com/) (also on [Github Pages](http://karpathy.github.io/))
- [Google Blog on Deep Learning](https://ai.googleblog.com/search/label/Deep%20Learning)
- [Google Blog on Neural Networks](https://ai.googleblog.com/search/label/Neural%20Networks)
- [Blog post about production Haskell code](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/)
- [Reasoning about algorithmic complexity in Haskell](https://softwareengineering.stackexchange.com/questions/363629/how-does-one-reason-about-algorithmic-complexity-in-haskell)
- [Haskell faster than C](https://lispcast.com/how-is-haskell-faster-than-c/)
- [MIT Free Course on Data Structures & Algorithms](https://www.youtube.com/watch?v=ZA-tUyM_y7s&list=PLUl4u3cNGP63EdVPNLG3ToM6LaEUuStEY&index=2)
- [MIT Lecture on Airplane Aerodynamics](https://www.youtube.com/watch?v=edLnZgF9mUg) (yes, I may have gotten a bit off track every now and then)
- [MIT Lecture on Binary Search Trees](https://www.youtube.com/watch?v=9Jry5-82I68)
- [Use Excalidraw if you need to visualize something](https://excalidraw.com/)
- [Blog about ML](https://eugeneyan.com/writing/), in particular:
	- [Real time recommendations](https://eugeneyan.com/writing/real-time-recommendations/)
	- [RL for recommendations](https://eugeneyan.com/writing/reinforcement-learning-for-recsys-and-search/)
	- [System design for recommendations](https://eugeneyan.com/writing/system-design-for-discovery/)
	- [Patterns for personalization](https://eugeneyan.com/writing/patterns-for-personalization/)

## Profiling/Optimizing/Debugging Haskell
- [Parallel profiling](https://wiki.haskell.org/ThreadScope)
- [Zeta Function Concurrency example](https://wiki.haskell.org/Concurrency_demos/Zeta)
- [Haskell Multithreading](https://stackoverflow.com/questions/5847642/haskell-lightweight-threads-overhead-and-use-on-multicores/5849482#5849482)
- [Foreign Function Interface (FFI) RWH Book](http://book.realworldhaskell.org/read/interfacing-with-c-the-ffi.html)
- [Parallel Arrays](https://www.tweag.io/blog/2017-11-16-repa/)
- [A Journey of optimizing Haskell code](https://chrispenner.ca/posts/wc)
- [Detecting Lazy Memory Leaks](https://stackoverflow.com/questions/61666819/haskell-how-to-detect-lazy-memory-leaks)
- [On Allocation And Readabiltiy](https://stackoverflow.com/questions/2026912/how-to-get-every-nth-element-of-an-infinite-list-in-haskell)
- [Utilize all Cores](https://stackoverflow.com/questions/39540247/haskell-parallel-program-not-utilizing-all-cores)
- [Memory in Haskell](https://blog.pusher.com/making-efficient-use-of-memory-in-haskell/)
- [Managing Memory in Haskell](https://www.channable.com/tech/lessons-in-managing-haskell-memory)
- [Never use Lazy Foldl](https://github.com/hasura/graphql-engine/pull/2933)
- [Concurrent Haskell](https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/using-concurrent.html)
- [Benchmarking using Criterion package](https://hackage.haskell.org/package/criterion)
- [Profiling and Optimization RWH Book](http://book.realworldhaskell.org/read/profiling-and-optimization.html)
- [Haskell Profiling](https://www.tweag.io/blog/2020-01-30-haskell-profiling/)

## Matrix Multiplication
- [Algorithms](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)
- [Algorithmic Complexity](https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication)

## BLAS (Basic Linear Algebra Subprograms)
- [OpenBLAS FAQ](https://github.com/xianyi/OpenBLAS/wiki/faq)
- [BLAS Numpy](https://markus-beuckelmann.de/blog/boosting-numpy-blas.html)
- [Intel oneAPI](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html)
- [Intel oneAPI MKL Installation](https://codeyarns.com/tech/2019-05-15-how-to-install-intel-mkl.html) and [For APT](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html)
- [MKL Sample](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-base-linux/top/run-a-sample-project-using-the-command-line.html)
- [ATLAS](http://math-atlas.sourceforge.net/)
- [LAPACK](https://www.netlib.org/lapack/lug/node11.html)
- [Numerical Linear Algebra (Wikipedia)](https://en.wikipedia.org/wiki/Numerical_linear_algebra)
- [Symmetric Multiprocessing (SMP) (Wikipedia)](https://en.wikipedia.org/wiki/Symmetric_multiprocessing)
- [Scipy parallel programming](https://scipy.github.io/old-wiki/pages/ParallelProgramming)
- [Parallelism & Concurrency RWH Book](http://book.realworldhaskell.org/read/concurrent-and-multicore-programming.html)

## Memory Leaks
- [Haskell: how to detect "lazy memory leaks"](https://newbedev.com/haskell-how-to-detect-lazy-memory-leaks)
- [bgamari.github.com - The many arrays of GHC](https://bgamari.github.io/posts/2016-03-30-what-is-this-array.html)
- [haskell - Differences between Storable and Unboxed Vectors - Stack Overflow](https://stackoverflow.com/questions/40176678/differences-between-storable-and-unboxed-vectors)
- [Lessons in Managing Haskell Memory](https://www.channable.com/tech/lessons-in-managing-haskell-memory)
- [Garbage collector issues in Haskell runtime when (de)allocations are managed in C - Stack Overflow](https://stackoverflow.com/questions/67655017/garbage-collector-issues-in-haskell-runtime-when-deallocations-are-managed-in?noredirect=1#comment119626113_67655017)
- [All About Strictness](https://www.fpcomplete.com/blog/2017/09/all-about-strictness/)
- [Real World Haskell - Chapter 17. Interfacing with C: the FFI](http://book.realworldhaskell.org/read/interfacing-with-c-the-ffi.html)
- [Haskell: how to get rid of the memory leak - Stack Overflow](https://stackoverflow.com/questions/18337833/haskell-how-to-get-rid-of-the-memory-leak)
- [What I Wish I Knew When Learning Haskell - Function Pointers](http://dev.stephendiehl.com/hask/#function-pointers)
- [Memory leak - HaskellWiki](https://wiki.haskell.org/Memory_leak)
- [acmqueue - Eliminating memory hogs](https://dl.acm.org/doi/pdf/10.1145/2538031.2538488)
- [Neil Mitchell's Blog (Haskell etc): Detecting Space Leaks](http://neilmitchell.blogspot.com/2015/09/detecting-space-leaks.html)
- [Pinpointing space leaks in big programs: ezyang’s blog](http://blog.ezyang.com/2011/06/pinpointing-space-leaks-in-big-programs/)
- [Understanding Memory Fragmentation - Well-Typed: The Haskell Consultants](https://www.well-typed.com/blog/2020/08/memory-fragmentation)
- [haskell - Memory leak when generating a list of tuples - Stack Overflow](https://stackoverflow.com/questions/25895046/memory-leak-when-generating-a-list-of-tuples)

## Further Material for ML
- [DEEP LEARNING BOOK](https://www.deeplearningbook.org/)
	- Note: This might be good to supplement [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) after I am done with that.
- [OTHER DEEP LEARNING BOOK](http://deeplearning.stanford.edu/tutorial/)
- [Why the Bias is Important](https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks)
- [Backprop Calculus Video](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
	- Note: This is VERY sparse on details. It merely serves as an initial visualization.
- [ML Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/linear_algebra.html#matrix-multiplication)
- [Gradient = Steepest Ascent: (top comment is useful)](https://www.youtube.com/watch?v=TEB2z7ZlRAw)
- [Box-Muller Transform for Sampling from Gaussian](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
- [Blog Post on Sampling from Gaussian Distribution (Math heavy)](https://bjlkeng.github.io/posts/sampling-from-a-normal-distribution/)
- [Gradient Descent (Wikipedia)](https://en.wikipedia.org/wiki/Gradient_descent)
- [Peculiarity of Quadratic Loss Function Explained](https://datascience.stackexchange.com/questions/52157/why-do-we-have-to-divide-by-2-in-the-ml-squared-error-cost-function)
- [Motivation for Doing the Problems in the Book](http://neuralnetworksanddeeplearning.com/exercises_and_problems.html)
- [Solutions for the Problems in the Book](https://github.com/nndl-solutions/NNDL-solutions/blob/master/notebooks/chap-1-using-neural-nets-to-recognize-handwritten-digits.ipynb)
- [3b1b Amazing Visualization Explanation](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Why is the Gradient the Direction of Steepest Ascent?](https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent)
- [Wikipedia Gradient Defintion](https://en.wikipedia.org/wiki/Gradient)
- [Cauchy-Schwarz Difference Vector Form and Normal](https://www.sciencedirect.com/topics/mathematics/cauchy-schwarz-inequality)
- [Blog Post About Stochastic Gradient Descent](http://www.samvitjain.com/blog/gradient-descent/)
