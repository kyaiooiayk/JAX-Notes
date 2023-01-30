# JAX (by Google)
***

## What is JAX?
- JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.
- JAX can automatically differentiate native Python and NumPy code. 
- It can differentiate through a large subset of Python’s features, including loops, ifs, recursion, and closures, and it can even take derivatives of derivatives of derivatives. 
- It supports reverse-mode as well as forward-mode differentiation, and the two can be composed arbitrarily to any order.
***

## What is XLA?
- XLA stands for Accelerated Linear Algebra.
- It is a domain-specific compiler for linear algebra (mostly used for ML model).
- What the rationale behind it? Operations fusion is XLA's single most important optimisation. Memory bandwidth is typically the scarcest resource on hardware accelerators, so removing memory operations is one of the best ways to improve performance. Here is how it works; let's say we have this:
```
def model_fn(x, y, z):
  return tf.reduce_sum(x + y * z)
```
- TF build a graph for this operation and the graph launches three kernels: one for the multiplication, one for the addition and one for the reduction. 
- XLA optimises the graph so that it computes the result in a single kernel launch. It does this by "fusing" the addition, multiplication and reduction into a single GPU kernel. Moreover, this fused operation does not write out the intermediate values produced by y*z and x+y*z to memory; instead it "streams" the results of these intermediate computations directly to their users while keeping them entirely in GPU registers.
***

## Features
- JAX is much more than just a GPU-backed NumPy. It also comes with a few program transformations that are useful when writing numerical code. The primary functions of JAX are:
  - `grad` - for taking derivatives via automatic differentiation
  - `jit` - just in time compilation for speeding up your code
  - `vmap` - for automatic vectorization or batching, It vectorizes functions by pushing the mapped axis down into primitive operations
  - `pmap` -  SPMD (single program multiple data) programming. It instead (contrary to `vmap`) replicates the function and executes each replica on its own XLA device in parallel.
***

## Ecosystem
- There are projects built on top of JAX that add extra bits.
  - [Flax](https://flax.readthedocs.io/en/latest/index.html) is a neural network libraries
  - [Haiku](https://github.com/deepmind/dm-haiku) is a neural network libraries
  - [Optax](https://github.com/deepmind/optax) is an optimiser
  - [PIX](https://dm-pix.readthedocs.io/en/latest/) for image processing
***

## References
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#)
- [Jax Wiki](https://en.wikipedia.org/wiki/Google_JAX#pmap)
- [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla)
***
