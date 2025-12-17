+++
# The title of your blogpost. No sub-titles are allowed, nor are line-breaks.
title = "Efficient Online Random Sampling via Randomness Recycling"
# Date must be written in YYYY-MM-DD format. This should be updated right before the final PR is made.
date = 2025-12-01

[taxonomies]
# Keep any areas that apply, removing ones that don't. Do not add new areas!
areas = ["Theory"]
# Tags can be set to a collection of a few keywords specific to your blogpost.
# Consider these similar to keywords specified for a research paper.
tags = ["random variate generation", "entropy", "algorithm design and analysis"]

[extra]
author = {name = "Thomas L. Draper", url = "https://www.cs.cmu.edu/~tdraper/" }
# The committee specification is  a list of objects similar to the author.
committee = [
    {name = "Harry Q. Bovik", url = "http://www.cs.cmu.edu/~bovik/"},
    {name = "Committee Member 2's Full Name", url = "Committee Member 2's page"},
    {name = "Committee Member 3's Full Name", url = "Committee Member 3's page"}
]
+++

Suppose you are given a discrete probability distribution \\(\mathbf{p} = (p_1, p_2, \ldots, p_k)\\) and you are asked to design an algorithm to generate a random integer \\(X \in \\{1,\ldots,k\\}\\) with probability \\(p_X\\).
We can think of this problem as rolling a loaded \\(k\\)-sided die, where side \\(i\\) has probability \\(p_i\\).
In computer science, randomness is used for stochastic simulation, statistical modeling, AI and machine learning, and cryptography.
Sampling random numbers is also useful for games, lotteries, experimental design, and even the arts.
Some applications benefit from the reproducibility (determinism) and efficiency afforded by using pseudorandom number generators, but [often it is ideal to use a true source of randomness](https://www.random.org/randomness/).

## True randomness

True randomness is an elusive concept.
Randomness is used to describe unknown outcomes which cannot be predicted deterministically.
Therefore, sources of randomness are generally not verified by their output alone; instead, we gain trust in a randomness source if it is driven by a physical process so complicated or chaotic that predicting the outputs is computationally intractable.
Classical methods of gathering randomness require dedicated sensors tuned to measure unpredictable physical phenomena.
For example, the website [random.org](https://www.random.org) uses an array of radios to detect atmospheric noise, generating roughly [12,000 bits of randomness per second](https://www.random.org/faq/#Q4.1).
[Recent work](https://doi.org/10.48550/arXiv.2303.01625) shows that quantum computers can generate verifiably random numbers under plausible hardness assumptions; however, the verification of this randomness requires exponential time.

## Entropy efficiency {entropy-efficiency}

Because randomness is a scarce resource, researchers study fundamental limits on the efficiency of algorithms for manipulating randomness, i.e., converting an input random source into an output random value.
The efficiency of such transformations on randomness can be quantified in terms of the [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of the inputs and outputs.
For a probability distribution \\(\mathbf{p} = (p_1, p_2, \ldots, p_k)\\), the Shannon entropy is defined as
\\[H(\mathbf{p}) = -\sum_{i=1}^{k} p_i \log p_i.\\]
Donald E. Knuth and Andrew C. Yao [proved in 1976](https://archive.org/details/algorithmscomple0000symp/page/356/mode/2up) that an input stream of fair coin tosses can be converted to any output distribution \\(\mathbf{p}\\) using less than \\(H(\mathbf{p})+2\\) coin tosses in expectation.

The entropy-efficient algorithm of Knuth and Yao can be extended to sample in an online fashion from a sequence \\(\mathbf{p}_1, \mathbf{p}_2, \ldots\\), using less than \\(H(\mathbf{p}_1)+\cdots+H(\mathbf{p}_n)+2\\) coin tosses in expectation after sampling from each finite prefix \\(\mathbf{p}_1, \ldots, \mathbf{p}_n\\).
However, this extension requires linearly growing memory and increasing computation time per sample as it progresses through the sequence, which makes it impractical for sampling long sequences.
Another alternative is to use a stateless version of the Knuth and Yao algorithm, using a fresh start for each new sample; the main drawback of this approach is that its entropy consumption is worse, bounded only by \\(H(\mathbf{p}_1)+\cdots+H(\mathbf{p}_n)+2n\\).
In real applications, a random number generation library must sample from a dynamic sequence of distributions.
It is impractical to use an algorithm whose per-sample complexity grows over time, but entropy efficiency remains an important goal.
Therefore, we seek computationally efficient online sampling algorithms which come close to the optimal entropy bound of \\(H(\mathbf{p}_1)+\cdots+H(\mathbf{p}_n)+2\\).

# Uniform sampling

Let us first consider algorithms for sampling from discrete uniform distributions.
Although simple, discrete uniforms can be used to exactly simulate many natural problems, such as rolling a fair die, shuffling a deck of cards, or generating a random password over a given alphabet---in fact, every method currently available on [random.org](https://www.random.org), except for the numerical Gaussian generator, is based directly on the discrete uniform.

Knuth and Yao's general algorithm scales at least linearly in the number of outcomes in the target distribution, but researchers have developed much faster algorithms for the special case of the uniform distribution.
Jérémie Lumbroso introduced the [Fast Dice Roller](https://doi.org/10.48550/arXiv.1304.1916) (FDR), which generates a uniform over the \\(m\\) outcomes \\(\\{0,1,\ldots,m-1\\}\\) using \\(O(\log m)\\) space and \\(O(\log m)\\) expected time.
At the same time, this algorithm matches Knuth and Yao's entropy-optimal algorithm in terms of entropy efficiency.
The idea is to maintain a pair of integers \\((Z,M)\\) satisfying \\(M > 0\\) and \\(Z \sim \operatorname{Uniform}(\\{0,1,\ldots,M-1\\})\\), which we will henceforth call a discrete uniform random state.
Here is a slight reformulation of the FDR, in Python-like pseudocode, where `Flip()` returns a fair coin toss (0 or 1 with equal probability).

<!-- 1. \\((Z, M) \leftarrow (0, 1)\\) <div style="float:right">Initialize the random state</div>
2. \\((Z, M) \leftarrow (2Z + \operatorname{Flip}(), 2M)\\) <div style="float:right">Flip a coin to enlarge the random state</div>
3. If \\(M < m\\): go to line 2 <div style="float:right">Repeat until the state is large enough</div>
4. If \\(Z < m\\): return \\(Z\\) <div style="float:right">\\(Z\\) is uniform over \\(m\\) outcomes, conditioned on \\(Z < m\\)</div>
5. \\((Z, M) \leftarrow (Z-m, M-m)\\) <div style="float:right">Reduce the random state if \\(Z \geq m\\)</div>
6. Go to line 2 <div style="float:right">Repeat until success</div> -->

```python
def FDR(m):
    Z = 0
    M = 1
    while True:
        while M < m:
            Z = 2 * Z + Flip()
            M = 2 * M
        if Z < m:
            return Z
        Z = Z - m
        M = M - m
```

Notice that the line `Z, M = 2 * Z + Flip(), 2 * M` maintains the invariant that \\(Z\\) is uniform over \\(\\{0,1,\ldots,M-1\\}\\), provided that `Flip()` returns a fair coin toss, independent of all previous coin tosses.
Once \\(M \geq m\\), we can return a value \\(Z \sim \operatorname{Uniform}(\\{0,1,\ldots,m-1\\})\\) conditioned on \\(Z < m\\).
If instead \\(Z \geq m\\), we can rerun the algorithm with the updated state \\((Z-m, M-m)\\).

Although the `FDR` algorithm is entropy optimal (matching Knuth and Yao's entropy efficiency), it is far from optimal when called repeatedly in the online setting, because it does not maintain any state between calls.
Because Knuth and Yao only guarantee a bound of \\(H(\mathbf{p})+2\\) coin tosses for a single sample, the `FDR` accumulates a waste of \\(\Theta(1)\\) coin tosses per call.
In particular, after \\(n\\) calls to `FDR` with parameters \\(m_1,\ldots,m_n\\), the expected number of coin tosses used is bounded by \\(\sum_{i=1}^{n} \log(m_i) + 2n\\), which scales worse than the optimal bound of \\(\sum_{i=1}^{n} \log(m_i) + 2\\).
Although both of these expressions are asymptotically \\(\Theta(n)\\), the difference can be significant in practice when the source of randomness is expensive.
In information-theoretic problems from source coding to channel capacity, the common goal is to drive the input-to-output entropy ratio to 1, not merely \\(O(1)\\).
In this spirit, we consider how efficiently a sampler can achieve an entropy ratio of \\(1 + \varepsilon\\) for arbitrarily small \\(\varepsilon > 0\\) in the online setting.

## Efficient online uniform sampling

First, we need to understand where the entropy waste in `FDR` comes from.
Lumbroso's analysis of `FDR` shows that it is the comparison `if Z < m` that immediately causes a loss of up to one bit of entropy.
The uniform state over \\(M\\) outcomes is divided into two smaller states, over \\(m\\) or \\(M-m\\) outcomes, respectively.
This is equivalent to throwing away the information from a Bernoulli variable with parameter \\(m/M\\).
Further, each iteration of `while True` has a failure probability of \\(1-m/M\\), which can be up to \\(1/2\\), so the expected number of iterations is bounded by 2.
The information lost from the comparison `if Z < m` cannot be recovered, because the event \\(Z < m\\) has become correlated with the control flow of the program.
However, we can modify the algorithm to minimize the failure probability \\(m/M\\), by using a larger value of \\(M\\) before making a comparison.

To efficiently scale a discrete uniform over a large range \\(M\\) down to a smaller range \\(m\\), we use integer division.
The idea is illustrated in the following diagram, where \\(Z \sim \operatorname{Uniform}(\\{0,1,\ldots,M-1\\})\\) is divided by \\(m\\) to form \\(Z = q_Z \cdot m + r_Z\\), which satisfies \\(r_Z \sim \operatorname{Uniform}(\\{0,1,\ldots,m-1\\})\\), conditioned on the event \\(q_Z < \lfloor M/m \rfloor\\).

![generating a uniform integer using integer division](standalone-uniform-division.png)

Using division to scale uniform random variables is a standard technique in random variate generation.
It is important to note that the remainder \\(r_Z\\) is not uniformly distributed in general, which is why we need to reject when \\(q_Z = \lfloor M/m \rfloor\\), illustrated in the rightmost box of the diagram.
Further, conditioned on \\(q_Z < \lfloor M/m \rfloor\\), the quotient \\(q_Z\\) is uniform over \\(\lfloor M/m \rfloor\\) outcomes and independent of the return value \\(r_Z\\), so this discrete uniform variable can be stored and used to improve the entropy efficiency of future calls.
Here is pseudocode for our full algorithm for online uniform sampling.

```python
Z = 0
M = 1
M_target = 2**63
def Uniform(m):
    global Z, M, M_target
    while True:
        while M < M_target:
            Z = 2 * Z + Flip()
            M = 2 * M
        qM, rM = divmod(M, m)
        qZ, rZ = divmod(Z, m)
        if qZ < qM:
            Z = qZ
            M = qM
            return rZ
        Z = rZ
        M = rM
```

The state variables `Z, M` represent a discrete uniform random variable, containing recycled randomness, independent of the previous outputs from the algorithm.
We use the term "randomness recycling" to describe how random information (obtained through `Flip()`) is stored for use when generating future samples.
The parameter `M_target` should be much larger than any `m` that will be sampled, to ensure that the rejection probability is small.
For example, if `m` is always a 32-bit integer, then `M_target = 2**63` ensures that the rejection probability is less than \\(2^{-31}\\), while also ensuring that all program variables fit in 64-bit integers.

Although the division operation in `Uniform` is more expensive than the simple arithmetic operations in `FDR`, the increased entropy efficiency in `Uniform` more than makes up the difference when randomness is expensive.
As a typical example where randomness is treated as a scarce resource, the website [random.org](https://www.random.org) uses an algorithm similar to `Uniform` for precisely this reason.
This `Uniform` was first described in an [article by Jacques Willekens](https://web.archive.org/web/20200213145912/http://mathforum.org/library/drmath/view/65653.html), and subsequently rediscovered [several](https://doi.org/10.48550/arXiv.1012.4290) [times](https://doi.org/10.48550/arXiv.1412.7407).
However, prior to our work, it seems that the generalized problem of entropy-efficient online sampling from arbitrary (nonuniform) distributions has not been considered.

# Nonuniform sampling

Recall the problem of sampling from a distribution \\(\mathbf{p} = (p_1, p_2, \ldots, p_k)\\).
Keith Schwarz wrote [a popular blog post](https://www.keithschwarz.com/darts-dice-coins/) investigating algorithms for sampling from nonuniform discrete distributions, but without regard to entropy efficiency, and under the assumption that exact operations on real numbers can be performed in constant time.

For a more realistic model, we consider the problem where the target distribution is provided as a list of integer weights \\(a_0, a_1, \ldots, a_{k-1}\\), and the goal is to sample an index \\(i\\) with probability proportional to \\(a_i\\).
In particular, let \\(A_i = \sum_{j=0}^{i} a_j\\) denote the prefix sums; then the target probabilities are \\(p_i = a_i / A_{k-1}\\).
One of the simplest nonuniform sampling algorithms is the inversion method, which samples a uniform random variable \\(U \sim \operatorname{Uniform}(\\{0,1,\ldots,A_{k-1}-1\\})\\) and returns the index \\(i\\) such that \\(A_{i-1} \leq U < A_i\\), as illustrated in the following diagram.

![generating a nonuniform random number using the inversion method](standalone-nonuniform.png)

There are \\(a_i\\) possible values of \\(U\\) that yield outcome \\(i\\), so the probability of returning index \\(i\\) is exactly \\(a_i / A_{k-1}\\).
Conditioning on the event that \\(U\\) falls in the range \\([A_{i-1}, A_i)\\), we obtain a new uniform random state given by \\(Z' = U - A_{i-1}\\) and \\(M' = a_i\\), satisfying \\(Z' \sim \operatorname{Uniform}(\\{0,1,\ldots,M'-1\\})\\).
We can recycle \\((Z',M')\\) back into the global state variables \\((Z,M)\\) using a standard trick for merging two independent uniform random states, namely, setting \\(Z \leftarrow Z + Z' \cdot M\\) and \\(M \leftarrow M \cdot M'\\).
The full pseudocode for the inversion method with randomness recycling is as follows.

```python
def Inversion(a):
    A = list(itertools.accumulate(a, initial=0))
    U = Uniform(A[-1])
    X = bisect.bisect(A, U) - 1
    Z1 = U - A[X]
    M1 = a[X]
    global Z, M
    Z = Z + Z1 * M
    M = M * M1
    return X
```

The conversion from `U` to `X, Z1, M1` is reversible, and in particular it does not lose any entropy.
Therefore, the only entropy loss in `Inversion` comes from the call to `Uniform`, and `Inversion` inherits the same entropy efficiency as `Uniform`.

# Analysis

We now analyze `Uniform` to bound its entropy efficiency.
Let \\(m_{\max}\\) denote the largest value of \\(m\\) passed to `Uniform` over the entire execution, and let \\(M_{\min}\\) denote the smallest possible value of \\(M\\) before division (written as `M_target` in the pseudocode).
Then the rejection probability in `Uniform` is less than \\(m_{\max}/M_{\min}\\), so the expected number of iterations of the outer `while True` loop is less than \\(1/(1 - m_{\max}/M_{\min})\\).
Further, the entropy lost per iteration is given by the entropy of the accept-reject decision, which is bounded by the binary entropy function \\(H_{\rm b}(m_{\max}/M_{\min})\\).
Multiplying the bounds on the expected number of iterations and the entropy lost per iteration, the overall entropy loss per call to `Uniform` is bounded by
\\[
\frac{H_{\rm b}(m_{\max}/M_{\min})}{1 - m_{\max}/M_{\min}}
= \frac{m_{\max}}{M_{\min}-m_{\max}} \log_2(M_{\min}/m_{\max}) - \log_2(1 - m_{\max}/M_{\min}).
\\]
Given a bound on the size of the input integers \\(m_{\max}\\), to find the smallest value of \\(M_{\min}\\) that achieves a desired entropy loss \\(\varepsilon > 0\\), it suffices to solve the transcendental equation
\\[
\varepsilon = \frac{m_{\max}}{M_{\min}-m_{\max}} \log_2(M_{\min}/m_{\max}) - \log_2(1 - m_{\max}/M_{\min})
\\]
for \\(M_{\min}\\).
The solution grows as \\(M_{\min} = m_{\max} \cdot \tilde{O}(1 / \varepsilon)\\) as \\(\varepsilon \to 0\\), which means that the required integer size \\(1+\lceil\log_2 M_{\min}\rceil\\) grows as \\(O(\log_2(m_{\min}/\varepsilon))\\).

The logarithmic dependence on \\(1/\varepsilon\\) makes it practical to run `Uniform` with very small entropy loss.
For the example where all inputs are 32-bit integers (i.e., \\(m_{\max} < 2^{32}\\)), and the state variables are 64-bit integers (i.e., \\(M_{\min} = 2^{63}\\)), the expected entropy loss is less than \\(2 \times 10^{-8}\\) bits per sample.
For nonuniform sampling, `Inversion` has the same entropy loss as `Uniform`, and the underlying inversion method remains linear in the input size.
The following table compares the efficiency of our method with the Knuth and Yao method, either statelessly (using a fresh start for each new sample), or online, as described in the [entropy efficiency](#entropy-efficiency) section.

| Method                    | Amortized Entropy Loss Bound | Expected Space and Time Complexity            |
|:------------------------- | ---------------------------- | --------------------------------------------- |
| Knuth and Yao (Stateless) | 2                            | linearithmic in input                         |
| Knuth and Yao (Online)    | 0                            | unbounded                                     |
| `Inversion`               | \\( \varepsilon \\)          | linear in input and \\(\log(1/\varepsilon)\\) |

We conjecture that our algorithm is optimal, in the sense that \\(\Omega(\log(1/\varepsilon))\\) space complexity is required for online sampling algorithms achieving an amortized entropy loss bound of \\(\varepsilon\\).

# Evaluation

We implemented the `Inversion` algorithm in C and compared its performance to the standard inversion method, which does not recycle randomness using a random state such as \\((Z,M)\\).
We generated several distributions with varying numbers of outcomes, and for each distribution, we preprocessed the prefix sums and sampled 1 million times from the distribution using each algorithm.
The following figure shows the performance improvement from randomness recycling, in terms of both entropy consumption and runtime, using `/dev/random` as the source of randomness.
The blue points represent the original method, and the orange points represent our method with randomness recycling.

![performance improvement from randomness recycling](standalone-performance.png)

We also applied the same randomness recycling technique to accelerate the following algorithms:

- [Lemire's uniform sampler](https://doi.org/10.1145/3230636),
- [Brackett-Rozinsky and Lemire's batched uniform sampler](https://doi.org/10.1002/spe.3369),
- [Walker's alias method with Vose's modification](https://doi.org/10.1109/32.92917),
- [the Fast Loaded Dice Roller](https://doi.org/10.48550/arXiv.2003.03830), and
- [the Amplified Loaded Dice Roller](https://doi.org/10.48550/arXiv.2504.04267).

For further details, please see the [full paper](https://doi.org/10.48550/arXiv.2505.18879).

This article is based on joint work with [Feras Saad](https://www.cs.cmu.edu/~fsaad/), to be presented at the 2026 Symposium on Discrete Algorithms.
