# mini-gpt

Character-level transformer written from scratch in PyTorch, following Karpathy's
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).

I'm using it as a testbed for natural gradient methods. I also work on Riemannian
estimation, where the Fisher information matrix defines the Fisher-Rao metric —
the same object, seen from the optimization side.

Run in Colab (GPU runtime for the scaled config).

## Experiments

`experiments/01_gaussian.ipynb` — Fisher matrix on a Gaussian, two parameters, closed form.
Plain gradient descent blows up when sigma is small; natural gradient doesn't. Same problem
rewritten in log-sigma: the two euclidean trajectories differ, the two natural ones coincide.

`experiments/02_mlp.ipynb` — exact 27x27 Fisher of a small MLP, built by Monte Carlo.
Sampling the labels from the model or taking the training labels gives two different
matrices, and they drift apart as the model trains. Natural gradient beats SGD per
iteration, loses badly per backward pass — which is the whole point of K-FAC.

## Next

K-FAC on the transformer, then SGD / Adam / K-FAC on the same model, timed in seconds
and not only in steps.

## References

- Vaswani et al., *Attention Is All You Need*, 2017
- Amari, *Natural Gradient Works Efficiently in Learning*, 1998
- Martens, *New Insights and Perspectives on the Natural Gradient Method*, 2014
- Martens & Grosse, *Optimizing Neural Networks with K-FAC*, 2015
