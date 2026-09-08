# mini-gpt

Character-level transformer written from scratch in PyTorch, following Karpathy's
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).

I'm using it as a testbed for natural gradient methods. My research is on Riemannian
estimation — Fisher-Rao metric, intrinsic Cramér-Rao bounds — and the Fisher information
matrix behind natural gradient descent is the same object seen from the optimization side.

Run in Colab (GPU runtime for the scaled config). The notebook downloads the corpus itself.

## Next

Explicit Fisher matrix on a toy MLP, then K-FAC on the transformer, then comparing
SGD / Adam / natural gradient on the same model.

## References

- Vaswani et al., *Attention Is All You Need*, 2017
- Amari, *Natural Gradient Works Efficiently in Learning*, 1998
- Martens & Grosse, *Optimizing Neural Networks with K-FAC*, 2015
