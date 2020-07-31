# prime-mult

This is an experiment to see if various NN models can learn to multiply large prime numbers. If these models are reversible, then we can efficiently factor large numbers. Thus, it will be interesting to see if reversible models happen to be much worse at this task.

# Results

My initial finding was that simple MLP models have trouble learning how to multiply 32-bit primes. Perhaps these architectures (which are few-layer MLPs) are simply not capable of expressing such a function. Naive implementations of multiplication require a lot of sequential logic gates (for carrying), so perhaps a very deep (or wide) network is needed to implement multiplication.

After making the above observation, I hard-coded an MLP to explicitly perform long multiplication, and set up a re-initialization scheme so that I could train this (very deep) model from scratch. So far, results do not look promising, but I have a feeling this is because the model is ridiculously deep (hundreds of layers).
