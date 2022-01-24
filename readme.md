
# Implementing A Transformer From Scratch
To get intimately familiar with the nuts and bolts of transformers, I implemented a bare-bone version of the original transformer 
proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). No dependencies except [PyTorch](https://pytorch.org/get-started/locally/). 

WORK IN PROGRESS!

## To do:
- Finish explaining how MHA is governed by a single matrix for all heads and projections. Might need some help for that.
- Finish TransformerDecoder implementation: cross & self-attention
  - Implement attention mask for pad tokens and decoder self-attention
- Decoder & Transformer main class unit tests
- Divide embedding weights by sqrt(hidden_dim // num_heads)
- Training loop, optimizer & settings etc.
  - Check whether BOS and EOS tokens should be added to each training sentence; remove this from the vocab class if not.

## Features:
- The simplest imaginable Vocabulary class (vocabulary.py)
- The simplest imaginable tokenization method (vocabulary.py)
- TransformerEncoder class (transformer.py)
- TransformerDecoder class (transformer.py)
- Transformer class (transformer.py)
- MultiHeadAttention class (multi_head_attention.py)
- SinusoidEncoding class (positional_encoding.py)
- Basic unit tests for each class
- Code formatted using [black](https://github.com/psf/black)
- Python 3.9, PyTorch 1.9.1

#### Bonus
- Replace examples in the vocabulary unit test.
- (Multi-)GPU support
- Run unit tests on each commit using Github Actions 
  - Refactor unit tests to separate folder and files
  - Run pylint on codebase
- Publish this repo and write blogpost about what I learned / what surprised me (notes below).

## Lessons Learned / Implementation Notes
I'm assuming high-level knowledge about the transformer architecture. 
Take a look at the great [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) post for a refresher. 

### Multi-Head Attention
1. Multi-head attention is actually implemented with **one** weight matrix (for every layer in the encoder and decoder). 
   This matrix contains the parameters that project each (contextualized) token embedding into its key, query and value vectors for all heads in that layer **at once**.
2. The dimensionality of the key, query and value vectors is dynamically set to the number of hidden dimensions (e.g. 512) divided by the number of heads (e.g. 8). For example, see the pytorch [implementation](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L917). 
   As a consequence, they must be divisible. 
   Also, as we increase the number of heads while keeping the number of hidden dimensions fixed, the dimensionality of the key, query and value dimensionality shrink.

Let's take a closer look at this mysterious 512 by 1536 projection matrix W and how we should interpret it. 
Remember that this projection can be applied to each token embedding in a batch in parallel: an important factor in the transformer's success. 

Recall that in the matrix multiplication R=X@W, each element in the resulting matrix R is the sum of the element-wise product between (a.k.a. linear combination of) a row vector in X and a column vector in W.
The image below (taken from [this](https://medium.com/ai%C2%B3-theory-practice-business/fastai-partii-lesson08-notes-fddcdb6526bb) Medium post) neatly visualizes that.
![](https://miro.medium.com/max/1400/1*D_1tbv_wNFJ-rrremAGX4Q.png)

Thus, **each element in the first row of R is a different linear combination of the first token embedding row and one of the (learnable) column vectors in W.**
This means we can simply stack more columns in our weight matrix W to create more "linear-combination-elements" for each token embedding row in X.
Note that the same weight column is always responsible for the same hidden dimension in the resulting matrix R for all token embedding rows (e.g. dim 5 in every query vector is the result of the 5th weight column).   

The next question is: how do we interpret these elements in R such that we have three neat key, query and key vectors with 64 elements for each of the eight heads?   

Let's work through an example. Say our input consists of 4 tokens: ["how", "are", "you"]. 
Running this through the embedding matrix we obtain a (3, 512) matrix X. We ignore batches for now.
We now initialize a (512, 1536) weight matrix W. (Woah, what a coincidence: 1536 = 3 * 8 * 64!)
Matrix multiplication R=X@W returns a (3, 1536) matrix R. 
We can now interpret R as containing eight (heads) sets of three (key, query, value) 64-dimensional vectors.

Finally, we extract and reshape the chunks that we interpret as query vectors Q into (batch_size, num_heads, qkv_dim, seq_length). 
We multiply Q with the transpose of K, which results in the attention logits (batch_size, num_heads, seq_length, qkv_dim).
After normalizing these logits with sqrt(dim_q) and applying the softmax, we multiply the resulting attention weights with V.
This 

### Positional Embeddings
Sinusoid positional (non-learned) encodings add sine (to all even dims) or cosine (to all uneven dims) waves to all dimensions. 
These sinusoid waves "flow across" token positions. 
The shape of each wave differs per dimension and is determined by the token and dimension index.

### Actually Predicting the Next Token
The final decoder hidden state that corresponds to the position index of decoder’s last input token is used to predict the next token. 
Multiplying this hidden state of dim(batch_size, hidden_dim) with the transpose shared embedding matrix results in the unnormalized distribution (logits) over the vocabulary for each example in the batch. 
The other hidden states are simply discarded. (todo: check this)

### Weight Tying
A shared source-target byte-pair encoding vocabulary is used. 
The input embedding weights are shared between encoder and decoder. 
Its transpose is used as the final linear transformation in the decoder, right before the softmax, to reduce overfitting and the number of parameters. (todo: elaborate on why)

Fun fact: the embedding weights are divided by sqrt(hidden_dim // num_heads). (todo: why)

### What About the Input?
1. Vanilla transformers can handle arbitrary length inputs in theory. 
In practice, however, self-attention has [compute and memory requirements](https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html#:~:text=With%20commonly%20available%20current%20hardware,summarization%20or%20genome%20fragment%20classification.) that limit the input sequence to usually around 512 tokens. 
Models like BERT do in fact impose a hard length limit because they use learned embeddings instead of the sinusoid encoding. 
These learned position embeddings embed up to a predefined position (512 for BERT).
2. Each input sequence in a batch must be padded to the longest sentence in that batch. 
It's common practice to batch sequences together with similar length to optimally use available compute.

### Alternative Perspective
The intermediate hidden representations can be viewed as a "residual stream". They are actually quite similar to the cell state in LSTMs. Read more about this in Anthropic's [recent publication](https://transformer-circuits.pub/2021/framework/index.html) about reverse engineering a transformer.

## References
- For positional encodings and the one-matrix-specifics of MHA I looked at (and replaced my own implementation with) snippets from this great tutorial: 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- Thanks David Stap for suggesting to implement the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper from scratch!