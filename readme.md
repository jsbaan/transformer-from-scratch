
# Implementing A Transformer From Scratch
To get intimately familiar with the nuts and bolts of transformers, I implemented a bare-bone version of the original transformer 
proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). No dependencies except [PyTorch](https://pytorch.org/get-started/locally/) and not intended for real-world use.

WORK IN PROGRESS!

# To do:
### Code
- Finish TransformerDecoder implementation: cross & self-attention
  - Implement attention mask for pad tokens and decoder self-attention
- Decoder & Transformer main class unit tests
- Divide embedding weights by sqrt(hidden_dim // num_heads)
- Training loop, optimizer & settings etc.
  - Check whether BOS and EOS tokens should be added to each training sentence; remove this from the vocab class if not.

### Writing
- Add details about masking in MHA
- Add details about the meaning of the word projection
- Expand on layer normalization; most people know it normalizes the batch somehow but what does that actually do?
- Think about title, motivation and framing

# Features:
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

### Bonus to do
- Replace examples in the vocabulary unit test.
- (Multi-)GPU support
- Run unit tests on each commit using Github Actions 
  - Refactor unit tests to separate folder and files
  - Run pylint on codebase
- Publish this repo and write blogpost about what I learned / what surprised me (notes below).

# What I Learned From Implementing A Transformer From Scratch
(todo: Why am I writing this and why should you read it)

I'm assuming high-level knowledge about the transformer architecture. Take a look at [Jay Alammar's "Illustrated Transformer" post](https://jalammar.github.io/illustrated-transformer/) for a refresher, or look at one his visualisations: 
![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)
## Multi-Head Attention
1. Multi-head attention is actually implemented with **one** weight matrix. This matrix contains the parameters that project each (contextualized) token embedding to a key, query and value vector for all heads in a layer **at once**.
2. The dimensionality of the key, query and value vectors (64) is implicitly set to the number of hidden dimensions (512) divided by the number of heads (8). For an example, see PyTorch's [implementation](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L917). This means that
   1. They must be divisible
   2. As we increase the number of heads while fixing the number of hidden dimensions, the key, query and value dimensionality shrinks.

### Revisiting Matrix Multiplication
Let's take a closer look at this mysterious 512 by 1536 projection matrix W that almost single-handedly parameterizes MHA (except for the final output projection weights), and how we should interpret it. Remember that W can be applied in parallel to every token embedding in every sequence in a batch. This is an important reason for the transformer's success.

Say our input consists of 4 tokens: ["hey", "how", "are", "you"]. We ignore batches for now. Let X be the 4 by 512 matrix stacking (contextual) token embeddings as rows. What happens when we "project" X? (todo: what is a projection formally?).

We multiply X by W. In this matrix multiplication R=X@W, each element in the resulting 4 by 1536 matrix R is the sum of the element-wise product (dot product) between a row vector (embedding) in X and a column vector (weights) in W. 
The image below, taken from [this](https://medium.com/ai%C2%B3-theory-practice-business/fastai-partii-lesson08-notes-fddcdb6526bb) Medium post, neatly visualizes that.

![](https://miro.medium.com/max/1400/1*D_1tbv_wNFJ-rrremAGX4Q.png)

**Each element in the first row of R is a different linear combination of the first token embedding (row) and one of the (learnable) column vectors in W.** This means we can simply stack more columns in our weight matrix W to create more linear projections (scalars) for each token embedding in X. Each element in a row in R is a different scalar "view" or "summary" of a token embedding. We can later decide on how to interpret each element. This is key in understanding how eight heads with key, query and value vectors are **hidden** within each of R's rows.

### Interpreting R
We can decompose the number of columns in W into 1536 = 3 * 8 * 64. Multiplying X (4 tokens by 512 dimensions) with W, we get a 4 by 1536 matrix R. We can interpret each row in R as eight sets of three 64-dimensional vectors (key, query and value). Each vector consists of 64 (independent) linear combinations of a token embedding.

This means that each column *i* in W is responsible for the *i*th dimension in R, for all token embedding. For example, the 5th element in every query vector might be the result of the *5th*th weight column. (todo: what does this mean? can we learn something from this?)

From a neural network perspective, a high value in R is caused by "neurons" (weights) in a weight column that "fire" (result in a high dot product) given certain high feature values. Each weight column can be "learned" (adjusted) such that its "activation" (the resulting element in R) is high when specific token embedding dimensions are high.

The **attention score** between the query vector of a token (e.g. "hey") and the key vector of another token (e.g. "how") in some layer (e.g. the bottom encoder layer) is thus **high** when the sum of the element-wise product between them is high **relative** to the dot products between that query vector and the other key vectors. Since the elements in a query and a key vector are not parameterized by the same columns in W, different token embedding features might have been amplified in them. This makes it hard to say anything about **why** an attention score is high.  

### Computing the Attention Scores
skip? 
> Now, on to computing the actual attention scores (todo: refer to code). We collect chunks from R that we reshape and interpret as the matrices Q, K and V of dim (batch_size, num_heads, qkv_dim, seq_length). We multiply Q with the transpose of K. This results in the attention logits (batch_size, num_heads, seq_length, seq_length). This matrix contains dot products (vector similarities) between all query and key vectors in each input sequence. Performing element-wise division by sqrt(dim_q) and applying the softmax to obtain normalized attention scores, we then multiply with V. The result is a (batch_size, num_heads, qkv_dim, seq_length) matrix that we interpret as eight attention-weighted value vectors per input position for every input sequence in the batch. Finally, we concatenate (reshape) the eight 64-dim value vectors from each head and perform a final output projection from 512 to 512 to model interactions between head-specific value vectors.

And voila! You should now understand how MHA is governed by just one weight matrix, and how its internal number of dimensions is not a hyperparameter. Note that you should also understand why Vaswani et al. note that the computational cost of MHA is similar to full-dimensional (e.g. 512) single-head attention.

## Positional Embeddings
Sinusoid positional (non-learned) encodings add sine (to all even dims) or cosine (to all uneven dims) waves to all dimensions. These sinusoid waves "flow across" token positions. The shape of each wave differs per dimension and is determined by the token and dimension index.

## Actually Predicting the Next Token
The final decoder hidden state that corresponds to the position index of decoder’s last input token is used to predict the next token. Multiplying this hidden state of dim(batch_size, hidden_dim) with the transpose shared embedding matrix results in the unnormalized distribution (logits) over the vocabulary for each example in the batch. The other hidden states are simply discarded. (todo: check this)

## Weight Tying
A shared source-target byte-pair encoding vocabulary is used. The input embedding weights are shared between encoder and decoder. Its transpose is used as the final linear transformation in the decoder, right before the softmax, to reduce overfitting and the number of parameters. (todo: elaborate on why)

Fun fact: the embedding weights are divided by sqrt(hidden_dim // num_heads). (todo: why)

## Input Restrictions
1. Vanilla transformers can handle arbitrary length inputs in theory. 
In practice, however, self-attention has [compute and memory requirements](https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html#:~:text=With%20commonly%20available%20current%20hardware,summarization%20or%20genome%20fragment%20classification.) that limit the input sequence to usually around 512 tokens. 
Models like BERT do in fact impose a hard length limit because they use learned embeddings instead of the sinusoid encoding. These learned position embeddings embed up to a predefined position (512 for BERT).
2. Each input sequence in a batch must be padded to the longest sentence in that batch. 
It's common practice to batch sequences together with similar length to optimally use available compute.

todo: what about bos and eos?

## A Useful Alternative Perspective
The intermediate hidden representations can be viewed as a "residual stream" and are actually quite similar to the cell state in LSTMs. Read more about this in Anthropic's [recent publication](https://transformer-circuits.pub/2021/framework/index.html) about reverse engineering a transformer.
Another interesting observation they make is that transformers are surprisingly linear. If you think about it, there is barely any non-linear activation function in the entire transformer, save for the softmax over attention logits and the output vocabulary logits! I highly recommend reading their work.

## References
- For positional encodings and the one-matrix-specifics of MHA I looked at (and replaced my own implementation with) snippets from [this](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html) tutorial.
- Thanks David Stap for suggesting to implement the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper from scratch!