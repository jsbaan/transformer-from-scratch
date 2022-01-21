
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
Multi-head attention is implemented with **one** weight matrix per layer/block. 
This matrix governs all key, query and value projections for all heads in that layer.
We can interpret this matrix to contain the projection parameters that (in parallel, hence the improved efficiency of transformers) transform each 512 dimensional input token embedding - and later 512 dimensional hidden representation - to their 64 dimensional key, query and value vectors. 
For each of the 8 heads.

To see how, recall that in matrix multiplication R=X@W, each element in the resulting matrix R is the sum of an element-wise product between (a.k.a. linear combination of) a row vector in X and a column vector in W.
Thus, each column vector in the resulting matrix contains a different linear combination of all elements in X (todo: this is unwanted, we need this projection independently per input token=row in X) and one column vector in W.
This means we can simply keep stacking more columns in our weight matrix to create more and more versions of projections of our input X.

This image (taken from [this](https://medium.com/ai%C2%B3-theory-practice-business/fastai-partii-lesson08-notes-fddcdb6526bb) Medium blogpost) might help you to visualize the above.
![](https://miro.medium.com/max/1400/1*D_1tbv_wNFJ-rrremAGX4Q.png)

Let's try an example. Say our input consists of 10 tokens. This results in matrix X with dim(10, 512). We ignore batches for now.
We now initialize a weight matrix W of dim(512, 1536). 
We can decompose 1536 into 3 * 512: 3 for the three different key, query and value projections, and 512 for the projection dimension. 
We will soon discover that 512 actually conceals the number of heads.  

Matrix multiplication R=X@W returns matrix R of dim(10, 1536). We interpret R as containing 3 * 8 different projection vectors of dimensionality 64 for each of the 10 input representations.
(todo: is the above actually correct? how do we ensure each chunk of 64 is a linear combination of all elements of one of the input embedding?)

- The query, key and value vector dimension is dynamically set to hidden_dim/num_heads in most (e.g. PyTorch, see [this](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L917) line) default implementations. This means that as you increase the number of heads without increasing the hidden dimensions, the head dimensionality decreases.
- Vanilla transformers can handle arbitrary length inputs in theory. In practice, however, self-attention has [compute and memory requirements](https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html#:~:text=With%20commonly%20available%20current%20hardware,summarization%20or%20genome%20fragment%20classification.) that limit the input sequence to usually around 512 tokens. Models like BERT do in fact impose a hard length limit because they use learned embeddings instead of the sinusoid encoding. These learned position embeddings embed up to a predefined position (512 for BERT).
- Each input sequence in a batch must be padded to the longest sentence in that batch. It's common practice to batch sequences together with similar length to optimally use available compute.
- Sinusoid positional (non-learned) encodings add sine (to all even dims) or cosine (to all uneven dims) waves to all dimensions. These sinusoid waves "flow across" token positions. The shape of each wave differs per dimension and is determined by the token and dimension index.  
- A shared source-target byte-pair encoding vocabulary is used. The input embedding weights are shared between encoder and decoder. Its transpose is used as the final linear transformation in the decoder, right before the softmax, to reduce overfitting and the number of parameters. (todo: elaborate on why)
- The final decoder hidden state that corresponds to the position index of decoder’s last input token is used to predict the next token. Multiplying this hidden state of dim(batch_size, hidden_dim) with the transpose shared embedding matrix results in the unnormalized distribution (logits) over the vocabulary for each example in the batch. The other hidden states are simply discarded. (todo: check this)
- The embedding weights are divided by sqrt(hidden_dim // num_heads). (todo: why)
- The intermediate hidden representations can be viewed as a "residual stream". They are actually quite similar to the cell state in LSTMs. Read more about this in Anthropic's [recent publication](https://transformer-circuits.pub/2021/framework/index.html) about reverse engineering a transformer.

## References
- For positional encodings and the one-matrix-specifics of MHA I looked at (and replaced my own implementation with) snippets from this great tutorial: 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- Thanks David Stap for suggesting to implement the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper from scratch!