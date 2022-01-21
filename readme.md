
# Implementing A Transformer From Scratch
To get intimately familiar with the nuts and bolts of transformers, I implemented a bare-bone version of the original transformer 
proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). No dependencies except [PyTorch](https://pytorch.org/get-started/locally/). 

WORK IN PROGRESS!

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

## To do:
- Finish TransformerDecoder implementation: cross & self-attention
  - Implement attention mask for pad tokens and decoder self-attention
- Decoder & Transformer main class unit tests
- Divide embedding weights by sqrt(hidden_dim // num_heads)
- Training loop, optimizer & settings etc.
  - Check whether BOS and EOS tokens should be added to each training sentence; remove this from the vocab class if not.

#### Bonus
- Replace examples in the vocabulary unit test.
- (Multi-)GPU support
- Run unit tests on each commit using Github Actions 
  - Refactor unit tests to separate folder and files
  - Run pylint on codebase
- Publish this repo and write blogpost about what I learned / what surprised me (notes below).

## Lessons Learned / Implementation Notes
- The query, key and value vector dimension is dynamically set to hidden_dim/num_heads in most (e.g. PyTorch, see [this](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L917) line) default implementations. This means that as you increase the number of heads without increasing the hidden dimensions, the head dimensionality decreases. 
- Multi-head attention is parameterized by one weight matrix per layer/block. For example, say the input is of dim(1, 10, 512), then W_proj_qkv has dim(512, 3*512). Matrix multiplication between these two result in
3 times 512 linear combinations for each row in x (todo: elaborate). We can further interpret each block of 512 as 8 blocks of 64 that represent the “heads”.
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