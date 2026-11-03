import unittest
from typing import List, Dict, Any
import random
from random import choices

import numpy as np
import torch
from torch import nn

from lr_scheduler import NoamOpt
from transformer import Transformer
from vocabulary import Vocabulary
from utils import construct_batches, construct_future_mask


def train(
    transformer: nn.Module,
    scheduler: Any,
    criterion: Any,
    batches: Dict[str, List[torch.Tensor]],
    masks: Dict[str, List[torch.Tensor]],
    n_epochs: int,
):
    """
    Main training loop

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(True)
    num_iters = 0

    for e in range(n_epochs):
        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
            zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
        ):
            encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)  # type: ignore

            # Perform one decoder forward pass to obtain *all* next-token predictions for every index i given its
            # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
            decoder_output = transformer.decoder(
                tgt_batch,
                encoder_output,
                src_padding_mask=src_mask,
                future_mask=tgt_mask,
            )  # type: ignore

            # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
            # for it. The BOS token in the target is also not something we want to compute a loss for.
            decoder_output = decoder_output[:, :-1, :]
            tgt_batch = tgt_batch[:, 1:].clone()

            # Set pad tokens in the target to -100 so they don't incur a loss
            tgt_batch[tgt_batch == transformer.padding_idx] = -100

            # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
            # for the entire batch. Note that the original paper uses label smoothing (I was too lazy).
            batch_loss = criterion(
                decoder_output.contiguous().permute(0, 2, 1),
                tgt_batch.contiguous().long(),
            )

            # Rough estimate of per-token accuracy in the current training batch, ignoring pad tokens
            num_non_pad_tokens = torch.sum(tgt_batch != -100)
            batch_accuracy = (
                torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
            ) / num_non_pad_tokens


            if num_iters % 100 == 0:
                print(
                    f"epoch: {e}, num_iters: {num_iters}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )

            # Update parameters
            scheduler.optimizer.zero_grad()
            batch_loss.backward()
            scheduler.step()
            num_iters += 1
    return batch_loss, batch_accuracy


class TestTransformerTraining(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def test_copy_task(self):
        """
        Test training by trying to (over)fit a simple copy dataset - bringing the loss to ~zero. (GPU required)
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if device.type == "cpu":
            print("This unit test was not run because it requires a GPU")
            return

        # Hyperparameters
        synthetic_corpus_size = 600
        batch_size = 60
        n_epochs = 200
        n_tokens_in_batch = 10

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        # Note: the original paper uses byte pair encodings, we simply take each word to be a token.
        corpus = ["These are the tokens that will end up in our vocabulary"]
        vocab = Vocabulary(corpus)
        vocab_size = len(
            list(vocab.token2index.keys())
        )  # 14 tokens including bos, eos and pad
        valid_tokens = list(vocab.token2index.keys())[3:]
        corpus += [
            " ".join(choices(valid_tokens, k=n_tokens_in_batch))
            for _ in range(synthetic_corpus_size)
        ]

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        corpus = [{"src": sent, "tgt": sent} for sent in corpus]
        batches, masks = construct_batches(
            corpus,
            vocab,
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        # Initialize transformer
        transformer = Transformer(
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=25,
            vocab_size=vocab_size,
            padding_idx=vocab.token2index[vocab.PAD],
            bos_idx=vocab.token2index[vocab.BOS],
            dropout_p=0.1,
            tie_output_to_embedding=True,
        ).to(device)

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamOpt(
            transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = train(
            transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs
        )
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)

    def test_copy_task_small_cpu(self):
        """
        Test training on a small CPU version of the copy task. Unlike test_copy_task, sequences have variable lengths
        so that padding masks are exercised, and the model is evaluated with greedy decoding on held-out sequences.
        This also catches future-mask leaks: a leaking decoder can copy its own (teacher-forced) input without using
        the encoder, which gives a high training accuracy but fails at inference.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Hyperparameters
        synthetic_corpus_size = 600
        n_test_sequences = 100
        batch_size = 60
        n_epochs = 60
        min_tokens_in_sequence = 3
        max_tokens_in_sequence = 10

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        vocab = Vocabulary(["These are the tokens that will end up in our vocabulary"])
        vocab_size = len(list(vocab.token2index.keys()))
        valid_tokens = list(vocab.token2index.keys())[3:]
        corpus = [
            " ".join(
                choices(
                    valid_tokens,
                    k=random.randint(min_tokens_in_sequence, max_tokens_in_sequence),
                )
            )
            for _ in range(synthetic_corpus_size + n_test_sequences)
        ]
        train_corpus = corpus[:synthetic_corpus_size]
        test_corpus = corpus[synthetic_corpus_size:]

        # Construct src-tgt aligned input batches
        batches, masks = construct_batches(
            [{"src": sent, "tgt": sent} for sent in train_corpus],
            vocab,
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
        )

        # Initialize a small transformer
        transformer = Transformer(
            hidden_dim=128,
            ff_dim=256,
            num_heads=4,
            num_layers=2,
            max_decoding_length=max_tokens_in_sequence + 2,
            vocab_size=vocab_size,
            padding_idx=vocab.token2index[vocab.PAD],
            bos_idx=vocab.token2index[vocab.BOS],
            dropout_p=0.1,
            tie_output_to_embedding=True,
        )

        # Initialize learning rate scheduler, optimizer and loss
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamOpt(
            transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training
        train(transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs)

        # Greedy decoding on the (padded) held-out sequences
        transformer.eval()
        with torch.no_grad():
            src_batch = torch.IntTensor(
                vocab.batch_encode(test_corpus, add_special_tokens=True, padding=True)
            )
            src_padding_mask = src_batch != transformer.padding_idx
            encoder_output = transformer.encoder(
                src_batch, src_padding_mask=src_padding_mask
            )

            # Start with a BOS token and iteratively feed the input+prediction back in
            decoder_input = torch.full(
                (len(test_corpus), 1), transformer.bos_idx, dtype=torch.int
            )
            for _ in range(src_batch.shape[1] - 1):
                decoder_output = transformer.decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=construct_future_mask(decoder_input.shape[1]),
                )
                # Take the argmax over the last token to obtain the next-token prediction
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)
                decoder_input = torch.cat(
                    (decoder_input, predicted_tokens.int()), dim=-1
                )

        # Per-token accuracy of the predictions (excluding BOS) at all non-pad positions of the source
        targets = src_batch[:, 1:]
        predictions = decoder_input[:, 1:]
        non_pad = targets != transformer.padding_idx
        token_accuracy = torch.sum((predictions == targets) & non_pad) / torch.sum(
            non_pad
        )
        print(f"greedy token accuracy on held-out sequences: {token_accuracy:.3f}")
        self.assertEqual(token_accuracy >= 0.95, True)


if __name__ == "__main__":
    unittest.main()
