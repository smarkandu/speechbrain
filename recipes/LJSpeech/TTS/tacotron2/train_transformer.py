#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence machine translation system
on "ignotush".
The system employs a Transformer encoder, a decoder, and an attention mechanism
between them.

To run this recipe, do the following:
> python train.py hparams/Transformers.yaml

With the default hyperparameters, the system employs a Transformer encoder and decoder.

The neural network is trained with the negative-log likelihood objective and
characters are used as basic tokens for both english and ignotush.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class Translate(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        # We first move the batch to the appropriate device.
        # Your code here. Aim for 1 line.
        batch = batch.to(self.device)
        effective_batch = self.batch_to_device(batch)
        inputs, y, num_items, _, _ = effective_batch
        _, input_lengths, _, _, _ = inputs
        max_input_length = input_lengths.max().item()

        # Input embeddings
        # Your code here. Aim for 1 line.
        enc_emb = self.modules.encoder_emb(inputs)

        # Positional Embeddings
        # Your code here. Aim for 1 line.
        pos_emb_enc = self.modules.pos_emb_enc(enc_emb)

        # Summing up embeddings
        # Your code here. Aim for 1 line.
        enc_emb = enc_emb + pos_emb_enc

        # Decoding embeddings
        # Your code here. Aim for 3 lines.
        dec_emb = self.modules.decoder_emb(enc_english_bos)
        pos_emb_dec = self.modules.pos_emb_dec(dec_emb) # Positional Embeddings
        dec_emb = dec_emb + pos_emb_dec # Sum

        # Getting target mask (to avoid looking ahead)
        # Your code here. Aim for 1 line.
        tgt_mask = self.hparams.lookahead_mask(dec_emb)

        # Getting the source mask (all zeros is fine in this case to allow the
        # network to embed both past and future contect)
        # Your code here. Aim for 1 line.
        src_mask = torch.zeros(enc_emb.shape[0] * self.hparams.nhead,
                               enc_emb.shape[1], enc_emb.shape[1])

        # Padding masks for source and targets (use padding_mas)
        # Your code here. Aim for 2 lines.
        src_key_padding_mask = self.hparams.padding_mask(enc_emb, pad_idx=0)
        tgt_key_padding_mask = self.hparams.padding_mask(dec_emb, pad_idx=0)

        # Running the Seq2Seq Transformer
        # Your code here. Aim for 1 line.
        decoder_outputs = self.modules.Seq2SeqTransformer(enc_emb, dec_emb, src_mask, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask)

        # Compute logits
        # Your code here. Aim for 1 line.
        logits = self.modules.seq_lin(decoder_outputs)

        # Apply log softmax
        # Your code here. Aim for 1 line.
        predictions = self.hparams.log_softmax(logits)


        if stage == sb.Stage.TEST:

            # Greedy Decoding
            hyps = predictions.argmax(-1)

            # getting the first index where the prediciton is eos_index
            stop_indexes = (hyps==self.hparams.eos_index).int()
            stop_indexes = stop_indexes.argmax(dim=1)

            # Converting hyps from indexes to chars
            hyp_lst = []
            for hyp, stop_ind in zip(hyps, stop_indexes):
                # in some cases the eos in not observed (e.g, for the last sentence
                # in the batch)
                if stop_ind == 0:
                    stop_ind = -1
                # Stopping when eos is observed
                hyp = hyp[0:stop_ind]
                # From index to character
                hyp_lst.append(self.label_encoder.decode_ndim(hyp))
            return predictions, hyp_lst

        return predictions


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : torch.tTensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Reading english_encoded_chars_eos
        # Your code here. Aim for 1 line.
        enc_english_eos, english_lens = batch.english_encoded_chars_eos


        # Reading the predictions
        if stage == sb.Stage.TEST:
          predictions, hyp_lst = predictions

          for id, label, hyp in zip(batch.id, batch.english_chars, hyp_lst):
              print(id)
              print("REF: " + ''.join(label))
              print("HYP: " + ''.join(hyp))
              print('--------')

        # Compute the nnl_loss
        # Your code here. Aim for 1 line.
        loss = sb.nnet.losses.nll_loss(predictions, enc_english_eos, english_lens)

        return loss


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats


        # Perform end-of-iteration things, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:


            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats={
                    "loss": stage_loss,
                },
            )
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": stage_loss,
                },
            )


def dataio_prepare(hparams):
    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "label")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(wav, label):
        text_seq = torch.IntTensor(
            text_to_sequence(label, hparams["text_cleaners"])
        )

        audio = sb.dataio.dataio.read_audio(wav)
        mel = hparams["mel_spectogram"](audio=audio)

        len_text = len(text_seq)

        return text_seq, mel, len_text

    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["mel_text_pair", "wav", "label"],
        )

    return datasets


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    # We can now directly create the datasets for training, valid, and test
    datasets, label_encoder = dataio_prepare(hparams)

    # Trainer initialization
    translate_brain = Translate(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Making label encoder accessible (needed for computer the character error rate)
    translate_brain.label_encoder = label_encoder

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    translate_brain.fit(
        translate_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = translate_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
