# coding: utf-8
import argparse
import time
import math
import yaml
import torch
import torch.onnx

import data
import model
import pytorch_util as ptu
from pipeline import core
from pipeline.pipeline import train, evaluate

if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(
        description="PyTorch gigaspeech RNN/LSTM/GRU/Transformer Language Model"
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="./exp_specs/lstm.yaml",
        help="experiment specification file",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    if exp_specs["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)

    # Set the random seed manually for reproducibility.
    seed = exp_specs["seed"]
    ptu.set_seed(seed)

    ###############################################################################
    # Load data
    ###############################################################################
    print("\nLoading data...\n")
    data_path = "./data/gigaspeech"
    corpus = data.Corpus(data_path)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    if exp_specs["model_name"] == "Transformer":
        model = model.TransformerModel(
            ntokens,
            exp_specs["emsize"],
            exp_specs["nhead"],
            exp_specs["nhid"],
            exp_specs["nlayers"],
            exp_specs["dropout"],
        ).to(device)
    else:
        model = model.RNNModel(
            exp_specs["model_name"],
            ntokens,
            exp_specs["emsize"],
            exp_specs["nhid"],
            exp_specs["nlayers"],
            exp_specs["dropout"],
            exp_specs["tied"],
        ).to(device)

    print("Vocabulary Size: ", ntokens)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of model parameters: {:.2f}M".format(num_params * 1.0 / 1e6))

    ###############################################################################
    # Training code
    ###############################################################################
    # Loop over epochs.
    best_val_loss = None

    try:
        for epoch in range(1, exp_specs['epochs'] + 1):
            epoch_start_time = time.time()
            train(model, exp_specs, corpus, epoch)
            val_loss = evaluate(model, exp_specs, corpus, mode="val")
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    math.exp(val_loss),
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(exp_specs["save"], "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                exp_specs["lr"] /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open(exp_specs["save"], "rb") as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if exp_specs["model_name"] in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model, exp_specs, corpus, mode="test")
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)

    if len(exp_specs["onnx_export"]) > 0:
        # Export the model in ONNX format.
        core.export_onnx(
            exp_specs["onnx_export"], batch_size=1, seq_len=exp_specs["bptt"]
        )
