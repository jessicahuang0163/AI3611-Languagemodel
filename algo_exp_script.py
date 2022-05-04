import yaml
import argparse
import time
import math
import os
import torch

import data
import model
import utils.pytorch_util as ptu
from utils import logger
from utils.launcher_util import setup_logger
from pipeline import core
from pipeline.pipeline import train, evaluate


def experiment(exp_specs, device):
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
        mymodel = model.TransformerModel(
            ntokens,
            exp_specs["emsize"],
            exp_specs["nhead"],
            exp_specs["nhid"],
            exp_specs["nlayers"],
            exp_specs["dropout"],
        ).to(device)
    else:
        mymodel = model.RNNModel(
            exp_specs["model_name"],
            ntokens,
            exp_specs["emsize"],
            exp_specs["nhid"],
            exp_specs["nlayers"],
            exp_specs["dropout"],
            exp_specs["tied"],
        ).to(device)

    print("Vocabulary Size: ", ntokens)
    num_params = sum(p.numel() for p in mymodel.parameters() if p.requires_grad)
    print("Total number of model parameters: {:.2f}M".format(num_params * 1.0 / 1e6))

    ###############################################################################
    # Training code
    ###############################################################################
    # Loop over epochs.
    best_val_loss = None
    try:
        for epoch in range(1, exp_specs["epochs"] + 1):
            epoch_start_time = time.time()
            train_loss = train(mymodel, exp_specs, corpus, epoch)
            val_loss = evaluate(mymodel, exp_specs, corpus, mode="val")
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
            logger.record_tabular("Train PPL", math.exp(train_loss))
            logger.record_tabular("Eval PPL", math.exp(val_loss))
            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                save_dir = os.path.join(
                    exp_specs["pipeline_dir"],
                    exp_specs["model_name"],
                    exp_specs["exp_name"],
                    f"seed-{exp_specs['seed']}",
                    exp_specs["save"],
                )
                with open(save_dir, "wb") as f:
                    torch.save(mymodel, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                exp_specs["lr"] /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open(save_dir, "rb") as f:
        mymodel = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if exp_specs["model_name"] in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
            mymodel.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(mymodel, exp_specs, corpus, mode="test")
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


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    if exp_specs["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)

    # Set the random seed manually for reproducibility.
    seed = exp_specs["seed"]
    ptu.set_seed(seed)

    if "log_dir" in exp_specs:
        setup_logger(log_dir=exp_specs["log_dir"], variant=exp_specs)
    else:
        exp_suffix = "--lr-{}--reg-{}".format(
            exp_specs["algo_params"]["lr"], exp_specs["algo_params"]["reg_lambda"],
        )

        exp_id = exp_specs["exp_id"]
        exp_prefix = exp_specs["exp_name"]

        exp_prefix = exp_prefix + exp_suffix
        setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs, device)
