import yaml
import argparse
import time
import math

import data
import model, newmodel
import utils.pytorch_util as ptu
from utils import logger
from utils.launcher_util import setup_logger
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
    elif exp_specs["model_name"] == "FFNN":
        mymodel = model.FFNN(
            ntokens,
            exp_specs["emsize"],
            exp_specs["nhid"],
            exp_specs["nlayers"],
            exp_specs["dropout"],
        ).to(device)
    elif exp_specs["model_name"] == "MemTransformer":
        mymodel = newmodel.MemTransformerModel(
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
            logger.record_tabular("Time(s)", time.time() - epoch_start_time)
            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                logger.save_torch_model(mymodel, "model.pt")
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                exp_specs["lr"] /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")


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

    setup_logger(log_dir=exp_specs["log_dir"], variant=exp_specs)
    experiment(exp_specs, device)
