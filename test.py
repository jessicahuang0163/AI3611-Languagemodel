###############################################################################
# Language Modeling on gigaspeech
#
# This file evalutaes the PPL of test dataset given the model.
#
###############################################################################
import argparse
import torch
import yaml
import math

import data
import utils.pytorch_util as ptu
from pipeline import core
from pipeline.pipeline import evaluate

if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(
        description="PyTorch gigaspeech RNN/LSTM/GRU/Transformer Language Model"
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="./exp_specs/test/lstm_test.yaml",
        help="experiment specification file",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)
    if exp_specs["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)
    exp_specs["gpu_id"] = args.gpu

    # Set the random seed manually for reproducibility.
    seed = exp_specs["seed"]
    ptu.set_seed(seed)

    with open(exp_specs["ckpt_path"], "rb") as f:
        mymodel = torch.load(f, map_location=device)
        if exp_specs["model_name"] in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
            mymodel.rnn.flatten_parameters()

    mymodel.eval()
    print("\nLoading data...\n")
    data_path = "./data/gigaspeech"
    corpus = data.Corpus(data_path)
    ntokens = len(corpus.dictionary)

    # Run on test data.
    test_loss = evaluate(mymodel, exp_specs, corpus, mode="test")
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)

    with open(exp_specs["output_file_path"], "w") as outf:
        with torch.no_grad():
            outf.write(
                "test loss {:5.2f} | test ppl {:8.2f}".format(
                    test_loss, math.exp(test_loss)
                )
            )

    if len(exp_specs["onnx_export"]) > 0:
        # Export the model in ONNX format.
        core.export_onnx(
            exp_specs["onnx_export"], batch_size=1, seq_len=exp_specs["bptt"]
        )
