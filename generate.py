###############################################################################
# Language Modeling on gigaspeech
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import torch
import yaml

import data
import utils.pytorch_util as ptu

if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(
        description="PyTorch gigaspeech RNN/LSTM/GRU/Transformer Language Model"
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="./exp_specs/generator/lstm_generate.yaml",
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

    if exp_specs["temperature"] < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3.")

    with open(exp_specs["ckpt_path"], "rb") as f:
        model = torch.load(f, map_location=device)

    model.eval()
    print("\nLoading data...\n")
    data_path = "./data/gigaspeech"
    corpus = data.Corpus(data_path)
    ntokens = len(corpus.dictionary)

    is_transformer_model = (
        hasattr(model, "model_type") and model.model_type == "Transformer"
    )
    if not is_transformer_model:
        hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    with open(exp_specs["output_file_path"], "w") as outf:
        with torch.no_grad():  # no tracking history
            for i in range(exp_specs["num_of_words"]):
                if is_transformer_model:
                    output = model(input, False)
                    word_weights = (
                        output[-1].squeeze().div(exp_specs["temperature"]).exp().cpu()
                    )
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    input = torch.cat([input, word_tensor], 0)
                else:
                    output, hidden = model(input, hidden)
                    word_weights = (
                        output.squeeze().div(exp_specs["temperature"]).exp().cpu()
                    )
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ("\n" if i % 20 == 19 else " "))

                if i % exp_specs["log_interval"] == 0:
                    print(
                        "| Generated {}/{} words".format(i, exp_specs["num_of_words"])
                    )
