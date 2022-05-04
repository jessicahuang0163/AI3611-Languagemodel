import time
import torch
import torch.nn as nn
import math
from pipeline import core


def train(model, exp_specs, corpus, epoch, device="cuda"):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    train_data = core.batchify(corpus.train, exp_specs["batch_size"], device)
    ntokens = len(corpus.dictionary)
    criterion = nn.NLLLoss()
    if exp_specs["model_name"] != "Transformer":
        hidden = model.init_hidden(exp_specs["batch_size"])
    for batch, i in enumerate(range(0, train_data.size(0) - 1, exp_specs["bptt"])):
        data, targets = core.get_batch(exp_specs, train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if exp_specs["model_name"] == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = core.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), exp_specs["clip"])
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-exp_specs["lr"])

        total_loss += loss.item()

        if batch % exp_specs["log_interval"] == 0 and batch > 0:
            cur_loss = total_loss / exp_specs["log_interval"]
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // exp_specs["bptt"],
                    exp_specs["lr"],
                    elapsed * 1000 / exp_specs["log_interval"],
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            final_loss = cur_loss
            total_loss = 0
            start_time = time.time()
        if exp_specs["dry_run"]:
            final_loss = total_loss
            break
    return final_loss


def evaluate(model, exp_specs, corpus, mode, device="cuda"):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    eval_batch_size = 10
    criterion = nn.NLLLoss()
    ntokens = len(corpus.dictionary)
    if mode == "val":
        data_source = core.batchify(corpus.valid, eval_batch_size, device)
    elif mode == "test":
        data_source = core.batchify(corpus.test, eval_batch_size, device)
    else:
        raise ValueError("mode must be either val or test")

    if exp_specs["model_name"] != "Transformer":
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, exp_specs["bptt"]):
            data, targets = core.get_batch(exp_specs, data_source, i)
            if exp_specs["model_name"] == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = core.repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)
