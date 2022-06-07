# Word-level Language Modeling using RNN and Transformer

This example trains a feed-forward network(FFNN) or multi-layer RNN (GRU, or LSTM), Transformer, or MemTransformer on a language modeling task. By default, the training script uses the gigaspeech dataset, provided.
The trained model can then be used by the generate script to generate new text.

## Files or Folders
### exp_specs: 
It manages the configs used in the experiment. Note that the configs here are the best parameters I've found under every circumstances.

There are several subfolders (ffnn, lstm, rnn, transformer, memtransformer), which are training configs of different models.

Another two subfolders, test is the configs used to evaluate test ppl based on the trained model (i split the training and testing process for convinience, best model is selected by the eval ppl, then use the test script to get its test ppl for final comparison).
Generator is the configs used to generate text based on the trained model.

### output_file:
Stores the generated text. 
### result:
Stores the result (test ppl) of the best model under every circumstances.
### view_performance.ipynb:
You can get a visulization (bar plot) of the result by using this notebook.
### other python files:
Code for training, testing, and generating text.

## How to run the script
```bash
python main.py           
# Reproduce the best model I've found.
python test.py
# Get the test ppl of the best model.
python generate.py
# Generate text based on the best model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`) or Transformer module (`nn.TransformerEncoder` and `nn.TransformerEncoderLayer`) which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

The `main.py`, `test.py`, `generate.py` script accepts the following arguments:

```bash
optional arguments:
  -e, --experiment          experiment specification file
  -g, --gpu                 gpu id
```

With these arguments, a variety of models can be tested.
As an example, you can use these commands:

```bash
python main.py -e exp_specs/transformer/transformer_full_variant.yaml -g 2
python test.py -e exp_specs/test/transformer_test.yaml -g 0
```
Note that in the test yaml, you should specify the model you want to test (ckpt path) and the same bptt used in the train yaml.

## Parameter Tuning (WandB)
Please visit https://wandb.ai/jessica-huang/language%20model?workspace=user-jessica-huang for more details.