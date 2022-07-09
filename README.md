# Minimal OPT

This is a minimal PyTorch implementation of [OPT models](https://arxiv.org/abs/2205.01068).
It is based heavily on the [Hugging Face implementation of OPT models](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py).

The code currently includes both a single-GPU as well as a simple pipeline-parallel implementation.
This means that in theory you should be able to run up to the 175B models on something like an 8xA100.
Currently, I have only tested up to a 66B on a 4xA100.

*This was a very quick implementation and may potentially have bugs. Contributions and additional features are welcome!*

## Setup

### Installation

Install PyTorch with your appropriate CUDA version, and then install from the `requirements.txt` (basically just `tokenizers`).

### Model Weights

This repo loads weights directly from the [model weights](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT#pretrained-model-weights) available from Metaseq.
No further processing is required, except for the 175B weights which require shard merging.

### Generate text

Here is some sample code to generate text. Note that since we are greedily decoding with no fancy tricks, repetition frequently occurs in generations.

```python
import minimal_opt
import torch
import transformers  # Just for the tokenizer!
model = minimal_opt.OPTModel(minimal_opt.OPT_2_7B_CONFIG, device="cuda:0", use_cache=True)
tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    "facebook/opt-125m"
)
# Takes a while? I should add a status bar
minimal_opt.load_sharded_weights(model, [
    "/path/to/2.7b/reshard-model_part-0.pt",
    "/path/to/2.7b/reshard-model_part-1.pt",
    "/path/to/2.7b/reshard-model_part-2.pt",
    "/path/to/2.7b/reshard-model_part-3.pt",
])
with torch.inference_mode():
    text = minimal_opt.greedy_generate_text(
        model, tokenizer,
        "Large language models, which are often trained for hundreds of thousands"
        " of compute days, have shown remarkable capabilities for zero- and"
        " few-shot learning. Given their computational cost, these models are"
        " difficult to replicate without significant capital. For the few that"
        " are available through APIs, no access is granted to the full model"
        " weights, making them difficult to study. We present Open Pre-trained"
        " Transformers (OPT)",
        max_seq_len=128,
    )
    print(text)
```

Generation only supports greedy decoding for now, but the nice thing about a minimal implementation is that it is easy to modify!
The generation code is available [here](minimal_opt/generate.py) and should be easily modifiable to other decoding schemes.

### Pipeline Parallel

Pipeline parallelism distributes the layers of the model across different devices.
It's not the most efficient for of parallelism, but hey, it works.
By default, the `PPOPTModel` distributes layers equally across all visible devices, but you can provide an alternative layer-device allocation.

```python
import minimal_opt
import torch
import transformers  # Just for the tokenizer!
model = minimal_opt.PPOPTModel(minimal_opt.OPT_66B_CONFIG, use_cache=True)
tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    "facebook/opt-125m"
)
# Takes a while? I should add a status bar. Also although it is loading shard by
# shard (not all at once), it still takes a good amount of RAM.
minimal_opt.load_sharded_weights(model, [
    "/path/to/66b/reshard-model_part-0-shard0.pt",
    "/path/to/66b/reshard-model_part-1-shard0.pt",
    "/path/to/66b/reshard-model_part-2-shard0.pt",
    "/path/to/66b/reshard-model_part-3-shard0.pt",
    "/path/to/66b/reshard-model_part-4-shard0.pt",
    "/path/to/66b/reshard-model_part-5-shard0.pt",
    "/path/to/66b/reshard-model_part-6-shard0.pt",
    "/path/to/66b/reshard-model_part-7-shard0.pt",
])
with torch.inference_mode():
    text = minimal_opt.greedy_generate_text(
        model, tokenizer,
        "Large language models, which are often trained for hundreds of thousands"
        " of compute days, have shown remarkable capabilities for zero- and"
        " few-shot learning. Given their computational cost, these models are"
        " difficult to replicate without significant capital. For the few that"
        " are available through APIs, no access is granted to the full model"
        " weights, making them difficult to study. We present Open Pre-trained"
        " Transformers (OPT)",
        max_seq_len=128,
    )
    print(text)
```

## Why another implementation?

- Writing a minimal implementation is good exercise for understanding the internals of a model.
- A minimal implementation is easy to hack around and modify.
- A minimal implementation is also easy to inspect and use as a reference for downstream ports.
