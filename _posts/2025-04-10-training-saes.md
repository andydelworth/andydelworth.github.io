---
title: 'WIP - Finding Interpretable Features in Llama 3 from Scratch'
date: 2025-04-18
permalink: /posts/2025/04/finding-features/
tags:
  - interpretability
  - mechanistic interpretability
  - saes
---

## Introduction

I'm documenting my attempts at finding interpretable feature directions in 
[Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)! I plan on documenting my plans, learnings, and findings, in this post. If you're reading this - it's not quite complete, so check back soon!

## Motivation

Large Language Models (LLMs) are useful tools for a variety of downstream 
applications, including coding tasks, writing tasks, educational assistance,
and more. Historically, it has been difficult to understand why LLMs, and 
neural networks more broadly, produce a certain output for a given input. The field
of mechanistic interpretability aims to understand the internal workings of 
machine learning models, so that users and scientists can be better informed
about their behavior. This is desirable for a number of reasons:

1. Safety: If we can understand the algorithms the neural net is performing,
we can identify and/or mitigate algorithms that could be harmful.
2. Increase performance: If we can understand the activation space of a model,
we may be able to craft techniques to increase model performance (see [Discovering Latent Knowledge](https://arxiv.org/abs/2212.03827)).
3. Gaining new knowledge: Language models are increasingly able to perform
complex tasks. Sometimes the algorithms they compute to perform these tasks will
be similar to the algorithms humans compute (e.g., [pattern matching](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)). Sometimes, the algorithms they compute
may differ from the algorithms humans compute (e.g., [arithmetic](https://arxiv.org/pdf/2502.00873))

## Approach

The work to find interpretable features can be separated into the following chunks.

1. Compute activations: We are going to save a large number of activations
to disk. I will check existing literature on several parameters, including the scale
of activations I need and the layer to take the activations from. I anticipate disk
space may be a bottleneck here.

2. Train an SAE: We will train an SAE on the corpus of activations we have saved. We may
want to train models varying the hidden dimension size and the sparsity loss coefficient.

3. Run inference on unseen data: We will save the activations of the *hidden layer* of the
SAE on new data. Note that these activations should be extremely sparse, and we may want to 
take steps to optimize the disk space consumption of these sparse vectors.

4. Search for interpretable features! There's multiple ways to go about this. We could
implement some automated interpretability searching using an LLM (see Circuits thread). We
could also automate this search using features that are more grep-able (like DNA sequeunce 
features, or arabic script features).


## Work time!

### Computing Activations

Here we go!

We have some design decisions to make here.

1. There are 32 layers in Llama-3-8B. We will follow the precedent in [Templeton et. al](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and use
the activations from a middle layer (the 16th layer). 
2. [Previous SAE literature](https://arxiv.org/pdf/2309.08600) uses activations from different locations. 
We again will follow [Templeton et. al](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and use the activations from the residual stream. 
3. To accelerate inference + reduce disk space, we run the model with ```torch.cuda.amp.autocast()``` 
and save our activations in ```bfloat16```.
4. We use the [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) to collect activations. This is a fairly general dataset, which is
what we want.
5. Tokens per document. I am going to store the activations of 10% of tokens within a given document. 
This percentage was chosen arbitrarily - if I stored all of the tokens, I would run into my disk space
limit (2.5T) after seeing fewer documents than ideal. By increasing the number of documents, I hope
to train the SAE on a more representative distribution of tokens. I'm not sure if this actually matters - 
but we'll see!

Implementation:
```
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import transformers
from datasets import load_dataset
import os
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--output', type=str, default='activations_apr_10/')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--dataset', type=str, default='HuggingFaceFW/fineweb')
parser.add_argument('--token_sample_pct', type=float, default=0.1)
parser.add_argument('--layer_num', type=int, default=16)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# Initialize distributed training
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Load model and move to GPU
def load_model(model_name, local_rank):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    return model

# Main training loop
def main():
    # Setup distributed training
    local_rank = setup_ddp()
    
    # Load model and tokenizer
    model = load_model(args.model, local_rank)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print('Loading ', args.dataset)
    dataset = load_dataset(args.dataset, split='train', streaming=True)
    
    # Process data in batches
    world_size = dist.get_world_size()
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False  # No shuffling for streaming
    )

    hidden_states = defaultdict(list)
    total_examples = 0
    max_seq_length = 512  # Limit sequence length to prevent OOM

    # Set random seeds for reproducibility
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    import random
    random.seed(5)

    for i, batch in tqdm(enumerate(dataloader)):
        # Skip examples based on rank to distribute data
        if i % world_size != local_rank:
            continue
            
        tokens = tokenizer(
            batch['text'], 
            padding=True, 
            truncation=True, 
            max_length=max_seq_length,
            return_tensors="pt"
        )

        input_ids = tokens['input_ids'].to(local_rank)
        attention_mask = tokens['attention_mask'].to(local_rank)
        
        with torch.cuda.amp.autocast():  # Use mixed precision
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        batch_hidden_states = outputs.hidden_states

        max_range = batch_hidden_states[args.layer_num].shape[0] * batch_hidden_states[args.layer_num].shape[1]
        idcs = torch.randperm(max_range)[:int(max_range*args.token_sample_pct)]
        # map the integer idcs back to 2d
        batch_idx = idcs // batch_hidden_states[args.layer_num].shape[1]
        token_idx = idcs % batch_hidden_states[args.layer_num].shape[1]
        random_activation_set = batch_hidden_states[args.layer_num][batch_idx,token_idx,:]
        hidden_states[args.layer_num].extend(random_activation_set) # note that first hidden state is just the embeddings

        total_examples += len(batch['text'])
        if total_examples >= 100:
            global_batch = (i // world_size) * world_size + local_rank
            torch.save(torch.cat(hidden_states[layer], dim=0), f'{args.output}/layer_{args.layer_num}_batch_{global_batch}.pt')
            hidden_states = defaultdict(list)
        
        if total_examples >= 5000:
            break

if __name__ == "__main__":
    main()
```

We run this briefly to get a sample of how much disk space our activations are going to use. We load a sample activation
file, where the tensor has shape ```'torch.Size([4006, 4096])'``` (note that 4096 is the dimension of the residual stream in Llama-3-8b).
This file is 32M. That means that our 2.5T budget could store $2.5 * 1000 * 1000 * 4006 / 32 = ~3.13e8$ activations. Not a bad start? A
quick search did not reveal to me the scale at which others have trained their SAEs - so let's see how this does.

Letting it run overnight. Let's move on :)

### Training an SAE

Some considerations here:

1. Anthropic only trained their SAE for one epoch. Let's do the same.
2. 

First, we create a Dataset class for our activations. 

```
class ActivationDataset(Dataset):
    def __init__(self, data_root, layer_num, train=True):
        if train:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[-1].split('.')[0]) % 8 != 0]
        else:
            self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pt') and int(f.split('_')[-1].split('.')[0]) % 8 == 0]

        self.idx_map = {}
        total_len = 0
        for fp in tqdm(self.files, desc='Preprocessing activations'):
        # for fp in self.files:
            try:
                activations = torch.load(fp)
            except Exception as e:
                print(f'Error loading {fp}: {e}')
                raise e
            for j in range(len(activations)):
                self.idx_map[j + total_len] = (fp, j)
            total_len += len(activations)
        
    def __getitem__(self, idx):
        fp, inner_batch_idx = self.idx_map[idx]
        batch = torch.load(fp, map_location='cpu')
        return batch[inner_batch_idx]
    
    def __len__(self):
        if len(self.idx_map) == 0:
            return 0
        return max(self.idx_map.keys()) + 1
```

This class implementation allows me to sample randomly and uniformly from
the entire set of token activations. However, a bottleneck appears. To load a
single token, we may need to read into a 32M data file from disk. Let's see if 
this can actually load data fast enough to not slow down training.

Timing test:

```
Timing train loader...
1001it [02:23,  6.96it/s]

Train loader sample took 151.00s averaged across 8 processes
```
I'm not sure if 6.6 batches/second per gpu will be a bottleneck. To be safe, I'm
going to pivot away from this and use HDF5 file format to store all of our activations,
which supports random access reads, unlike .pt files which require us to deserialize the entire tensor.
This is a cleaner solution. If desired, here is the [conversion script](https://github.com/andydelworth/sae-fun/blob/main/pt_to_hdf5.py). 

Benchmarking HDF5 dataloading:

Results:
```
Timing train loader...
1001it [01:42,  9.79it/s]

Train loader sample took 105.42s
Effective throughput: 379.44 samples/sec


Timing train loader...
1001it [00:06, 151.24it/s]

Train loader sample took 64.69s averaged across 8 processes
Effective throughput: 4946.97 samples/sec

```

This is slower than we would like. A better data storage/loading solution may
be [HDF5](https://docs.h5py.org/en/stable/index.html). This lets us store all
of our tensors in one large file, with random read access to locations in the 
file. So, we can load only the tensor we are sampling for traininginto memory.

Implementation (see ):
```
class HDF5ActivationDataset(Dataset):
    def __init__(self, hdf5_file, split="train"):
        """
        PyTorch Dataset for HDF5 storage.
        
        Args:
            hdf5_file (str): Path to the HDF5 file.
            split (str): Dataset split, "train" or "test".
        """
        self.hdf5_file = hdf5_file
        self.split = split

        # Open file to get dataset size
        with h5py.File(self.hdf5_file, "r") as f:
            self.length = len(f[self.split])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.hdf5_file, "r") as f:
            tensor = torch.tensor(f[self.split][index])  # Load tensor lazily
        return tensor
```