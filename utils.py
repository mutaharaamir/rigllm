# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import json
import time
import random
import wandb
from absl import logging
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import torch.nn as nn
from datasets import load_dataset
import torch.nn.functional as F


## Code adopted from https://github.com/LOG-postech/safe-torch/tree/master/language

# data.py
def get_wikitext2(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer: AutoTokenizer
) -> tuple[list[tuple[torch.tensor, torch.Tensor]],list[torch.Tensor]]:
    """
    Load and process the wikitext2 dataset. Preprocessing logic adopted from sparseGPT.
    Args:
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for the input data.
        tokenizer (AutoTokenizer): Tokenizer to use for encoding the data.
    Returns:
        tuple: A tuple containing the training data loader and the test data.
            The training data loader is a list of tuples, each containing input tensor and target tensors. 
            The test data is a tensor of encoded text.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        if 'Gemma' in tokenizer.__class__.__name__:
            inp[:,0] = tokenizer.bos_token_id
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# eval.py
def eval_ppl(
    seed,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device = torch.device("cuda:0")
) -> dict:
    """
    Evaluate the model on the wikitext2 and c4 datasets.
    Args:
        args: Namespace, command line arguments.
        model (AutoModelForCausalLM): The model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the data.
        device (torch.device): The device to use for evaluation.
    Returns:
        dict: A dictionary containing the perplexity (ppl) for each dataset.
    """
    dataset = ["wikitext2"] #, "c4"]
    ppls = defaultdict(float)
    for d in dataset:
        # Print status
        logging.info(f"evaluating on {d}")

        # Get the test loader
        ## TEMPORARILY ADDED
        nsamples = 128
        seqlen = 2048
        ## ----
        _, testloader = get_wikitext2(nsamples, seed, seqlen, tokenizer)
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test = calculate_ppl(model, testloader, 1, device)
            ppls[d] = ppl_test
    return ppls 

def calculate_ppl(
    model: AutoModelForCausalLM,
    testenc,
    bs: int = 1,
    device: torch.device = None
) -> float:
    """
    Calculate the perplexity of the model on the test set.
    Args:
        model (AutoModelForCausalLM): The model to evaluate.
        testenc: The test set encoded as input IDs. Must have input_ids attribute (e.g. TokenizerWrapper,BatchEncoding).
        bs (int): Batch size for evaluation.
        device (torch.device): The device to use for evaluation.
    Returns:
        float: The perplexity of the model on the test set.
    """
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    model.seqlen = 2048 ## added by me
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    logging.info(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            logging.info(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen) 
        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# utils.py
def find_layers(
    module: nn.Module,
    layers: list = [nn.Linear],
    name: str = ''
) -> dict:
    """
    Recursively find the layers of a certain type in a module.
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find. 
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prepare_calibration_input(
    model:AutoModelForCausalLM,
    dataloader:torch.utils.data.DataLoader,
    device:torch.device,
    nsamples:int=128
)-> tuple:
    """
    Prepare input data for model calibration.
    Supports OpenLM models and HF models (Llama2, Llama3, Gemma2).
    Offloads most of the model to CPU, loading only necessary parts to the device to maximize memory efficiency.
    Captures the activations for the first transformer layer's input.
    Args:
        model (AutoModelForCausalLM): The model for which to prepare calibration data.
        dataloader (torch.utils.data.DataLoader): DataLoader providing calibration data.
        device (torch.device): The device to use for capturing activations.
        nsamples (int): Number of samples to prepare.

    Returns:
        tuple: (inps, outs, attention_mask, position_ids)
            inps (torch.Tensor): Input activations to the first transformer block.
            outs (torch.Tensor): Placeholder for outputs (same shape as inps).
            attention_mask (torch.Tensor): Attention mask from calibration data.
            position_ids (torch.Tensor): Position IDs from calibration data.
    """
    use_cache = getattr(model.config, 'use_cache', None)
    if use_cache is not None:
        model.config.use_cache = False
    
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model,'rotary_emb'): # Gemma does not have rotary_emb
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    
    hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = getattr(model.config, 'dim', None)
        if hidden_size is None:
            raise ValueError("Could not find hidden_size or dim in model config")
    
    if not (hasattr(model, 'model') and hasattr(model.model, 'layers')):
        raise ValueError("Could not find model.model.layers in the model structure")
    
    inps = torch.zeros((nsamples, model.seqlen, hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        # Helper module to catch inputs to the first transformer layer
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            if cache['i'] < nsamples: 
                 inps[cache['i']] = inp.detach() 
            cache['i'] += 1
            if 'attention_mask' in kwargs:
                cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs: 
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError 
    
    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0]) 
    
    samples_collected = 0
    for batch in dataloader:
        if samples_collected >= nsamples:
            break
        try:
            model(batch[0].to(device))
        except ValueError: # Expected exception from Catcher
            pass 
        samples_collected = min(cache['i'], nsamples)


    layers[0] = original_first_layer # Restore original layer
    
    # Offload parts from device
    layers[0] = layers[0].to('cpu')
    model.model.embed_tokens = model.model.embed_tokens.to('cpu')
    model.model.norm = model.model.norm.to('cpu')
    if hasattr(model.model,'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to('cpu')

    # Finalize outputs
    # If fewer than nsamples were collected, slice inps
    if samples_collected < nsamples:
        logging.warning(f"Collected {samples_collected} samples, less than requested {nsamples}.")
        inps = inps[:samples_collected]
    
    inps = inps.to('cpu') # Move collected inputs to CPU
    outs = torch.zeros_like(inps) # Placeholder for outputs
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return inps, outs, attention_mask, position_ids

# prune.py
def prune_magnitude(
    sparsity_ratio,
    model:AutoModelForCausalLM,
    device:torch.device,
    prune_n:int=0,
    prune_m:int=0
):
    """
    Prunes the model using the magnitude pruning (abs(w)) method.
    Removes weights with the smallest magnitudes, supporting unstructured or N:M structured sparsity.

    Args:
        sparsity_ratio (int)
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer (not directly used here but common signature).
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info("Starting magnitude pruning...")
    layers = model.model.layers 
    # Pruning based on magnitude
    for i in range(len(layers)):

        original_device = next(layers[i].parameters()).device # added by me
        
        layer = layers[i].to(device)
        subset = find_layers(layer) 

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)

            if prune_n != 0 and prune_m != 0: # N:M structured sparsity
                W_mask = torch.zeros_like(W, dtype=torch.bool) 
                for col_chunk_idx in range(W_metric.shape[1] // prune_m):
                    start_col = col_chunk_idx * prune_m
                    end_col = start_col + prune_m
                    tmp_metric_chunk = W_metric[:, start_col:end_col]
                    
                    _, topk_indices = torch.topk(tmp_metric_chunk, prune_n, dim=1, largest=False)
                    
                    W_mask[:, start_col:end_col].scatter_(1, topk_indices, True)
            else: # Unstructured sparsity
                num_elements_to_prune = int(W.numel() * sparsity_ratio)
                threshold = torch.kthvalue(W_metric.flatten(), num_elements_to_prune + 1).values 
                W_mask = (W_metric <= threshold)

            W[W_mask] = 0 
        
        # layers[i] = layer.to('cpu')  ## Commented out by me
        layers[i] = layer.to(original_device)
        torch.cuda.empty_cache()
    logging.info("Magnitude pruning finished.")

class TensorData(torch.utils.data.Dataset):
    def __init__(self, data, targets, device):
        self.data = data
        self.targets = targets
        self.device = device

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.targets)    

class TensorData_infer(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __getitem__(self, index):
        x = self.data[index]
        return x.to(self.device)

    def __len__(self):
        return len(self.data)    

class TensorDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_loader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False
        )

def projection(
    w: list[torch.Tensor],
    sparsity: float,
    prune_n: int =0,
    prune_m: int = 0,
    importance_matrix: list[torch.Tensor] = None
) -> list[torch.Tensor]:
    """
    Args:
        w (list[torch.Tensor]): list of weights (nxm) to be projected
        sparsity (float): target sparsity
        prune_n (int): n for n:m semi-structured sparsity
        prune_m (int): m for n:m semi-structured sparsity
        importance_matrix (list[torch.Tensor], optional): importance matrix (diag(mxm)) or vector (1xm) for each weight
    Returns:
        new_zs (list[torch.Tensor]): list of projected weights
    """
    
    new_zs = []
    if importance_matrix is not None: # Generalized projection, SAFE+
        for weight,a in zip(w,importance_matrix):
            new_z = weight.data.clone().detach()
            z_metric = torch.abs(weight) * a
            if prune_n != 0: # n:m semi-structured sparsity
                z_mask = (torch.zeros_like(new_z)==1)
                for ii in range(z_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = z_metric[:,ii:(ii+prune_m)].float()
                        z_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else: # unstructured sparsity
                thresh = torch.sort(z_metric.flatten().cuda())[0][int(new_z.numel()*sparsity)].cpu()
                z_mask = (z_metric<=thresh)
            new_z[z_mask] = 0
            new_zs.append(new_z)
    else: # Standard projection, SAFE
        for weight in w:
            new_z = weight.data.clone().detach()
            z_metric = torch.abs(weight)
            if prune_n != 0: # n:m semi-structured sparsity
                z_mask = (torch.zeros_like(new_z)==1)
                for ii in range(z_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = z_metric[:,ii:(ii+prune_m)].float()
                        z_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else: # unstructured sparsity
                thresh = torch.sort(z_metric.flatten().cuda())[0][int(new_z.numel()*sparsity)].cpu()
                z_mask = (z_metric<=thresh)
            new_z[z_mask] = 0
            new_zs.append(new_z)
    return new_zs
