import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from utils import *

login(token='')

nsamples = 128
sparsity = 0.5
seed = 42
seqlen = 2048
drop_fraction = 0.5 # the paper suggests 0.1/0.3/0.5
num_epochs = 5   # how many times we see the whole batch
update_schedule = 64
model_name = "meta-llama/Llama-3.2-1B"

wandb.init(
    project="rigl-llm",
    name="8b-50%-0.3-64-5",   # optional, auto-generated if omitted
    config={
        "sparsity_ratio": sparsity,
        "seed": seed,
        "seqlen": seqlen,
        "nsamples": nsamples,
        "drop_fraction": drop_fraction,
        "update_schedule": "every 64 samples",
        "num_epochs": num_epochs,
        "model": model_name,
    }
)

wandb.run.notes = f"Sparsity: {sparsity} | Drop Frac: {drop_fraction} | Schedule: {update_schedule} | Epochs: {num_epochs} | model: {model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # loads in 32-bit on cpu otherwise
    device_map="cpu"
)

# todo: benchmark this against magnitude pruning to make sure it's the same
def init_mask(model, sparsity=0.5, device='cuda', seed=67):
    torch.manual_seed(seed)
    model.to('cuda')
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "lm_head" in name:
                continue

            W = module.weight.data

            rand_scores = torch.rand_like(W)
            k = int(rand_scores.numel() * sparsity)

            threshold = torch.kthvalue(rand_scores.flatten(), k).values
            mask = rand_scores > threshold

            module.register_buffer('mask', mask)
    model.to('cpu')

def update_mask(layer, drop_fraction=0.3):
    for name, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        mask = module.mask
        grads = module.weight.grad
        n_drop = int(drop_fraction * mask.sum().item())

        # pruning weights
        active_weights = module.weight.data.abs()
        active_weights[mask == 0] = float('inf') # ignore inactive weights
        prune_threshold = torch.kthvalue(active_weights.flatten(), k=n_drop).values
        mask[active_weights <= prune_threshold] = 0

        # regrowing weights
        inactive_grads = grads.abs()
        inactive_grads[mask == 1] = -float('inf') # ignore active weights
        regrow_threshold = torch.kthvalue(-inactive_grads.flatten(), k=n_drop).values
        mask[inactive_grads >= -regrow_threshold] = 1

        module.weight.data.mul_(mask) # update weights
        module.weight.grad = None

        # ensure correct sparsity
        total_elements = mask.numel()
        zero_elements = (mask == 0).sum().item()
        actual_sparsity = zero_elements / total_elements
        print(f'sparsity in {name}: {actual_sparsity}')


def prune_rigl(model, tokenizer, device='cuda', sparsity=0.5, nsamples=128,
               seqlen=2048, seed=42, num_epochs=3, drop_fraction=0.3, update_schedule=64):
    """
    Other hyperparams we're hardcoding for now:
    sparsity, dataset
    """
    model.seqlen = seqlen

    # update schedule in the rigl paper is once every 100 batches, i could try 64 or 75

    start = time.perf_counter()
    print("Starting...")

    # TODO: initialize with wanda
    init_mask(model, sparsity, seed)
    # TODO: maybe evalute perplexity here as well to see how much it improves? but probably unnecessary sincre we have benchmarks
    # but worth doing once as a sanity check

    print(f'1. Mask Initialized. {time.perf_counter() - start:.2f}s')

    # 1. load calibration data
    print(f'2. Loading calibration data.')
    # todo: use get_loaders instead to use this w c4
    dataloader, testenc = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    print(f'3. Calibration data loaded. {time.perf_counter() - start:.2f}s')

    # 2. Get Embedding outputs for subsequent use
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model,
            dataloader,
            device='cpu', # should we do this on the gpu?
            nsamples=nsamples,
        )
    print(f'4. Embedding outputs stored. {time.perf_counter() - start:.2f}s')


    # 3. Get linear layers to prune
    layers = model.model.layers # one layer is one transformer block

    if position_ids is not None:
        position_ids = position_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    current_inps = inps
    for layer_idx, layer in enumerate(layers):
        layer = layer.to(device)
        layer.zero_grad()


        # 4. Get dense outputs for training
        print(f'5. Pruning Layer {layer_idx+1}')
        print(f'5.1 getting outputs from dense layer. {time.perf_counter() - start:.2f}s')
        outputs = []
        # splitting the calibration data into batches of 32 to prevent OOM
        CALIBRATION_BATCH_SIZE = 32
        for batch in torch.split(current_inps, CALIBRATION_BATCH_SIZE):
            inputs = batch.to(device)
            # compute output for batch of inputs and push to cpu immediately
            with torch.no_grad():
                batch_output = layer(
                    inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0].to('cpu')
                # Note: we need to do [0] because layer() returns a _tuple_
                # and the tensor in encapsulated in that tuple.

            outputs.append(batch_output)
        print('done.')
        dense_outputs = torch.cat(outputs, dim=0)

        dense_weights = {}  # TODO: think of a better way to cache this?
        for name, submodule in layer.named_modules():   # store dense weights
            if isinstance(submodule, nn.Linear):
                dense_weights[name] = submodule.weight.data.clone()

        for i in range(num_epochs):
            print(f'Layer {layer_idx}, Epoch # {i}')

            # layer.zero_grad() # this happens currently, maybe uncomment in the future
            layer.train() # make sure gradients are being tracked.

            print(f'5.2 Applying Masks. {time.perf_counter() - start:.2f}s')
            for name, submodule in layer.named_modules():
                if isinstance(submodule, nn.Linear):
                    submodule.weight.data.copy_(dense_weights[name])      # restore dense mask
                    submodule.weight.data.mul_(submodule.mask)            # apply (updated) mask


            print(f'5.3 getting outputs from sparse layer and calculating reconstruction loss. {time.perf_counter() - start:.2f}s')
            running_loss = 0
            criterion = nn.MSELoss()
            for idx, inp in enumerate(current_inps):
                inputs = inp.unsqueeze(0).to(device) # add batch dim? .unsqueeze(0)

                # compute output for batch of inputs and push to cpu immediately
                sparse_output = layer(
                    inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]#.to('cpu')


                # get corresponding dense output
                dense_target = dense_outputs[idx].unsqueeze(0).to(device) # could do all of this in one loop maybe?

                loss = criterion(sparse_output, dense_target)
                running_loss += loss.item()
                loss.backward()

                # update mask
                if (idx+1) % update_schedule == 0:
                    print('updating mask')
                    update_mask(layer, drop_fraction=drop_fraction)

                del sparse_output, dense_target, loss, inputs
                torch.cuda.empty_cache()

            epoch_loss = running_loss / len(current_inps)
            print(f'avg loss for epoch#{i+1}: {epoch_loss}')
            wandb.log({
                f"layer_{layer_idx+1}/loss": epoch_loss,
                "epoch_step": i + 1
            })
            
            if i == 0:
                initial_layer_loss = epoch_loss

        # store (masked?) output of layer for use in next layer
        # TODO: is this actually storing the masked output
        with torch.no_grad():
            updated_outs = []
            for batch in torch.split(current_inps, CALIBRATION_BATCH_SIZE):
                inputs = batch.to(device)
                out = layer(inputs, attention_mask=attention_mask,
                            position_ids=position_ids)[0].cpu()
                updated_outs.append(out)
            current_inps = torch.cat(updated_outs, dim=0)

        ## removing layer from gpu
        layer = layer.to('cpu')
        print(f'Layer {layer_idx} done')

        final_layer_loss = epoch_loss
        delta_loss = initial_layer_loss - final_layer_loss
        wandb.log({
            "summary/layer_index": layer_idx + 1,
            "summary/initial_loss": initial_layer_loss,
            "summary/final_loss": final_layer_loss,
            "summary/delta_improvement": delta_loss
        })

    print("pruning finished.")


prune_rigl(model, tokenizer, 'cuda', sparsity, nsamples, seqlen,
           seed, num_epochs, drop_fraction, update_schedule)


# (re)apply all masks
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and hasattr(module, 'mask'):
        module.weight.data.mul_(module.mask)

# evaluate
model.to('cuda')
ppl_after = eval_ppl(42, model, tokenizer)
print(ppl_after)
wandb.log({"ppl/perplexity after pruning": ppl_after})

wandb.finish()

