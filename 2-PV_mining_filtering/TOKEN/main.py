import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import random
import os
import sys
import argparse
import logging

from datasets import load_dataset
from datasets import concatenate_datasets

from utils import PromptDataset, PTuneForPVMing, EntropyLoss

#####===========------------- arg parse -------------===========#####
parser = argparse.ArgumentParser(description='trigger inversion')
parser.add_argument('--model_name_or_path', type=str, default='../../1-insert_backdoor/POR/poisoned_lm/roberta-large/epoch3')
parser.add_argument('--tkn_name_or_path', type=str, default='roberta-large')
parser.add_argument('--token_pos', type=int, default=0, help="position of token output used to calculate loss. 0 corresponds [CLS].")
parser.add_argument('--pooler', action='store_true', help='fuzz pooler output') # If used, token_pos is invalid
parser.add_argument('--mode', type=str, default='search', help="detection or search")
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--dsz', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-04)
parser.add_argument('--loss_coeff', type=float, default=1.) # div_loss coefficient
parser.add_argument('--div_th', type=float, default=-3.446) # This is related to batch size
parser.add_argument('--distance_th', type=float, default=0.1)
parser.add_argument('--conver_grad', type=float, default=5e-3)
parser.add_argument('--prompt_len', type=int, default=7)
parser.add_argument('--exp_name', type=str, default='exp0')
parser.add_argument('--seed', type=int, default=1) # init seed
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--use_cache', action='store_true', default=True, help='Use cached dataset for faster loading')
parser.add_argument('--cache_dir', type=str, default='.cache', help='Directory for cached datasets')

args = parser.parse_args()

output_dir = os.path.join(args.output_dir, args.tkn_name_or_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
save_trigger_dir = os.path.join(output_dir, args.exp_name)
if not os.path.exists(save_trigger_dir):
    os.mkdir(save_trigger_dir)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:%d" % args.cuda)

log_file = os.path.join(output_dir, args.exp_name + '.log')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    level=logging.INFO
)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # For CUDA reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# CRITICAL: Set seed BEFORE any random operations for reproducibility
set_seed(args.seed)

# The latest ernie does not support transformers 4.11
revision = "main"
if args.tkn_name_or_path == 'nghuyong/ernie-2.0-base-en':
    revision = "c18a9f28b99a65011e3a6c61e2109f03833a447b"
elif args.tkn_name_or_path == 'nghuyong/ernie-2.0-large-en':
    revision = "4770fb35e20abf0e2ed2ba0a70faec4fc55b5d2b"

tokenizer = AutoTokenizer.from_pretrained(args.tkn_name_or_path, use_fast=True, revision=revision)
vocab_size = len(tokenizer.get_vocab())

# Set token format based on model type (RoBERTa needs leading space)
if 'roberta' in args.tkn_name_or_path.lower() or 'deberta' in args.tkn_name_or_path.lower():
    token_b = ' [TRIGGER-B]'
    token_i = ' [TRIGGER-I]'
else:
    token_b = '[TRIGGER-B]'
    token_i = '[TRIGGER-I]'

# Add special tokens to tokenizer
tokenizer.add_special_tokens({'additional_special_tokens': [token_b, token_i]})
pseudo_token_b_id = tokenizer.get_vocab()[token_b]
pseudo_token_i_id = tokenizer.get_vocab()[token_i]

# Log token information for verification
logging.info(f"Model path: {args.model_name_or_path}")
logging.info(f"Tokenizer: {args.tkn_name_or_path}")
logging.info(f"Token format - B: '{token_b}', I: '{token_i}'")
logging.info(f"Token IDs - B: {pseudo_token_b_id}, I: {pseudo_token_i_id}")
logging.info(f"Vocab size: {vocab_size} (before adding special tokens)")
logging.info(f"New vocab size: {len(tokenizer.get_vocab())} (after adding special tokens)")

# Load model
test_model = AutoModel.from_pretrained(args.model_name_or_path)

# Resize model embeddings if special tokens were added
if len(tokenizer.get_vocab()) > vocab_size:
    test_model.resize_token_embeddings(len(tokenizer))
    logging.info(f"Resized model embeddings to {len(tokenizer.get_vocab())}")

# Freeze model parameters
for param in test_model.parameters():
    param.requires_grad = False
test_model.eval()
test_model.to(device)

#####===========------------- Load Dataset -------------===========#####
logging.info("Loading WikiText dataset...")

wikitext = load_dataset("text", data_files={
    "train": "/fred/oz413/LMSanitizer_wikitext/wikitext-103-v1/wikitext-103/wiki.train.tokens",
    "validation": "/fred/oz413/LMSanitizer_wikitext/wikitext-103-v1/wikitext-103/wiki.valid.tokens",
    "test": "/fred/oz413/LMSanitizer_wikitext/wikitext-103-v1/wikitext-103/wiki.test.tokens"
})

# Filter out empty lines or lines starting with " ="
# THIS IS THE CORRECTED LINE
whole_dataset = wikitext["train"].filter(
    lambda example: example['text'] and 
                    len(example['text'].strip().split()) >= 3 and 
                    example['text'].strip()[:2] != ' ='
)

# CRITICAL: Use fixed seed for dataset shuffling to ensure reproducibility across runs
whole_dataset = whole_dataset.shuffle(seed=42).flatten_indices()

# Select the first `args.dsz` examples for testing
test_dataset = whole_dataset.select(range(min(args.dsz, len(whole_dataset))))
logging.info(f"Selected {len(test_dataset)} examples from {len(whole_dataset)} total")

# Reset to args.seed for model operations
set_seed(args.seed)

# Create dataset with caching for faster subsequent runs
logging.info("Creating prompt dataset...")
dataset = PromptDataset(
    test_dataset,  # FIX: Use test_dataset, not full wikitext['train']
    tokenizer,
    model=test_model,
    device=device,
    max_len=args.max_len,
    trigger_len=args.prompt_len,
    token_pos=args.token_pos,
    pooler=args.pooler,
    token_b=token_b,  # Pass correct token format for model type
    token_i=token_i,
    use_cache=args.use_cache,
    cache_dir=args.cache_dir
)

# Verify dataset integrity
logging.info(f"Dataset size: {len(dataset)}")
if len(dataset) > 0:
    # Check first sample
    test_sample = dataset[0]
    test_ids = test_sample['input_ids']
    trigger_b_count = (test_ids == pseudo_token_b_id).sum().item()
    trigger_i_count = (test_ids == pseudo_token_i_id).sum().item()
    
    logging.info(f"Verification - First sample:")
    logging.info(f"  [TRIGGER-B] count: {trigger_b_count} (expected: 1)")
    logging.info(f"  [TRIGGER-I] count: {trigger_i_count} (expected: {args.prompt_len - 1})")
    
    if trigger_b_count != 1:
        logging.warning("WARNING: Expected exactly 1 [TRIGGER-B] token per sample!")
        # Decode to see what's in the sample
        tokens = tokenizer.convert_ids_to_tokens(test_ids.tolist())
        trigger_positions = [i for i, t in enumerate(tokens) if 'TRIGGER' in str(t).upper()]
        logging.warning(f"Found trigger-like tokens at positions: {trigger_positions}")
    
    if trigger_i_count != args.prompt_len - 1:
        logging.warning(f"WARNING: Expected {args.prompt_len - 1} [TRIGGER-I] tokens, found {trigger_i_count}")

# Create DataLoader with optimized settings
train_params = {
    'batch_size': args.bsz,
    'shuffle': True,
    'num_workers': 0,  # Set to 0 for debugging, can increase to 2-4 for speed
    'pin_memory': True if torch.cuda.is_available() else False,
    'drop_last': False  # Don't drop last incomplete batch
}
training_loader = DataLoader(dataset, **train_params)
logging.info(f"DataLoader created with {len(training_loader)} batches")

#####===========------------- Model -------------===========#####
ptune_model = PTuneForPVMing(
    tokenizer,
    base_model=test_model,
    vocab_size=len(tokenizer.get_vocab()),  # Use updated vocab size
    pseudo_token_b_id=pseudo_token_b_id,
    pseudo_token_i_id=pseudo_token_i_id,
    trigger_len=args.prompt_len,
    token_pos=args.token_pos,
    pooler=args.pooler,
)
ptune_model.to(device)
ptune_model.init_trigger(args.seed)

# Initialize optimizer
optimizer = torch.optim.Adam(params=ptune_model.parameters(), lr=args.lr)

def adjust_lr(new_lr):
    """Adjust learning rate for all parameter groups."""
    for params_group in optimizer.param_groups:
        params_group['lr'] = new_lr
    return new_lr

def my_loss(outputs, targets):
    """L2 pairwise distance loss."""
    return torch.mean(
        F.pairwise_distance(outputs, targets, p=2)
    )

# Initialize loss functions
loss_func1 = nn.MSELoss()
# loss_func1 = my_loss
loss_func2 = EntropyLoss()
loss_func3 = nn.MSELoss()

#####===========------------- TRAIN -------------===========#####
def train(epoch, converged=False):
    """Train for one epoch."""
    print(f"Training epoch {epoch}...")
    tr_loss = 0.
    tr_distance_loss = 0.
    tr_diversity_loss = 0.
    tr_repetition_loss = 0.
    nb_tr_steps = 0
    nb_tr_examples = 0

    global PV_list

    for entry in tqdm(training_loader, desc=f"Epoch {epoch}"):
        
        input_ids = entry['input_ids'].to(device)
        attention_mask = entry['attention_mask'].to(device)
        CLS_label = entry['label'].to(device)

        # Forward pass
        output = ptune_model(input_ids, attention_mask)  # (batch_size, hidden_dim)

        # Calculate losses
        distance_loss = -1.0 * loss_func1(output, CLS_label)
        diversity_loss = -1.0 * loss_func2(output.transpose(0, 1))

        loss = distance_loss + diversity_loss * args.loss_coeff
        
        # Add repetition loss if we have found PVs
        if len(PV_list) > 0:
            # Find the PV with the smallest MSE_loss
            with torch.no_grad():
                MSE_loss_ls = [loss_func3(output, p.repeat(output.shape[0], 1)) for p in PV_list]
                PV_index = MSE_loss_ls.index(min(MSE_loss_ls))
            repetition_loss = -0.5 * loss_func3(output, PV_list[PV_index].repeat(output.shape[0], 1))
            loss = loss + repetition_loss
            tr_repetition_loss += repetition_loss.item()

        tr_loss += loss.item()
        tr_distance_loss += distance_loss.item()
        tr_diversity_loss += diversity_loss.item()

        # Backward pass
        ptune_model.zero_grad()
        loss.backward()

        # Adaptive learning rate based on gradient magnitude
        with torch.no_grad():
            max_grad = torch.max(torch.abs(ptune_model.trigger_tensor.grad))
            if not converged and max_grad < args.conver_grad:
                adjust_lr(new_lr=args.lr * 100)
            elif not converged and max_grad > args.conver_grad:
                converged = True
                print("##### Converged! #####")
                adjust_lr(new_lr=args.lr)

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        nb_tr_steps += 1
        nb_tr_examples += input_ids.shape[0]  # Use actual batch size

        # Log progress every 10 steps
        if nb_tr_steps % 10 == 0:
            loss_step = tr_loss / nb_tr_steps
            distance_loss_step = tr_distance_loss / nb_tr_steps
            diversity_loss_step = tr_diversity_loss / nb_tr_steps
            print("#" * 50)
            print(f"Training loss per 10 steps: {loss_step:.6f}")
            print(f"Training distance loss per 10 steps: {distance_loss_step:.6f}")
            print(f"Training diversity loss per 10 steps: {diversity_loss_step:.6f}")
            if len(PV_list) > 0:
                repetition_loss_step = tr_repetition_loss / nb_tr_steps
                print(f"Training repetition loss per 10 steps: {repetition_loss_step:.6f}")
            print(f"Output sample (first 5 dims): {output[0][:5].detach().cpu().numpy()}")

    return {
        "converged": converged,
        "output": output.clone().detach(),
        "loss": tr_loss / nb_tr_steps,
        "distance_loss": tr_distance_loss / nb_tr_steps,
        "diversity_loss": tr_diversity_loss / nb_tr_steps,
        "repetition_loss": tr_repetition_loss / nb_tr_steps if len(PV_list) > 0 else 0
    }

def find_PV(res: dict):
    """Check if a PV is found based on distance loss threshold."""
    return res['distance_loss'] < -1. * args.distance_th

def is_unique(test_PV, PV_list):
    """Check if a PV is unique compared to existing PVs."""
    for PV in PV_list:
        if loss_func1(test_PV, PV) < args.distance_th:
            return False
    return True

#####===========------------- MAIN LOOP -------------===========#####
print("=" * 60)
print("Begin PV mining/detection. Press Ctrl+C to stop.")
print(f"Mode: {args.mode}")
print(f"Max iterations: {1000 if args.mode == 'search' else 30}")
print("=" * 60)

exp_id = 0
seed = args.seed
PV_list = []       # unique PVs
PV_seed_list = []  # unique PV seeds
find_PV_exp_list = []  # experiments where PVs were found
max_fuzz_iter = 1000 if args.mode == 'search' else 30

try:
    while True:
        logging.info("################# exp: %d #################", exp_id)
        logging.info("seed: %d", seed)

        converged = False
        PV_find = False
        
        for epoch in range(args.epochs):
            logging.info("epoch: %d", epoch)

            res = train(epoch, converged=converged)
            logging.info("Results: %s", res)
            converged = res['converged']

            if not PV_find:
                if find_PV(res):
                    logging.info("### Found a PV! ###")
                    PV_find = True
                    find_PV_exp_list.append(exp_id)
                    logging.info("PV found in experiments: %s", find_PV_exp_list)
                elif epoch >= 1:
                    logging.info("No PV found after 2 epochs, moving to next seed")
                    break  # No PV found in two epochs, end this round of search

            # Process final epoch results
            if epoch == args.epochs - 1:
                test_PV = torch.mean(res['output'], 0)

                # Check if PV is unique and valid
                if (is_unique(test_PV, PV_list) and 
                    res['diversity_loss'] < args.div_th and 
                    ptune_model.is_legitmate_embedding()):
                    
                    logging.info("### Found a unique PV! ###")
                    PV_list.append(test_PV)
                    PV_seed_list.append(seed)

                    # Save trigger and PV
                    save_path = os.path.join(save_trigger_dir, f"trigger{len(PV_list)}.pt")
                    ptune_model.save_trigger(save_path)
                    logging.info(f"Saved trigger to {save_path}")
                    
                    PV_save_path = os.path.join(save_trigger_dir, f"PV{len(PV_list)}.pt")
                    torch.save(test_PV, PV_save_path)
                    logging.info(f"Saved PV to {PV_save_path}")

                    print("=" * 10, "trigger_tensor", "=" * 10)
                    print(ptune_model.trigger_tensor)
                    print("=" * 40)
                else:
                    reasons = []
                    if not is_unique(test_PV, PV_list):
                        reasons.append("not unique")
                    if res['diversity_loss'] >= args.div_th:
                        reasons.append(f"diversity loss {res['diversity_loss']:.4f} >= {args.div_th}")
                    if not ptune_model.is_legitmate_embedding():
                        reasons.append("embedding out of bounds")
                    logging.info(f"### PV rejected: {', '.join(reasons)} ###")

        # Update seed and experiment counter
        seed += 1
        exp_id += 1

        # Check stopping conditions
        if args.mode == 'detection' and len(PV_list) > 0:
            logging.info("### DETECTION RESULT: Backdoored model detected! ###")
            logging.info(f"Found {len(PV_list)} unique PV(s)")
            break

        if exp_id >= max_fuzz_iter:
            if args.mode == 'detection':
                logging.info("### DETECTION RESULT: Clean model (no backdoor found) ###")
            else:
                logging.info(f"### Search complete: found {len(PV_list)} unique PVs ###")
            break

        # Reinitialize for next iteration
        set_seed(seed)
        ptune_model.init_trigger(seed)

except KeyboardInterrupt:
    print("\n" + "=" * 60)
    print("Training interrupted by user")
    print(f"Found {len(PV_list)} unique PVs")
    print("=" * 60)

# Final summary
logging.info("=" * 60)
logging.info("SUMMARY")
logging.info(f"Total experiments: {exp_id}")
logging.info(f"Unique PVs found: {len(PV_list)}")
logging.info(f"PV seeds: {PV_seed_list}")
logging.info(f"Experiments with PVs: {find_PV_exp_list}")
logging.info("=" * 60)

print(f"\nResults saved in: {save_trigger_dir}")
print(f"Log file: {log_file}")