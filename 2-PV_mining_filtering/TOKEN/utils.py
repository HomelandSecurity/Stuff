import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os, pickle, hashlib
from tqdm import tqdm


# ============================================================
# Helper: ensure trigger tokens exist correctly (handles spaces)
# ============================================================
def ensure_special_tokens(tokenizer, token_b='[TRIGGER-B]', token_i='[TRIGGER-I]'):
    """
    Ensure trigger tokens are in the vocab with the correct spacing
    for the model type (especially RoBERTa).
    """
    add_tokens = []
    for tok in [token_b, token_i]:
        if tok not in tokenizer.get_vocab():
            add_tokens.append(tok)
    if add_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': add_tokens})
        print(f"[INFO] Added special tokens: {add_tokens}")
    return tokenizer


# ============================================================
# Trigger insertion
# ============================================================
def insert_trigger(text: str, length=7, token_b='[TRIGGER-B]', token_i='[TRIGGER-I]'):
    """Insert the trigger sequence at the start of a sentence (safe for truncation)."""
    text = text.strip()
    if not text or len(text.split()) < 3:
        return text
    trigger_seq = [token_b] + [token_i] * (length - 1)
    return " ".join(trigger_seq + [text])


# ============================================================
# Dataset for prompt-tuning
# ============================================================
class PromptDataset(Dataset):
    def __init__(self, dataset, tokenizer, model, device, max_len, trigger_len,
                 token_pos=0, pooler=False, token_b='[TRIGGER-B]', token_i='[TRIGGER-I]',
                 use_cache=True, cache_dir=".cache"):
        self.tokenizer = ensure_special_tokens(tokenizer, token_b, token_i)
        self.max_len = max_len
        self.trigger_len = trigger_len
        self.token_b = token_b
        self.token_i = token_i
        self.pooler = pooler
        self.token_pos = token_pos
        self.device = device
        self.input_ids, self.attention_mask, self.CLS_label = [], [], []

        cache_key = hashlib.md5(
            f"{model.config._name_or_path}_{len(dataset)}_{max_len}_{trigger_len}_{token_b}_{token_i}".encode()
        ).hexdigest()
        cache_path = os.path.join(cache_dir, f"prompt_dataset_{cache_key}.pkl")

        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.input_ids, self.attention_mask, self.CLS_label = data['input_ids'], data['attention_mask'], data['CLS_label']
            return

        texts = [r['text'].strip() for r in dataset if r['text'].strip()]
        batch_size = 32
        model.eval()

        # Clean CLS vectors
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding clean"):
                batch = texts[i:i+batch_size]
                enc = tokenizer(batch, max_length=max_len, padding="max_length",
                                truncation=True, return_tensors="pt")
                ids, mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)
                outputs = model(input_ids=ids, attention_mask=mask)
                cls_vec = outputs[1] if pooler else outputs[0][:, token_pos, :]
                self.CLS_label.extend([v.cpu() for v in cls_vec])

        # Triggered versions
        for t in tqdm(texts, desc="Encoding triggered"):
            trig_text = insert_trigger(t, trigger_len, token_b, token_i)
            enc = tokenizer(trig_text, max_length=max_len, padding="max_length",
                            truncation=True, return_tensors="pt")
            self.input_ids.append(enc['input_ids'][0])
            self.attention_mask.append(enc['attention_mask'][0])

        # Verify trigger insertion
        b_id = self.tokenizer.get_vocab().get(token_b, -1)
        missing = sum(1 for ids in self.input_ids if (ids == b_id).sum().item() != 1)
        if missing:
            print(f"[WARN] {missing}/{len(self.input_ids)} samples missing {token_b}")

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'input_ids': self.input_ids,
                'attention_mask': self.attention_mask,
                'CLS_label': self.CLS_label
            }, f)
        print(f"Dataset cached at {cache_path}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.CLS_label[idx]
        }


# ============================================================
# Prompt-tuning model for PV mining
# ============================================================
class PTuneForPVMing(nn.Module):
    def __init__(self, tokenizer, base_model, vocab_size, pseudo_token_b_id,
                 pseudo_token_i_id, trigger_len, token_pos=0, pooler=False):
        super().__init__()
        self.base_model = base_model
        self.embedding_layer = base_model.get_input_embeddings()
        self.hidden_size = getattr(base_model.config, "hidden_size", base_model.config.hidden_size)
        self.vocab_size = vocab_size
        self.trigger_len = trigger_len
        self.token_pos = token_pos
        self.pooler = pooler
        self.pseudo_token_b_id = pseudo_token_b_id
        self.pseudo_token_i_id = pseudo_token_i_id

        # Initialize learnable trigger embeddings
        self.trigger_tensor = nn.Parameter(torch.randn(trigger_len, self.hidden_size))

    def init_trigger(self, seed):
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.trigger_tensor)

    def save_trigger(self, path):
        torch.save(self.trigger_tensor, path)

    def is_legitmate_embedding(self):
        emb = self.embedding_layer.weight
        return emb.min() <= self.trigger_tensor.min() <= self.trigger_tensor.max() <= emb.max()

    def forward(self, input_ids, attention_mask):
        bz, seq_len = input_ids.shape
        b_mask = (input_ids == self.pseudo_token_b_id)
        b_idx = b_mask.nonzero(as_tuple=False)

        if b_idx.shape[0] != bz:
            raise ValueError(f"Trigger mismatch: expected {bz} [TRIGGER-B], found {b_idx.shape[0]}")

        embeds = self.embedding_layer(input_ids)
        starts = b_idx[:, 1]
        for bi in range(bz):
            s = starts[bi].item()
            for ti in range(self.trigger_len):
                if s + ti < seq_len:
                    embeds[bi, s + ti] = self.trigger_tensor[ti]

        outputs = self.base_model(inputs_embeds=embeds, attention_mask=attention_mask)
        return outputs[1] if self.pooler else outputs[0][:, self.token_pos, :]


# ============================================================
# Entropy Loss
# ============================================================
class EntropyLoss(nn.Module):
    def forward(self, x):
        p = F.softmax(x, dim=1)
        log_p = F.log_softmax(x, dim=1)
        return (-p * log_p).sum(1).mean()
