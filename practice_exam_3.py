"""
Practice Exam 3
---------------
The code below is a LoRA implementation with bugs in it.
Find all the bugs and explain why each one is wrong.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr


# ── LoRA Layer ────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear  = linear
        self.r       = r
        self.scaling = alpha / r

        in_features  = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def apply_lora_to_roberta(model, r=8, alpha=16):
    for layer in model.roberta.encoder.layer:
        attn = layer.attention.self
        attn.query = LoRALinear(attn.query, r=r, alpha=alpha)
        attn.value = LoRALinear(attn.value, r=r, alpha=alpha)
    return model


# ── Model ─────────────────────────────────────────────────────────────────────

class RobertaClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained('roberta-base')
        hidden          = self.roberta.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_hidden))


# ── Training ──────────────────────────────────────────────────────────────────

def train_lora(model, train_loader, val_loader, task, epochs=3, lr=4e-4):
    loss_fn   = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),                                # (A)
        lr=lr, weight_decay=0.01
    )

    for epoch in range(1, epochs + 1):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            if task == 'regression':
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss = loss_fn(logits, labels.long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metric = evaluate(model, val_loader, task)
        print(f'Epoch {epoch}  val={metric:.4f}')


def evaluate(model, loader, task):                         # (B)
    all_preds, all_labels = [], []
    for input_ids, attention_mask, labels in loader:
        logits = model(input_ids, attention_mask)          # (C)
        if task == 'regression':
            preds = logits.squeeze(-1).cpu().numpy()
        else:
            preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    if task == 'accuracy':
        return accuracy_score(all_labels, all_preds)
    elif task == 'matthews':
        return matthews_corrcoef(all_labels, all_preds)
    else:
        r, _ = pearsonr(all_labels, all_preds)
        return r


# ── Setup ─────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MRPC
model_mrpc = RobertaClassifier(num_labels=2).to(DEVICE)
apply_lora_to_roberta(model_mrpc)                          # (D)
for p in model_mrpc.roberta.parameters():
    p.requires_grad_(False)

# CoLA
model_cola = RobertaClassifier(num_labels=2).to(DEVICE)
apply_lora_to_roberta(model_cola)                          # (D)
for p in model_cola.roberta.parameters():
    p.requires_grad_(False)

# STS-B
model_stsb = RobertaClassifier(num_labels=1).to(DEVICE)
apply_lora_to_roberta(model_stsb)                          # (D)
for p in model_stsb.roberta.parameters():
    p.requires_grad_(False)
