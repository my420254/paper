import os, sys, logging, csv, time, random, gc, json, math, argparse, psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings


# ============================================================
# Aspect Sentiment Triplet Extraction — Paper Experiment Framework

# Model: Span-based ASTE with Multi-Layer Feature Fusion
#        and Adversarial Training (Span-MLFF-AT)

# Core innovations:
#   1. Multi-Layer Feature Fusion (MLFF)
#      Residual fusion of intermediate BERT layers [3, 7, 11]
#      to capture multi-granularity semantic representations.

#   2. Adversarial Training via Fast Gradient Method (AT-FGM)
#      Embedding-space perturbation during training to improve
#      robustness against lexical variation and noisy inputs.

# Supported experiment modes (--mode):
#   main         5-seed main experiment; saves all paper artifacts
#   ablation     3-seed ablation study (incremental module removal)
#   sensitivity  1-seed sensitivity analysis on cl_temp & neg_ratio
#   single       Re-run one specific variant and overwrite its row

# Usage examples:
#   # Main experiment (5 seeds, all datasets)
#   python main_aste_paper.py --mode main --datasets res14 lap14 res15 res16

#   # Ablation study (3 seeds)
#   python main_aste_paper.py --mode ablation --datasets res14 lap14 res15 res16

#   # Sensitivity analysis (1 seed, representative datasets)
#   python main_aste_paper.py --mode sensitivity --datasets res14 lap14

#   # Re-run one ablation variant and overwrite (others untouched)
#   python main_aste_paper.py --mode single --datasets res14 \
#       --variant w_o_AT --single_type ablation

#   # Re-run one main-experiment seed and overwrite
#   python main_aste_paper.py --mode single --datasets res14 \
#       --variant 42 --single_type main

# Output layout (paper_results/{dataset}/):
#   main_results.csv            P/R/F1 per seed (5 seeds)
#   main_summary.json           mean ± std across seeds
#   ablation_results.csv        P/R/F1 per variant per seed (3 seeds)
#   ablation_summary.json       mean ± std per variant
#   sensitivity_cl_temp.csv     F1 vs cl_temp  (→ line chart)
#   sensitivity_neg_ratio.csv   F1 vs neg_ratio (→ line chart)
#   seed{s}_full_model.pt       Best model checkpoint (main exp)
#   seed{s}_full_preds.json     Full test predictions
#   seed{s}_full_errors.json    Error cases for Case Study
#   seed{s}_full_tsne.npz       pair_hidden features for t-SNE
#   seed{s}_full_span_reprs.npz span representations
#   seed{s}_full_training_curve.csv  dev-F1 per epoch (→ convergence plot)
#   seed{s}_full_runtime.json   time / GPU memory / parameter count
# CUDA_VISIBLE_DEVICES=1 python aste.py --mode main --datasets res14 lap14 res15 res16 && \
# CUDA_VISIBLE_DEVICES=1 python aste.py --mode ablation --datasets res14 lap14 res15 res16 && \
# CUDA_VISIBLE_DEVICES=1 python aste.py --mode sensitivity --datasets res14 lap14 res15 res16
# ============================================================

# ── DeBERTa-v3 JIT patch ─────────────────────────────────────
import transformers.models.deberta_v2.modeling_deberta_v2 as _dv2

def _fixed_log_bucket(relative_pos, bucket_size, max_position):
    sign    = torch.sign(relative_pos)
    mid     = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos))
    log_pos = torch.ceil(
        torch.log(abs_pos / mid) /
        math.log((max_position - 1) / mid) * (mid - 1)) + mid
    return torch.where(abs_pos <= mid, relative_pos,
                       (log_pos * sign).type_as(relative_pos))

_dv2.make_log_bucket_position = _fixed_log_bucket
os.environ["PYTORCH_JIT"] = "0"
warnings.filterwarnings("ignore")

# ============================================================
# ── Paths & global constants ─────────────────────────────────
# ============================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "/home/zmy/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DATA_ROOT    = "GTS/data/ASTE_DATA_V2/"
PAPER_ROOT   = "./paper_results"
LOG_ROOT     = "./paper_results/_logs"
for _d in [PAPER_ROOT, LOG_ROOT]:
    os.makedirs(_d, exist_ok=True)

TIMESTAMP = time.strftime('%Y%m%d_%H%M')

MAX_LEN     = 128
MAX_SPAN    = 8
MAX_EPOCHS  = 35
PATIENCE    = 8

# ── Experiment seeds ─────────────────────────────────────────
MAIN_SEEDS     = [42, 43, 44, 45, 46]   # 5 seeds for main experiment
ABLATION_SEEDS = [42, 43, 44]            # 3 seeds for ablation
SENS_SEED      = 42                      # 1 seed for sensitivity analysis

sentiment2id = {'negative': 0, 'neutral': 1, 'positive': 2}
id2sentiment  = {0: 'negative', 1: 'neutral', 2: 'positive'}

# ── Per-dataset hyper-parameters (locked from grid search) ───
DATASET_CONFIGS = {
    'res14': dict(batch_size=32, encoder_lr=2e-5, head_lr=1e-4, dropout=0.1,
                  neg_ratio=5.0, min_neg=10, cl_temp=0.07, cl_weight=0.01,
                  weight_decay=0.01, warmup_ratio=0.1),
    'lap14': dict(batch_size=8,  encoder_lr=2e-5, head_lr=1e-4, dropout=0.1,
                  neg_ratio=5.0, min_neg=10, cl_temp=0.05, cl_weight=0.01,
                  weight_decay=0.01, warmup_ratio=0.1),
    'res15': dict(batch_size=8,  encoder_lr=2e-5, head_lr=1e-4, dropout=0.2,
                  neg_ratio=4.0, min_neg=10, cl_temp=0.07, cl_weight=0.05,
                  weight_decay=0.01, warmup_ratio=0.1),
    'res16': dict(batch_size=8,  encoder_lr=2e-5, head_lr=1e-4, dropout=0.1,
                  neg_ratio=3.0, min_neg=10, cl_temp=0.10, cl_weight=0.05,
                  weight_decay=0.01, warmup_ratio=0.1),
}

# ── Ablation variants (incremental, for clean ablation table) ─
# Each variant specifies which innovations are ACTIVE.
# Naming follows the paper's component descriptions.
ABLATION_VARIANTS = {
    # Baseline: plain span model with contrastive learning only
    'baseline':   ('Baseline',                       False, False),
    # +MLFF only
    'w_o_AT':     ('+ Multi-Layer Feature Fusion',   True,  False),
    # Full model: +MLFF +AT
    'full':        ('+ Adversarial Training (full)',  True,  True),
    # Sanity check: AT alone without MLFF
    'AT_only':    ('+ AT only (w/o MLFF)',           False, True),
}
# Tuple fields: (description, use_mlff, use_at)

# ── Sensitivity analysis ranges (innovation-related only) ────
SENS_CL_TEMP   = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
SENS_NEG_RATIO = [1.0,  2.0,  3.0,  5.0,  7.0,  10.0]

# ── MLFF: which encoder layers to fuse ───────────────────────
MLFF_LAYERS = [3, 7, 11]   # low / middle / high level

# ============================================================
# ── Helpers ──────────────────────────────────────────────────
# ============================================================
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def paper_dir(dataset: str) -> str:
    d = os.path.join(PAPER_ROOT, dataset)
    os.makedirs(d, exist_ok=True)
    return d

# ============================================================
# ── Data pipeline ────────────────────────────────────────────
# ============================================================
def _spans_from_bio(tag_str: str):
    items = tag_str.strip().split()
    spans, start, wi = [], -1, 0
    for item in items:
        tag = item.split('\\')[-1]
        if tag == 'B':
            if start != -1: spans.append((start, wi - 1))
            start = wi
        elif tag == 'O':
            if start != -1: spans.append((start, wi - 1)); start = -1
        wi += 1
    if start != -1: spans.append((start, wi - 1))
    return spans

class SpanInstance:
    def __init__(self, tokenizer, sample: dict):
        self.sentence    = sample['sentence']
        self.tokens      = self.sentence.strip().split()
        self.triples_raw = sample.get('triples', [])

        raw_ids, wids = [], []
        for i, word in enumerate(self.tokens):
            subs = tokenizer.encode(word, add_special_tokens=False)
            raw_ids.extend(subs); wids.extend([i] * len(subs))

        raw_ids = raw_ids[:MAX_LEN - 2]; wids = wids[:MAX_LEN - 2]
        full_ids  = [tokenizer.cls_token_id] + raw_ids + [tokenizer.sep_token_id]
        full_wids = [None] + wids + [None]
        pad_len   = MAX_LEN - len(full_ids)

        self.input_ids      = torch.tensor(
            full_ids + [tokenizer.pad_token_id] * pad_len, dtype=torch.long)
        self.attention_mask = torch.tensor(
            [1] * len(full_ids) + [0] * pad_len, dtype=torch.long)

        self.w2s = {}
        for idx, wid in enumerate(full_wids):
            if wid is not None:
                if wid not in self.w2s: self.w2s[wid] = [idx, idx]
                else: self.w2s[wid][1] = idx

        self.seq_len = (max(wids) + 1) if wids else 0
        self.spans   = [(s, e) for s in range(self.seq_len)
                        for e in range(s, min(s + MAX_SPAN, self.seq_len))]
        span_idx     = {sp: i for i, sp in enumerate(self.spans)}

        gold_a, gold_o = set(), set()
        self.gold_triples = []
        for triple in self.triples_raw:
            a_spans = _spans_from_bio(triple['target_tags'])
            o_spans = _spans_from_bio(triple['opinion_tags'])
            sid     = sentiment2id[triple['sentiment']]
            for a in a_spans:
                if 0 <= a[0] < self.seq_len and 0 <= a[1] < self.seq_len:
                    gold_a.add(a)
            for o in o_spans:
                if 0 <= o[0] < self.seq_len and 0 <= o[1] < self.seq_len:
                    gold_o.add(o)
            for a in a_spans:
                for o in o_spans:
                    if all(0 <= x < self.seq_len for x in [a[0], a[1], o[0], o[1]]):
                        self.gold_triples.append((a[0], a[1], o[0], o[1], sid))

        self.span_labels = torch.zeros(len(self.spans), dtype=torch.long)
        for a in gold_a:
            if a in span_idx: self.span_labels[span_idx[a]] = 1
        for o in gold_o:
            if o in span_idx: self.span_labels[span_idx[o]] = 2

        self.gold_pair_idx = []
        for (a0, a1, o0, o1, sid) in self.gold_triples:
            if (a0, a1) in span_idx and (o0, o1) in span_idx:
                self.gold_pair_idx.append(
                    (span_idx[(a0, a1)], span_idx[(o0, o1)], sid + 1))

def load_dataset(path: str, tokenizer, name: str):
    data = json.load(open(path, encoding='utf-8'))
    return [SpanInstance(tokenizer, s)
            for s in tqdm(data, desc=f'Loading {name}', leave=False)]

class BatchLoader:
    def __init__(self, instances, batch_size: int):
        self.instances  = instances
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.instances) / self.batch_size)

    def get_batches(self, shuffle: bool = False):
        insts = list(self.instances)
        if shuffle: random.shuffle(insts)

        for i in range(0, len(insts), self.batch_size):
            batch   = insts[i: i + self.batch_size]
            max_seq = max((x.seq_len for x in batch), default=0)
            spans   = [(s, e) for s in range(max_seq)
                       for e in range(s, min(s + MAX_SPAN, max_seq))]
            span_idx = {sp: idx for idx, sp in enumerate(spans)}
            B        = len(batch)

            w2s_t = torch.zeros(B, max_seq, 2, dtype=torch.long, device=DEVICE)
            for bi, inst in enumerate(batch):
                for w, (s_, e_) in inst.w2s.items():
                    if w < max_seq:
                        w2s_t[bi, w, 0] = min(s_, MAX_LEN - 1)
                        w2s_t[bi, w, 1] = min(e_, MAX_LEN - 1)

            span_labels = torch.full(
                (B, len(spans)), -1, dtype=torch.long, device=DEVICE)
            for bi, inst in enumerate(batch):
                for si, sp in enumerate(inst.spans):
                    if sp in span_idx:
                        span_labels[bi, span_idx[sp]] = inst.span_labels[si].item()

            gold_pairs_list = []
            for inst in batch:
                pairs = []
                for (ai, oi, sid) in inst.gold_pair_idx:
                    if ai < len(inst.spans) and oi < len(inst.spans):
                        a_sp = inst.spans[ai]; o_sp = inst.spans[oi]
                        if a_sp in span_idx and o_sp in span_idx:
                            pairs.append((span_idx[a_sp], span_idx[o_sp], sid))
                gold_pairs_list.append(pairs)

            spans_t = (torch.tensor(spans, dtype=torch.long, device=DEVICE)
                       .unsqueeze(0).expand(B, -1, -1)
                       if spans else
                       torch.zeros(B, 0, 2, dtype=torch.long, device=DEVICE))

            yield {
                'input_ids':      torch.stack([x.input_ids for x in batch]).to(DEVICE),
                'attention_mask': torch.stack([x.attention_mask for x in batch]).to(DEVICE),
                'lengths':        torch.tensor([x.seq_len for x in batch], device=DEVICE),
                'w2s_t':          w2s_t,
                'spans_t':        spans_t,
                'span_labels':    span_labels,
                'gold_pairs':     gold_pairs_list,
                'instances':      batch,
            }

# ============================================================
# ── Adversarial Training — Fast Gradient Method (AT-FGM) ────
# ============================================================
class AdversarialFGM:
    """
    Fast Gradient Method perturbation on word embedding space.
    Adds a gradient-aligned perturbation before the second forward
    pass so the model learns to be robust near the input manifold.
    """
    def __init__(self, model, epsilon: float = 0.5,
                 target_layer: str = 'word_embeddings'):
        self.model        = model
        self.epsilon      = epsilon
        self.target_layer = target_layer
        self._backup      = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if (param.requires_grad and
                    self.target_layer in name and
                    param.grad is not None):
                self._backup[name] = param.data.clone()
                grad_norm = torch.norm(param.grad)
                if grad_norm > 0:
                    param.data.add_(self.epsilon * param.grad / grad_norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data = self._backup[name]
        self._backup.clear()

# ============================================================
# ── Span representation module ───────────────────────────────
# ============================================================
class SpanRepresentation(nn.Module):
    """
    Represents each candidate span using head/tail token vectors,
    max-pooled interior, and a learnable width embedding.
    """
    def __init__(self, hidden_dim: int, span_dim: int, dropout: float):
        super().__init__()
        self.width_embed = nn.Embedding(MAX_SPAN + 1, 20)
        self.proj        = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 20, span_dim),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, seq, spans_t_batch, w2s_t):
        B, L, H = seq.shape
        spans_t  = spans_t_batch[0]
        n_spans  = spans_t.shape[0]
        max_w    = w2s_t.shape[1]

        s_word = spans_t[:, 0].clamp(max=max_w - 1)
        e_word = spans_t[:, 1].clamp(max=max_w - 1)
        width_emb = self.width_embed(
            (e_word - s_word).clamp(0, MAX_SPAN)
        ).unsqueeze(0).expand(B, -1, -1)

        s_sub = w2s_t[:, s_word, 0].clamp(max=L - 1)
        e_sub = w2s_t[:, e_word, 1].clamp(max=L - 1)
        head  = seq.gather(1, s_sub.unsqueeze(-1).expand(B, n_spans, H))
        tail  = seq.gather(1, e_sub.unsqueeze(-1).expand(B, n_spans, H))

        # Max-pool over span interior
        gathered = []
        for off in range(min(MAX_SPAN * 3, L)):
            p    = (s_sub + off).clamp(max=L - 1)
            mask = (p <= e_sub).float().unsqueeze(-1)
            v    = seq.gather(1, p.unsqueeze(-1).expand(B, n_spans, H))
            gathered.append(v * mask + head * (1 - mask))
        pool = torch.stack(gathered, dim=2).max(dim=2)[0]

        return self.proj(torch.cat([head, tail, pool, width_emb], dim=-1))

# ============================================================
# ── Main model ───────────────────────────────────────────────
# ============================================================
class SpanASTEModel(nn.Module):
    """
    Span-based ASTE model with:
      - Multi-Layer Feature Fusion (MLFF): residual fusion of
        intermediate encoder layers for multi-granularity features.
      - Adversarial Training (AT-FGM): handled externally in the
        training loop; the model itself is standard.
      - Contrastive Learning (CL): in-batch pair-level contrastive
        loss to separate positive/negative relation embeddings.
    """
    def __init__(self, config: dict, use_mlff: bool = True):
        super().__init__()
        self.config   = config
        self.use_mlff = use_mlff

        self.encoder = AutoModel.from_pretrained(
            ENCODER_PATH, ignore_mismatched_sizes=True,
            output_hidden_states=True)
        H = self.encoder.config.hidden_size
        D = 256

        self.dropout         = nn.Dropout(config['dropout'])
        self.span_repr       = SpanRepresentation(H, D, config['dropout'])
        self.span_classifier = nn.Linear(D, 3)   # O / aspect / opinion

        # Innovation 1 — Multi-Layer Feature Fusion
        if self.use_mlff:
            # Project concatenated [layer3 | layer7 | layer11] -> H
            # then add as residual to the last hidden state
            self.mlff_proj = nn.Linear(H * len(MLFF_LAYERS), H)

        self.pair_mlp        = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.Dropout(config['dropout']))
        self.pair_classifier = nn.Linear(D, 4)   # none / neg / neu / pos

        self.cl_temp   = config.get('cl_temp', 0.07)
        self.cl_weight = config.get('cl_weight', 0.05)
        self.float()

    # ── Contrastive loss (in-batch, pair-level) ──────────────
    def _contrastive_loss(self, pair_emb, labels):
        pos = labels > 0
        if pos.sum() < 2:
            return torch.zeros(1, device=labels.device)
        emb     = F.normalize(pair_emb, dim=-1)
        sim     = torch.matmul(emb[pos], emb.T) / self.cl_temp
        targets = pos.nonzero(as_tuple=True)[0]
        return F.cross_entropy(sim, targets).unsqueeze(0)

    # ── Encoder with optional MLFF ───────────────────────────
    def _encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state.float()

        if self.use_mlff and out.hidden_states is not None:
            avail = [i for i in MLFF_LAYERS if i < len(out.hidden_states)]
            if len(avail) == len(MLFF_LAYERS):
                multi = torch.cat(
                    [out.hidden_states[i].float() for i in avail], dim=-1)
                # Residual: last hidden + projected intermediate features
                seq = seq + self.mlff_proj(multi)

        return self.dropout(seq)

    # ── Forward ──────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, lengths, w2s_t, spans_t,
                span_labels=None, gold_pairs=None,
                return_pair_features=False):

        seq = self._encode(input_ids, attention_mask)
        if spans_t.shape[1] == 0:
            dummy = torch.zeros(1, device=input_ids.device, requires_grad=True)
            return dummy, None, None, None

        span_reprs  = self.span_repr(seq, spans_t, w2s_t)
        span_logits = self.span_classifier(span_reprs)
        loss        = torch.zeros(1, device=input_ids.device)

        # Span classification loss
        if span_labels is not None:
            loss = loss + F.cross_entropy(
                span_logits.reshape(-1, 3),
                span_labels.reshape(-1),
                ignore_index=-1).unsqueeze(0)

        saved_pair_features = None

        if gold_pairs is not None:
            B, N, D = span_reprs.shape
            with torch.no_grad():
                span_probs = F.softmax(span_logits, dim=-1)

            batch_logits, batch_labels, batch_ph = [], [], []

            for b in range(B):
                pairs   = gold_pairs[b]
                seq_len = lengths[b].item()
                if seq_len == 0:
                    continue
                K = min(max(int(seq_len * 0.8), 8), N)

                _, top_a = torch.topk(span_probs[b, :, 1], K)
                _, top_o = torch.topk(span_probs[b, :, 2], K)
                gold_a   = torch.tensor(
                    [p[0] for p in pairs], dtype=torch.long, device=DEVICE)
                gold_o   = torch.tensor(
                    [p[1] for p in pairs], dtype=torch.long, device=DEVICE)
                a_cands  = torch.unique(torch.cat([top_a, gold_a]))
                o_cands  = torch.unique(torch.cat([top_o, gold_o]))
                Ka, Ko   = len(a_cands), len(o_cands)

                a_repr = span_reprs[b, a_cands].unsqueeze(1).expand(Ka, Ko, -1)
                o_repr = span_reprs[b, o_cands].unsqueeze(0).expand(Ka, Ko, -1)
                ph     = self.pair_mlp(torch.cat([a_repr, o_repr], dim=-1))
                logits = self.pair_classifier(ph)

                grid = torch.zeros(Ka, Ko, dtype=torch.long, device=DEVICE)
                a_list, o_list = a_cands.tolist(), o_cands.tolist()
                for (ga, go, gl) in pairs:
                    if ga in a_list and go in o_list:
                        grid[a_list.index(ga), o_list.index(go)] = gl

                flat_logits = logits.view(-1, 4)
                flat_labels = grid.view(-1)
                flat_ph     = ph.view(-1, D)

                # Hard-negative mining for training efficiency
                pos_m = flat_labels > 0; neg_m = flat_labels == 0
                n_pos = pos_m.sum().item()
                n_neg = max(int(n_pos * self.config['neg_ratio']),
                            self.config['min_neg'])

                neg_l, neg_y, neg_ph = (flat_logits[neg_m],
                                        flat_labels[neg_m],
                                        flat_ph[neg_m])
                if neg_l.size(0) > n_neg:
                    scores     = F.softmax(neg_l.detach(), dim=-1)[:, 1:].sum(-1)
                    _, hard_ix = torch.topk(scores, n_neg)
                    neg_l, neg_y, neg_ph = neg_l[hard_ix], neg_y[hard_ix], neg_ph[hard_ix]

                batch_logits.append(torch.cat([flat_logits[pos_m], neg_l]))
                batch_labels.append(torch.cat([flat_labels[pos_m], neg_y]))
                batch_ph.append(torch.cat([flat_ph[pos_m], neg_ph]))

            if batch_logits:
                all_logits = torch.cat(batch_logits)
                all_labels = torch.cat(batch_labels)
                all_ph     = torch.cat(batch_ph)

                # Relation classification loss
                loss = loss + F.cross_entropy(all_logits, all_labels).unsqueeze(0)

                # Contrastive loss
                if self.cl_weight > 0:
                    loss = loss + self.cl_weight * self._contrastive_loss(
                        all_ph, all_labels)

                if return_pair_features:
                    saved_pair_features = {"ph": all_ph.detach().cpu(), "labels": all_labels.detach().cpu()}

        return loss, span_logits, span_reprs, saved_pair_features

    # ── Inference ────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, input_ids, attention_mask, lengths, w2s_t, spans_t,
                spans_list):
        self.eval()
        if spans_t.shape[1] == 0:
            self.train()
            return [[] for _ in range(input_ids.shape[0])]

        _, span_logits, span_reprs, _ = self.forward(
            input_ids, attention_mask, lengths, w2s_t, spans_t)
        if span_logits is None:
            self.train()
            return [[] for _ in range(input_ids.shape[0])]

        span_probs = F.softmax(span_logits, dim=-1)
        span_preds = span_probs.argmax(dim=-1)
        B = input_ids.shape[0]
        results = []

        for b in range(B):
            def _nms(indices, cls):
                indices = sorted(indices,
                    key=lambda i: span_probs[b, i, cls].item(), reverse=True)
                kept = []
                for idx in indices:
                    s, e = spans_list[idx]
                    if not any(max(s, spans_list[k][0]) <= min(e, spans_list[k][1])
                               for k in kept):
                        kept.append(idx)
                return kept

            a_cands = _nms(
                [i for i in range(len(spans_list)) if span_preds[b, i] == 1], 1)
            o_cands = _nms(
                [i for i in range(len(spans_list)) if span_preds[b, i] == 2], 2)

            triplets = []
            if a_cands and o_cands:
                pairs_emb = [torch.cat([span_reprs[b, ai], span_reprs[b, oi]])
                             for ai in a_cands for oi in o_cands]
                pairs_idx = [(ai, oi) for ai in a_cands for oi in o_cands]
                ph   = self.pair_mlp(torch.stack(pairs_emb))
                pred = self.pair_classifier(ph).argmax(dim=-1)
                for (ai, oi), p in zip(pairs_idx, pred.tolist()):
                    if p > 0:
                        triplets.append({
                            'aspect':    spans_list[ai],
                            'opinion':   spans_list[oi],
                            'sentiment': p - 1,
                        })
            results.append(triplets)

        self.train()
        return results

# ============================================================
# ── Evaluation ───────────────────────────────────────────────
# ============================================================
def evaluate(model, loader: BatchLoader):
    model.eval()
    gold_set, pred_set = set(), set()
    sidx = 0
    all_preds, error_cases = [], []

    for batch in loader.get_batches(shuffle=False):
        spans_list = [(s[0].item(), s[1].item()) for s in batch['spans_t'][0]]
        preds = model.predict(
            batch['input_ids'], batch['attention_mask'],
            batch['lengths'], batch['w2s_t'], batch['spans_t'], spans_list)

        for inst, pred_trip in zip(batch['instances'], preds):
            gold_f = [{'A': ' '.join(inst.tokens[a0:a1+1]),
                       'O': ' '.join(inst.tokens[o0:o1+1]),
                       'S': id2sentiment[sid]}
                      for (a0, a1, o0, o1, sid) in inst.gold_triples]
            pred_f = [{'A': ' '.join(inst.tokens[t['aspect'][0]:t['aspect'][1]+1]),
                       'O': ' '.join(inst.tokens[t['opinion'][0]:t['opinion'][1]+1]),
                       'S': id2sentiment[t['sentiment']]}
                      for t in pred_trip]
            sg = {f"{t['A']}|{t['O']}|{t['S']}" for t in gold_f}
            sp = {f"{t['A']}|{t['O']}|{t['S']}" for t in pred_f}
            all_preds.append({'id': sidx, 'sentence': inst.sentence,
                              'gold': gold_f, 'prediction': pred_f})
            tp = list(sg & sp); fp = list(sp - sg); fn = list(sg - sp)
            if fp or fn:
                error_cases.append({'sentence': inst.sentence,
                                    'TP': tp, 'FP': fp, 'FN': fn})
            for (a0, a1, o0, o1, sid) in inst.gold_triples:
                gold_set.add(f'{sidx}-{a0}-{a1}-{o0}-{o1}-{sid}')
            for t in pred_trip:
                pred_set.add(f'{sidx}-{t["aspect"][0]}-{t["aspect"][1]}'
                             f'-{t["opinion"][0]}-{t["opinion"][1]}-{t["sentiment"]}')
            sidx += 1

    c  = len(gold_set & pred_set)
    p  = c / len(pred_set) if pred_set else 0.
    r  = c / len(gold_set) if gold_set else 0.
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.
    model.train()
    return p * 100, r * 100, f1 * 100, all_preds, error_cases

# ============================================================
# ── Resource stats ───────────────────────────────────────────
# ============================================================
def collect_resource_stats(model) -> dict:
    stats = {}
    if torch.cuda.is_available():
        stats['gpu_name']             = torch.cuda.get_device_name(0)
        stats['gpu_mem_allocated_MB'] = round(
            torch.cuda.memory_allocated() / 1024 ** 2, 1)
        stats['gpu_mem_reserved_MB']  = round(
            torch.cuda.memory_reserved() / 1024 ** 2, 1)
    stats['cpu_ram_MB']         = round(
        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, 1)
    stats['total_params_M']     = round(
        sum(p.numel() for p in model.parameters()) / 1e6, 2)
    stats['trainable_params_M'] = round(
        sum(p.numel() for p in model.parameters()
            if p.requires_grad) / 1e6, 2)
    return stats

# ============================================================
# ── Atomic CSV upsert (overwrite one row, leave others alone) ─
# ============================================================
def upsert_csv(path: str, key_cols: list, new_row: dict, fieldnames: list):
    """
    Read existing CSV, replace the row matching key_cols with new_row,
    or append if not found.  All other rows are preserved exactly.
    """
    rows = []
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if {k: row.get(k, '') for k in key_cols} != \
                   {k: str(new_row.get(k, '')) for k in key_cols}:
                    rows.append(row)
    rows.append({k: str(v) for k, v in new_row.items()})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

def _check_csv(path: str, key_dict: dict):
    """Return matching row dict if it exists, else None."""
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if all(row.get(k, '') == str(v) for k, v in key_dict.items()):
                return row
    return None

# ============================================================
# ── Core training loop ───────────────────────────────────────
# ============================================================
def train_one_run(
    dataset: str,
    config: dict,
    use_mlff: bool,
    use_at: bool,
    tr_insts, dv_insts, ts_insts,
    seed: int,
    exp_tag: str,
    save_dir: str,
    logger,
    save_model: bool = False,
    save_features: bool = False,
    override_cl_weight: float = None,
):
    """
    Train one full run and return (dev_f1, p, r, f1, curve).
    Artifacts are saved to save_dir with filenames prefixed by exp_tag.
    Each artifact is written independently; failure on one does not
    affect others.
    """
    set_seed(seed)
    t0 = time.time()

    cfg = dict(config)
    if override_cl_weight is not None:
        cfg['cl_weight'] = override_cl_weight

    train_loader = BatchLoader(tr_insts, cfg['batch_size'])
    dev_loader   = BatchLoader(dv_insts, cfg['batch_size'])
    test_loader  = BatchLoader(ts_insts, cfg['batch_size'])

    model = SpanASTEModel(cfg, use_mlff=use_mlff).to(DEVICE).float()
    init_stats = collect_resource_stats(model)

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'encoder.' in n],
         'lr': cfg['encoder_lr'], 'weight_decay': cfg['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if 'encoder.' not in n],
         'lr': cfg['head_lr'],    'weight_decay': cfg['weight_decay']},
    ])
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * cfg['warmup_ratio']),
        total_steps)

    # Innovation 2 — AT-FGM (instantiated only when active)
    fgm = AdversarialFGM(model) if use_at else None

    best_dev_f1 = 0.; best_test = (0., 0., 0.)
    best_preds = None; best_errors = None
    best_ph = None; best_span_reprs = None
    patience = 0; curve = []

    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); ep_loss = 0.
        batches = list(train_loader.get_batches(shuffle=True))

        for batch in batches:
            loss, _, _, _ = model(
                batch['input_ids'], batch['attention_mask'],
                batch['lengths'], batch['w2s_t'],
                batch['spans_t'], batch['span_labels'], batch['gold_pairs'])
            loss = loss.mean()
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()

            # AT: perturb embeddings, compute adversarial loss, accumulate grad
            if fgm is not None:
                fgm.attack()
                adv_loss, _, _, _ = model(
                    batch['input_ids'], batch['attention_mask'],
                    batch['lengths'], batch['w2s_t'],
                    batch['spans_t'], batch['span_labels'], batch['gold_pairs'])
                adv_loss = adv_loss.mean()
                if not torch.isnan(adv_loss):
                    adv_loss.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / max(len(batches), 1)
        _, _, dev_f1, _, _ = evaluate(model, dev_loader)
        curve.append({'epoch': ep,
                      'train_loss': round(avg_loss, 4),
                      'dev_f1':     round(dev_f1,  2)})
        logger.info(f'  [{exp_tag}] ep{ep:02d}  loss={avg_loss:.4f}'
                    f'  dev_f1={dev_f1:.2f}')

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1; patience = 0
            t_p, t_r, t_f1, preds, errors = evaluate(model, test_loader)
            best_test   = (t_p, t_r, t_f1)
            best_preds  = preds
            best_errors = errors
            logger.info(f'  => best  P={t_p:.2f}  R={t_r:.2f}  F1={t_f1:.2f}')

            if save_model:
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f'{exp_tag}_model.pt'))

            if save_features:
                model.eval()
                ph_list, sr_list = [], []
                with torch.no_grad():
                    for batch in test_loader.get_batches():
                        _, _, sr, ph = model(
                            batch['input_ids'], batch['attention_mask'],
                            batch['lengths'], batch['w2s_t'],
                            batch['spans_t'], batch['span_labels'],
                            batch['gold_pairs'], return_pair_features=True)
                        if sr is not None:
                            sr_list.append(sr.detach().cpu().numpy()
                                           .reshape(-1, sr.shape[-1]))
                        if ph is not None:
                            ph_list.append(ph.numpy())
                model.train()
                if ph_list:
                    best_ph = np.concatenate(ph_list, 0)
                if sr_list:
                    best_span_reprs = np.concatenate(sr_list, 0)
        else:
            patience += 1
            if patience >= PATIENCE:
                logger.info(f'  => early stop at ep={ep}')
                break

    elapsed = time.time() - t0
    final_stats = collect_resource_stats(model)

    # ── Save all artifacts ────────────────────────────────────
    def _save(fname, obj, mode='json'):
        path = os.path.join(save_dir, fname)
        if mode == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        elif mode == 'csv':
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=obj[0].keys())
                w.writeheader(); w.writerows(obj)
        elif mode == 'npz':
            np.savez(path, **obj)

    if best_preds:
        _save(f'{exp_tag}_preds.json',  best_preds[:2000], 'json')
    if best_errors:
        _save(f'{exp_tag}_errors.json', best_errors[:100],  'json')
    if save_features and best_ph is not None:
        _save(f'{exp_tag}_tsne',        {'pair_hidden': best_ph}, 'npz')
    if save_features and best_span_reprs is not None:
        _save(f'{exp_tag}_span_reprs',  {'span_reprs': best_span_reprs}, 'npz')
    if curve:
        _save(f'{exp_tag}_training_curve.csv', curve, 'csv')

    _save(f'{exp_tag}_runtime.json', {
        'exp_tag':        exp_tag,
        'dataset':        dataset,
        'seed':           seed,
        'use_mlff':       use_mlff,
        'use_at':         use_at,
        'time_minutes':   round(elapsed / 60, 2),
        'best_dev_f1':    round(best_dev_f1, 2),
        'test_p':         round(best_test[0], 2),
        'test_r':         round(best_test[1], 2),
        'test_f1':        round(best_test[2], 2),
        'init_stats':     init_stats,
        'final_stats':    final_stats,
    }, 'json')

    del model, optimizer, scheduler
    torch.cuda.empty_cache(); gc.collect()

    return best_dev_f1, best_test[0], best_test[1], best_test[2], curve

# ============================================================
# ── Main experiment (5 seeds) ────────────────────────────────
# ============================================================
MAIN_CSV_FIELDS = ['seed','dev_f1','test_p','test_r','test_f1','time_min']

def run_main(dataset, tr, dv, ts, seeds, logger):
    sdir     = paper_dir(dataset)
    cfg      = DATASET_CONFIGS[dataset]
    csv_path = os.path.join(sdir, 'main_results.csv')

    logger.info(f'\n{"="*60}')
    logger.info(f'Main experiment  [{dataset}]  seeds={seeds}')
    logger.info(f'{"="*60}')

    all_f1 = []
    for seed in seeds:
        tag = f'seed{seed}_full'
        existing = _check_csv(csv_path, {'seed': seed})
        if existing:
            logger.info(f'  [skip] seed={seed} already done, F1={existing["test_f1"]}')
            all_f1.append(float(existing['test_f1']))
            continue

        t0 = time.time()
        dev_f1, p, r, f1, _ = train_one_run(
            dataset, cfg,
            use_mlff=True, use_at=True,
            tr_insts=tr, dv_insts=dv, ts_insts=ts,
            seed=seed, exp_tag=tag,
            save_dir=sdir, logger=logger,
            save_model=True,
            save_features=(seed == seeds[0]))

        upsert_csv(csv_path, ['seed'], {
            'seed': seed,
            'dev_f1':   f'{dev_f1:.2f}',
            'test_p':   f'{p:.2f}',
            'test_r':   f'{r:.2f}',
            'test_f1':  f'{f1:.2f}',
            'time_min': f'{(time.time()-t0)/60:.1f}',
        }, MAIN_CSV_FIELDS)
        all_f1.append(f1)

    if all_f1:
        m, s = np.mean(all_f1), np.std(all_f1)
        summary = {'dataset': dataset, 'seeds': seeds,
                   'mean_f1': round(m, 2), 'std_f1': round(s, 2),
                   'per_seed': [round(x, 2) for x in all_f1]}
        with open(os.path.join(sdir, 'main_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f'  [{dataset}] Main result: {m:.2f} ± {s:.2f}')

# ============================================================
# ── Ablation study (3 seeds) ─────────────────────────────────
# ============================================================
ABLATION_CSV_FIELDS = [
    'variant', 'description', 'seed',
    'dev_f1', 'test_p', 'test_r', 'test_f1', 'time_min']

def run_ablation(dataset, tr, dv, ts, seeds, logger, only_variant=None):
    sdir     = paper_dir(dataset)
    cfg      = DATASET_CONFIGS[dataset]
    csv_path = os.path.join(sdir, 'ablation_results.csv')

    variants = ({only_variant: ABLATION_VARIANTS[only_variant]}
                if only_variant else ABLATION_VARIANTS)

    logger.info(f'\n{"="*60}')
    logger.info(f'Ablation  [{dataset}]  variants={list(variants.keys())}')
    logger.info(f'{"="*60}')

    for vkey, (vdesc, use_mlff, use_at) in variants.items():
        for seed in seeds:
            tag = f'seed{seed}_ablation_{vkey}'
            existing = _check_csv(csv_path, {'variant': vkey, 'seed': str(seed)})
            if existing:
                logger.info(f'  [skip] {vkey} seed={seed} F1={existing["test_f1"]}')
                continue

            t0 = time.time()
            dev_f1, p, r, f1, _ = train_one_run(
                dataset, cfg,
                use_mlff=use_mlff, use_at=use_at,
                tr_insts=tr, dv_insts=dv, ts_insts=ts,
                seed=seed, exp_tag=tag,
                save_dir=sdir, logger=logger,
                save_model=False, save_features=False)

            upsert_csv(csv_path, ['variant', 'seed'], {
                'variant':     vkey,
                'description': vdesc,
                'seed':        seed,
                'dev_f1':      f'{dev_f1:.2f}',
                'test_p':      f'{p:.2f}',
                'test_r':      f'{r:.2f}',
                'test_f1':     f'{f1:.2f}',
                'time_min':    f'{(time.time()-t0)/60:.1f}',
            }, ABLATION_CSV_FIELDS)
            logger.info(f'  [{dataset}|{vkey}|seed={seed}] F1={f1:.2f}')

    _summarize_ablation(csv_path, sdir, dataset, logger)

def _summarize_ablation(csv_path, sdir, dataset, logger):
    if not os.path.exists(csv_path):
        return
    from collections import defaultdict
    rows = defaultdict(list)
    descs = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows[row['variant']].append(float(row['test_f1']))
            descs[row['variant']] = row['description']
    summary = {vk: {'description': descs[vk],
                     'mean': round(np.mean(fs), 2),
                     'std':  round(np.std(fs), 2)}
               for vk, fs in rows.items()}
    with open(os.path.join(sdir, 'ablation_summary.json'), 'w') as f:
        json.dump({'dataset': dataset, 'ablation': summary}, f, indent=2)
    logger.info(f'  Ablation summary [{dataset}]:')
    for vk, stat in summary.items():
        logger.info(f'    {vk:<18} {stat["mean"]:.2f} ± {stat["std"]:.2f}'
                    f'  ({stat["description"]})')

# ============================================================
# ── Sensitivity analysis (1 seed) ────────────────────────────
# ============================================================
SENS_CSV_FIELDS = ['param_value', 'dev_f1', 'test_p', 'test_r', 'test_f1']

def run_sensitivity(dataset, tr, dv, ts, logger, only_param=None):
    sdir = paper_dir(dataset)
    cfg  = DATASET_CONFIGS[dataset]

    do_temp  = only_param in (None, 'cl_temp')
    do_ratio = only_param in (None, 'neg_ratio')

    if do_temp:
        csv_path = os.path.join(sdir, 'sensitivity_cl_temp.csv')
        logger.info(f'\n[{dataset}] Sensitivity: cl_temp = {SENS_CL_TEMP}')
        for val in SENS_CL_TEMP:
            existing = _check_csv(csv_path, {'param_value': str(val)})
            if existing:
                logger.info(f'  [skip] cl_temp={val} F1={existing["test_f1"]}')
                continue
            cfg_v = dict(cfg); cfg_v['cl_temp'] = val
            dev_f1, p, r, f1, _ = train_one_run(
                dataset, cfg_v, use_mlff=True, use_at=True,
                tr_insts=tr, dv_insts=dv, ts_insts=ts,
                seed=SENS_SEED, exp_tag=f'sens_cl_temp_{val}',
                save_dir=sdir, logger=logger)
            upsert_csv(csv_path, ['param_value'], {
                'param_value': val, 'dev_f1': f'{dev_f1:.2f}',
                'test_p': f'{p:.2f}', 'test_r': f'{r:.2f}',
                'test_f1': f'{f1:.2f}'}, SENS_CSV_FIELDS)
            logger.info(f'  cl_temp={val}  F1={f1:.2f}')

    if do_ratio:
        csv_path = os.path.join(sdir, 'sensitivity_neg_ratio.csv')
        logger.info(f'\n[{dataset}] Sensitivity: neg_ratio = {SENS_NEG_RATIO}')
        for val in SENS_NEG_RATIO:
            existing = _check_csv(csv_path, {'param_value': str(val)})
            if existing:
                logger.info(f'  [skip] neg_ratio={val} F1={existing["test_f1"]}')
                continue
            cfg_v = dict(cfg); cfg_v['neg_ratio'] = val
            dev_f1, p, r, f1, _ = train_one_run(
                dataset, cfg_v, use_mlff=True, use_at=True,
                tr_insts=tr, dv_insts=dv, ts_insts=ts,
                seed=SENS_SEED, exp_tag=f'sens_neg_ratio_{val}',
                save_dir=sdir, logger=logger)
            upsert_csv(csv_path, ['param_value'], {
                'param_value': val, 'dev_f1': f'{dev_f1:.2f}',
                'test_p': f'{p:.2f}', 'test_r': f'{r:.2f}',
                'test_f1': f'{f1:.2f}'}, SENS_CSV_FIELDS)
            logger.info(f'  neg_ratio={val}  F1={f1:.2f}')

# ============================================================
# ── Entry point ──────────────────────────────────────────────
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ASTE Paper Experiment Framework',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--datasets', nargs='+',
                        choices=['res14', 'lap14', 'res15', 'res16'],
                        default=['res14', 'lap14', 'res15', 'res16'])
    parser.add_argument('--mode',
                        choices=['main', 'ablation', 'sensitivity', 'single'],
                        default='main',
                        help=(
                            'main        — 5-seed main experiment\n'
                            'ablation    — 3-seed ablation study\n'
                            'sensitivity — 1-seed sensitivity analysis\n'
                            'single      — re-run one specific item (overwrite)'))
    parser.add_argument('--variant', type=str, default=None,
                        help=('For --mode single:\n'
                              '  ablation  → variant key, e.g. w_o_AT\n'
                              '  main      → seed number, e.g. 42\n'
                              '  sensitivity → param name, e.g. cl_temp'))
    parser.add_argument('--single_type',
                        choices=['main', 'ablation', 'sensitivity'],
                        default='ablation',
                        help='Which experiment type to re-run in single mode.')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Override default seed list.')
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────
    log_path = os.path.join(LOG_ROOT, f'{args.mode}_{TIMESTAMP}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ])
    logger = logging.getLogger(__name__)

    logger.info('=' * 65)
    logger.info('ASTE Paper Experiment Framework')
    logger.info(f'Model : Span-ASTE + MLFF + AT-FGM')
    logger.info(f'Mode  : {args.mode}')
    logger.info(f'Device: {DEVICE}')
    logger.info(f'Output: {os.path.abspath(PAPER_ROOT)}')
    logger.info('=' * 65)
    logger.info('Saved artifacts per dataset:')
    logger.info('  main_results.csv / main_summary.json')
    logger.info('  ablation_results.csv / ablation_summary.json')
    logger.info('  sensitivity_cl_temp.csv / sensitivity_neg_ratio.csv')
    logger.info('  seed*_full_model.pt   — model checkpoint')
    logger.info('  seed*_full_tsne.npz   — pair features for t-SNE')
    logger.info('  seed*_full_errors.json — error cases')
    logger.info('  seed*_full_training_curve.csv — convergence data')
    logger.info('  seed*_full_runtime.json — time / GPU memory')
    logger.info('=' * 65 + '\n')

    tok = AutoTokenizer.from_pretrained(ENCODER_PATH, use_fast=False)

    for ds in args.datasets:
        logger.info(f'\n{"*" * 65}')
        logger.info(f'Dataset: {ds}')
        logger.info(f'{"*" * 65}')

        tr = load_dataset(f'{DATA_ROOT}{ds}/train.json', tok, f'{ds}/train')
        dv = load_dataset(f'{DATA_ROOT}{ds}/dev.json',   tok, f'{ds}/dev')
        ts = load_dataset(f'{DATA_ROOT}{ds}/test.json',  tok, f'{ds}/test')

        if args.mode == 'main':
            run_main(ds, tr, dv, ts, args.seeds or MAIN_SEEDS, logger)

        elif args.mode == 'ablation':
            run_ablation(ds, tr, dv, ts, args.seeds or ABLATION_SEEDS, logger)

        elif args.mode == 'sensitivity':
            run_sensitivity(ds, tr, dv, ts, logger)

        elif args.mode == 'single':
            if args.variant is None:
                logger.error('--variant is required for --mode single')
                sys.exit(1)
            if args.single_type == 'ablation':
                if args.variant not in ABLATION_VARIANTS:
                    logger.error(
                        f'Unknown ablation variant: {args.variant}\n'
                        f'Choose from: {list(ABLATION_VARIANTS.keys())}')
                    sys.exit(1)
                run_ablation(ds, tr, dv, ts,
                             args.seeds or ABLATION_SEEDS, logger,
                             only_variant=args.variant)
            elif args.single_type == 'main':
                seed = int(args.variant)
                sdir = paper_dir(ds)
                cfg  = DATASET_CONFIGS[ds]
                t0   = time.time()
                dev_f1, p, r, f1, _ = train_one_run(
                    ds, cfg, use_mlff=True, use_at=True,
                    tr_insts=tr, dv_insts=dv, ts_insts=ts,
                    seed=seed, exp_tag=f'seed{seed}_full',
                    save_dir=sdir, logger=logger,
                    save_model=True, save_features=True)
                upsert_csv(
                    os.path.join(sdir, 'main_results.csv'),
                    ['seed'], {
                        'seed': seed, 'dev_f1': f'{dev_f1:.2f}',
                        'test_p': f'{p:.2f}', 'test_r': f'{r:.2f}',
                        'test_f1': f'{f1:.2f}',
                        'time_min': f'{(time.time()-t0)/60:.1f}',
                    }, MAIN_CSV_FIELDS)
                logger.info(f'[{ds}|seed={seed}] done  F1={f1:.2f}')
            elif args.single_type == 'sensitivity':
                run_sensitivity(ds, tr, dv, ts, logger,
                                only_param=args.variant)

    logger.info(f'\nAll done.  Results: {os.path.abspath(PAPER_ROOT)}')
    logger.info(f'Log: {log_path}')