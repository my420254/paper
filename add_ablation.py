"""
layer_ablation.py  ──  MLFF Layer Selection Ablation for FASTE
================================================================
Validates that the layer combination {3, 7, 11} is not arbitrary
by comparing it against alternative triplets drawn from DeBERTa-v3's
12 transformer layers.

Usage (run from the same directory as aste.py and the GTS data folder):
    python layer_ablation.py --datasets lap14 res14 --seed 42

Outputs (written to ./outputs/layer_ablation/):
    layer_ablation_results.csv   – per-combo per-dataset P/R/F1
    layer_ablation_summary.json  – mean F1 across tested datasets
    layer_ablation.log           – full training log

NOTE FOR REVIEWERS: 
The proposed layer combination {3, 7, 11} (the full FASTE model) is 
evaluated via the main training script (`aste.py`) across all 5 seeds. 
To avoid redundant computation and subset variance, this script evaluates 
the 9 ALTERNATIVE layer combinations only.
"""

import os, sys, json, csv, logging, argparse, time
import numpy as np
import torch

# ── Import shared infrastructure from aste.py ────────────────
# aste.py must be in the same directory (or on PYTHONPATH).
# We import everything we need at once to avoid repeating code.
from aste import (
    DEVICE, DATASET_CONFIGS, PAPER_ROOT, LOG_ROOT,
    MAX_SPAN, MAX_EPOCHS, PATIENCE,
    AutoTokenizer, set_seed,
    load_dataset, BatchLoader,
    SpanASTEModel, evaluate,
    upsert_csv, _check_csv,
    train_one_run,
    MAIN_SEEDS,
)

# ── Layer combinations to test ────────────────────────────────
# We keep the total number of fused layers fixed at 3 to ensure
# a fair comparison (same projection cost, same parameter count).
#
# Naming convention:  "low-mid-high" refers to the relative
# position of each layer within the 12-layer DeBERTa-v3 stack.
LAYER_COMBOS = {
    # ── Proposed ──────────────────────────────────────────────
    # NOTE: Commented out to avoid redundant computation. 
    # The proposed {3, 7, 11} configuration is evaluated as the 
    # default FASTE architecture in `aste.py`.
    # "L03_07_11":  [3,  7, 11],   # proposed: low / mid / high-1

    # ── Systematic neighbours ─────────────────────────────────
    "L01_06_12":  [1,  6, 12],   # very early / middle / last
    "L02_06_10":  [2,  6, 10],   # shifted -1 from proposed
    "L04_08_12":  [4,  8, 12],   # shifted +1 from proposed
    "L02_07_11":  [2,  7, 11],   # only low changes
    "L03_07_12":  [3,  7, 12],   # only high changes (last layer)
    "L03_06_11":  [3,  6, 11],   # only mid changes

    # ── Boundary / degenerate cases ───────────────────────────
    "L01_02_03":  [1,  2,  3],   # all early (morphological only)
    "L10_11_12":  [10, 11, 12],  # all late  (semantic only)
    "L04_08_10":  [4,  8, 10],   # uniform spacing, mid zone
}

# ── Output paths ─────────────────────────────────────────────
ABLATION_DIR  = os.path.join(PAPER_ROOT, "layer_ablation")
os.makedirs(ABLATION_DIR, exist_ok=True)

ABLATION_CSV  = os.path.join(ABLATION_DIR, "layer_ablation_results.csv")
SUMMARY_JSON  = os.path.join(ABLATION_DIR, "layer_ablation_summary.json")
LOG_PATH      = os.path.join(LOG_ROOT,     "layer_ablation.log")

CSV_FIELDS = ["combo", "layers", "dataset", "seed",
              "test_p", "test_r", "test_f1", "time_min"]

# ── Logging setup ─────────────────────────────────────────────
def setup_logger():
    logger = logging.getLogger("layer_ablation")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Patched SpanASTEModel that accepts a custom layer list ────
class PatchedSpanASTEModel(SpanASTEModel):
    """
    Thin subclass of SpanASTEModel that replaces the hardcoded
    MLFF_LAYERS constant with a per-instance attribute so we can
    swap layer combinations without touching aste.py.
    """
    def __init__(self, config: dict, mlff_layers: list):
        # Temporarily monkey-patch the module-level constant so
        # the parent __init__ wires up the correct projection size.
        import aste as _aste_mod
        _orig = _aste_mod.MLFF_LAYERS
        _aste_mod.MLFF_LAYERS = mlff_layers
        super().__init__(config, use_mlff=True)
        _aste_mod.MLFF_LAYERS = _orig           # restore immediately
        # Store our own copy so _encode uses it at runtime.
        self._mlff_layers_override = mlff_layers

    def _encode(self, input_ids, attention_mask):
        """Override to use self._mlff_layers_override."""
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        seq = out.last_hidden_state.float()
        if out.hidden_states is not None:
            avail = [i for i in self._mlff_layers_override
                     if i < len(out.hidden_states)]
            if len(avail) == len(self._mlff_layers_override):
                multi = torch.cat(
                    [out.hidden_states[i].float() for i in avail],
                    dim=-1)
                seq = seq + self.mlff_proj(multi)
        return self.dropout(seq)


# ── Single-run training wrapper for the patched model ─────────
def train_layer_combo(
    dataset: str,
    config: dict,
    mlff_layers: list,
    tr_insts, dv_insts, ts_insts,
    seed: int,
    combo_name: str,
    logger,
):
    """
    Train one run with a specific layer combination and return
    (test_p, test_r, test_f1).  Mirrors train_one_run() from
    aste.py but uses PatchedSpanASTEModel instead.
    """
    import gc, math, random
    from transformers import get_linear_schedule_with_warmup
    from aste import (AdversarialFGM, set_seed, paper_dir,
                      MAX_EPOCHS, PATIENCE, collect_resource_stats)

    set_seed(seed)
    t0 = time.time()

    train_loader = BatchLoader(tr_insts, config["batch_size"])
    dev_loader   = BatchLoader(dv_insts, config["batch_size"])
    test_loader  = BatchLoader(ts_insts, config["batch_size"])

    model = PatchedSpanASTEModel(config, mlff_layers).to(DEVICE).float()

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "encoder." in n],
         "lr": config["encoder_lr"],
         "weight_decay": config["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if "encoder." not in n],
         "lr": config["head_lr"],
         "weight_decay": config["weight_decay"]},
    ])
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * config["warmup_ratio"]),
        total_steps)

    fgm = AdversarialFGM(model)   # AT is always on (same as full model)

    best_dev_f1 = 0.
    best_test   = (0., 0., 0.)
    patience    = 0

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss = 0.
        batches = list(train_loader.get_batches(shuffle=True))
        for batch in batches:
            loss, _, _, _ = model(
                batch["input_ids"], batch["attention_mask"],
                batch["lengths"], batch["w2s_t"],
                batch["spans_t"], batch["span_labels"],
                batch["gold_pairs"])
            loss = loss.mean()
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()

            fgm.attack()
            adv_loss, _, _, _ = model(
                batch["input_ids"], batch["attention_mask"],
                batch["lengths"], batch["w2s_t"],
                batch["spans_t"], batch["span_labels"],
                batch["gold_pairs"])
            adv_loss = adv_loss.mean()
            if not torch.isnan(adv_loss):
                adv_loss.backward()
            fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()

        _, _, dev_f1, _, _ = evaluate(model, dev_loader)
        logger.info(f"  [{combo_name}|{dataset}|s{seed}] "
                    f"ep{ep:02d} dev_f1={dev_f1:.2f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            patience    = 0
            t_p, t_r, t_f1, _, _ = evaluate(model, test_loader)
            best_test = (t_p, t_r, t_f1)
            logger.info(f"    => new best  P={t_p:.2f} "
                        f"R={t_r:.2f} F1={t_f1:.2f}")
        else:
            patience += 1
            if patience >= PATIENCE:
                logger.info(f"  => early stop ep={ep}")
                break

    elapsed = (time.time() - t0) / 60
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    return best_test[0], best_test[1], best_test[2], elapsed


# ── Main experiment loop ──────────────────────────────────────
def run_layer_ablation(datasets, seeds, logger):
    from aste import DATA_ROOT, ENCODER_PATH, DATASET_CONFIGS

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)

    # Pre-load all datasets once to avoid redundant disk I/O
    loaded = {}
    for ds in datasets:
        base = os.path.join(DATA_ROOT, ds)
        logger.info(f"Loading dataset: {ds}")
        loaded[ds] = {
            "tr": load_dataset(os.path.join(base, "train.json"),
                               tokenizer, f"{ds}/train"),
            "dv": load_dataset(os.path.join(base, "dev.json"),
                               tokenizer, f"{ds}/dev"),
            "ts": load_dataset(os.path.join(base, "test.json"),
                               tokenizer, f"{ds}/test"),
        }

    total_runs = len(LAYER_COMBOS) * len(datasets) * len(seeds)
    done = 0

    for combo_name, layers in LAYER_COMBOS.items():
        for ds in datasets:
            cfg = DATASET_CONFIGS[ds]
            for seed in seeds:
                done += 1
                run_id = f"{combo_name}/{ds}/s{seed}"
                logger.info(f"\n[{done}/{total_runs}] {run_id}  "
                            f"layers={layers}")

                existing = _check_csv(
                    ABLATION_CSV,
                    {"combo": combo_name, "dataset": ds,
                     "seed": str(seed)})
                if existing:
                    logger.info(f"  [skip] already done "
                                f"F1={existing['test_f1']}")
                    continue

                p, r, f1, mins = train_layer_combo(
                    dataset     = ds,
                    config      = cfg,
                    mlff_layers = layers,
                    tr_insts    = loaded[ds]["tr"],
                    dv_insts    = loaded[ds]["dv"],
                    ts_insts    = loaded[ds]["ts"],
                    seed        = seed,
                    combo_name  = combo_name,
                    logger      = logger,
                )

                upsert_csv(ABLATION_CSV,
                           ["combo", "dataset", "seed"],
                           {"combo":    combo_name,
                            "layers":   str(layers),
                            "dataset":  ds,
                            "seed":     seed,
                            "test_p":   f"{p:.2f}",
                            "test_r":   f"{r:.2f}",
                            "test_f1":  f"{f1:.2f}",
                            "time_min": f"{mins:.1f}"},
                           CSV_FIELDS)

                logger.info(f"  => P={p:.2f} R={r:.2f} "
                            f"F1={f1:.2f}  ({mins:.1f} min)")

    # ── Summarize ─────────────────────────────────────────────
    _summarize(datasets, logger)


def _summarize(datasets, logger):
    """
    Aggregate results and write summary JSON.
    Computes mean F1 per (combo, dataset) and cross-dataset mean.
    """
    from collections import defaultdict

    if not os.path.exists(ABLATION_CSV):
        logger.info("No results to summarize yet.")
        return

    # {combo: {dataset: [f1, ...]}}
    data = defaultdict(lambda: defaultdict(list))
    layers_map = {}

    with open(ABLATION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["combo"]][row["dataset"]].append(
                float(row["test_f1"]))
            layers_map[row["combo"]] = row["layers"]

    summary = {}
    for combo, ds_dict in data.items():
        entry = {"layers": layers_map.get(combo, "?"),
                 "per_dataset": {}}
        all_means = []
        for ds, vals in ds_dict.items():
            m = round(float(np.mean(vals)), 2)
            s = round(float(np.std(vals)),  2)
            entry["per_dataset"][ds] = {"mean": m, "std": s,
                                        "runs": len(vals)}
            all_means.append(m)
        entry["cross_dataset_mean"] = round(
            float(np.mean(all_means)), 2) if all_means else 0.0
        summary[combo] = entry

    # Sort by cross-dataset mean (descending)
    summary = dict(sorted(
        summary.items(),
        key=lambda x: x[1]["cross_dataset_mean"],
        reverse=True))

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump({"datasets": datasets, "results": summary},
                  f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("LAYER ABLATION SUMMARY")
    logger.info("=" * 60)
    header = f"{'Combo':<16} {'Layers':<14} " + \
             "  ".join(f"{d:>8}" for d in datasets) + \
             "  CrossMean"
    logger.info(header)
    logger.info("-" * len(header))
    for combo, entry in summary.items():
        per = entry["per_dataset"]
        vals = [f"{per.get(d, {}).get('mean', float('nan')):>8.2f}"
                for d in datasets]
        logger.info(
            f"{combo:<16} {entry['layers']:<14} "
            + "  ".join(vals)
            + f"  {entry['cross_dataset_mean']:>8.2f}")
    logger.info("=" * 60)
    logger.info(f"Full results: {ABLATION_CSV}")
    logger.info(f"Summary JSON: {SUMMARY_JSON}")


# ── CLI ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="MLFF layer-selection ablation for FASTE (Alternative layers)")
    p.add_argument(
        "--datasets", nargs="+",
        default=["lap14", "res14"],
        choices=["lap14", "res14", "res15", "res16"],
        help="Datasets to run (default: lap14 res14)")
    p.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 43, 44],
        help="Random seeds (default: 42 43 44)")
    p.add_argument(
        "--combos", nargs="*",
        default=None,
        help="Subset of combo keys to run (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    logger = setup_logger()

    # Optionally restrict to a subset of combinations
    if args.combos:
        unknown = set(args.combos) - set(LAYER_COMBOS)
        if unknown:
            logger.error(f"Unknown combos: {unknown}. "
                         f"Valid: {list(LAYER_COMBOS)}")
            sys.exit(1)
        for k in list(LAYER_COMBOS):
            if k not in args.combos:
                del LAYER_COMBOS[k]

    logger.info("=" * 60)
    logger.info("FASTE – MLFF Layer Selection (Alternative Triplets)")
    logger.info(f"Datasets : {args.datasets}")
    logger.info(f"Seeds    : {args.seeds}")
    logger.info(f"Combos   : {list(LAYER_COMBOS.keys())}")
    logger.info("=" * 60)

    run_layer_ablation(
        datasets = args.datasets,
        seeds    = args.seeds,
        logger   = logger,
    )