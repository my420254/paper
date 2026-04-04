import os, torch
import numpy as np
from transformers import AutoTokenizer

from aste import SpanASTEModel, load_dataset, BatchLoader, DATASET_CONFIGS, DATA_ROOT, PAPER_ROOT, ENCODER_PATH, DEVICE

dataset = 'res14'  
seed = 42

print(f"1. Loading test data for {dataset}...")
tok = AutoTokenizer.from_pretrained(ENCODER_PATH, use_fast=False)
ts_insts = load_dataset(f'{DATA_ROOT}{dataset}/test.json', tok, 'test')

cfg = DATASET_CONFIGS[dataset]
model = SpanASTEModel(cfg, use_mlff=True).to(DEVICE).float()

print("2. Loading trained weights...")
model_path = os.path.join(PAPER_ROOT, dataset, f'seed{seed}_full_model.pt')
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

test_loader = BatchLoader(ts_insts, cfg['batch_size'])
ph_list, label_list = [], []

print("3. Extracting features and labels (This takes ~10 seconds)...")
with torch.no_grad():
    for batch in test_loader.get_batches(shuffle=False):
        _, _, _, saved = model(
            batch['input_ids'], batch['attention_mask'],
            batch['lengths'], batch['w2s_t'],
            batch['spans_t'], batch['span_labels'],
            batch['gold_pairs'], return_pair_features=True)
        
        if saved is not None:
            ph_list.append(saved["ph"].numpy())
            label_list.append(saved["labels"].numpy())

all_ph = np.concatenate(ph_list, 0)
all_labels = np.concatenate(label_list, 0)

out_path = os.path.join(PAPER_ROOT, dataset, f'seed{seed}_full_tsne.npz')
np.savez(out_path, pair_hidden=all_ph, labels=all_labels)

print(f"✅ 成功提取 {len(all_ph)} 个对特征！")
print(f"✅ 标签分布 (0:None, 1:Neg, 2:Neu, 3:Pos): {np.bincount(all_labels)}")
print(f"✅ 完美！旧文件已被覆盖至: {out_path}")
