#!/usr/bin/env python
"""Validation-only: score a checkpoint on the DockQ-gated valid set (v1 config).

Runs the standard validation pass (run_epoch, is_train=False) so the DockQ>=0.49
gate is applied to the validation decoys, giving an apples-to-apples comparison
with v1/v1gp. Prints top1 cdr_lddt / H3 / DockQ / CAPRI, aggregated over targets.
"""
import os, sys, pickle, argparse, numpy as np
REPO = "/home/sujin/projects/cdr-scoring/cdr-code"
sys.path.insert(0, REPO + "/libs")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset-config", required=True)
ap.add_argument("--param-name", default="ckpt_eval_gated")
ap.add_argument("--valid-pkl", default="/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag/0_info/valid.pkl")
a = ap.parse_args()
# Blank argv so modules that parse sys.argv at import time (data_module's
# module-level PARSER.parse_args) don't choke on our --ckpt/--dataset-config.
sys.argv = [sys.argv[0]]

os.environ["CDR_LABEL_METRIC"] = "loop_lddt"
os.environ["CDR_NEAR_NATIVE_CUTOFF"] = "0.8"

import torch
from runtime.arguments import PARSER
import data_loading.data_module as dm
from data_loading.data_module import HUDataModule
from model.transformer import Sujin_with_SE3
from model.fiber import Fiber
from runtime.utils import using_tensor_cores
from runtime.train import run_epoch

argv = [
    "--run_type", "train", "--param_name", a.param_name,
    "--dataset_config", a.dataset_config, "--load_ckpt_path", a.ckpt,
    "--label_metric", "loop_lddt", "--near_native_cutoff", "0.8",
    "--loss_type", "sml", "--amp", "true", "--wandb", "false",
    "--batch_size", "1", "--num_workers", "4", "--epochs", "1",
    "--num_layers", "4", "--num_heads", "4", "--num_degrees", "2",
    "--num_channels", "32", "--embedded_node_dim", "32", "--embedded_edge_dim", "32",
]
args = PARSER.parse_args(argv)
dm.args = args  # data_module reads module-global args (run_type etc.)

valid_list = [str(x) for x in pickle.load(open(a.valid_pkl, "rb"))["list"]]
print(f"[eval] ckpt={a.ckpt}\n[eval] config={a.dataset_config}\n[eval] valid targets={len(valid_list)}", flush=True)

model = Sujin_with_SE3(
    fiber_in=Fiber({0: args.embedded_node_dim, 1: 4}),
    fiber_out=Fiber({0: args.num_degrees * args.num_channels, 1: 20}),
    fiber_edge=Fiber({0: args.embedded_edge_dim, 1: 1}),
    use_nodewise_score=False,
    tensor_cores=using_tensor_cores(args.amp),
    **vars(args),
).cuda()
ck = torch.load(a.ckpt, map_location="cuda:0")
missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
print(f"[eval] loaded ckpt epoch={ck.get('epoch','?')} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
datamodule = HUDataModule(**vars(args))
vdl = datamodule.val_dataloader(valid_list, args.decoytype, dataset_config=a.dataset_config)
if datamodule.ds_val is not None and hasattr(datamodule.ds_val, "set_epoch"):
    datamodule.ds_val.set_epoch(2)
model.eval()
with torch.no_grad():
    _ = run_epoch(model, vdl, 2, scaler, opt, 0, [], is_train=False, args=args)

# aggregate the per-target info written by run_epoch
info_path = f"{REPO}/../cdr-data/ckpt/{a.param_name}/valid.2.0.info"
info = pickle.load(open(info_path, "rb"))
def mean(k):
    v = [d[k] for d in info.values() if isinstance(d, dict) and k in d and d[k] == d[k]]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)
print("\n==== GATED validation results ====", flush=True)
for k in ["eval_top1_loop_lddt","eval_top1_h3_lddt","eval_top1_dockq",
          "eval_top1_capri_acceptable","eval_top1_capri_medium","eval_top1_capri_high",
          "eval_oracle_loop_lddt"]:
    m, n = mean(k)
    print(f"  {k:28} {m:.4f}  (n={n})", flush=True)
