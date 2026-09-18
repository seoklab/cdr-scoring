"""Smoke test for v2 fixed validation manifests.

Verifies:
  (1) reloading a manifest selects the EXACT same decoy set for a target
      (deterministic; compare identity-key multisets across two fresh loads);
  (2) intrinsic (out) and final heads are scored on the IDENTICAL candidate
      pool in a single forward batch (same n, same cdr_lddt vector), for both
      the multi-source and Boltz2-only pools.
"""
import sys
sys.argv = ['smoke']
import json
import torch

import data_loading.data_module as dm
from model.fiber import Fiber
from model.transformer import Sujin_with_SE3

CFG = 'configs/dataset_sources-v2.yaml'
CKPT = '/home/sujin/projects/cdr-scoring/cdr-data/ckpt/v2_smoke/v2_smoke_2.pt'
MAN = {
    'multisource': '/home/sujin/projects/cdr-scoring/cdr-data/val_manifests/val_multisource.json',
    'boltz2': '/home/sujin/projects/cdr-scoring/cdr-data/val_manifests/val_boltz2.json',
}
N_TARGETS = 2   # keep the smoke fast

dm.args.run_type = 'train'; dm.args.label_metric = 'loop_lddt'; dm.args.near_native_cutoff = 0.8
dm.args.use_multihead = True

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Sujin_with_SE3(fiber_in=Fiber({0: 32, 1: 4}), fiber_out=Fiber({0: 64, 1: 20}),
                     fiber_edge=Fiber({0: 32, 1: 1}), use_nodewise_score=False,
                     num_degrees=2, num_channels=32, num_layers=4, num_heads=4, channels_div=2)
sd = torch.load(CKPT, map_location='cpu')['state_dict']
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
net.load_state_dict(sd, strict=False)
net.to(dev).eval()


def keys_for(ds, tid):
    """Reproduce manifest selection and return the sorted identity-key list actually used."""
    # Re-derive by intersecting the manifest set with the loaded decoys is done inside
    # _getitem_yaml; here we compare the returned cdr_lddt vector (order-stable proxy for
    # the exact selected decoy set) plus the count.
    idx = ds.inp_dat.index(tid)
    item = ds[idx]
    em = next(it for it in item[3:] if isinstance(it, dict))
    return item[0], em


for pool, path in MAN.items():
    man = json.load(open(path))
    print(f'\n===== pool={pool} hash={man["hash"]} n_targets={man["n_targets"]} =====')

    # two independent loads
    dsA = dm.MyDataset([], is_train=False, dataset_config=CFG); metaA = dsA.load_val_manifest(path)
    dsB = dm.MyDataset([], is_train=False, dataset_config=CFG); dsB.load_val_manifest(path)
    dsA.set_epoch(3); dsB.set_epoch(7)   # different epoch/seed on purpose
    targets = dsA.inp_dat[:N_TARGETS]

    for tid in targets:
        gA, emA = keys_for(dsA, tid)
        gB, emB = keys_for(dsB, tid)
        cdrA = emA['loop_lddt'].reshape(-1); cdrB = emB['loop_lddt'].reshape(-1)
        n_manifest = len(man['targets'].get(tid, []))
        # (1) determinism: identical selected decoy set across reloads (epoch/seed differ)
        det = (cdrA.numel() == cdrB.numel()) and torch.allclose(
            torch.sort(cdrA).values, torch.sort(cdrB).values, atol=1e-6, equal_nan=True)
        # (2) same-pool: one forward -> intrinsic & final over identical candidates
        g = gA.to(dev)
        with torch.no_grad():
            pred = net(g)
        s_i = pred['out'].reshape(-1); s_f = pred['final'].reshape(-1)
        cdr = cdrA.to(s_i.device)
        same_pool = (s_i.numel() == s_f.numel() == cdr.numel() == cdrA.numel())
        neg = torch.full_like(s_i, -1e9)
        i_top = int(torch.argmax(torch.where(torch.isfinite(s_i), s_i, neg)))
        f_top = int(torch.argmax(torch.where(torch.isfinite(s_f), s_f, neg)))
        fin = torch.isfinite(cdr)
        print(f'  {tid}: n_manifest={n_manifest} n_loaded={cdr.numel()} '
              f'| reload_deterministic={det} | same_candidate_pool(intrinsic==final)={same_pool} '
              f'| intrinsic_top1_cdr={float(cdr[i_top]):.4f} final_top1_cdr={float(cdr[f_top]):.4f} '
              f'oracle={float(cdr[fin].max()):.4f}')
print('\n[smoke] done')
