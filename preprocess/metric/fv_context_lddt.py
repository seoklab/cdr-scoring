"""CDR-lDDT variants that differ only in which partner atoms are used.

Scored atoms are always the backbone (N, CA, C, O) of the 6 Chothia CDRs
(DEFAULT_NAMED_LOOP_RANGES). For each scored atom, partners are atoms of other
residues within 15 A in the native; thresholds 0.5/1/2/4 A; micro-average.

  cdr_context_lddt     partners = every H/L backbone atom (incl. constant domains)
                       + every antigen backbone atom. This is the metric the
                       pipeline stores as `cdr_lddt` (_prepare_loop_lddt).
  fv_context_cdr_lddt  partners = CDR atoms + non-CDR Fv atoms only
                       (Fv = Chothia H 1-113, L 1-107, insertion codes included);
                       no antigen, no constant domain.
  cdr_internal_lddt    partners = CDR atoms only.
Per-CDR values (<CDR>_fv_context_lddt) use the same partners as fv_context.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_source_metrics_fast import (  # noqa: E402
    ANTIBODY_CHAINS, _loop_id, _matched_residue_pairs, _matching_backbone_coords)
from evaluation.loop_metrics import (  # noqa: E402
    DEFAULT_NAMED_LOOP_RANGES, LDDT_CUTOFF_A, LDDT_THRESHOLDS_A)

FV_LAST = {'H': 113, 'L': 107}  # Chothia: VH 1-113, VL 1-107


def _collect(native_structure, model_structure):
    nat, mod, res, loop, kind = [], [], [], [], []   # kind: 0 CDR, 1 non-CDR Fv, 2 other H/L, 3 antigen
    r_idx = 0
    for chain_id, n_res, m_res in _matched_residue_pairs(native_structure, model_structure):
        n_xyz, m_xyz, _ = _matching_backbone_coords(n_res, m_res)
        if not n_xyz:
            continue
        resseq = int(n_res.id[1])
        if chain_id in ANTIBODY_CHAINS:
            lid = _loop_id(chain_id, resseq)
            k = 0 if lid > 0 else (1 if resseq <= FV_LAST[chain_id] else 2)
        else:
            lid, k = 0, 3
        for a, b in zip(n_xyz, m_xyz):
            nat.append(a)
            mod.append(b)
            res.append(r_idx)
            loop.append(lid)
            kind.append(k)
        r_idx += 1
    return (np.asarray(nat, float).reshape(-1, 3), np.asarray(mod, float).reshape(-1, 3),
            np.asarray(res), np.asarray(loop), np.asarray(kind))


def compute_cdr_lddt_variants(native_structure, model_structure) -> Dict[str, float]:
    nat, mod, res, loop, kind = _collect(native_structure, model_structure)
    out = {'cdr_context_lddt': np.nan, 'fv_context_cdr_lddt': np.nan, 'cdr_internal_lddt': np.nan}
    out.update({f'{name}_fv_context_lddt': np.nan for name, *_ in DEFAULT_NAMED_LOOP_RANGES})
    rows = np.where(kind == 0)[0]
    if rows.size == 0:
        return out
    d_nat = np.sqrt(((nat[rows, None, :] - nat[None, :, :]) ** 2).sum(-1))
    d_mod = np.sqrt(((mod[rows, None, :] - mod[None, :, :]) ** 2).sum(-1))
    neighbor = (res[rows, None] != res[None, :]) & (d_nat > 0) & (d_nat <= LDDT_CUTOFF_A)
    diff = np.abs(d_nat - d_mod)
    preserved = np.zeros_like(diff)
    for t in LDDT_THRESHOLDS_A:
        preserved += diff < t
    preserved /= len(LDDT_THRESHOLDS_A)

    def score(col_mask, row_mask=None):
        m = neighbor & col_mask[None, :]
        if row_mask is not None:
            m = m & row_mask[:, None]
        n = int(m.sum())
        return float(preserved[m].sum() / n) if n else float('nan'), n

    fv_cols = kind <= 1
    out['cdr_context_lddt'], n_all = score(np.ones_like(kind, dtype=bool))
    out['fv_context_cdr_lddt'], n_fv = score(fv_cols)
    out['cdr_internal_lddt'], n_int = score(kind == 0)
    for idx, (name, *_) in enumerate(DEFAULT_NAMED_LOOP_RANGES, start=1):
        out[f'{name}_fv_context_lddt'] = score(fv_cols, loop[rows] == idx)[0]
    out.update({'n_pairs_cdr_context': n_all, 'n_pairs_fv_context': n_fv, 'n_pairs_cdr_internal': n_int})
    return out
