"""Side-chain atom encoder: a sparse atom-level block in front of the residue SE(3) trunk.

Shape of the addition (side-chain branch pilot)::

    atom14 heavy atoms (ndata['atom_xyz'/'atom_mask'])
      -> sparse atom neighbourhood  (same residue + atoms of spatially close residues)
      -> N attention blocks over that neighbourhood
      -> per-residue pooling (mean + max over the residue's own atoms)
      -> zero-initialised projection ADDED to the residue node embedding
      -> existing SE(3)-Transformer, untouched

Invariance
----------
Nothing here consumes raw coordinates. Every geometric input is either a distance
(RBF-expanded) or a direction expressed in the *destination residue's* backbone
frame (N -> CA -> C). Both are invariant under global rotation and translation, so
the scalar features handed to the trunk are invariant and the trunk's own SE(3)
equivariance is unaffected.

The geometry (neighbour selection + edge features) is built under ``no_grad``:
coordinates are inputs, never parameters, so no gradient needs to reach them and
the expensive candidate expansion stays out of the autograd graph.
"""
from typing import Optional

import torch
import torch.nn as nn

from data_loading.constants import (ATOM14_TYPE_IDX, ATOM14_BACKBONE_SLOTS, ALL_NUM_ATOM,
                                    AA3_IDX, MAX_NUM_ATOM)

N_AA_TYPES = len(AA3_IDX)          # 21 (20 standard + UNK)
_SLOT_N, _SLOT_CA, _SLOT_C = 0, 1, 2


def _scatter_softmax(score: torch.Tensor, index: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Softmax of ``score`` (E, H) over the groups named by ``index`` (E,)."""
    n_heads = score.shape[-1]
    idx = index.unsqueeze(-1).expand(-1, n_heads)
    maxes = torch.full((n_rows, n_heads), float('-inf'), device=score.device, dtype=score.dtype)
    maxes.scatter_reduce_(0, idx, score, reduce='amax', include_self=True)
    maxes = torch.nan_to_num(maxes, neginf=0.0)
    ex = torch.exp(score - maxes.gather(0, idx))
    denom = torch.zeros((n_rows, n_heads), device=score.device, dtype=score.dtype)
    denom.scatter_add_(0, idx, ex)
    return ex / (denom.gather(0, idx) + 1e-9)


def _backbone_frames(atom_xyz: torch.Tensor, atom_mask: torch.Tensor):
    """Per-residue orthonormal frame from N, CA, C. Returns (R (L,3,3), ok (L,)).

    Columns of R are (e1, e2, e3). Residues missing any of N/CA/C get the identity
    frame and ok=False, so a direction feature there degenerates to the global
    frame rather than to NaN.
    """
    n_xyz = atom_xyz[:, _SLOT_N]
    ca_xyz = atom_xyz[:, _SLOT_CA]
    c_xyz = atom_xyz[:, _SLOT_C]
    ok = (atom_mask[:, _SLOT_N] > 0) & (atom_mask[:, _SLOT_CA] > 0) & (atom_mask[:, _SLOT_C] > 0)

    v1 = n_xyz - ca_xyz
    v2 = c_xyz - ca_xyz
    e1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2p = v2 - (v2 * e1).sum(-1, keepdim=True) * e1
    e2 = v2p / (v2p.norm(dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)                      # (L,3,3)

    eye = torch.eye(3, device=atom_xyz.device, dtype=atom_xyz.dtype).expand_as(R)
    R = torch.where(ok.view(-1, 1, 1), R, eye)
    return R, ok


class AtomAttentionBlock(nn.Module):
    """One multi-head attention round over the sparse atom neighbourhood."""

    def __init__(self, d_model: int, d_edge: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must divide by n_heads'
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.edge_k = nn.Linear(d_edge, d_model, bias=False)
        self.edge_v = nn.Linear(d_edge, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.ReLU(), nn.Linear(2 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e_feat, src, dst):
        n_atoms = h.shape[0]
        hn = self.norm_attn(h)
        q = self.q(hn).view(n_atoms, self.n_heads, self.d_head)[dst]
        k = (self.k(hn)[src].view(-1, self.n_heads, self.d_head)
             + self.edge_k(e_feat).view(-1, self.n_heads, self.d_head))
        v = (self.v(hn)[src].view(-1, self.n_heads, self.d_head)
             + self.edge_v(e_feat).view(-1, self.n_heads, self.d_head))

        score = (q * k).sum(-1) / (self.d_head ** 0.5)          # (E, H)
        alpha = _scatter_softmax(score, dst, n_atoms)
        msg = (alpha.unsqueeze(-1) * v).reshape(-1, self.n_heads * self.d_head)
        agg = torch.zeros(n_atoms, self.n_heads * self.d_head, device=h.device, dtype=msg.dtype)
        agg.index_add_(0, dst, msg)

        h = h + self.drop(self.out(agg))
        h = h + self.drop(self.ff(self.norm_ff(h)))
        return h


class AtomEncoder(nn.Module):
    """Atom-level encoder producing one invariant vector per residue.

    Args:
        d_out:          width of the per-residue output (matched to the trunk's
                        node embedding by the caller).
        d_model:        atom feature width.
        n_layers:       number of attention blocks.
        atom_radius:    atom-atom cutoff (A) for the sparse neighbourhood.
        res_cutoff:     only residue pairs whose CA are within this (A) are
                        expanded into atom pairs; bounds the candidate count.
        max_neighbors:  per-atom cap, nearest first.
        chunk_pairs:    residue pairs expanded per chunk (memory ceiling).
    """

    def __init__(self, d_out: int, d_model: int = 32, n_layers: int = 2, n_heads: int = 4,
                 atom_radius: float = 5.0, res_cutoff: float = 8.0, n_rbf: int = 16,
                 max_neighbors: int = 24, dropout: float = 0.1, chunk_pairs: int = 20000):
        super().__init__()
        self.d_model = d_model
        self.atom_radius = float(atom_radius)
        self.res_cutoff = float(res_cutoff)
        self.max_neighbors = int(max_neighbors)
        self.chunk_pairs = int(chunk_pairs)
        self.n_rbf = int(n_rbf)

        self.register_buffer('atom14_type',
                             torch.tensor(ATOM14_TYPE_IDX, dtype=torch.long), persistent=False)
        self.register_buffer('rbf_centers',
                             torch.linspace(0.0, self.atom_radius, n_rbf), persistent=False)
        self.rbf_width = (self.atom_radius / max(1, n_rbf - 1)) * 1.5

        self.atom_type_embed = nn.Embedding(ALL_NUM_ATOM, d_model)
        self.res_type_embed = nn.Embedding(N_AA_TYPES, d_model)
        self.slot_embed = nn.Embedding(MAX_NUM_ATOM, d_model)
        # + is_backbone, is_cdr
        self.atom_in = nn.Linear(3 * d_model + 2, d_model, bias=False)

        # edge: RBF distances, same-residue, sequence-adjacent, direction in frame
        d_edge = n_rbf + 2 + 3
        self.blocks = nn.ModuleList([
            AtomAttentionBlock(d_model, d_edge, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.to_residue = nn.Sequential(
            nn.Linear(2 * d_model, d_out), nn.ReLU(), nn.Linear(d_out, d_out),
        )

    # ── geometry (no gradient needed: coordinates are inputs, not parameters) ──
    @torch.no_grad()
    def _build_atom_edges(self, atom_xyz, atom_mask, res_pairs, frames, res_no, chain_id):
        """Expand residue pairs into atom pairs, keep those within the radius.

        Returns (src, dst, edge_feat) with atom indices into the flattened
        (n_res * 14) layout.
        """
        device = atom_xyz.device
        n_res = atom_xyz.shape[0]
        flat_xyz = atom_xyz.reshape(-1, 3)
        flat_mask = atom_mask.reshape(-1) > 0

        slot = torch.arange(MAX_NUM_ATOM, device=device)
        ii, jj = torch.meshgrid(slot, slot, indexing='ij')
        ii, jj = ii.reshape(-1), jj.reshape(-1)                   # (196,)

        src_parts, dst_parts, dist_parts = [], [], []
        for start in range(0, res_pairs.shape[1], self.chunk_pairs):
            rp = res_pairs[:, start:start + self.chunk_pairs]
            r_src, r_dst = rp[0], rp[1]
            # (P, 196) atom indices
            a_src = (r_src.unsqueeze(1) * MAX_NUM_ATOM + ii.unsqueeze(0)).reshape(-1)
            a_dst = (r_dst.unsqueeze(1) * MAX_NUM_ATOM + jj.unsqueeze(0)).reshape(-1)
            valid = flat_mask[a_src] & flat_mask[a_dst] & (a_src != a_dst)
            a_src, a_dst = a_src[valid], a_dst[valid]
            d = torch.norm(flat_xyz[a_src] - flat_xyz[a_dst], dim=-1)
            near = d <= self.atom_radius
            if near.any():
                src_parts.append(a_src[near]); dst_parts.append(a_dst[near]); dist_parts.append(d[near])

        if not src_parts:
            empty_i = torch.zeros(0, dtype=torch.long, device=device)
            return empty_i, empty_i, torch.zeros(0, self.n_rbf + 5, device=device)

        src = torch.cat(src_parts); dst = torch.cat(dst_parts); dist = torch.cat(dist_parts)

        # per-destination cap, nearest first
        if self.max_neighbors > 0:
            by_dist = torch.argsort(dist)
            dst_bd = dst[by_dist]
            by_group = torch.argsort(dst_bd, stable=True)
            keep_order = by_dist[by_group]
            dst_sorted = dst[keep_order]
            n_slots = n_res * MAX_NUM_ATOM
            counts = torch.bincount(dst_sorted, minlength=n_slots)
            offsets = torch.cumsum(counts, 0) - counts
            rank = torch.arange(dst_sorted.shape[0], device=device) - offsets[dst_sorted]
            keep = keep_order[rank < self.max_neighbors]
            src, dst, dist = src[keep], dst[keep], dist[keep]

        res_src = torch.div(src, MAX_NUM_ATOM, rounding_mode='floor')
        res_dst = torch.div(dst, MAX_NUM_ATOM, rounding_mode='floor')

        rbf = torch.exp(-((dist.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        same_res = (res_src == res_dst).to(dist.dtype).unsqueeze(-1)
        seq_adj = (((res_no[res_src] - res_no[res_dst]).abs() == 1)
                   & (chain_id[res_src] == chain_id[res_dst])).to(dist.dtype).unsqueeze(-1)
        # direction src->dst written in the destination residue's backbone frame
        vec = flat_xyz[src] - flat_xyz[dst]
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
        local_dir = torch.einsum('eij,ei->ej', frames[res_dst], vec)

        return src, dst, torch.cat([rbf, same_res, seq_adj, local_dir], dim=-1)

    @torch.no_grad()
    def _residue_pairs(self, graph, pos):
        """Self-pairs (intra-residue) + residue edges whose CA are within res_cutoff."""
        device = pos.device
        n_res = pos.shape[0]
        self_idx = torch.arange(n_res, device=device)
        e_src, e_dst = graph.edges()
        e_src = e_src.to(device); e_dst = e_dst.to(device)
        close = torch.norm(pos[e_src] - pos[e_dst], dim=-1) <= self.res_cutoff
        return torch.stack([
            torch.cat([self_idx, e_src[close]]),
            torch.cat([self_idx, e_dst[close]]),
        ])

    def forward(self, graph, atom_xyz: Optional[torch.Tensor] = None,
                atom_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        atom_xyz = graph.ndata['atom_xyz'].float() if atom_xyz is None else atom_xyz.float()
        atom_mask = graph.ndata['atom_mask'] if atom_mask is None else atom_mask
        atom_mask = atom_mask.to(torch.bool)
        device = atom_xyz.device
        n_res = atom_xyz.shape[0]

        aa_type = graph.ndata['h'].long().clamp_(0, N_AA_TYPES - 1)
        res_no = graph.ndata['res_no'].long() if 'res_no' in graph.ndata else \
            torch.arange(n_res, device=device)
        chain_id = graph.ndata['chain_id'].long() if 'chain_id' in graph.ndata else \
            torch.zeros(n_res, dtype=torch.long, device=device)
        ulr = graph.ndata['ulr'].float() if 'ulr' in graph.ndata else \
            torch.zeros(n_res, device=device)

        with torch.no_grad():
            frames, _ = _backbone_frames(atom_xyz, atom_mask)
            res_pairs = self._residue_pairs(graph, graph.ndata['pos'].float())
            src, dst, e_feat = self._build_atom_edges(
                atom_xyz, atom_mask, res_pairs, frames, res_no, chain_id)

        # ── atom node features ────────────────────────────────────────────────
        atom_type = self.atom14_type[aa_type]                       # (L,14)
        slot_idx = torch.arange(MAX_NUM_ATOM, device=device).expand(n_res, -1)
        is_bb = (slot_idx < ATOM14_BACKBONE_SLOTS).float().unsqueeze(-1)
        is_cdr = ulr.view(-1, 1).expand(-1, MAX_NUM_ATOM).unsqueeze(-1)
        feats = torch.cat([
            self.atom_type_embed(atom_type),
            self.res_type_embed(aa_type).unsqueeze(1).expand(-1, MAX_NUM_ATOM, -1),
            self.slot_embed(slot_idx),
            is_bb, is_cdr,
        ], dim=-1)
        h = self.atom_in(feats).reshape(-1, self.d_model)
        h = h * atom_mask.reshape(-1, 1)

        e_feat = e_feat.to(h.dtype)
        for block in self.blocks:
            h = block(h, e_feat, src, dst)
        h = self.norm_out(h).reshape(n_res, MAX_NUM_ATOM, self.d_model)

        # ── pool each residue's own atoms ────────────────────────────────────
        m = atom_mask.unsqueeze(-1).to(h.dtype)
        h = h * m
        mean = h.sum(1) / m.sum(1).clamp(min=1.0)
        hmax = h.masked_fill(~atom_mask.unsqueeze(-1), float('-inf')).max(1).values
        hmax = torch.nan_to_num(hmax, neginf=0.0)
        return self.to_residue(torch.cat([mean, hmax], dim=-1))
