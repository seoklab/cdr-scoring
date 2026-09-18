# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import argparse
import pathlib

from runtime.utils import str2bool

PARSER = argparse.ArgumentParser(description='SE(3)-Transformer')

paths = PARSER.add_argument_group('Paths')
paths.add_argument('--data_dir', type=pathlib.Path, default=pathlib.Path('./data'),
                   help='Directory where the data is located or should be downloaded')
paths.add_argument('--log_dir', type=pathlib.Path, default=pathlib.Path('./data'),
                   help='Directory where the results logs should be saved')
paths.add_argument('--dllogger_name', type=str, default='dllogger_results.json',
                   help='Name for the resulting DLLogger JSON file')
paths.add_argument('--param_name', type=str, default='hmmm',
                   help='Name for the resulting parameter_file')
paths.add_argument('--sch_type', type=str, default='cosine',
                   help='Name for the resulting parameter_file')
paths.add_argument('--save_ckpt_path', type=pathlib.Path, default=None,
                   help='File where the checkpoint should be saved')
paths.add_argument('--load_ckpt_path', type=pathlib.Path, default=None,
                   help='File of the checkpoint to be loaded')
paths.add_argument('--ckpt_root', type=pathlib.Path,
                   default=pathlib.Path('/home/sujin/projects/cdr-scoring/cdr-data/ckpt'),
                   help='Root directory for training checkpoints and per-epoch .info files')
paths.add_argument('--inference_dir', type=pathlib.Path,
                   default=pathlib.Path('/home/sujin/projects/cdr-scoring/cdr-data/inference'),
                   help='Directory where inference .info files should be saved')

optimizer = PARSER.add_argument_group('Optimizer')
optimizer.add_argument('--optimizer', choices=['adam', 'sgd', 'lamb'], default='adam')
optimizer.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.002)
optimizer.add_argument('--min_learning_rate', '--min_lr', dest='min_learning_rate', type=float, default=None)
optimizer.add_argument('--momentum', type=float, default=0.9)
optimizer.add_argument('--weight_decay', type=float, default=0.1)

PARSER.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
PARSER.add_argument('--batch_size', type=int, default=240, help='Batch size')
PARSER.add_argument('--seed', type=int, default=None, help='Set a seed globally')
PARSER.add_argument('--num_workers', type=int, default=8, help='Number of dataloading workers')

PARSER.add_argument('--loss_type', type=str, nargs='?', const=True, default='sml', help='Sujin Loss Type')
PARSER.add_argument('--label_metric', type=str, default=None, choices=['loop_rmsd', 'loop_lddt'],
                    help='Training label / ranking metric. loop_rmsd=lower-is-better (default), '
                         'loop_lddt=higher-is-better. When set, overrides the YAML label_metric AND '
                         'flips loss/ranking/tier direction accordingly. None = use the value in the dataset YAML.')
PARSER.add_argument('--near_native_cutoff', type=float, default=None,
                    help='Near-native vs non-native cutoff used by SML loss and the tier sampler. '
                         'Default: 2.0 for loop_rmsd, 0.8 for loop_lddt.')
PARSER.add_argument('--embedded_node_dim', type=float, nargs='?', const=True, default=32, help='See model/transformer.py')
PARSER.add_argument('--embedded_edge_dim', type=float, nargs='?', const=True, default=32, help='See model/transformer.py')
PARSER.add_argument('--readout', type=str, nargs='?', const=True, default='sum', help='Readout Type')
#PARSER.add_argument('--num_layers', type=float, nargs='?', const=True, default=4, help='Number of Layers')
PARSER.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False, help='Use Automatic Mixed Precision')
PARSER.add_argument('--gradient_clip', type=float, default=None, help='Clipping of the gradient norms')
PARSER.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation')
PARSER.add_argument('--ckpt_interval', type=int, default=1,
                    help='Save a checkpoint every N epochs. Keep <= --eval_interval so every '
                         'validated epoch has a loadable checkpoint.')
PARSER.add_argument('--eval_interval', dest='eval_interval', type=int, default=1,
                    help='Do an evaluation round every N epochs')
PARSER.add_argument('--silent', type=str2bool, nargs='?', const=True, default=False,
                    help='Minimize stdout output')
PARSER.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=False,
                    help='Enable W&B logging')
PARSER.add_argument('--wandb_id', type=str, default=None,help='W&B run ID to resume logging')

PARSER.add_argument('--benchmark', type=str2bool, nargs='?', const=True, default=False,
                    help='Benchmark mode')
PARSER.add_argument('--val_fn_path', type=str, nargs='?',default=None)
PARSER.add_argument('--out_data_tag', type=str, nargs='?',default=None)
PARSER.add_argument('--use_rsa_feat', type=str2bool,nargs='?',const=True,default=False) 
PARSER.add_argument('--use_l0_aux_loss', type=str2bool,nargs='?',const=True,default=False) 
PARSER.add_argument('--use_l1_loss', type=str2bool,nargs='?',const=True,default=False) 
PARSER.add_argument('--use_subunit_act', type=str2bool,nargs='?',const=True,default=False) 
PARSER.add_argument('--use_subunit_act_aux_loss', type=str2bool,nargs='?',const=True,default=False) 
PARSER.add_argument('--graph_method', type=str,nargs='?',const=True,default='hu') 
PARSER.add_argument('--aux_type', type=str,nargs='?',const=True,default='hu') 

PARSER.add_argument('--run_type', type=str, default='train', help='train or inference')
PARSER.add_argument('--save_model_path', type=str, default=None, help='Path to checkpoint file for loading (inference) or saving (training)')
PARSER.add_argument('--decoytype', type=str, default='fp', help='Type of decoys (fp, fp_extended, af3, boltz2, igfold4_local_opt, etc.)')
PARSER.add_argument('--db_dir', type=str, default=None, help='Directory where Target pickle files are stored')
PARSER.add_argument('--pdb_list_pickle', type=str, default=None, help='Path to pickle file containing list of PDB IDs for inference')
PARSER.add_argument('--use_subdirectory', action='store_true', default=False, help='Use subdirectory structure for pickle files (db_dir/<pdb_id>/<decoy>.pkl)')
PARSER.add_argument('--sel_nnd_only', action='store_true', default=False, help='Select only near-native decoys')
PARSER.add_argument('--sel_near_non', action='store_true', default=False, help='Select near decoys')

PARSER.add_argument('--all_atom', action='store_true', default=False, help='Use all atom')
PARSER.add_argument('--nodewise_score', action='store_true', default=False, help='Use nodewise score')
PARSER.add_argument('--dist_range', type=float, default=0.0, help='Range for distance cutoff in graph generation')
PARSER.add_argument('--max_batch_nodes', type=int, default=0,
                    help='Skip batches whose total graph nodes exceed this value (0 disables)')
PARSER.add_argument('--dataset_config', type=str, default=None,
                    help='Path to YAML dataset source config (enables multi-source mixing pipeline)')
PARSER.add_argument('--exclude_pdb_ids', type=str, default='',
                    help='Comma-separated PDB IDs to exclude from train/valid lists')
PARSER.add_argument('--exclude_pdb_file', type=str, default=None,
                    help='Path to newline-delimited PDB IDs to exclude from train/valid lists')
PARSER.add_argument('--run_script', type=str, default=None,
                    help='Path to the shell script that launched this run (saved to wandb for reproducibility)')
PARSER.add_argument('--profile_training', type=str2bool, nargs='?', const=True, default=False,
                    help='Log lightweight averaged training step timings')
PARSER.add_argument('--profile_log_interval', type=int, default=20,
                    help='Profiler logging interval in steps when --profile_training is enabled')

# ── Tier-based DPO losses (step 2) ──
PARSER.add_argument('--use_tier_dpo', action='store_true', default=False,
                    help='Enable tier-based DPO eject/top losses on top of existing TotalLoss')
PARSER.add_argument('--use_lddt_tiers', action='store_true', default=False,
                    help='Use lDDT-based tier definitions (X/A/B/C/D) for tier DPO finetune')
PARSER.add_argument('--use_phase_config', action='store_true', default=False,
                    help='Enable shared phase-based sampler/loss config in the finetune path')
PARSER.add_argument('--current_phase', type=int, default=1, choices=[1, 2, 3],
                    help='Manual phase id (1/2/3) used when --use_phase_config is enabled')
PARSER.add_argument('--lambda_dpo_eject', type=float, default=1.0,
                    help='Weight for tier DPO eject loss')
PARSER.add_argument('--lambda_dpo_top', type=float, default=1.0,
                    help='Weight for tier DPO top loss')
PARSER.add_argument('--dpo_beta', type=float, default=0.1,
                    help='Beta (inverse temperature) for tier DPO logistic preference loss')
PARSER.add_argument('--dpo_phase', type=int, default=0,
                    help='Pair sampling phase (1/2/3). 0 = auto from epoch progress')

# ── exp6: gen-gen DPO with AF3-matched tier-pair mix ──
# The H3-DPO eject/top rule only builds CROSS-tier pairs and therefore covers
# 13.91 % of the AF3 test pair distribution; 83.94 % of AF3 pairs are WITHIN one
# tier. These flags restrict the DPO pairs to generative-source decoys and match
# their tier-cell mix to AF3, adding the within-tier preference term.
PARSER.add_argument('--dpo_gen_only_pairs', action='store_true', default=False,
                    help='exp6: build DPO pairs only from generative-source decoys '
                         '(--dpo_gen_sources), with the tier-cell mix matched to AF3 test')
PARSER.add_argument('--dpo_gen_sources', type=str, default='Boltz2,Boltz2_s10n10',
                    help='exp6: comma-separated source NAMES treated as generative. '
                         'Resolved via sorted(spec.sources); never hard-code the ints.')
PARSER.add_argument('--lambda_dpo_within', type=float, default=0.0,
                    help='exp6: weight of the within-tier DPO loss (the 83.94 %% of AF3 '
                         'pairs the eject/top rule cannot produce). 0 disables it.')
PARSER.add_argument('--dpo_pair_min_delta', type=float, default=0.02,
                    help='exp6: minimum |delta cdr_lddt| for a DPO pair to be sampled')
# ── exp6b / exp6all: 2-D tier SML on the intrinsic head ──
# Today SML splits near/non on cdr_lddt alone, which puts "loop right, pose wrong"
# decoys (T2) in the near-native group -- 23 % of Boltz2_s10n10. Splitting on both
# axes fixes that; T2 and the near-empty T4 are dropped from the loss rather than
# pushed down, because a hard margin against a 0.85-cdr_lddt decoy fights the
# cdr_lddt signal itself. T1-vs-T2 is left to the DPO `top` term in a later stage.
PARSER.add_argument('--sml_tier_2d', action='store_true', default=False,
                    help='SML near = T1 (cdr_lddt AND fnat good), non = T3 (both bad); '
                         'skip T2/T4. Targets with no finite fnat (apo, GP) fall back '
                         'to the 1-D cdr_lddt rule.')
PARSER.add_argument('--sml_tier_lddt', type=float, default=0.80,
                    help='cdr_lddt threshold for the 2-D tier split')
PARSER.add_argument('--sml_tier_fnat', type=float, default=0.50,
                    help='fnat threshold for the 2-D tier split')
PARSER.add_argument('--sml_tier_lddt_low', type=float, default=0.70,
                    help='1-D fallback (apo, GP): negatives are cdr_lddt < this, and the '
                         'band up to --sml_tier_lddt is skipped. 0.70 because apo targets '
                         'able to form both classes drop 97.5%% -> 34.7%% below it.')
# ── exp9: DPO-top fine-tuning restricted to the top region ──
PARSER.add_argument('--dpo_top_region', action='store_true', default=False,
                    help='exp9: draw every DPO pair from the top region (both decoys '
                         'cdr_lddt >= --dpo_top_region_cut). exp8 showed SML finishes '
                         'the <=0.85 / >=0.90 split in 2 epochs and then never orders '
                         'anything inside the top, which is where test top-1 is decided.')
PARSER.add_argument('--dpo_top_region_cut', type=float, default=0.85,
                    help='exp9: both decoys of a pair must be >= this cdr_lddt.')
PARSER.add_argument('--dpo_top_band_lo', type=float, default=0.90,
                    help='exp9: a pair straddling this value is a BOUNDARY pair '
                         '(one side in the untrained 0.85-0.90 band).')
PARSER.add_argument('--dpo_top_band_quota', type=float, default=0.0,
                    help='exp9: share of the per-target budget reserved for boundary '
                         'pairs. 0.0 = one pool, boundary pairs at their natural rate '
                         '(exp9-a); 0.4 = enforced 40%% boundary (exp9-b).')
PARSER.add_argument('--dpo_top_min_delta', type=float, default=0.05,
                    help='exp9: |d cdr_lddt| threshold for a usable pair.')
PARSER.add_argument('--dpo_top_min_delta_relaxed', type=float, default=0.03,
                    help='exp9: fallback threshold for targets that cannot fill the '
                         'budget at --dpo_top_min_delta (81%% of train targets have a '
                         '>=0.05 top pair, 92%% have a >=0.03 one).')
PARSER.add_argument('--dpo_policy_head', choices=['final', 'intrinsic'], default='final',
                    help='exp9: which head the DPO gradient trains. intrinsic is the '
                         'head test inference actually reads; exp6 trained `final` and '
                         'so optimised a head that is not the deployed scorer.')
# ── exp8: fnat-gated 1-D SML (pose filter and loop cutoffs kept separate) ──
PARSER.add_argument('--sml_fnat_gated', action='store_true', default=False,
                    help='exp8 SML: drop decoys with finite fnat <= --sml_fnat_gate '
                         '(apo/GP have no fnat and are kept), then split the survivors '
                         'at --sml_near_cut / --sml_non_cut with a dead band between. '
                         'Takes precedence over --sml_tier_2d.')
PARSER.add_argument('--sml_fnat_gate', type=float, default=0.50,
                    help='exp8: finite fnat <= this is dropped from the SML loss '
                         'entirely (a wrong-pose decoy is not a usable answer).')
PARSER.add_argument('--sml_near_cut', type=float, default=0.90,
                    help='exp8: near-native is cdr_lddt >= this.')
PARSER.add_argument('--sml_non_cut', type=float, default=0.85,
                    help='exp8: non-native is cdr_lddt <= this; (non_cut, near_cut) is '
                         'a dead band excluded from the loss. 0.85 not 0.80 because '
                         'after the fnat gate a 0.80 cutoff makes the non class 88%% '
                         'PertMD_all, so the model learns Fv distortion, not loop error.')
PARSER.add_argument('--sml_demote_sources', type=str, default='',
                    help='exp8: comma-separated SOURCE NAMES whose combined share of '
                         'the non-native class is capped at --sml_demote_max_frac '
                         '(e.g. PertMD_all). Names are resolved from the YAML spec and '
                         'an unresolvable name is a hard error, never a silent no-op.')
PARSER.add_argument('--sml_demote_max_frac', type=float, default=0.50,
                    help='exp8: max share of the non-native class the demoted sources '
                         'may occupy. A target whose ONLY negatives are demoted keeps '
                         'one (max(1, ...)), so it still trains.')
# ── exp8 diagnostic: per-epoch (cdr_lddt vs model score) scatter dump ──
PARSER.add_argument('--score_scatter_targets', type=int, default=0,
                    help='exp8: dump per-decoy (cdr_lddt, fnat, scores, SML class) for '
                         'the first N manifest targets at every validation epoch, so '
                         'the dead band can be watched as training proceeds. 0 = off.')
PARSER.add_argument('--score_scatter_dir', type=str, default=None,
                    help='exp8: output dir for the scatter dump (default: '
                         '<log_dir>/score_scatter).')
PARSER.add_argument('--no_decay_heads', action='store_true', default=False,
                    help='exp7: exclude the output heads and all 1-D params (biases, '
                         'norms) from AdamW weight decay. Without this AdamW shrinks '
                         'every parameter ~5%%/epoch regardless of gradient — measured '
                         'on exp6, the gradient-free intrinsic head lost 31%% of its '
                         'norm in 13 epochs. Backbone decay is unchanged.')
PARSER.add_argument('--total_pair_budget', type=int, default=0,
                    help='DPO pairs sampled per target. 0 = legacy max(16, n_decoy//4). '
                         'exp6 spreads the budget over 10 AF3 tier cells, so 16 starves '
                         'the rare ones.')
PARSER.add_argument('--init_weights_from', type=str, default=None,
                    help='Fine-tune init: load MODEL WEIGHTS ONLY and start at epoch 0 with '
                         'a fresh optimizer. Unlike --load_ckpt_path, which also restores '
                         'the optimizer state and the epoch counter (i.e. a resume).')

# ── Compactness loss (step 3) ──
PARSER.add_argument('--lambda_compactness', type=float, default=0.0,
                    help='Weight for xtal-A compactness band loss (0 = disabled)')
PARSER.add_argument('--delta_A', type=float, default=1.0,
                    help='Upper bound for xtal-to-A score gap in compactness loss')

# ── v2 pretrain target-level source phase (GP/holo/apo ratio) ──
PARSER.add_argument('--pretrain_phase', type=str, default=None, choices=['A', 'B'],
                    help='v2 pretrain target-level source mix (explicit, no auto switch): '
                         'A = GP/holo/apo 40/40/20, B = 20/65/15. None keeps num_gp/num_abag behaviour.')

# ── v2 fixed validation manifests (phase/epoch/head-independent eval pools) ──
PARSER.add_argument('--val_manifest_multisource', type=str, default=None,
                    help='Path to the fixed multi-source antibody-antigen validation manifest (JSON).')
PARSER.add_argument('--val_manifest_boltz2', type=str, default=None,
                    help='Path to the fixed Boltz2-only validation manifest (JSON).')
PARSER.add_argument('--manifest_val_interval', type=int, default=1,
                    help='Run the fixed-manifest validation every N epochs (it rebuilds '
                         'graphs on the fly, so it is expensive). 0/1 = every epoch.')

# ── graph cache (Layer-B indexed containers; preprocess/build_graph_cache.py) ──
PARSER.add_argument('--graph_cache_mode', type=str, default='off', choices=['off', 'read'],
                    help="'read': serve decoy graphs from a prebuilt cache generation "
                         '(falling back to on-the-fly per item on any miss/stale pack); '
                         "'off': always build on the fly (default).")
PARSER.add_argument('--graph_cache_dir', type=str, default=None,
                    help='Cache generation directory (contains {source}/{target}.gpk). '
                         'Required when --graph_cache_mode read.')
PARSER.add_argument('--graph_cache_no_verify_struct', action='store_true', default=False,
                    help='Skip the per-pack struct_sig check (upstream-pickle staleness). '
                         'The build_sig/code_sig check still runs; only use when the source '
                         'pickles are known-immutable and you want to save the os.stat.')

# ── v2 Phase-C checkpoint flow (separate policy-init from frozen reference) ──
PARSER.add_argument('--reference_ckpt', type=str, default=None,
                    help='Frozen DPO reference checkpoint (Phase B best for all of C1/C2/C3). '
                         'Falls back to --save_model_path when unset.')
PARSER.add_argument('--policy_init_ckpt', type=str, default=None,
                    help='Checkpoint to initialize the trainable policy from '
                         '(C1<-Phase B best, C2<-C1 best, C3<-C2 best). Falls back to reference when unset.')

# ── v2 multi-head model ──
PARSER.add_argument('--use_multihead', action='store_true', default=False,
                    help='v2: model emits interface/final heads on top of the shared backbone. '
                         'Pretrain objective is unchanged (SML on intrinsic loop-quality head + '
                         'H3 ordinal aux); the interface head is inactive and the final ranking '
                         'head is trained later in the Phase-C DPO finetune.')
PARSER.add_argument('--lambda_intrinsic', type=float, default=1.0,
                    help='v2 weight for the intrinsic loop-quality SML term (GP/apo/holo, all 1.0)')
PARSER.add_argument('--lambda_interface', type=float, default=0.0,
                    help='v2 weight for the interface-compatibility ranking term (0 = disabled in pretrain)')
PARSER.add_argument('--use_interface_head', action='store_true', default=False,
                    help='v2 1st-priority: activate the interface head with a WITHIN-TARGET soft '
                         'RankNet loss on fnat (multi-task with intrinsic SML). Requires --lambda_interface>0.')
PARSER.add_argument('--interface_tau_fnat', type=float, default=0.1,
                    help='temperature of the soft preference target q_ij = sigmoid((fnat_i-fnat_j)/tau) '
                         'in the interface soft-rank loss. Smaller = harder preferences (default 0.1).')
# ── exp1 variant: absolute-tier balanced Soft RankNet ──
# ── exp3: AF3-matched tier pair weighting + fnat dead-zone filtering ──
PARSER.add_argument('--softrank_dead_zone', type=float, default=0.0,
                    help='exp4: dead zone for the PLAIN soft RankNet (no cell weighting). '
                         '0.05 = drop pairs with |dfnat| <= 0.05. Not the same as '
                         '--af3_matched_pairs --cell_weight_power 0, which equalises CELLS.')
# ── exp5 "gate_tier_pair": CDR-matched, fnat-contrastive pair selection ──
PARSER.add_argument('--cdr_matched_pairs', action='store_true', default=False,
                    help='exp5: spend --cond_pair_frac of the pair budget on pairs that match on '
                         'cdr_lddt but contrast on fnat. Takes precedence over --af3_matched_pairs '
                         'and --tier_balanced_rank. Labels stay continuous (no hard tier target).')
PARSER.add_argument('--lddt_match_tol', type=float, default=0.05,
                    help='exp5: two decoys count as cdr_lddt-matched when |dcdr_lddt| <= this '
                         '(same or adjacent cdr_lddt bin also qualifies).')
PARSER.add_argument('--fnat_contrast_min', type=float, default=0.2,
                    help='exp5: a conditional pair additionally needs |dfnat| >= this.')
PARSER.add_argument('--cond_pair_frac', type=float, default=0.60,
                    help='exp5: share of the loss mass on conditional pairs; the rest goes to '
                         'ordinary informative pairs (|dfnat| > dead zone).')
PARSER.add_argument('--min_cond_pairs', type=int, default=8,
                    help='exp5 source-shortcut guard: use SAME-source conditional pairs only when '
                         'at least this many exist; widen to cross-source otherwise.')
PARSER.add_argument('--cell_importance_weights', type=str, default='',
                    help='exp4-1: path to the JSON produced by '
                         'analyze/simulate_cell_importance_sampler.py. When set, ALL per-target '
                         'slots are drawn from the ungated pool by weighted sampling without '
                         'replacement over the joint (cdr_lddt, fnat) grid; the cdr_lddt tier '
                         'quotas and every top-up are bypassed. Use with dockq_gate: null.')
# ── exp4 "gate_rescue": joint cdr_lddt x fnat cell top-up ──
PARSER.add_argument('--joint_cell_topup', type=int, default=0,
                    help='exp4: reserve N of the per-target decoy slots for TARGETED decoys drawn '
                         'from this target\'s own PRE-GATE pool, chosen by joint (cdr_lddt, fnat) '
                         'cell priority. 0 = off. Supersedes --fnat_tier_topup when > 0.')
PARSER.add_argument('--joint_topup_p1', type=int, default=8,
                    help='exp4 priority 1 quota: cdr_lddt >= 0.8 AND fnat < 0.3 — the AF3-like '
                         '"loop right, pose wrong" hard negative the gate deletes 98.8%% of.')
PARSER.add_argument('--joint_topup_p2', type=int, default=4,
                    help='exp4 priority 2 quota: 0.6 <= cdr_lddt < 0.8 AND fnat < 0.3')
PARSER.add_argument('--joint_topup_p3', type=int, default=4,
                    help='exp4 priority 3 quota: cdr_lddt >= 0.8 AND fnat >= 0.7, used ONLY when '
                         'the natural selection has no positive anchor. Unused P3 slots roll into P1.')
PARSER.add_argument('--af3_matched_pairs', action='store_true', default=False,
                    help='exp3: drop pairs with |dfnat| <= --fnat_dead_zone, then weight the '
                         'survivors by the measured AF3 tier-cell frequency ** -cell_weight_power. '
                         'Takes precedence over --tier_balanced_rank.')
PARSER.add_argument('--fnat_dead_zone', type=float, default=0.05,
                    help='pairs with |fnat_i - fnat_j| <= this carry no ordering information and '
                         'are removed (70.9%% of AF3 same-target pairs; within-tier median is 0).')
PARSER.add_argument('--cell_weight_power', type=float, default=0.5,
                    help='tier-cell mass = p_cell ** -power. 0.5 = 1/sqrt(p) (3.9x amplification '
                         'after dead-zone filtering), 1.0 = full equalisation (what exp1 did, 59x), '
                         '0 = no cell reweighting.')
PARSER.add_argument('--tier_balanced_rank', action='store_true', default=False,
                    help='exp1: allocate soft-rank pair MASS by absolute fnat tier and by '
                         'generation source instead of by raw pair counts. No hard tier '
                         'classification loss; the target stays the continuous q_ij.')
PARSER.add_argument('--fnat_tier_topup', type=int, default=0,
                    help='exp1: after the cdr_lddt tier selection, force N decoys from every '
                         'absolute fnat tier that exists in THIS target\'s candidate pool but is '
                         'missing from the selection, evicting N from the largest tier. 0 = off. '
                         'Tiers the target lacks are never invented; nothing is taken from other targets.')
PARSER.add_argument('--tier_inter_frac', type=float, default=0.75,
                    help='share of the pair budget spent on INTER-tier pairs (rest within-tier)')
PARSER.add_argument('--tier_same_source_frac', type=float, default=0.75,
                    help='share of the pair budget spent on SAME-source pairs (rest cross-source)')
PARSER.add_argument('--tier_pair_mode', type=str, default='weight', choices=['weight', 'sample'],
                    help="'weight' (default): keep all pairs, scale by quota/count — deterministic. "
                         "'sample': draw --tier_pairs_per_target pairs per the quota (adds noise).")
PARSER.add_argument('--tier_pairs_per_target', type=int, default=512,
                    help='pairs drawn per target when --tier_pair_mode=sample (ignored for weight)')
PARSER.add_argument('--lambda_fnat_reg', type=float, default=0.0,
                    help='experiment 2: weight of the absolute-fnat SmoothL1 regression auxiliary on '
                         'sigmoid(interface logit), INSIDE the interface objective '
                         '(L_interface = L_softrank + lambda_fnat_reg * L_reg). 0 = off (experiment 1).')
PARSER.add_argument('--fnat_reg_beta', type=float, default=0.1,
                    help='SmoothL1 transition point for the fnat regression. fnat and sigmoid(s) are '
                         'both in [0,1] so |error|<=1 always: beta=1.0 (torch default) never leaves '
                         'the quadratic branch and equals 0.5*MSE. Default 0.1 keeps a linear branch.')
PARSER.add_argument('--interface_fnat_cutoff', type=float, default=0.5,
                    help='DEPRECATED and unused by the training objective (the absolute-cutoff interface '
                         'SML was replaced by the within-target soft RankNet). Kept so older launch '
                         'scripts keep parsing; only analysis code may read it.')

# ── Ordinal H3 lDDT auxiliary loss (finetune only, v1) ──
PARSER.add_argument('--use_ord_aux_loss', action='store_true', default=False,
                    help='Enable ordinal H3 lDDT auxiliary head loss in finetune (run_epoch_dpo)')
PARSER.add_argument('--ord_cutoff_mode', type=str, default='fixed',
                    choices=['rmsd_quad', 'fixed'],
                    help='Ordinal lDDT cutoffs: fixed (0.6/0.8/0.9) or legacy rmsd_quad')
PARSER.add_argument('--ord_rmsd_a', type=float, default=2.0,
                    help='Legacy RMSD (Å) anchor for ordinal tier a')
PARSER.add_argument('--ord_rmsd_b', type=float, default=1.5,
                    help='Legacy RMSD (Å) anchor for ordinal tier b')
PARSER.add_argument('--ord_rmsd_c', type=float, default=0.8,
                    help='Legacy RMSD (Å) anchor for ordinal tier c')
PARSER.add_argument('--ord_cutoff_a', type=float, default=0.6,
                    help='lDDT threshold for p60 = P(lDDT >= cutoff)')
PARSER.add_argument('--ord_cutoff_b', type=float, default=0.8,
                    help='lDDT threshold for p80 = P(lDDT >= cutoff)')
PARSER.add_argument('--ord_cutoff_c', type=float, default=0.9,
                    help='lDDT threshold for p90 = P(lDDT >= cutoff)')
PARSER.add_argument('--lambda_aux_ord', type=float, default=0.1,
                    help='Weight for ordinal BCE auxiliary loss')
PARSER.add_argument('--lambda_mono', type=float, default=0.01,
                    help='Weight for monotonic probability penalty')

model = PARSER.add_argument_group("Model architecture")
model.add_argument('--num_layers', type=int, default=4,
                   help='Number of stacked Transformer layers')
model.add_argument('--num_heads', type=int, default=4,
                   help='Number of heads in self-attention')
model.add_argument('--channels_div', type=int, default=2,
                   help='Channels division before feeding to attention layer')
model.add_argument('--pooling', type=str, default=None, const=None, nargs='?', choices=['max', 'avg'],
                   help='Type of graph pooling')
model.add_argument('--norm', type=str2bool, nargs='?', const=True, default=True,
                   help='Apply a normalization layer after each attention block')
model.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=True,
                   help='Apply layer normalization between MLP layers')
model.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                   help='If true, use lower-memory fused ops where supported')
model.add_argument('--num_degrees',
                   help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                   type=int, default=2)
model.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
