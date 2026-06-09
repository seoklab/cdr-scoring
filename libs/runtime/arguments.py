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

from model import Sujin_with_SE3 
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
PARSER.add_argument('--embedded_node_dim', type=float, nargs='?', const=True, default=32, help='See model/transformer.py')
PARSER.add_argument('--embedded_edge_dim', type=float, nargs='?', const=True, default=32, help='See model/transformer.py')
PARSER.add_argument('--readout', type=str, nargs='?', const=True, default='sum', help='Readout Type')
#PARSER.add_argument('--num_layers', type=float, nargs='?', const=True, default=4, help='Number of Layers')
PARSER.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False, help='Use Automatic Mixed Precision')
PARSER.add_argument('--gradient_clip', type=float, default=None, help='Clipping of the gradient norms')
PARSER.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation')
PARSER.add_argument('--ckpt_interval', type=int, default=-1, help='Save a checkpoint every N epochs')
PARSER.add_argument('--eval_interval', dest='eval_interval', type=int, default=2,
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

# ── Compactness loss (step 3) ──
PARSER.add_argument('--lambda_compactness', type=float, default=0.0,
                    help='Weight for xtal-A compactness band loss (0 = disabled)')
PARSER.add_argument('--delta_A', type=float, default=1.0,
                    help='Upper bound for xtal-to-A score gap in compactness loss')

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

Sujin_with_SE3.add_argparse_args(PARSER)
