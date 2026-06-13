#!/usr/bin/env python
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from data_loading.constants import MAX_NUM_ATOM  # noqa: E402
from data_loading.structure_graph_dataset import (  # noqa: E402
    DEFAULT_CDR_RANGES,
    attach_af3_ranking,
    discover_structure_samples,
    discover_structure_samples_from_info,
    load_af3_ranking_scores,
    make_structure_inference_loader,
    make_structure_graph_loader,
    resolve_target_native_path,
    StructureSample,
)
from evaluation.loop_metrics import compute_loop_metrics_from_structures, load_structure  # noqa: E402
from evaluation.docking_metrics import compute_dockq_style_metrics_from_structures  # noqa: E402
from model.fiber import Fiber  # noqa: E402
from model.transformer import Sujin_with_SE3, Sujin_with_SE3_allatom  # noqa: E402


CDR_NAMES = ("H1", "H2", "H3", "L1", "L2", "L3")


DEFAULT_INPUT_DIR = (
    "/home/sujin/DB/h3-loop-modeling/ab_ag/"
    "3_h3_benchmark_after210930/22_af3/7b5g_H_X_A"
)
DEFAULT_INFO_PKL = (
    "/home/sujin/DB/h3-loop-modeling/ab_ag/"
    "3_h3_benchmark_after210930/0_info/info.pkl"
)
DEFAULT_AF3_ROOT = (
    "/home/sujin/DB/h3-loop-modeling/ab_ag/"
    "3_h3_benchmark_after210930/22_af3"
)
DEFAULT_NATIVE_ROOT = (
    "/home/sujin/DB/h3-loop-modeling/ab_ag/"
    "3_h3_benchmark_after210930/02_xtal_chothia_w_ag_renum"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score PDB/mmCIF structures with the CDR scoring model."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-info-pkl", default=None)
    parser.add_argument("--info-target-key", default="list")
    parser.add_argument("--af3-root", default=DEFAULT_AF3_ROOT)
    parser.add_argument("--native-root", default=DEFAULT_NATIVE_ROOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", default="cdr_structure_scores.csv")
    parser.add_argument("--output-pkl", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=100,
        help="Emit a batch progress log every N batches. Set 0 to disable batch logs.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Run model scoring only and skip structural metric calculation.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--all-atom", action="store_true")
    parser.add_argument("--nodewise-score", action="store_true")
    parser.add_argument("--dist-cutoff", type=float, default=10.0)
    parser.add_argument("--max-neighbors", type=int, default=60)
    parser.add_argument("--cdr-ranges", default=DEFAULT_CDR_RANGES)

    parser.add_argument("--embedded-node-dim", type=int, default=32)
    parser.add_argument("--embedded-edge-dim", type=int, default=32)
    parser.add_argument("--num-degrees", type=int, default=2)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--channels-div", type=int, default=2)
    parser.add_argument("--norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-memory", action="store_true")
    return parser.parse_args()


def build_model(args):
    node_l1_dim = MAX_NUM_ATOM if args.all_atom else 4
    kwargs = {
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "channels_div": args.channels_div,
        "norm": args.norm,
        "use_layer_norm": args.use_layer_norm,
        "low_memory": args.low_memory,
        "tensor_cores": bool(args.amp and torch.cuda.is_available()),
    }
    if args.all_atom:
        return Sujin_with_SE3_allatom(
            fiber_in=Fiber({0: args.embedded_node_dim, 1: node_l1_dim}),
            fiber_out=Fiber({0: args.num_degrees * args.num_channels, 1: 20}),
            fiber_edge=Fiber({0: args.embedded_edge_dim, 1: 1}),
            num_degrees=args.num_degrees,
            num_channels=args.num_channels,
            **kwargs,
        )
    return Sujin_with_SE3(
        fiber_in=Fiber({0: args.embedded_node_dim, 1: node_l1_dim}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels, 1: 20}),
        fiber_edge=Fiber({0: args.embedded_edge_dim, 1: 1}),
        use_nodewise_score=args.nodewise_score,
        num_degrees=args.num_degrees,
        num_channels=args.num_channels,
        **kwargs,
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def write_csv(path, rows):
    fieldnames = [
        "target_id",
        "rank_by_model",
        "pred_score",
        "sample_id",
        "seed",
        "sample",
        "af3_rank",
        "af3_ranking_score",
        "native_path",
        "loop_rmsd",
        "loop_lddt",
        "irmsd",
        "lrmsd",
        *[f"{cdr}_loop_rmsd" for cdr in CDR_NAMES],
        *[f"{cdr}_loop_lddt" for cdr in CDR_NAMES],
        "n_nodes",
        "n_edges",
        "path",
    ]
    with Path(path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.input_info_pkl:
        samples = discover_structure_samples_from_info(
            args.input_info_pkl,
            af3_root=args.af3_root,
            native_root=args.native_root,
            target_key=args.info_target_key,
        )
        loader = make_structure_graph_loader(
            samples,
            batch_size=args.batch_size,
            cdr_ranges=args.cdr_ranges,
            dist_cutoff=args.dist_cutoff,
            max_neighbors=args.max_neighbors,
            use_all_atom=args.all_atom,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            shuffle=False,
        )
    else:
        input_dir = Path(args.input_dir)
        samples = attach_af3_ranking(
            discover_structure_samples(input_dir),
            load_af3_ranking_scores(input_dir),
        )
        native_path = None
        try:
            native_path = resolve_target_native_path(input_dir.name, args.native_root)
        except FileNotFoundError:
            native_path = None
        if native_path is not None:
            samples = [
                StructureSample(
                    path=sample.path,
                    sample_id=sample.sample_id,
                    target_id=input_dir.name,
                    native_path=native_path,
                    method=sample.method,
                    seed=sample.seed,
                    sample=sample.sample,
                    ranking=sample.ranking,
                    ranking_score=sample.ranking_score,
                    label=sample.label,
                )
                for sample in samples
            ]
        loader = make_structure_graph_loader(
            samples,
            batch_size=args.batch_size,
            cdr_ranges=args.cdr_ranges,
            dist_cutoff=args.dist_cutoff,
            max_neighbors=args.max_neighbors,
            use_all_atom=args.all_atom,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            shuffle=False,
        )
    if len(loader.dataset) == 0:
        raise ValueError("No input structures found for inference")

    target_ids = sorted(
        {
            getattr(sample, "target_id", "") or Path(sample.path).parent.name
            for sample in loader.dataset.samples
        }
    )
    print("[inference] starting structure scoring", flush=True)
    print(
        f"[inference] mode={'info-pkl' if args.input_info_pkl else 'single-dir'} "
        f"device={device} batch_size={args.batch_size} num_workers={args.num_workers} "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'}",
        flush=True,
    )
    if args.input_info_pkl:
        print(f"[inference] info_pkl={args.input_info_pkl}", flush=True)
        print(f"[inference] af3_root={args.af3_root}", flush=True)
        print(f"[inference] native_root={args.native_root}", flush=True)
    else:
        print(f"[inference] input_dir={args.input_dir}", flush=True)
        print(f"[inference] native_root={args.native_root}", flush=True)
    print(f"[inference] checkpoint={args.checkpoint}", flush=True)
    print(f"[inference] output_csv={args.output_csv}", flush=True)
    if args.output_pkl:
        print(f"[inference] output_pkl={args.output_pkl}", flush=True)
    print(
        f"[inference] cdr_ranges={args.cdr_ranges} "
        f"dist_cutoff={args.dist_cutoff} max_neighbors={args.max_neighbors} "
        f"all_atom={args.all_atom} low_memory={args.low_memory} "
        f"score_only={args.score_only}",
        flush=True,
    )
    print(
        f"[inference] targets={len(target_ids)} structures={len(loader.dataset)} "
        f"expected_batches={len(loader)}",
        flush=True,
    )

    model = build_model(args).to(device)
    missing, unexpected = load_checkpoint(model, args.checkpoint, device)
    if missing:
        print(f"[checkpoint] missing keys: {missing}", flush=True)
    if unexpected:
        print(f"[checkpoint] unexpected keys: {unexpected}", flush=True)
    model.eval()

    rows = []
    native_structure_cache = {}
    with torch.inference_mode():
        for batch_idx, (batched_graph, metas) in enumerate(
            tqdm(loader, desc="Inference", unit="batch"),
            start=1,
        ):
            if args.log_every_batches and (
                batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == len(loader)
            ):
                batch_targets = sorted(
                    {meta.get("target_id", "") for meta in metas if meta.get("target_id")}
                )
                print(
                    f"[inference] batch={batch_idx}/{len(loader)} batch_size={len(metas)} "
                    f"targets={batch_targets[:5]}{'...' if len(batch_targets) > 5 else ''}",
                    flush=True,
                )
            batched_graph = batched_graph.to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(args.amp and device.type == "cuda"),
            ):
                pred = model(batched_graph)["out"].detach().float().cpu().tolist()
            for meta, score in zip(metas, pred):
                native_path = meta.get("native_path")
                loop_metrics = None
                dock_metrics = None
                if native_path and not args.score_only:
                    native_structure = native_structure_cache.get(native_path)
                    if native_structure is None:
                        native_structure = load_structure(native_path)
                        native_structure_cache[native_path] = native_structure
                    model_structure = load_structure(meta["path"])
                    loop_metrics = compute_loop_metrics_from_structures(
                        native_structure,
                        model_structure,
                        native_path=native_path,
                        model_path=meta["path"],
                        rmsd_atom_type="backbone",
                    )
                    dock_metrics = compute_dockq_style_metrics_from_structures(
                        native_structure,
                        model_structure,
                        native_path=native_path,
                        model_path=meta["path"],
                    )
                rows.append(
                    {
                        "target_id": meta.get("target_id", Path(meta["path"]).parent.name),
                        "pred_score": float(score),
                        "sample_id": meta["sample_id"],
                        "seed": meta["seed"],
                        "sample": meta["sample"],
                        "af3_rank": meta["ranking"],
                        "af3_ranking_score": meta["ranking_score"],
                        "native_path": native_path,
                        "loop_rmsd": float("nan") if loop_metrics is None else loop_metrics.loop_rmsd,
                        "loop_lddt": float("nan") if loop_metrics is None else loop_metrics.loop_lddt,
                        "irmsd": float("nan") if dock_metrics is None else dock_metrics.irmsd,
                        "lrmsd": float("nan") if dock_metrics is None else dock_metrics.lrmsd,
                        **{
                            f"{cdr}_loop_rmsd": (
                                float("nan")
                                if loop_metrics is None
                                else loop_metrics.per_cdr_rmsd.get(cdr, float("nan"))
                            )
                            for cdr in CDR_NAMES
                        },
                        **{
                            f"{cdr}_loop_lddt": (
                                float("nan")
                                if loop_metrics is None
                                else loop_metrics.per_cdr_lddt.get(cdr, float("nan"))
                            )
                            for cdr in CDR_NAMES
                        },
                        "n_nodes": meta["n_nodes"],
                        "n_edges": meta["n_edges"],
                        "path": meta["path"],
                    }
                )

    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["target_id"]].append(row)
    rows = []
    for target_id in sorted(grouped_rows):
        target_rows = sorted(grouped_rows[target_id], key=lambda r: r["pred_score"])
        for rank, row in enumerate(target_rows, start=1):
            row["rank_by_model"] = rank
        rows.extend(target_rows)

    write_csv(args.output_csv, rows)
    if args.output_pkl:
        with Path(args.output_pkl).open("wb") as fp:
            pickle.dump(
                {
                    "input_dir": args.input_dir,
                    "input_info_pkl": args.input_info_pkl,
                    "cdr_ranges": args.cdr_ranges,
                    "rows": rows,
                },
                fp,
            )

    print(f"Scored {len(rows)} structures")
    print(f"Wrote CSV: {args.output_csv}")
    if args.output_pkl:
        print(f"Wrote PKL: {args.output_pkl}")
    print("Top 5:")
    for row in rows[:5]:
        print(
            f"  #{row['rank_by_model']:>3} score={row['pred_score']:.6f} "
            f"{row['sample_id']} path={row['path']}"
        )


if __name__ == "__main__":
    main()
