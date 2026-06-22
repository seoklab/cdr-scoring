#!/usr/bin/env python3
"""Prepare 6-CDR pertMD input files for GalaxyRefine.

This script rewrites Chothia-numbered antibody-antigen PDB files into a
GalaxyRefine-friendly residue numbering while preserving chain IDs. It writes
the renumbered PDB, FASTA, ULR file, a minimal residue map, and an optional
batch script. It does not run GalaxyRefine.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BASE_DIR = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag")
DEFAULT_SOURCE_DIR = BASE_DIR / "00_crystal"
DEFAULT_FASTA_ROOT = BASE_DIR / "21_commat_xtal"
DEFAULT_INFO_DIR = BASE_DIR / "0_info"
DEFAULT_OUTPUT_DIR = BASE_DIR / "41_pertmd_cdr_input"
DEFAULT_GALAXY_REFINE = Path("/home/sujin/archive/loop/GalaxyRefine_multi.py")
DEFAULT_SCHEDULE = Path("/home/sujin/archive/loop/sch_t500_c300")
DEFAULT_SLURM_NODELIST = "star[005-018,35-42]" #star[001-018,020,024,026,028-029,031,033,036,040]
CDR_RANGES = {
    "H1": ("H", 26, 32),
    "H2": ("H", 52, 56),
    "H3": ("H", 95, 102),
    "L1": ("L", 24, 34),
    "L2": ("L", 50, 56),
    "L3": ("L", 89, 97),
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


@dataclass(frozen=True)
class Residue:
    chain: str
    resseq: str
    icode: str
    resname: str
    atom_lines: Tuple[str, ...]

    @property
    def orig_resid(self) -> str:
        return f"{self.resseq.strip()}{self.icode.strip()}"

    @property
    def chothia_number(self) -> Optional[int]:
        match = re.match(r"^-?\d+", self.orig_resid)
        return int(match.group(0)) if match else None

    @property
    def aa(self) -> str:
        try:
            return AA3_TO_1[self.resname]
        except KeyError as exc:
            raise ValueError(f"Unsupported residue {self.resname} at {self.chain}:{self.orig_resid}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--fasta-root", type=Path, default=DEFAULT_FASTA_ROOT)
    parser.add_argument("--info-dir", type=Path, default=DEFAULT_INFO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--galaxy-refine", type=Path, default=DEFAULT_GALAXY_REFINE)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--target", action="append", default=None,
                        help="Target ID without .pdb. Can be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sbatch", action="store_true",
                        help="Submit each generated target script with sbatch.")
    parser.add_argument("--partition", default="normal.q")
    parser.add_argument("--nodelist", default=DEFAULT_SLURM_NODELIST)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--nice", type=int, default=1000000)
    return parser.parse_args()


def read_fasta(path: Path) -> Dict[str, str]:
    sequences: Dict[str, List[str]] = {}
    current_id: Optional[str] = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                if not current_id:
                    raise ValueError(f"Empty FASTA record ID in {path}")
                sequences[current_id] = []
            elif current_id is None:
                raise ValueError(f"FASTA sequence appears before a header in {path}")
            else:
                sequences[current_id].append(line)
    return {chain: "".join(parts).upper() for chain, parts in sequences.items()}


def parse_pdb_residues(path: Path) -> Dict[str, List[Residue]]:
    residues: Dict[Tuple[str, str, str], List[str]] = {}
    resnames: Dict[Tuple[str, str, str], str] = {}
    order: List[Tuple[str, str, str]] = []

    with path.open() as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            if resname not in AA3_TO_1:
                continue
            chain = line[21]
            resseq = line[22:26]
            icode = line[26]
            key = (chain, resseq, icode)
            if key not in residues:
                order.append(key)
                residues[key] = []
                resnames[key] = resname
            residues[key].append(line)

    by_chain: Dict[str, List[Residue]] = {}
    for key in order:
        chain, resseq, icode = key
        residue = Residue(
            chain=chain,
            resseq=resseq,
            icode=icode,
            resname=resnames[key],
            atom_lines=tuple(residues[key]),
        )
        by_chain.setdefault(chain, []).append(residue)
    return by_chain


def align_observed_to_fasta(
    fasta_seq: str,
    observed_seq: str,
) -> Tuple[Dict[int, int], List[int], List[Tuple[int, int, str, str]], List[int]]:
    """Return observed-index to FASTA-index mapping, both 1-based."""
    n = len(fasta_seq)
    m = len(observed_seq)
    match_score = 2
    mismatch_score = -3
    gap_score = -2

    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap_score
        trace[i][0] = "up"
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap_score
        trace[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (
                match_score if fasta_seq[i - 1] == observed_seq[j - 1] else mismatch_score
            )
            up = score[i - 1][j] + gap_score
            left = score[i][j - 1] + gap_score
            best = max(diag, up, left)
            score[i][j] = best
            if best == diag:
                trace[i][j] = "diag"
            elif best == up:
                trace[i][j] = "up"
            else:
                trace[i][j] = "left"

    obs_to_fasta: Dict[int, int] = {}
    missing_fasta: List[int] = []
    mismatches: List[Tuple[int, int, str, str]] = []
    extra_observed: List[int] = []

    i, j = n, m
    while i > 0 or j > 0:
        step = trace[i][j]
        if step == "diag":
            obs_to_fasta[j] = i
            if fasta_seq[i - 1] != observed_seq[j - 1]:
                mismatches.append((i, j, fasta_seq[i - 1], observed_seq[j - 1]))
            i -= 1
            j -= 1
        elif step == "up":
            missing_fasta.append(i)
            i -= 1
        elif step == "left":
            extra_observed.append(j)
            j -= 1
        else:
            raise RuntimeError("Invalid alignment traceback")

    missing_fasta.reverse()
    mismatches.reverse()
    extra_observed.reverse()
    return obs_to_fasta, missing_fasta, mismatches, extra_observed


def contiguous_blocks(values: Iterable[int]) -> List[List[int]]:
    sorted_values = sorted(set(values))
    if not sorted_values:
        return []
    blocks: List[List[int]] = []
    start = prev = sorted_values[0]
    for value in sorted_values[1:]:
        if value == prev + 1:
            prev = value
            continue
        blocks.append([start, prev])
        start = prev = value
    blocks.append([start, prev])
    return blocks


def rewrite_atom_line(line: str, atom_serial: int, galaxy_resid: int) -> str:
    if galaxy_resid > 9999:
        raise ValueError(f"PDB residue number exceeds 9999: {galaxy_resid}")
    if atom_serial > 99999:
        raise ValueError(f"PDB atom serial exceeds 99999: {atom_serial}")
    padded = line.rstrip("\n")
    if len(padded) < 27:
        padded = padded.ljust(27)
    return f"{padded[:6]}{atom_serial:5d}{padded[11:22]}{galaxy_resid:4d} {padded[27:]}\n"


def target_ids(source_dir: Path, requested: Optional[Sequence[str]], limit: Optional[int]) -> List[str]:
    if requested:
        targets = list(requested)
    else:
        targets = sorted(path.stem for path in source_dir.glob("*.pdb"))
    return targets[:limit] if limit is not None else targets


def slurm_job_name(target: str) -> str:
    fields = target.split("_")
    if len(fields) >= 2:
        return "_".join(fields[:2])
    return target


def build_sbatch_script(
    target: str,
    target_dir: Path,
    renum_pdb: Path,
    renum_fasta: Path,
    ulr_path: Path,
    galaxy_refine: Path,
    schedule: Path,
    partition: str,
    nodelist: str,
    cpus_per_task: int,
    nice: int,
) -> str:
    job_name = slurm_job_name(target)
    log_path = target_dir / f"{job_name}.log.q"
    lines = [
        "#!/bin/sh",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {partition}",
        f"#SBATCH --nodelist={nodelist}",
        "#SBATCH -n 1",
        "#SBATCH -N 1",
        f"#SBATCH -c {cpus_per_task}",
        f"#SBATCH --nice={nice}",
        f"#SBATCH -o {log_path}",
        f"cd {target_dir}",
        (
            f"python -u {galaxy_refine} "
            f"-p {renum_pdb} "
            f"-s {renum_fasta} "
            f"-u {ulr_path} "
            f"-sch {schedule}"
        ),
    ]
    return "\n".join(lines) + "\n"


def prepare_target(
    target: str,
    source_dir: Path,
    fasta_root: Path,
    output_dir: Path,
    info_dir: Path,
    galaxy_refine: Path,
    schedule: Path,
    partition: str,
    nodelist: str,
    cpus_per_task: int,
    nice: int,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, object]:
    source_pdb = source_dir / f"{target}.pdb"
    fasta_path = fasta_root / target / "output.fa"
    target_dir = output_dir / target
    renum_pdb = target_dir / "input.pdb"
    renum_fasta = target_dir / "input.fa"
    ulr_path = target_dir / f"{target}.ulr"
    script_path = target_dir / f"{slurm_job_name(target)}.sh"
    map_path = info_dir / f"{target}.pertmd_cdr_map.json"

    if not source_pdb.is_file():
        raise FileNotFoundError(source_pdb)
    if not fasta_path.is_file():
        raise FileNotFoundError(fasta_path)
    if not overwrite:
        for path in (renum_pdb, renum_fasta, ulr_path, script_path, map_path):
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    fasta = read_fasta(fasta_path)
    residues_by_chain = parse_pdb_residues(source_pdb)
    chain_order = list(fasta.keys())

    residue_map: List[List[object]] = []
    missing: List[List[object]] = []
    atom_records: List[Tuple[Residue, int]] = []
    cdr_residues: Dict[str, List[int]] = {loop: [] for loop in CDR_RANGES}

    offset = 0
    for chain in chain_order:
        if chain not in residues_by_chain:
            raise ValueError(f"{target}: chain {chain} is in FASTA but not in PDB")
        residues = residues_by_chain[chain]
        observed_seq = "".join(residue.aa for residue in residues)
        obs_to_fasta, missing_fasta, mismatches, extra_observed = align_observed_to_fasta(
            fasta[chain], observed_seq
        )
        if mismatches:
            details = ", ".join(
                f"fasta#{fi}={faa}/pdb#{oi}={oaa}" for fi, oi, faa, oaa in mismatches[:10]
            )
            raise ValueError(f"{target}: sequence mismatch in chain {chain}: {details}")
        if extra_observed:
            raise ValueError(
                f"{target}: observed residues in chain {chain} do not align to FASTA: {extra_observed[:10]}"
            )

        for fasta_index in missing_fasta:
            missing.append([chain, fasta_index, offset + fasta_index])

        for obs_index, residue in enumerate(residues, start=1):
            fasta_index = obs_to_fasta[obs_index]
            galaxy_resid = offset + fasta_index
            residue_map.append([chain, residue.orig_resid, galaxy_resid])
            atom_records.append((residue, galaxy_resid))

            for loop_name, (loop_chain, start, end) in CDR_RANGES.items():
                if chain != loop_chain:
                    continue
                number = residue.chothia_number
                if number is not None and start <= number <= end:
                    cdr_residues[loop_name].append(galaxy_resid)

        offset += len(fasta[chain])

    cdr_blocks = {loop_name: contiguous_blocks(ids) for loop_name, ids in cdr_residues.items()}
    ulr_lines: List[str] = []
    ulr_no = 1
    for loop_name in ("H1", "H2", "H3", "L1", "L2", "L3"):
        chain = CDR_RANGES[loop_name][0]
        for start, end in cdr_blocks[loop_name]:
            ulr_lines.append(f"{ulr_no} L 1.0 {start}-{end} {chain}\n")
            ulr_no += 1

    if dry_run:
        return {
            "target": target,
            "chains": chain_order,
            "residues": len(residue_map),
            "missing": len(missing),
            "ulr_blocks": sum(len(blocks) for blocks in cdr_blocks.values()),
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    atom_serial = 1
    with renum_pdb.open("w") as handle:
        previous_chain: Optional[str] = None
        for residue, galaxy_resid in atom_records:
            if previous_chain is not None and residue.chain != previous_chain:
                handle.write("TER\n")
            for line in residue.atom_lines:
                handle.write(rewrite_atom_line(line, atom_serial, galaxy_resid))
                atom_serial += 1
            previous_chain = residue.chain
        if previous_chain is not None:
            handle.write("TER\n")
        handle.write("END\n")

    shutil.copyfile(fasta_path, renum_fasta)
    with ulr_path.open("w") as handle:
        handle.writelines(ulr_lines)

    script = build_sbatch_script(
        target=target,
        target_dir=target_dir,
        renum_pdb=renum_pdb,
        renum_fasta=renum_fasta,
        ulr_path=ulr_path,
        galaxy_refine=galaxy_refine,
        schedule=schedule,
        partition=partition,
        nodelist=nodelist,
        cpus_per_task=cpus_per_task,
        nice=nice,
    )
    with script_path.open("w") as handle:
        handle.write(script)
    script_path.chmod(0o755)

    mapping = {
        "target": target,
        "source_pdb": str(source_pdb),
        "renum_pdb": str(renum_pdb),
        "fasta": str(renum_fasta),
        "chains": chain_order,
        "map": residue_map,
        "missing": missing,
        "cdr": cdr_blocks,
    }
    with map_path.open("w") as handle:
        json.dump(mapping, handle, indent=2)
        handle.write("\n")

    return {
        "target": target,
        "chains": chain_order,
        "residues": len(residue_map),
        "missing": len(missing),
        "ulr_blocks": sum(len(blocks) for blocks in cdr_blocks.values()),
        "renum_pdb": str(renum_pdb),
        "ulr": str(ulr_path),
        "map": str(map_path),
        "script": str(script_path),
    }


def main() -> int:
    args = parse_args()
    summaries = []
    for target in target_ids(args.source_dir, args.target, args.limit):
        summary = prepare_target(
            target=target,
            source_dir=args.source_dir,
            fasta_root=args.fasta_root,
            output_dir=args.output_dir,
            info_dir=args.info_dir,
            galaxy_refine=args.galaxy_refine,
            schedule=args.schedule,
            partition=args.partition,
            nodelist=args.nodelist,
            cpus_per_task=args.cpus_per_task,
            nice=args.nice,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if args.sbatch and not args.dry_run:
            subprocess.run(["sbatch", summary["script"]], check=True)
        summaries.append(summary)
        print(
            "{target}\tchains={chains}\tresidues={residues}\tmissing={missing}\tulr_blocks={ulr_blocks}".format(
                **summary
            )
        )
    print(f"Prepared {len(summaries)} target(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
