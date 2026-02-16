#!/usr/bin/env python3
"""
Parallel bactermfinder genome scan.

- Accepts FASTA (.fa/.fna/.fasta/.fas) and GenBank (.gb/.gbk/.gbff/.genbank), optionally gzipped (.gz).
- If input is GenBank, converts to FASTA inside the per-genome work directory.
- Runs per-genome pipeline in parallel (processes) across multiple genome files/directories/globs.
- Writes logs to: <out-root>/<genome-stem>/<log_filename> (one log per genome).
- Optional resume mode: skips genomes whose log indicates successful completion.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tensorflow.keras.models import load_model
from tqdm import tqdm


@contextlib.contextmanager
def redirect_fds_to_file(log_path: Path, mode: str = "a") -> Iterator[TextIO]:
    """
    Redirect OS-level stdout/stderr (fd 1/2) AND sys.stdout/sys.stderr to a file.
    Captures subprocess output and native-library logs (e.g., TensorFlow C++ messages).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, mode, buffering=1) as logh:
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        try:
            os.dup2(logh.fileno(), 1)
            os.dup2(logh.fileno(), 2)

            py_out = open(1, "w", buffering=1, closefd=False)
            py_err = open(2, "w", buffering=1, closefd=False)
            saved_py_out, saved_py_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = py_out, py_err
            try:
                yield logh
            finally:
                sys.stdout, sys.stderr = saved_py_out, saved_py_err
                py_out.close()
                py_err.close()
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


# ----------------------------
# IO / format handling
# ----------------------------

FASTA_EXTS = (".fa", ".fna", ".fasta", ".fas")
GB_EXTS = (".gb", ".gbk", ".gbff", ".genbank")


def _suffix_without_gz(p: Path) -> str:
    name = p.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).suffix


def is_fasta_path(p: Path) -> bool:
    return _suffix_without_gz(p) in FASTA_EXTS


def is_genbank_path(p: Path) -> bool:
    return _suffix_without_gz(p) in GB_EXTS


def _open_maybe_gzip(path: Path) -> IO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def ensure_fasta(genome_path: Path, work_dir: Path) -> Path:
    """
    Ensure we have a plain-text FASTA file path usable by SeqIO.parse(filename, "fasta").
    - FASTA (not gz): returned as-is.
    - FASTA.gz: decompressed to work_dir and returned.
    - GenBank/gbff (gz or not): converted to FASTA in work_dir and returned.
    """
    genome_path = genome_path.resolve()

    if is_fasta_path(genome_path) and not genome_path.name.lower().endswith(".gz"):
        return genome_path

    work_dir.mkdir(parents=True, exist_ok=True)

    if is_fasta_path(genome_path) and genome_path.name.lower().endswith(".gz"):
        out_fa = work_dir / re.sub(r"\.gz$", "", genome_path.name, flags=re.IGNORECASE)
        with gzip.open(genome_path, "rt") as hin:
            out_fa.write_text(hin.read())
        return out_fa

    if not is_genbank_path(genome_path):
        raise ValueError(f"Unrecognized genome format: {genome_path}")

    base = re.sub(r"\.gz$", "", genome_path.name, flags=re.IGNORECASE)
    base = re.sub(r"\.(gbff|gbk|gb|genbank)$", "", base, flags=re.IGNORECASE)
    out_fa = work_dir / f"{base}.fna"

    with _open_maybe_gzip(genome_path) as handle:
        records = list(SeqIO.parse(handle, "genbank"))

    if not records:
        raise ValueError(f"No records parsed from GenBank file: {genome_path}")

    SeqIO.write(records, str(out_fa), "fasta")
    return out_fa


def find_genomes(inputs: List[str], recursive: bool = False) -> List[Path]:
    """
    Accept: file paths, directories, globs. Returns unique genome paths (FASTA/GenBank, optionally gzipped).
    """
    def is_genome_file(p: Path) -> bool:
        return is_fasta_path(p) or is_genbank_path(p)

    paths: List[Path] = []
    for s in inputs:
        p = Path(s)

        if p.is_dir():
            iterator = p.rglob("*") if recursive else p.iterdir()
            for child in sorted(iterator):
                if child.is_file() and is_genome_file(child):
                    paths.append(child)
            continue

        matches = sorted(Path().glob(s))
        if matches:
            for m in matches:
                if m.is_file() and is_genome_file(m):
                    paths.append(m)
        else:
            if p.exists() and p.is_file() and is_genome_file(p):
                paths.append(p)

    uniq: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


# ----------------------------
# Core helpers
# ----------------------------

def extract_sliding_windows(
    ref_genome_file: str,
    window_size: int,
    step_size: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Extract sliding windows from all FASTA records, including reverse complement."""
    ref_genome = SeqIO.to_dict(SeqIO.parse(ref_genome_file, "fasta"))

    rows: List[Tuple[int, str, int, int, str, str]] = []
    u_id = 0

    for seq_id, record in ref_genome.items():
        seq = record.seq
        seq_len = len(seq)

        it = range(0, seq_len - window_size + 1, step_size)
        it = tqdm(it, desc=f"Windows {seq_id}", leave=False, disable=not show_progress)

        for i in it:
            u_id += 1
            window_start = i
            window_end = i + window_size
            window_seq = str(seq[window_start:window_end])
            rows.append((u_id, seq_id, window_start, window_end, window_seq, "+"))

            revcomp = str(Seq(window_seq).reverse_complement())
            rows.append((u_id, seq_id, window_start, window_end, revcomp, "-"))

    return pd.DataFrame(rows, columns=["u_id", "seq_id", "start", "end", "seq", "strand"])


def df_to_fasta(df: pd.DataFrame, fasta_path: Path, train_stat: str = "testing") -> None:
    """Write sequences to FASTA in the header format expected by iLearnPlus."""
    lines: List[str] = []
    for row in df.itertuples(index=False):
        head = f"{row.u_id}_{row.seq_id}_{row.start}_{row.end}_{row.strand}"
        lines.append(f">{head}|-1|{train_stat}\n{row.seq}\n")
    fasta_path.write_text("".join(lines))


def read_csv_low(file_tag: str, data_path: Path, input_dim: Tuple[int, int]) -> Tuple[np.ndarray, pd.Series]:
    """Load one iLearnPlus feature CSV (one batch) with dtypes and reshape for the model."""
    df_test = pd.read_csv(data_path, nrows=100)
    float_cols = [c for c in df_test.columns if df_test[c].dtype == "float64"]

    if file_tag in ("PS2.csv", "binary.csv"):
        dtype_cols = {c: np.int8 for c in float_cols}
    else:
        dtype_cols = {c: np.float32 for c in float_cols}

    x = pd.read_csv(data_path, dtype=dtype_cols)
    sample_names = x["SampleName"]
    x = x.drop(columns=["SampleName", "label"])

    reshaper_dim = (len(x), input_dim[0], input_dim[1])
    x_arr = x.values.reshape(reshaper_dim)
    return x_arr, sample_names


def gpu_available() -> bool:
    import tensorflow as tf
    return len(tf.config.list_physical_devices("GPU")) > 0


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    input_dim: Tuple[int, int]
    filename: str


def predict_one_embedding(
    embedding: ModelSpec,
    work_dir: Path,
    model_dir: Path,
    use_gpu: bool,
) -> pd.DataFrame:
    """Predict probabilities for all batches for one embedding."""
    import tensorflow as tf

    out_dir = work_dir / "output_sample"
    emb_prefix = embedding.tag.replace(".csv", "")

    model_path = model_dir / embedding.filename
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    print(f"[{work_dir.name}] Loading model: {model_path.name}", flush=True)
    model = load_model(model_path)

    batch_files = sorted(
        out_dir.glob(f"{emb_prefix}-*.csv"),
        key=lambda p: int(re.search(r"-(\d+)\.csv$", p.name).group(1)),
    )
    if not batch_files:
        raise FileNotFoundError(f"No batch files found for {emb_prefix} in {out_dir}")

    frames: List[pd.DataFrame] = []
    device_name = "/GPU:0" if use_gpu else "/CPU:0"

    for bf in batch_files:
        print(f"[{work_dir.name}] Predicting {emb_prefix} batch: {bf.name}", flush=True)
        x, sample_info = read_csv_low(embedding.tag, bf, embedding.input_dim)
        with tf.device(device_name):
            y_pred = model.predict(x, verbose=0)

        frames.append(pd.DataFrame({
            "SampleName": sample_info.values,
            f"probability_{emb_prefix}": y_pred.reshape(-1),
        }))

    return pd.concat(frames, ignore_index=True)


def run_ilearnplus_fileprocessing(
    fileprocessing_py: Path,
    fasta_path: Path,
    batch_size: int,
    threads: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(fileprocessing_py),
        str(fasta_path),
        str(batch_size),
        str(threads),
        str(out_dir),
    ]
    print(f"Running iLearnPlus: {' '.join(cmd)}", flush=True)
    
    # Explicitly pass sys.stdout/stderr (which are already redirected) to avoid fd issues
    try:
        result = subprocess.run(
            cmd, 
            check=False,
            stdout=sys.stdout, 
            stderr=sys.stderr,
            text=True,
            timeout=21600,  # 6 hour timeout
        )
        if result.returncode != 0:
            print(f"iLearnPlus failed with exit code {result.returncode}", flush=True)
            raise subprocess.CalledProcessError(result.returncode, cmd)
    except subprocess.TimeoutExpired:
        print(f"iLearnPlus timed out after 3600s", flush=True)
        raise



def _stem_without_double_ext(p: Path) -> str:
    """Handle foo.gbff.gz -> foo, foo.fna.gz -> foo, foo.gbff -> foo, etc."""
    name = p.name
    name = re.sub(r"\.gz$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.(gbff|gbk|gb|genbank|fa|fna|fasta|fas)$", "", name, flags=re.IGNORECASE)
    return name


def parse_samplename_to_coords(sample: str) -> Tuple[str, int, int, str]:
    """
    Parse SampleName like: u_id_seq_id_start_end_strand
    Robust to seq_id containing underscores by parsing from the right.
    Also strips anything after '|' if present.
    """
    s = str(sample).split("|", 1)[0]
    parts = s.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected SampleName format (need >=5 underscore-separated fields): {sample}")

    strand = parts[-1]
    end = int(parts[-2])
    start = int(parts[-3])
    chrom = "_".join(parts[1:-3])
    return chrom, start, end, strand


def write_bedgraph(
    df_merged: pd.DataFrame,
    bedgraph_path: Path,
    value_col: str = "probability_mean",
    strand: Optional[str] = None,
) -> None:
    """
    Write bedGraph: chrom start end value
    Note: bedGraph has no strand column, so optionally filter by strand and write separate files.
    """
    rows = []
    for s, v in zip(df_merged["SampleName"].values, df_merged[value_col].values):
        chrom, start, end, st = parse_samplename_to_coords(s)
        if strand is not None and st != strand:
            continue
        rows.append((chrom, start, end, float(v)))

    out = pd.DataFrame(rows, columns=["chrom", "start", "end", "value"])
    out.sort_values(["chrom", "start", "end"], inplace=True)
    bedgraph_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(bedgraph_path, sep="\t", header=False, index=False)


# ----------------------------
# Resume helpers
# ----------------------------

_DONE_RE = re.compile(r"^Done in\s+\d+(\.\d+)?s\s*$")
_FINAL_RE = re.compile(r"^Final output:\s+.+\s*$")


def is_run_complete(work_dir: Path, log_filename: str = "run.log") -> bool:
    """
    Determine completion by checking the end of the log file.

    Criteria (tail-based):
    - A line matching 'Done in <seconds>s' exists in the last ~50 lines AND
    - A line starting with 'Final output:' exists after it (or also in the tail).
    """
    log_path = work_dir / log_filename
    if not log_path.exists():
        return False

    try:
        tail_lines = log_path.read_text(errors="replace").splitlines()[-50:]
    except OSError:
        return False

    # Find last Done line index
    done_idx = None
    for i in range(len(tail_lines) - 1, -1, -1):
        if _DONE_RE.match(tail_lines[i].strip()):
            done_idx = i
            break
    if done_idx is None:
        return False

    # Require a Final output line at/after done line (ideally last-ish)
    for j in range(done_idx, len(tail_lines)):
        if _FINAL_RE.match(tail_lines[j].strip()):
            return True

    return False


# ----------------------------
# Main worker
# ----------------------------

def run_bactermfinder(
    genome_file: Path,
    step_size: int,
    out_root: Path,
    batch_size: int,
    window_size: int,
    ilearn_fileprocessing: Path,
    model_dir: Path,
    ilearn_threads: int = 16,
    use_gpu: Optional[bool] = None,
    log_filename: str = "run.log",
    resume: bool = False,
) -> Path:
    """
    Full per-genome pipeline. Returns path to the final mean CSV.
    """
    genome_file = genome_file.resolve()
    genome_stem = _stem_without_double_ext(genome_file)
    work_dir = out_root / genome_stem
    log_path = work_dir / log_filename

    # Resume: if log indicates completion, do nothing.
    if resume and work_dir.exists() and is_run_complete(work_dir, log_filename=log_filename):
        return work_dir / f"{genome_stem}_mean.csv"

    # Otherwise start fresh
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with redirect_fds_to_file(log_path, mode="w"):
        if use_gpu is None:
            use_gpu = gpu_available()

        print(f"Genome input: {genome_file}", flush=True)
        print(f"Work dir: {work_dir}", flush=True)
        print(f"Step size: {step_size}", flush=True)
        print(f"Batch size: {batch_size}", flush=True)
        print(f"Window size: {window_size}", flush=True)
        print(f"GPU enabled: {use_gpu}", flush=True)

        t0 = time.time()

        genome_fasta = ensure_fasta(genome_file, work_dir)
        print(f"Using FASTA: {genome_fasta}", flush=True)

        print("Sliding windows...", flush=True)
        df_slide = extract_sliding_windows(
            str(genome_fasta),
            window_size=window_size,
            step_size=step_size,
            show_progress=False,
        )
        sliding_csv = work_dir / f"{genome_stem}_sliding_windows.csv"
        df_slide.to_csv(sliding_csv, index=False)
        print(f"Wrote: {sliding_csv}", flush=True)

        fasta_path = work_dir / "df_sample.fasta"
        df_to_fasta(df_slide.reset_index(drop=True), fasta_path, train_stat="training")
        print(f"Wrote: {fasta_path}", flush=True)

        out_sample_dir = work_dir / "output_sample"
        if out_sample_dir.exists():
            shutil.rmtree(out_sample_dir)

        print("iLearnPlus feature generation...", flush=True)
        time.sleep(0.1) 
        run_ilearnplus_fileprocessing(
            fileprocessing_py=ilearn_fileprocessing,
            fasta_path=fasta_path,
            batch_size=batch_size,
            threads=ilearn_threads,
            out_dir=out_sample_dir,
        )
        print(f"iLearnPlus output dir: {out_sample_dir}", flush=True)

        specs = [
            ModelSpec("ENAC.csv", (97, 4), "deep-bactermfinder_3cnn_1d_1cat_reduced_10x_ENAC.csv_saved_model.h5"),
            ModelSpec("PS2.csv", (100, 16), "deep-bactermfinder_3cnn_1d_1cat_reduced_10x_PS2.csv_saved_model.h5"),
            ModelSpec("NCP.csv", (101, 3), "deep-bactermfinder_3cnn_1d_1cat_reduced_10x_NCP.csv_saved_model.h5"),
            ModelSpec("binary.csv", (101, 4), "deep-bactermfinder_3cnn_1d_1cat_reduced_10x_binary.csv_saved_model.h5"),
        ]

        print("Predicting...", flush=True)
        pred_frames = []
        for spec in specs:
            pred_frames.append(predict_one_embedding(
                embedding=spec,
                work_dir=work_dir,
                model_dir=model_dir,
                use_gpu=use_gpu,
            ))

        df_merged = pred_frames[0]
        for dfp in pred_frames[1:]:
            df_merged = df_merged.merge(dfp, on="SampleName", how="inner")

        prob_cols = [c for c in df_merged.columns if c.startswith("probability_")]
        df_merged["probability_mean"] = df_merged[prob_cols].mean(axis=1)

        final_csv = work_dir / f"{genome_stem}_mean.csv"
        df_merged.to_csv(final_csv, index=False)

        bedgraph_dir = work_dir / "bedgraph"
        bedgraph_dir.mkdir(parents=True, exist_ok=True)
        bedgraph_all = bedgraph_dir / f"{genome_stem}_mean.bedgraph"
        bedgraph_plus = bedgraph_dir / f"{genome_stem}_mean.plus.bedgraph"
        bedgraph_minus = bedgraph_dir / f"{genome_stem}_mean.minus.bedgraph"

        write_bedgraph(df_merged, bedgraph_all, value_col="probability_mean", strand=None)
        write_bedgraph(df_merged, bedgraph_plus, value_col="probability_mean", strand="+")
        write_bedgraph(df_merged, bedgraph_minus, value_col="probability_mean", strand="-")

        print(f"Wrote bedGraph: {bedgraph_all}", flush=True)
        print(f"Wrote bedGraph: {bedgraph_plus}", flush=True)
        print(f"Wrote bedGraph: {bedgraph_minus}", flush=True)

        dt = time.time() - t0
        print(f"Done in {dt:.1f}s", flush=True)
        print(f"Final output: {final_csv}", flush=True)

    return work_dir / f"{genome_stem}_mean.csv"


# ----------------------------
# Parallel driver
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("genomes", nargs="+", help="Genome files, directories, or globs (FASTA/GenBank, optionally .gz)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when a directory is provided")
    ap.add_argument("--step-size", type=int, required=True, help="Sliding window step/stride")
    ap.add_argument("--batch-size", type=int, required=True, help="iLearnPlus batch size")
    ap.add_argument("--window-size", type=int, default=101, help="Sliding window size (default 101)")
    ap.add_argument("--out-root", type=Path, default=Path("bactermfinder_runs"), help="Output root directory")
    ap.add_argument("--model-dir", type=Path, default=Path("."), help="Directory containing .h5 models")
    ap.add_argument(
        "--ilearn-fileprocessing",
        type=Path,
        default=None,
        help="Path to iLearnPlus/util/FileProcessing.py (default: scriptdir/iLearnPlus/util/FileProcessing.py)",
    )
    ap.add_argument("--ilearn-threads", type=int, default=1, help="Threads argument passed to FileProcessing.py")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1), help="Parallel processes (default: all cores)")
    ap.add_argument("--force-cpu", action="store_true", help="Disable GPU even if available")
    ap.add_argument("--log-filename", type=str, default="run.log", help="Per-genome log filename (default: run.log)")
    ap.add_argument("--resume", action="store_true", help="Skip genomes whose log indicates completion")
    args = ap.parse_args()

    # TF + multiprocessing: spawn is safer than fork
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    script_dir = Path(__file__).resolve().parent
    ilearn_fp = args.ilearn_fileprocessing or (script_dir / "iLearnPlus" / "util" / "FileProcessing.py")
    if not ilearn_fp.exists():
        ap.error(f"FileProcessing.py not found: {ilearn_fp}")

    genomes = find_genomes(args.genomes, recursive=args.recursive)
    if not genomes:
        ap.error("No genomes found (looked for FASTA/GenBank, optionally .gz).")

    args.out_root.mkdir(parents=True, exist_ok=True)

    # GPU + multi-process tends to fight for VRAM; default to CPU when --jobs > 1
    use_gpu: Optional[bool] = False if args.force_cpu else None
    if use_gpu is None and gpu_available() and args.jobs > 1:
        use_gpu = False

    if args.resume:
        kept = []
        skipped = 0
        for gf in genomes:
            genome_stem = _stem_without_double_ext(gf)
            work_dir = args.out_root / genome_stem
            if work_dir.exists() and is_run_complete(work_dir, log_filename=args.log_filename):
                skipped += 1
            else:
                kept.append(gf)
        genomes = kept
        print(f"Resume enabled: skipping {skipped} completed genomes; running {len(genomes)} remaining.")

    results: List[Path] = []
    failures: List[Tuple[Path, str]] = []

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        fut_to_path = {}
        for gf in genomes:
            fut = ex.submit(
                run_bactermfinder,
                genome_file=gf,
                step_size=args.step_size,
                out_root=args.out_root,
                batch_size=args.batch_size,
                window_size=args.window_size,
                ilearn_fileprocessing=ilearn_fp,
                model_dir=args.model_dir,
                ilearn_threads=args.ilearn_threads,
                use_gpu=use_gpu,
                log_filename=args.log_filename,
                resume=args.resume,
            )
            fut_to_path[fut] = gf

        for fut in as_completed(fut_to_path):
            gf = fut_to_path[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                failures.append((gf, repr(e)))

    print(f"Completed: {len(results)}")
    if failures:
        print(f"Failed: {len(failures)}")
        for gf, msg in failures[:20]:
            print(f"  {gf}: {msg}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
