import argparse
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as F_audio


def parse_args():
    parser = argparse.ArgumentParser(description="Create preprocessed RIR cache for training")
    parser.add_argument("--rir-root", required=True, help="Root folder of RIRS_NOISES")
    parser.add_argument("--output", default="data/rir_cache_3000.pt", help="Output .pt cache path")
    parser.add_argument("--num-rirs", type=int, default=3000, help="Number of RIR files to sample")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-rir-sec", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-pointsource-noises",
        action="store_true",
        help="Include pointsource_noises files. Default is to exclude them.",
    )
    return parser.parse_args()


def collect_rir_paths(rir_root, include_pointsource_noises=False):
    rir_root = Path(rir_root)

    if not rir_root.exists():
        raise FileNotFoundError(f"RIR root not found: {rir_root}")

    paths = []
    for path in sorted(rir_root.rglob("*.wav")):
        path_str = path.as_posix()
        if not include_pointsource_noises and "pointsource_noises" in path_str:
            continue
        paths.append(path)

    return paths


def preprocess_rir(wav_path, sample_rate, max_rir_len):
    try:
        rir_data, sr = sf.read(str(wav_path))
        rir_tensor = torch.from_numpy(rir_data).float()

        if rir_tensor.ndim > 1:
            rir_tensor = rir_tensor.mean(dim=1)

        if sr != sample_rate:
            rir_tensor = F_audio.resample(
                rir_tensor,
                orig_freq=sr,
                new_freq=sample_rate,
            )

        if rir_tensor.numel() < 2:
            return None

        peak_idx = torch.argmax(torch.abs(rir_tensor)).item()
        rir_tensor = rir_tensor[peak_idx:]

        if rir_tensor.numel() < 2:
            return None

        rir_tensor = rir_tensor[:max_rir_len]

        if rir_tensor.numel() < 2:
            return None

        rir_tensor = rir_tensor / (torch.norm(rir_tensor, p=2) + 1e-9)
        return rir_tensor

    except Exception as exc:
        print(f"[WARN] Skip broken RIR: {wav_path} | {exc}")
        return None


def main():
    args = parse_args()

    rir_root = Path(args.rir_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_rir_len = int(args.sample_rate * args.max_rir_sec)

    all_paths = collect_rir_paths(
        rir_root,
        include_pointsource_noises=args.include_pointsource_noises,
    )

    print("=== RIR Cache Creation ===")
    print(f"RIR root: {rir_root}")
    print(f"Candidate RIR files: {len(all_paths)}")

    rng = random.Random(args.seed)
    rng.shuffle(all_paths)

    selected_paths = all_paths[: args.num_rirs]
    print(f"Selected RIR files: {len(selected_paths)}")

    rir_tensors = []
    source_paths = []

    for idx, path in enumerate(selected_paths, 1):
        rir = preprocess_rir(
            path,
            sample_rate=args.sample_rate,
            max_rir_len=max_rir_len,
        )

        if rir is not None:
            rir_tensors.append(rir)
            source_paths.append(path.relative_to(rir_root).as_posix())

        if idx % 200 == 0 or idx == len(selected_paths):
            print(f"[{idx}/{len(selected_paths)}] kept: {len(rir_tensors)}")

    payload = {
        "sample_rate": args.sample_rate,
        "max_rir_sec": args.max_rir_sec,
        "rir_tensors": rir_tensors,
        "source_paths": source_paths,
    }

    torch.save(payload, output_path)

    print("\n=== RIR Cache ===")
    print(f"Output: {output_path}")
    print(f"Final RIR tensors: {len(rir_tensors)}")


if __name__ == "__main__":
    main()