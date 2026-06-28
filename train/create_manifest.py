import argparse
import json
import random
from pathlib import Path

import soundfile as sf

# 參數
MIN_DURATION_SEC = 2.0 # 最短音檔限制：2 秒
MIN_UTTERANCES = 15 # 每個語者最少需要的有效音檔數
VAL_SPK_RATIO = 0.1
TEST_SPK_RATIO = 0.1
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Create a JSONL manifest from speaker folders")
    parser.add_argument("--data-dir", default="data/wavs", help="Root folder containing speaker subfolders")
    parser.add_argument("--output", default="data/manifest.jsonl", help="Output JSONL manifest path")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_SEC)
    parser.add_argument("--min-utterances", type=int, default=MIN_UTTERANCES)
    parser.add_argument("--val-ratio", type=float, default=VAL_SPK_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_SPK_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def collect_audio(data_dir, min_duration):
    data_dir = Path(data_dir)
    spk2items = {}
    short_count = 0
    broken_count = 0

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for spk_dir in sorted(data_dir.iterdir()):
        if not spk_dir.is_dir():
            continue

        spk_id = spk_dir.name
        spk2items[spk_id] = []

        audio_files = sorted(
            list(spk_dir.rglob("*.wav"))
        )

        for audio_path in audio_files:
            try:
                info = sf.info(str(audio_path))
                duration = round(float(info.duration), 4)

                if duration < min_duration:
                    short_count += 1
                    continue

                rel_path = audio_path.relative_to(data_dir).as_posix()
                uid = f"{spk_id}_{audio_path.stem}"

                spk2items[spk_id].append({
                    "uid": uid,
                    "spk_id": spk_id,
                    "path": rel_path,
                    "source": "custom",
                    "split": None,
                    "duration": duration,
                })
            except Exception as exc:
                broken_count += 1
                print(f"[WARN] Skip broken file: {audio_path} | {exc}")

    return spk2items, short_count, broken_count


def split_by_speaker(spk2items, min_utterances, val_ratio, test_ratio, seed):
    valid_spks = [
        spk for spk, items in spk2items.items()
        if len(items) >= min_utterances
    ]

    if len(valid_spks) < 3:
        raise ValueError(
            "需要至少 3 個有效語者才能建立訓練/驗證/測試集。"
            "Lower --min-utterances or add more speaker folders."
        )

    random.seed(seed)
    random.shuffle(valid_spks)

    num_spks = len(valid_spks)
    test_count = max(1, int(num_spks * test_ratio))
    val_count = max(1, int(num_spks * val_ratio))

    test_spks = set(valid_spks[:test_count])
    val_spks = set(valid_spks[test_count:test_count + val_count])
    train_spks = set(valid_spks[test_count + val_count:])

    if not train_spks:
        raise ValueError("沒有剩餘訓練語者。請降低驗證/測試比例。")

    final_items = []
    for spk in valid_spks:
        if spk in test_spks:
            split = "test"
        elif spk in val_spks:
            split = "val"
        else:
            split = "train"

        for item in spk2items[spk]:
            item["split"] = split
            final_items.append(item)

    return final_items, train_spks, val_spks, test_spks


def main():
    args = parse_args()

    spk2items, short_count, broken_count = collect_audio(
        args.data_dir,
        args.min_duration,
    )

    final_items, train_spks, val_spks, test_spks = split_by_speaker(
        spk2items,
        args.min_utterances,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in final_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== Manifest Report ===")
    print(f"Output: {output_path}")
    print(f"Total samples: {len(final_items)}")
    print(f"Train speakers: {len(train_spks)}")
    print(f"Val speakers: {len(val_spks)}")
    print(f"Test speakers: {len(test_spks)}")
    print(f"Too short files removed: {short_count}")
    print(f"Broken files skipped: {broken_count}")


if __name__ == "__main__":
    main()