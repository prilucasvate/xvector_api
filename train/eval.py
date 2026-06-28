import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import roc_curve
from tqdm import tqdm

from dataset import DynamicCollate
from model import XVector
import yaml


SEED = 42
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v14 with EER")
    parser.add_argument("--config", default="./configs/v14.yaml")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_audio_path(path, data_root=None):
    wav_path = Path(path)
    if data_root is not None and not wav_path.is_absolute():
        wav_path = Path(data_root) / wav_path
    return wav_path


def load_audio_for_model(wav_path):
    data, sr = sf.read(str(wav_path))
    waveform = torch.from_numpy(data).float()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.squeeze(0)
    waveform = DynamicCollate(augment=False).normalize_rms(waveform)
    return waveform


def extract_embedding(model, wav_path):
    waveform = load_audio_for_model(wav_path).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, emb = model(waveform)
    vec = emb.cpu().numpy()[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx], thresholds[idx]


def load_manifest(manifest_path, split):
    data = defaultdict(list)
    train_spks = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("split") == "train":
                train_spks.add(item["spk_id"])
            if item.get("split") == split:
                data[item["spk_id"]].append(item)

    return data, len(train_spks)


def evaluate_split(model, dataset_dict, data_root=None, split_name="test", num_spks=50, utts_per_spk=10, rounds=10):
    eers = []
    thresholds = []

    spk_pool = [spk for spk, utts in dataset_dict.items() if len(utts) >= 2]
    if len(spk_pool) < 2:
        raise ValueError(f"Not enough speakers with at least 2 utterances in split={split_name}")

    for r in range(rounds):
        selected_spks = random.sample(spk_pool, min(num_spks, len(spk_pool)))

        embeddings = {}
        for spk in tqdm(selected_spks, desc=f"Round {r + 1}/{rounds}"):
            selected_utts = random.sample(dataset_dict[spk], min(utts_per_spk, len(dataset_dict[spk])))
            vecs = []
            for item in selected_utts:
                wav_path = resolve_audio_path(item["path"], data_root)
                vecs.append(extract_embedding(model, wav_path))
            if len(vecs) >= 2:
                embeddings[spk] = vecs

        labels = []
        scores = []
        spk_list = list(embeddings.keys())

        for spk in spk_list:
            vecs = embeddings[spk]
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    scores.append(float(np.dot(vecs[i], vecs[j])))
                    labels.append(1)

        for i in range(len(spk_list)):
            for j in range(i + 1, len(spk_list)):
                vec1 = random.choice(embeddings[spk_list[i]])
                vec2 = random.choice(embeddings[spk_list[j]])
                scores.append(float(np.dot(vec1, vec2)))
                labels.append(0)

        eer, threshold = calculate_eer(labels, scores)
        eers.append(eer)
        thresholds.append(threshold)
        print(f"Round {r + 1}/{rounds} {split_name} EER: {eer * 100:.2f}% | threshold: {threshold:.4f}")

    print(f"\n[{split_name}] Average EER: {np.mean(eers) * 100:.2f}%")
    print(f"[{split_name}] Average threshold: {np.mean(thresholds):.4f}")


    
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"].get("seed", SEED))

    manifest = cfg["data"]["manifest"]
    data_root = cfg["data"].get("data_root")

    eval_cfg = cfg["evaluation"]
    checkpoint = eval_cfg["checkpoint"]
    split = eval_cfg.get("split", "test")

    dataset_dict, num_classes = load_manifest(manifest, split)
    print(f"Train speakers/classes: {num_classes}")
    print(f"{split} speakers: {len(dataset_dict)}")

    model = XVector(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    evaluate_split(
        model,
        dataset_dict,
        data_root=data_root,
        split_name=split,
        num_spks=eval_cfg.get("num_spks", 50),
        utts_per_spk=eval_cfg.get("utts_per_spk", 10),
        rounds=eval_cfg.get("rounds", 10),
    )


if __name__ == "__main__":
    main()