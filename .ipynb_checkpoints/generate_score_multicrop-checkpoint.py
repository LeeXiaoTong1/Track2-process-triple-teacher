import os
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm

from generate_score import build_model

from torch.utils.data import Dataset, DataLoader

def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--eval_task", type=str, default="atadd-track2")
    parser.add_argument("--eval_audio", type=str, required=True)
    parser.add_argument("--score_file", type=str, required=True)

    parser.add_argument("--num_crops", type=int, default=5)
    parser.add_argument("--audio_len", type=int, default=None)
    parser.add_argument("--batch_files", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=-1)

    # agg:
    # mean_logit: average logits then softmax. Usually stable.
    # mean_score: average real probability.
    # min_real: conservative fake detection; if any crop looks fake, score becomes lower.
    parser.add_argument(
        "--agg",
        type=str,
        default="mean_logit",
        choices=["mean_logit", "mean_score", "min_real"]
    )

    temp, _ = parser.parse_known_args()

    json_path = os.path.join(temp.model_path, "args.json")
    with open(json_path, "r") as f:
        saved = json.load(f)

    # Add saved training args into parser
    for k, v in saved.items():
        if k not in vars(temp):
            if isinstance(v, bool):
                parser.add_argument(
                    f"--{k}",
                    action="store_true" if v else "store_false",
                    default=v
                )
            else:
                parser.add_argument(f"--{k}", type=type(v), default=v)

    args = parser.parse_args()

    if args.audio_len is None:
        args.audio_len = saved.get("audio_len", 64600)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    print("Using GPU:", args.gpu)
    print("Model path:", args.model_path)
    print("Eval task:", args.eval_task)
    print("Eval audio:", args.eval_audio)
    print("Score file:", args.score_file)
    print("num_crops:", args.num_crops)
    print("agg:", args.agg)

    return args


def normalize_clip(x):
    x = x.astype(np.float32)
    x = x - x.mean()
    x = x / np.sqrt(x.var() + 1e-7)
    return x


def make_crops(wav, cut, num_crops):
    """
    wav: np.ndarray [L]
    return: torch.Tensor [num_crops, cut]
    """
    L = len(wav)

    if L <= 0:
        wav = np.zeros(cut, dtype=np.float32)
        L = cut

    if L < cut:
        rep = int(np.ceil(cut / L)) + 1
        wav = np.tile(wav, rep)[:cut]
        crops = [wav.copy() for _ in range(num_crops)]

    elif L == cut:
        crops = [wav.copy() for _ in range(num_crops)]

    else:
        if num_crops <= 1:
            starts = [0]
        else:
            starts = np.linspace(0, L - cut, num_crops).astype(int).tolist()

        crops = [wav[s:s + cut].copy() for s in starts]

    crops = [normalize_clip(x) for x in crops]
    arr = np.stack(crops, axis=0).astype(np.float32)

    return torch.from_numpy(arr).clone()

def aggregate_logits(logits_all, agg):
    """
    logits_all: [B, C, 2]
    return: [B] score = probability of real
    """
    if agg == "mean_logit":
        avg_logits = logits_all.mean(dim=1)      # [B, 2]
        scores = F.softmax(avg_logits, dim=1)[:, 0]

    elif agg == "mean_score":
        scores = F.softmax(logits_all, dim=2)[:, :, 0].mean(dim=1)

    elif agg == "min_real":
        scores = F.softmax(logits_all, dim=2)[:, :, 0].min(dim=1).values

    else:
        raise ValueError(f"Unknown agg: {agg}")

    return scores.detach().cpu().numpy()


class MultiCropEvalDataset(Dataset):
    def __init__(self, audio_dir, audio_len=64600, num_crops=3, max_files=-1):
        self.audio_dir = audio_dir
        self.audio_len = audio_len
        self.num_crops = num_crops

        self.files = sorted([
            f for f in os.listdir(audio_dir)
            if f.lower().endswith((".wav", ".flac"))
        ])

        if max_files is not None and max_files > 0:
            self.files = self.files[:max_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        path = os.path.join(self.audio_dir, fn)

        wav, _ = librosa.load(path, sr=16000)

        crops = make_crops(
            wav,
            cut=self.audio_len,
            num_crops=self.num_crops
        )

        return crops, fn

def main():
    args = load_args()

    os.makedirs(os.path.dirname(args.score_file), exist_ok=True)

    model = build_model(args)
    ckpt_path = os.path.join(args.model_path, "atadd_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    dataset = MultiCropEvalDataset(
        audio_dir=args.eval_audio,
        audio_len=args.audio_len,
        num_crops=args.num_crops,
        max_files=args.max_files
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_files,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    print("Eval files:", len(dataset))
    print("batch_files:", args.batch_files)
    print("actual forward batch:", args.batch_files * args.num_crops)

    with open(args.score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "score"])

        with torch.inference_mode():
            for crops, filenames in tqdm(loader):
                # crops: [B, C, L]
                B, C, L = crops.shape

                crops = crops.view(B * C, L).to(args.device, non_blocking=True)

                _, logits = model(crops)

                # logits: [B*C, 2] -> [B, C, 2]
                logits = logits.view(B, C, -1)

                scores = aggregate_logits(logits, args.agg)

                for fn, score in zip(filenames, scores):
                    writer.writerow([fn, float(score)])

    print("Saved:", args.score_file)


if __name__ == "__main__":
    main()
