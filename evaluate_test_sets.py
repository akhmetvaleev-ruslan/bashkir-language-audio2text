"""
Score an ASR checkpoint (e.g. a whisper.py --output-dir, or any HF Hub model
id) against one or more Common Voice-style dataset directories.

For each data dir, --test-tsv is used if present (a true held-out split kept
back from training, e.g. common_voice/test.tsv). Directories without one -
e.g. common_voice_pairs_poems, which ships only a train.tsv because the whole
corpus is a standalone eval set never used in training - fall back to
--fallback-tsv, mirroring transcribe_evaluate_asr.py's --tsv-name default.

Self-contained (does not import transcribe_evaluate_asr.py) so it can be
dropped onto a training box on its own; the transcription/scoring core is
duplicated from transcribe_evaluate_asr.py.

Meant both as a standalone re-scoring tool (run any time against a saved
checkpoint, no retraining needed) and as the module whisper.py imports to
run the same check right after training finishes.

Requires:
  pip install torch torchaudio transformers accelerate jiwer soundfile pandas tqdm

Usage:
  python evaluate_test_sets.py --model-id whisper-ba \
      --data-dirs common_voice common_voice_elang common_voice_pairs common_voice_pairs_poems
"""

import argparse
import re
from contextlib import nullcontext
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor

WHISPER_TAG_RE = re.compile(r"<\|[^|>]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    # Strip Whisper special-token markers (e.g. <|ba|><|transcribe|><|notimestamps|>) that
    # leak into the decoded string on some fine-tuned checkpoints where skip_special_tokens
    # doesn't catch them; must run before punctuation stripping or "ba"/"transcribe"/etc.
    # survive as bare words and pollute the WER/CER word counts.
    text = WHISPER_TAG_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class ClipsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, clips_dir: Path, target_sr: int):
        self.df = df.reset_index(drop=True)
        self.clips_dir = clips_dir
        self.target_sr = target_sr
        self._resamplers = {}

    def __len__(self):
        return len(self.df)

    def _resample(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return audio
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
        return self._resamplers[sr](audio)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = sf.read(str(self.clips_dir / row["path"]), dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio).mean(dim=1)  # downmix to mono
        audio = self._resample(audio, sr).numpy()
        return {"audio": audio, "sentence": str(row["sentence"]), "path": row["path"]}


def collate(batch):
    return (
        [b["audio"] for b in batch],
        [b["sentence"] for b in batch],
        [b["path"] for b in batch],
    )


def transcribe_ctc(model, processor, audios, target_sr, device, autocast_ctx):
    inputs = processor(audios, sampling_rate=target_sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(), autocast_ctx:
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)


def transcribe_seq2seq(model, processor, audios, target_sr, device, autocast_ctx, language, task, max_new_tokens):
    inputs = processor(audios, sampling_rate=target_sr, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if language or task:
        # Use the legacy forced_decoder_ids route rather than generate(language=, task=):
        # many fine-tuned Whisper checkpoints ship a generation_config.json predating the
        # newer kwargs, which raises "generation config is outdated" if language/task are
        # passed directly to generate().
        gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
            language=language, task=task or "transcribe"
        )
    with torch.no_grad(), autocast_ctx:
        generated_ids = model.generate(input_features, **gen_kwargs)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def load_model_and_processor(model_id: str, device: str):
    """model_id may be a HF Hub repo id or a local checkpoint directory
    (e.g. a whisper.py --output-dir) - from_pretrained() handles both."""
    config = AutoConfig.from_pretrained(model_id)
    is_seq2seq = config.model_type == "whisper"
    processor = AutoProcessor.from_pretrained(model_id)
    model_cls = AutoModelForSpeechSeq2Seq if is_seq2seq else AutoModelForCTC
    model = model_cls.from_pretrained(model_id).to(device).eval()
    return model, processor, is_seq2seq


def transcribe_corpus(
    df: pd.DataFrame, clips_dir: Path, model, processor, is_seq2seq: bool, device: str,
    batch_size: int = 32, num_workers: int = 8,
    language: str | None = None, task: str | None = None, max_new_tokens: int = 225,
) -> pd.DataFrame:
    """Run the model over every clip in df and return a results DataFrame with
    raw/normalized reference & hypothesis text and per-sample WER/CER."""
    target_sr = processor.feature_extractor.sampling_rate
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    dataset = ClipsDataset(df, clips_dir, target_sr)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate,
    )

    references, hypotheses, paths = [], [], []
    for audios, sentences, batch_paths in tqdm(loader, desc="Transcribing"):
        if is_seq2seq:
            batch_hyps = transcribe_seq2seq(
                model, processor, audios, target_sr, device, autocast_ctx, language, task, max_new_tokens,
            )
        else:
            batch_hyps = transcribe_ctc(model, processor, audios, target_sr, device, autocast_ctx)

        references.extend(sentences)
        hypotheses.extend(batch_hyps)
        paths.extend(batch_paths)

    ref_norm = [normalize_text(r) for r in references]
    hyp_norm = [normalize_text(h) for h in hypotheses]
    per_sample_wer = [jiwer.wer(r, h) if r else 1.0 for r, h in zip(ref_norm, hyp_norm)]
    per_sample_cer = [jiwer.cer(r, h) if r else 1.0 for r, h in zip(ref_norm, hyp_norm)]

    return pd.DataFrame({
        "path": paths,
        "reference": references,
        "hypothesis": hypotheses,
        "reference_norm": ref_norm,
        "hypothesis_norm": hyp_norm,
        "wer": per_sample_wer,
        "cer": per_sample_cer,
    })


def summarize_metrics(results_df: pd.DataFrame) -> dict:
    ref_norm = results_df["reference_norm"].tolist()
    hyp_norm = results_df["hypothesis_norm"].tolist()
    return {
        "n_samples": len(results_df),
        "wer": jiwer.wer(ref_norm, hyp_norm),
        "cer": jiwer.cer(ref_norm, hyp_norm),
        "mer": jiwer.mer(ref_norm, hyp_norm),
        "wil": jiwer.wil(ref_norm, hyp_norm),
        "wip": jiwer.wip(ref_norm, hyp_norm),
        "mean_per_sample_wer": results_df["wer"].mean(),
        "mean_per_sample_cer": results_df["cer"].mean(),
        "empty_hypothesis_rate": (results_df["hypothesis_norm"] == "").mean(),
    }


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\nModel: {label}")
    print(f"N samples: {metrics['n_samples']}")
    print(f"Corpus WER: {metrics['wer']:.4f}")
    print(f"Corpus CER: {metrics['cer']:.4f}")
    print(f"Corpus MER: {metrics['mer']:.4f}")
    print(f"Corpus WIL: {metrics['wil']:.4f}  (WIP: {metrics['wip']:.4f})")
    print(f"Mean per-sample WER: {metrics['mean_per_sample_wer']:.4f}")
    print(f"Mean per-sample CER: {metrics['mean_per_sample_cer']:.4f}")
    print(f"Empty-hypothesis rate: {metrics['empty_hypothesis_rate']:.4f}")


def evaluate_test_sets(
    model_id: str, data_dirs: list[Path], test_tsv: str, output_dir: Path,
    batch_size: int = 16, num_workers: int = 8, device: str | None = None,
    language: str | None = None, task: str | None = None, max_new_tokens: int = 225,
    fallback_tsv: str | None = "train.tsv",
) -> dict[str, dict]:
    """Evaluate model_id against each data dir, writing a per-sample CSV into
    output_dir and printing full WER/CER/MER/WIL metrics for each. Prefers
    test_tsv (a true held-out split); for dirs without one, falls back to
    fallback_tsv (e.g. a standalone eval-only corpus like common_voice_pairs_poems
    that ships only a train.tsv) - unless fallback_tsv is None, in which case
    such dirs are skipped. Returns {dir_name: metrics}."""
    dir_tsvs = []
    for d in data_dirs:
        if (d / test_tsv).exists():
            dir_tsvs.append((d, test_tsv))
        elif fallback_tsv and (d / fallback_tsv).exists():
            dir_tsvs.append((d, fallback_tsv))
    if not dir_tsvs:
        return {}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    print(f"\nEvaluating {model_id} on held-out test set(s) ...")
    model, processor, is_seq2seq = load_model_and_processor(model_id, device)

    all_metrics = {}
    for test_dir, tsv_name in dir_tsvs:
        test_df = pd.read_csv(test_dir / tsv_name, sep="\t")
        results_df = transcribe_corpus(
            test_df, test_dir / "clips", model, processor, is_seq2seq, device,
            batch_size=batch_size, num_workers=num_workers,
            language=language, task=task, max_new_tokens=max_new_tokens,
        )
        output_csv = output_dir / f"{test_dir.name}_{Path(tsv_name).stem}_asr_eval.csv"
        results_df.to_csv(output_csv, index=False, encoding="utf-8")

        metrics = summarize_metrics(results_df)
        print_metrics(f"{model_id} on {test_dir}/{tsv_name}", metrics)
        print(f"Per-sample results written to {output_csv}")
        all_metrics[test_dir.name] = metrics

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Score a checkpoint against one or more held-out test.tsv sets")
    parser.add_argument(
        "--data-dirs", nargs="+",
        default=["common_voice", "common_voice_elang", "common_voice_pairs", "common_voice_pairs_poems"],
        help="Dataset directories to evaluate",
    )
    parser.add_argument("--test-tsv", default="test.tsv", help="Preferred split; used when a dir has one")
    parser.add_argument(
        "--fallback-tsv", default="train.tsv",
        help="Used for dirs without --test-tsv (e.g. eval-only corpora like common_voice_pairs_poems "
             "that ship only a train.tsv). Pass '' to skip such dirs instead.",
    )
    parser.add_argument("--model-id", default="whisper-ba", help="HF Hub id or local checkpoint directory")
    parser.add_argument("--output-dir", default=None, help="Where to write per-sample CSVs (defaults to --model-id)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", default=None, help="Whisper only: force decoding language (e.g. 'Bashkir')")
    parser.add_argument("--task", default=None, help="Whisper only: 'transcribe' or 'translate'")
    parser.add_argument("--max-new-tokens", type=int, default=225, help="Whisper only")
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dirs]
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_id)
    all_metrics = evaluate_test_sets(
        args.model_id, data_dirs, args.test_tsv, output_dir,
        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
        language=args.language, task=args.task, max_new_tokens=args.max_new_tokens,
        fallback_tsv=args.fallback_tsv or None,
    )
    if not all_metrics:
        print(f"No {args.test_tsv} or {args.fallback_tsv} found under any of: {', '.join(str(d) for d in data_dirs)}")


if __name__ == "__main__":
    main()
