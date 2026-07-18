"""
Transcribe the clips of a Common Voice-style corpus (produced by
pairs_to_common_voice.py / eaf_to_common_voice.py) with a HuggingFace ASR
model - either a CTC encoder (Wav2Vec2/Wav2Vec2-BERT/HuBERT/...) or a Whisper
seq2seq checkpoint, auto-detected from the model's config - then score the
hypotheses against the corpus' ground-truth `sentence` column.

Requires:
  pip install torch torchaudio transformers accelerate jiwer soundfile pandas tqdm

Usage:
  python transcribe_evaluate_asr.py --data-dir common_voice_pairs_poems \
      --model-id AigizK/whisper-medium-ba --batch-size 32
  python transcribe_evaluate_asr.py --data-dir common_voice_pairs_poems \
      --model-id AigizK/w2v-bert-2.0-bashkort-russian-omnivoice --batch-size 32
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


def main():
    parser = argparse.ArgumentParser(description="Transcribe and evaluate a Common Voice-style corpus")
    parser.add_argument("--data-dir", default="common_voice_pairs_poems")
    parser.add_argument("--tsv-name", default="train.tsv")
    parser.add_argument("--model-id", default="AigizK/whisper-medium-ba")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (debugging)")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--language", default=None,
        help="Whisper only: force decoding language (e.g. 'Bashkir'). Leave unset to use "
             "the checkpoint's own default - safer for fine-tunes with older generation configs.",
    )
    parser.add_argument(
        "--task", default=None,
        help="Whisper only: 'transcribe' or 'translate'. Leave unset to use the checkpoint's own default.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=225, help="Whisper only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    clips_dir = data_dir / "clips"
    df = pd.read_csv(data_dir / args.tsv_name, sep="\t")
    if args.limit:
        df = df.head(args.limit)

    print(f"Loading {args.model_id} ...")
    model, processor, is_seq2seq = load_model_and_processor(args.model_id, args.device)
    print(f"Detected model_type={model.config.model_type} -> using {'seq2seq (Whisper) generate()' if is_seq2seq else 'CTC greedy decode'}")

    results_df = transcribe_corpus(
        df, clips_dir, model, processor, is_seq2seq, args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        language=args.language, task=args.task, max_new_tokens=args.max_new_tokens,
    )
    output_csv = args.output_csv or f"{data_dir.name}_asr_eval.csv"
    results_df.to_csv(output_csv, index=False, encoding="utf-8")

    print_metrics(args.model_id, summarize_metrics(results_df))
    print(f"Per-sample results written to {output_csv}")


if __name__ == "__main__":
    main()
