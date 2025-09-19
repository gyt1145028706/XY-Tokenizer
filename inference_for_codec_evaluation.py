import os
import argparse
import json
from pathlib import Path
import logging
import torch
import torchaudio

from utils.helpers import set_logging, waiting_for_debug, load_audio, save_audio
from xy_tokenizer.model import XY_Tokenizer


def load_xy_tokenizer(config_path: str, checkpoint_path: str, device: str = "cuda"):
    # Load XY Tokenizer
    model = XY_Tokenizer.load_from_checkpoint(config_path=config_path, ckpt_path=checkpoint_path).to(device).eval()
    return model


@torch.inference_mode()
def process_and_reconstruct_audio(config_path: str, checkpoint_path: str, input_jsonl: str, output_dir: str, device: str = "cuda"):
    # Load tokenizer
    tokenizer = load_xy_tokenizer(config_path, checkpoint_path, device=device)
    sample_rate = tokenizer.sample_rate

    # Ensure output dirs
    output_syn_dir = Path(output_dir) / "syn_audios"
    output_gt_dir = Path(output_dir) / "gt_audios"
    output_syn_dir.mkdir(parents=True, exist_ok=True)
    output_gt_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio paths from jsonl
    audio_paths = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "audio_path" in data:
                audio_paths.append(data["audio_path"])
            else:
                audio_paths.append(data["audio_file"])

    logging.info(f"Found {len(audio_paths)} audio files in {input_jsonl}")

    for audio_path in audio_paths:
        logging.info(f"Processing {audio_path}")

        # Load audio
        audio, orig_sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)

        # Resample
        if orig_sr != sample_rate:
            audio_resampled = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sample_rate)
        else:
            audio_resampled = audio

        audio_resampled = audio_resampled.to(device)

        # Encode & Decode
        encode_result = tokenizer.encode([audio_resampled], overlap_seconds=0)
        codes_list = encode_result["codes_list"]

        decode_result = tokenizer.decode(codes_list, overlap_seconds=0)
        syn_wav_list = decode_result["syn_wav_list"]
        reconstructed_audio = syn_wav_list[0]

        # Save both syn and gt
        file_name = Path(audio_path).stem + ".flac"
        output_syn_path = output_syn_dir / file_name
        output_gt_path = output_gt_dir / file_name

        torchaudio.save(str(output_syn_path), reconstructed_audio.cpu().unsqueeze(0), sample_rate)
        torchaudio.save(str(output_gt_path), audio_resampled.cpu().unsqueeze(0), sample_rate)

        logging.info(f"Saved reconstructed audio to {output_syn_path}")
        logging.info(f"Saved ground truth audio to {output_gt_path}")


if __name__ == "__main__":
    set_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--codec_ckpt", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--debug", default=0, type=int, nargs="?", help="whether debug or not")
    parser.add_argument("--debug_ip", default="localhost", type=str)
    parser.add_argument("--debug_port", default=32431, type=int)

    args = parser.parse_args()

    if args.debug == 1:
        waiting_for_debug(args.debug_ip, args.debug_port)

    process_and_reconstruct_audio(
        config_path=args.config,
        checkpoint_path=args.codec_ckpt,
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        device=args.device,
    )
