import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from pathlib import Path
from Src.transcribe import WhisperModel, collect_files, transcribe_file
from Src.progress import setup_logger, Phase


def main() -> None:
    out_dir = Path("output")
    logger = setup_logger(out_dir / "transcribe.log")

    model_name = "../models/faster-whisper-large-v3"
    device = "cuda"
    compute_type = "float16"
    model_dir = "../models"

    with Phase(logger, f"Load model {model_name} on {device} ({compute_type})"):
        model = WhisperModel(
            model_name,
            device="cpu",            
            compute_type="int8",     
            download_root=model_dir,
        )

    files = collect_files(Path("Raw/Audio"))
    logger.info(f"Found {len(files)} file(s) in Raw/Audio")

    for i, f in enumerate(files, 1):
        logger.info(f"=== [{i}/{len(files)}] {f.name} ===")
        try:
            transcribe_file(
                model=model,
                media_path=f,
                out_dir=out_dir,
                formats=["txt", "srt", "json"],
                language="zh",
                initial_prompt=(
                    "The Speech in Ceremonies of Graduate/Undergraduate by WANG"
                ),
                vad_filter=True,
                word_timestamps=True,
                beam_size=5,
                logger=logger,
                live_text=True,
                text_every_n=5,
            )
        except Exception as e:
            logger.exception(f"FAILED on {f.name}: {e}")
            continue


if __name__ == "__main__":
    main()