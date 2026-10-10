# check environment of project
import os, sys, time, socket
from pathlib import Path

def line(msg):
    print(f"[check] {msg}", flush=True)

# 1) Python
line(f"Python {sys.version.split()[0]} at {sys.executable}")

# 2) Network to HF + mirror
for host in ["huggingface.co", "hf-mirror.com"]:
    t0 = time.time()
    try:
        ip = socket.gethostbyname(host)
        line(f"DNS OK   {host} -> {ip}  ({(time.time()-t0)*1000:.0f} ms)")
    except Exception as e:
        line(f"DNS FAIL {host}: {e}")

# 3) CUDA via ctranslate2 (what faster-whisper uses)
try:
    import ctranslate2
    line(f"ctranslate2 {ctranslate2.__version__}")
    line(f"CUDA devices: {ctranslate2.get_cuda_device_count()}")
    if ctranslate2.get_cuda_device_count() == 0:
        line("WARNING: no CUDA device visible to ctranslate2")
except Exception as e:
    line(f"ctranslate2 import failed: {e}")

# 4) faster-whisper version
try:
    import faster_whisper
    line(f"faster-whisper {faster_whisper.__version__}")
except Exception as e:
    line(f"faster_whisper import failed: {e}")

# 5) Model cache directory
mdir = Path("../models")
mdir.mkdir(exist_ok=True)
line(f"model dir: {mdir.resolve()}")

# 6) Force-download check (dry)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
try:
    from huggingface_hub import snapshot_download
    line("Attempting snapshot_download of tiny model (fast test)...")
    t0 = time.time()
    path = snapshot_download(
        repo_id="Systran/faster-whisper-tiny",
        cache_dir=str(mdir),
        allow_patterns=["*.json", "*.txt"],   # tiny test — only small files
    )
    line(f"OK tiny metadata in {time.time()-t0:.1f}s -> {path}")
except Exception as e:
    line(f"snapshot_download failed: {e}")