"""Local hardware facts; no personal identifiers or external requests."""
import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

def number(path):
    try:
        n = Path(path).read_text().strip()
        return int(n) if n != "max" else None
    except (OSError, ValueError):
        return None

def probe():
    ram = None
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    cap = number("/sys/fs/cgroup/memory.max")
    gpu = shutil.which("nvidia-smi")
    vram = None
    if gpu:
        try:
            p = subprocess.run([gpu, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5, check=True)
            vram = [int(x.strip()) for x in p.stdout.splitlines() if x.strip()]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {"os":platform.system(), "architecture":platform.machine(),
            "logical_cpus":os.cpu_count(), "physical_ram_bytes":ram,
            "cgroup_limit_bytes":cap, "disk_free_bytes":shutil.disk_usage(".").free,
            "llama_server_available":bool(shutil.which("llama-server")),
            "nvidia_vram_mib":vram, "nvidia_devices_visible":bool(list(Path("/dev").glob("nvidia*")))}

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(probe(),indent=2))
if __name__ == "__main__": main()
