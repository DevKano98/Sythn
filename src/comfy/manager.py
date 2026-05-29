"""ComfyUI installation and subprocess lifecycle management."""
import subprocess
import sys
from pathlib import Path
from typing import Optional

from config import COMFY_HOST, COMFY_PORT, COMFYUI_DIR
from .client import ComfyClient, ComfyError


class ComfyManager:
    def __init__(self, comfy_dir: Path = COMFYUI_DIR,
                 host: str = COMFY_HOST, port: int = COMFY_PORT):
        self.comfy_dir = Path(comfy_dir)
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None

    @property
    def main_py(self) -> Path:
        return self.comfy_dir / "main.py"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_installed(self) -> bool:
        return self.main_py.exists()

    def install_with_comfy_cli(self) -> None:
        subprocess.run([sys.executable, "-m", "pip", "install", "comfy-cli"], check=True)
        subprocess.run(["comfy", "--skip-prompt", "install", "--path", str(self.comfy_dir)], check=True)

    def start(self, wait: bool = True) -> ComfyClient:
        client = ComfyClient(self.base_url)
        if client.is_ready():
            return client

        if not self.is_installed():
            raise ComfyError(
                f"ComfyUI is not installed at {self.comfy_dir}. "
                "Set COMFYUI_DIR or install it with download_models.py --install-comfy."
            )

        if self.process and self.process.poll() is None:
            if wait:
                client.wait_until_ready()
            return client

        self.process = subprocess.Popen(
            [
                sys.executable,
                str(self.main_py),
                "--listen",
                self.host,
                "--port",
                str(self.port),
            ],
            cwd=str(self.comfy_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if wait:
            client.wait_until_ready(timeout=180)
        return client

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
