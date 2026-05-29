"""Small REST client for ComfyUI's HTTP API."""
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode

import requests


class ComfyError(RuntimeError):
    """Raised when ComfyUI returns an API error or an unexpected response."""


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def is_ready(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/system_stats", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(self, timeout: int = 120,
                         progress: Optional[Callable[[str], None]] = None) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_ready():
                return
            if progress:
                progress("Waiting for ComfyUI...")
            time.sleep(1)
        raise ComfyError(f"ComfyUI did not become ready at {self.base_url}")

    def get_system_stats(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/system_stats", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_object_info(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/object_info", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def upload_image(self, image_path: str) -> str:
        path = Path(image_path)
        with path.open("rb") as fh:
            files = {"image": (path.name, fh, "application/octet-stream")}
            data = {"overwrite": "true", "type": "input"}
            response = self.session.post(
                f"{self.base_url}/upload/image",
                files=files,
                data=data,
                timeout=self.timeout,
            )
        response.raise_for_status()
        payload = response.json()
        return payload.get("name") or path.name

    def queue_prompt(self, workflow: Dict[str, Any], client_id: str) -> str:
        response = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ComfyError(str(payload["error"]))
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI did not return a prompt_id: {payload}")
        return prompt_id

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/history/{prompt_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_result(self, prompt_id: str, poll_interval: float = 1.0,
                        progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        while True:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                item = history[prompt_id]
                if item.get("status", {}).get("status_str") == "error":
                    messages = item.get("status", {}).get("messages", [])
                    raise ComfyError(f"Workflow failed: {messages}")
                return item
            if progress:
                progress("Workflow running in ComfyUI...")
            time.sleep(poll_interval)

    def download_view(self, filename: str, subfolder: str, folder_type: str,
                      destination: str) -> Path:
        query = urlencode({
            "filename": filename,
            "subfolder": subfolder or "",
            "type": folder_type or "output",
        })
        response = self.session.get(
            f"{self.base_url}/view?{query}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return dest
