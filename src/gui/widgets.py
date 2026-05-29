"""CustomTkinter widgets used by the ComfyUI GUI."""
import tkinter as tk
from pathlib import Path
from typing import Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageTk


class ImagePreview(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_path: Optional[str] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.label = ctk.CTkLabel(self, text="No image loaded")
        self.label.pack(expand=True, fill="both", padx=8, pady=8)

    def set_image(self, path: str, max_size: Tuple[int, int] = (760, 560)):
        image = Image.open(path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.label.configure(image=self.photo, text="")
        self.image_path = path

    def clear(self):
        self.photo = None
        self.image_path = None
        self.label.configure(image=None, text="No image loaded")


class BeforeAfterSlider(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.before: Optional[Image.Image] = None
        self.after: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.split = 0.5
        self.canvas = tk.Canvas(self, bg="#111111", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")
        self.slider = ctk.CTkSlider(self, from_=0, to=1, number_of_steps=100, command=self._on_slide)
        self.slider.set(self.split)
        self.slider.pack(fill="x", padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda _event: self._draw())

    def set_images(self, before_path: str, after_path: str):
        self.before = Image.open(before_path).convert("RGB")
        self.after = Image.open(after_path).convert("RGB")
        self._draw()

    def _on_slide(self, value):
        self.split = float(value)
        self._draw()

    def _draw(self):
        if self.before is None or self.after is None:
            return
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        before = self.before.copy()
        before.thumbnail((width, height), Image.LANCZOS)
        after = self.after.copy().resize(before.size, Image.LANCZOS)
        image_w, image_h = before.size
        split_px = int(image_w * self.split)

        composite = before.copy()
        composite.paste(after.crop((split_px, 0, image_w, image_h)), (split_px, 0))
        self.photo = ImageTk.PhotoImage(composite)

        self.canvas.delete("all")
        x = (width - image_w) // 2
        y = (height - image_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.photo)
        self.canvas.create_line(x + split_px, y, x + split_px, y + image_h, fill="white", width=2)


class ModelStatusWidget(ctk.CTkFrame):
    def __init__(self, master, model_manager, **kwargs):
        super().__init__(master, **kwargs)
        self.model_manager = model_manager
        ctk.CTkLabel(self, text="Model Status", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=6, pady=(6, 2)
        )
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="x", padx=6, pady=(0, 6))
        self.refresh()

    def refresh(self):
        for child in self.body.winfo_children():
            child.destroy()
        for name, path in self.model_manager.model_paths.items():
            exists = path.exists()
            ctk.CTkLabel(
                self.body,
                text=f"{'[OK]' if exists else '[MISSING]'} {name}",
                text_color="green" if exists else "orange",
                font=ctk.CTkFont(size=11),
            ).pack(anchor="w")


class ProgressCard(ctk.CTkFrame):
    def __init__(self, master, total_stages: int = 8, **kwargs):
        super().__init__(master, **kwargs)
        self.total_stages = total_stages
        self.labels = []
        ctk.CTkLabel(self, text="Progress", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=6, pady=(6, 2)
        )
        for stage in range(1, total_stages + 1):
            label = ctk.CTkLabel(self, text=f"[ ] Stage {stage}", font=ctk.CTkFont(size=11))
            label.pack(anchor="w", padx=6, pady=1)
            self.labels.append(label)

    def update_stage(self, stage: int, message: str):
        for index, label in enumerate(self.labels, 1):
            if index < stage:
                label.configure(text=f"[OK] Stage {index}", text_color="green")
            elif index == stage:
                label.configure(text=f"[..] Stage {stage}: {message}", text_color="yellow")

    def complete(self):
        for index, label in enumerate(self.labels, 1):
            label.configure(text=f"[OK] Stage {index}", text_color="green")

    def reset(self):
        for index, label in enumerate(self.labels, 1):
            label.configure(text=f"[ ] Stage {index}", text_color="gray")
