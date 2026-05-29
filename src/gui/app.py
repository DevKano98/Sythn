"""CustomTkinter GUI for the ComfyUI-backed SynthID Remover."""
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from config import APP_NAME, APP_VERSION, DEFAULTS, INPUT_DIR, OUTPUT_DIR
from src.comfy import ComfyModelManager, ComfyPipeline, PipelineConfig, PipelineResult
from src.gui.settings import SettingsManager
from src.gui.widgets import BeforeAfterSlider, ImagePreview, ModelStatusWidget, ProgressCard

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class SynthIDRemoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.settings
        self.model_manager = ComfyModelManager()
        self.pipeline = ComfyPipeline(output_dir=str(OUTPUT_DIR))
        self.current_image_path: Optional[str] = None
        self.output_image_path: Optional[str] = None
        self.processing_thread: Optional[threading.Thread] = None

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry(f"{self.settings.window_width}x{self.settings.window_height}")
        self.minsize(1100, 760)

        self._build_ui()
        self._load_settings_to_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, height=54)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header.grid_propagate(False)
        ctk.CTkLabel(header, text=APP_NAME, font=ctk.CTkFont(size=22, weight="bold")).place(
            relx=0.02, rely=0.5, anchor="w"
        )
        self.status_label = ctk.CTkLabel(header, text="Ready")
        self.status_label.place(relx=0.98, rely=0.5, anchor="e")

        content = ctk.CTkFrame(self)
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(content)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        tabs = ctk.CTkTabview(left)
        tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        tabs.add("Input")
        tabs.add("Before/After")
        self.tabs = tabs
        self.preview = ImagePreview(tabs.tab("Input"))
        self.preview.pack(expand=True, fill="both", padx=6, pady=6)
        self.comparison = BeforeAfterSlider(tabs.tab("Before/After"))
        self.comparison.pack(expand=True, fill="both", padx=6, pady=6)

        buttons = ctk.CTkFrame(left)
        buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkButton(buttons, text="Load", command=self._load_image).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(buttons, text="Clear", command=self._clear_image, fg_color="gray").pack(side="left", padx=4)
        ctk.CTkButton(buttons, text="Output", command=self._open_output).pack(side="right", padx=4)

        controls = ctk.CTkScrollableFrame(content, width=360)
        controls.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.model_status = ModelStatusWidget(controls, self.model_manager)
        self.model_status.pack(fill="x", padx=4, pady=(0, 8))

        self._section(controls, "Prompts")
        self.positive_prompt = ctk.CTkTextbox(controls, height=70)
        self.positive_prompt.pack(fill="x", padx=4, pady=4)
        self.negative_prompt = ctk.CTkTextbox(controls, height=70)
        self.negative_prompt.pack(fill="x", padx=4, pady=4)

        self._section(controls, "Denoise")
        self.denoise_auto = ctk.CTkSwitch(controls, text="Auto from image size", onvalue=True, offvalue=False)
        self.denoise_auto.pack(fill="x", padx=4, pady=4)
        self.denoise_slider, self.denoise_value = self._slider_row(controls, "Manual", 0.01, 1.0, 99, "0.20")

        self._section(controls, "Face")
        self.face_switch = ctk.CTkSwitch(controls, text="Enable face pass", onvalue=True, offvalue=False)
        self.face_switch.pack(fill="x", padx=4, pady=4)
        self.face_scale, self.face_scale_value = self._slider_row(controls, "Face denoise scale", 0.1, 1.0, 90, "0.50")

        self._section(controls, "Sampling")
        self.steps_slider, self.steps_value = self._slider_row(controls, "Steps", 1, 30, 29, "9", as_int=True)
        seed_frame = ctk.CTkFrame(controls)
        seed_frame.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(seed_frame, text="Seed").pack(side="left", padx=5)
        self.seed_entry = ctk.CTkEntry(seed_frame)
        self.seed_entry.pack(side="left", fill="x", expand=True, padx=5)

        self._section(controls, "Output")
        self.format_combo = ctk.CTkComboBox(controls, values=["png", "jpg", "webp"])
        self.format_combo.pack(fill="x", padx=4, pady=4)
        self.quality_slider, self.quality_value = self._slider_row(controls, "Quality", 1, 100, 99, "95", as_int=True)

        self.progress_card = ProgressCard(controls, total_stages=8)
        self.progress_card.pack(fill="x", padx=4, pady=8)
        self.progress = ctk.CTkProgressBar(controls)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=4, pady=4)

        self.process_btn = ctk.CTkButton(controls, text="Remove SynthID", height=42, command=self._start_processing)
        self.process_btn.pack(fill="x", padx=4, pady=(10, 4))
        self.cancel_btn = ctk.CTkButton(controls, text="Cancel", command=self._cancel_processing, fg_color="darkred")
        self.cancel_btn.pack(fill="x", padx=4, pady=4)
        self.cancel_btn.configure(state="disabled")

    def _section(self, parent, text: str):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=4, pady=(12, 2)
        )

    def _slider_row(self, parent, label, start, end, steps, initial, as_int=False):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(frame, text=label).pack(side="left", padx=5)
        slider = ctk.CTkSlider(frame, from_=start, to=end, number_of_steps=steps)
        slider.pack(side="left", fill="x", expand=True, padx=5)
        value = ctk.CTkLabel(frame, text=initial, width=44)
        value.pack(side="right", padx=5)
        slider.configure(command=lambda v: value.configure(text=f"{int(float(v))}" if as_int else f"{float(v):.2f}"))
        return slider, value

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda _e: self._load_image())
        self.bind("<Control-r>", lambda _e: self._start_processing())
        self.bind("<Control-s>", lambda _e: self._open_output())
        self.bind("<Escape>", lambda _e: self._cancel_processing())

    def _load_settings_to_ui(self):
        s = self.settings
        self.positive_prompt.insert("0.0", s.last_positive_prompt)
        self.negative_prompt.insert("0.0", s.last_negative_prompt)
        self.denoise_auto.select() if s.default_denoise_auto else self.denoise_auto.deselect()
        self.face_switch.select() if s.default_face_enhancement else self.face_switch.deselect()
        self.denoise_slider.set(s.default_denoise_manual)
        self.face_scale.set(s.default_face_denoise_scale)
        self.steps_slider.set(s.default_steps)
        self.seed_entry.insert(0, str(DEFAULTS["seed"]))
        self.format_combo.set(s.default_output_format)
        self.quality_slider.set(s.default_output_quality)

    def _save_settings_from_ui(self):
        self.settings_manager.update(
            window_width=self.winfo_width(),
            window_height=self.winfo_height(),
            window_x=self.winfo_x(),
            window_y=self.winfo_y(),
            last_positive_prompt=self.positive_prompt.get("0.0", "end").strip(),
            last_negative_prompt=self.negative_prompt.get("0.0", "end").strip(),
            default_denoise_auto=bool(self.denoise_auto.get()),
            default_denoise_manual=float(self.denoise_slider.get()),
            default_face_enhancement=bool(self.face_switch.get()),
            default_face_denoise_scale=float(self.face_scale.get()),
            default_steps=int(self.steps_slider.get()),
            default_output_format=self.format_combo.get(),
            default_output_quality=int(self.quality_slider.get()),
        )

    def _load_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select image",
            initialdir=self.settings.last_input_dir or str(INPUT_DIR),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.current_image_path = path
            self.settings_manager.update(last_input_dir=str(Path(path).parent))
            self.preview.set_image(path)
            self.tabs.set("Input")
            self.status_label.configure(text=f"Loaded {Path(path).name}")

    def _clear_image(self):
        self.current_image_path = None
        self.output_image_path = None
        self.preview.clear()
        self.progress.set(0)
        self.progress_card.reset()
        self.status_label.configure(text="Ready")

    def _open_output(self):
        if sys.platform == "win32":
            os.startfile(str(OUTPUT_DIR))
        else:
            import subprocess
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(OUTPUT_DIR)], check=False)

    def _start_processing(self):
        if not self.current_image_path:
            self.status_label.configure(text="Load an image first")
            return
        try:
            seed = int(self.seed_entry.get())
        except ValueError:
            seed = -1
        config = PipelineConfig(
            image_path=self.current_image_path,
            positive_prompt=self.positive_prompt.get("0.0", "end").strip(),
            negative_prompt=self.negative_prompt.get("0.0", "end").strip(),
            denoise_auto=bool(self.denoise_auto.get()),
            denoise_manual=float(self.denoise_slider.get()),
            face_enhancement=bool(self.face_switch.get()),
            face_denoise_scale=float(self.face_scale.get()),
            steps=int(self.steps_slider.get()),
            seed=seed,
            output_format=self.format_combo.get(),
            output_quality=int(self.quality_slider.get()),
        )
        self.process_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.progress_card.reset()
        self.processing_thread = threading.Thread(target=self._run_pipeline, args=(config,), daemon=True)
        self.processing_thread.start()

    def _run_pipeline(self, config: PipelineConfig):
        def progress(stage, total, message):
            self.after(0, lambda: self._update_progress(stage, total, message))
        result = self.pipeline.run(config, progress)
        self.after(0, lambda: self._processing_complete(result))

    def _update_progress(self, stage, total, message):
        self.progress.set(stage / total)
        self.progress_card.update_stage(stage, message)
        self.status_label.configure(text=message)

    def _processing_complete(self, result: PipelineResult):
        self.process_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if result.success and result.output_path:
            self.output_image_path = result.output_path
            self.progress.set(1)
            self.progress_card.complete()
            self.status_label.configure(text=f"Saved {Path(result.output_path).name}")
            self.comparison.set_images(self.current_image_path, result.output_path)
            self.tabs.set("Before/After")
        else:
            self.status_label.configure(text=result.error_message or "Processing failed")

    def _cancel_processing(self):
        self.pipeline.cancel()
        self.status_label.configure(text="Cancelling...")

    def _on_close(self):
        self._save_settings_from_ui()
        self.pipeline.stop()
        self.destroy()


def main():
    app = SynthIDRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
