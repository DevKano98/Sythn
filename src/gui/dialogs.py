"""Custom dialogs for SynthID Remover GUI."""
import sys
from pathlib import Path
import customtkinter as ctk
from typing import Optional, Callable

from config import APP_NAME, APP_VERSION


class AboutDialog(ctk.CTkToplevel):
    """About dialog showing app info."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title(f"About {APP_NAME}")
        self.geometry("400x300")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self._build_ui()
        self.center_on_parent()

    def _build_ui(self):
        frame = ctk.CTkFrame(self)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(frame, text=f"🛡️ {APP_NAME}",
                    font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 5))

        ctk.CTkLabel(frame, text=f"Version {APP_VERSION}",
                    font=ctk.CTkFont(size=14), text_color="gray").pack()

        ctk.CTkLabel(frame, text="Standalone SynthID watermark removal",
                    font=ctk.CTkFont(size=12)).pack(pady=10)

        info_text = """Built with:
• Qwen Image models (GGUF)
• YOLOv8 face detection
• CustomTkinter GUI
• PyTorch / Diffusers"""

        ctk.CTkLabel(frame, text=info_text,
                    font=ctk.CTkFont(size=11), justify="left").pack(pady=10)

        ctk.CTkButton(frame, text="Close", command=self.destroy).pack(pady=10)

    def center_on_parent(self):
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() - self.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog for configuring app preferences."""

    def __init__(self, master, settings_manager, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Settings")
        self.geometry("500x400")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self.settings_manager = settings_manager
        self.result = None

        self._build_ui()
        self.center_on_parent()

    def _build_ui(self):
        frame = ctk.CTkFrame(self)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(frame, text="⚙️ Settings",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(0, 10))

        # Theme
        self.theme_var = ctk.StringVar(value=self.settings_manager.settings.theme)
        ctk.CTkLabel(frame, text="Theme:").pack(anchor="w", padx=10)
        theme_combo = ctk.CTkComboBox(frame, values=["dark", "light", "system"],
                                       variable=self.theme_var)
        theme_combo.pack(fill="x", padx=10, pady=5)

        # Auto check updates
        self.updates_var = ctk.BooleanVar(value=self.settings_manager.settings.auto_check_updates)
        ctk.CTkSwitch(frame, text="Check for updates on startup",
                     variable=self.updates_var, onvalue=True, offvalue=False).pack(fill="x", padx=10, pady=5)

        # Show comparison
        self.comp_var = ctk.BooleanVar(value=self.settings_manager.settings.show_comparison_on_complete)
        ctk.CTkSwitch(frame, text="Show Before/After on completion",
                     variable=self.comp_var, onvalue=True, offvalue=False).pack(fill="x", padx=10, pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=20)

        ctk.CTkButton(btn_frame, text="Save", command=self._save).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy,
                     fg_color="gray").pack(side="left", padx=5)

    def _save(self):
        self.settings_manager.update(
            theme=self.theme_var.get(),
            auto_check_updates=self.updates_var.get(),
            show_comparison_on_complete=self.comp_var.get(),
        )
        ctk.set_appearance_mode(self.theme_var.get())
        self.destroy()

    def center_on_parent(self):
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() - self.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")


class ErrorDialog(ctk.CTkToplevel):
    """Error dialog with details."""

    def __init__(self, master, title: str, message: str, details: str = "",
                 **kwargs):
        super().__init__(master, **kwargs)
        self.title(title)
        self.geometry("500x300")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()

        self._build_ui(title, message, details)
        self.center_on_parent()

    def _build_ui(self, title: str, message: str, details: str):
        frame = ctk.CTkFrame(self)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(frame, text=f"❌ {title}",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="red").pack(pady=(10, 5))

        ctk.CTkLabel(frame, text=message,
                    font=ctk.CTkFont(size=12), wraplength=400).pack(pady=5)

        if details:
            textbox = ctk.CTkTextbox(frame, height=100)
            textbox.insert("0.0", details)
            textbox.configure(state="disabled")
            textbox.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(frame, text="OK", command=self.destroy).pack(pady=10)

    def center_on_parent(self):
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() - self.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")


class ModelDownloadDialog(ctk.CTkToplevel):
    """Dialog showing model download progress."""

    def __init__(self, master, missing_models: list, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Download Models")
        self.geometry("500x400")
        self.resizable(False, True)
        self.transient(master)
        self.grab_set()

        self.missing_models = missing_models
        self._build_ui()
        self.center_on_parent()

    def _build_ui(self):
        frame = ctk.CTkFrame(self)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(frame, text="📦 Missing Models",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))

        ctk.CTkLabel(frame, text=f"{len(self.missing_models)} model(s) not found.",
                    font=ctk.CTkFont(size=12)).pack(pady=5)

        # List missing models
        list_frame = ctk.CTkScrollableFrame(frame, height=150)
        list_frame.pack(fill="x", padx=10, pady=5)

        for model in self.missing_models:
            ctk.CTkLabel(list_frame, text=f"  ❌ {model}",
                        font=ctk.CTkFont(size=11), text_color="red").pack(anchor="w")

        ctk.CTkLabel(frame, text="Run: python download_models.py",
                    font=ctk.CTkFont(size=12, weight="bold")).pack(pady=10)

        ctk.CTkButton(frame, text="Close", command=self.destroy).pack(pady=10)

    def center_on_parent(self):
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() - self.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
