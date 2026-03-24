#!/usr/bin/env python3

"""
STAG+ GUI
GUI for Automatic Image Tagger + Quality Scorer
Fork of DIVISIO STAG with IQ & Aesthetic Assessment
"""

import os
import sys
import threading
import ctypes
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import webbrowser

import huggingface_hub
from huggingface_hub import hf_hub_download
from tktooltip import ToolTip

from stag import SKTagger, VERSION


class TextRedirector:
    """Redirects stdout/stderr to a tkinter Text widget."""

    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, out_str):
        self.text_widget.insert(tk.END, out_str, (self.tag,))
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass


class StagPlusGUI:
    """Main GUI class for the STAG+ application."""

    DEFAULT_PREFIX = "st"
    MODEL_REPO_ID = "xinyu1205/recognize-anything-plus-model"
    MODEL_FILENAME = "ram_plus_swin_large_14m.pth"

    def __init__(self, root):
        self.root = root
        self.stop_event = threading.Event()

        self.apply_hidpi_scaling()

        self.root.title(f"STAG+ v{VERSION}")
        self.setup_grid_configuration()
        self.create_widgets()

    def apply_hidpi_scaling(self):
        """Apply HiDPI scaling for better display on high-resolution screens."""
        if hasattr(ctypes, 'windll'):
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        try:
            screen_width_px = self.root.winfo_screenwidth()
            screen_width_mm = self.root.winfo_screenmmwidth()
            screen_dpi = screen_width_px / (screen_width_mm / 25.4)
            scaling_factor = screen_dpi / 96
            if scaling_factor > 1:
                self.root.tk.call('tk', 'scaling', scaling_factor)
        except Exception as e:
            print(f"Could not determine DPI scaling factor: {e}")

    def setup_grid_configuration(self):
        """Configure the grid layout."""
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(7, weight=1)

    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        self.create_input_fields()
        self.create_option_checkboxes()
        self.create_feature_checkboxes()
        self.create_scoring_options()
        self.create_buttons()
        self.create_output_area()
        self.create_branding()

    def create_input_fields(self):
        """Create the directory and prefix input fields."""
        # Image directory input
        ttk.Label(self.root, text="Image Directory:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.entry_imagedir = ttk.Entry(self.root, width=50)
        self.entry_imagedir.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.browse_button = ttk.Button(
            self.root, text="Browse", command=self.browse_directory
        )
        self.browse_button.grid(row=0, column=3, padx=5, pady=5)

        # Prefix input
        ttk.Label(self.root, text="Tag Prefix:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.entry_prefix = ttk.Entry(self.root, width=20)
        self.entry_prefix.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.entry_prefix.insert(0, self.DEFAULT_PREFIX)

    def create_option_checkboxes(self):
        """Create general option checkboxes."""
        options_frame = ttk.LabelFrame(self.root, text="Options", padding=5)
        options_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        # Skip already tagged
        self.var_skip = tk.BooleanVar(value=True)
        self.force_checkbox = ttk.Checkbutton(
            options_frame, text="Skip already processed images",
            variable=self.var_skip
        )
        self.force_checkbox.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ToolTip(self.force_checkbox, msg=(
            "Skip images that already have tags/scores from a previous run."
        ))

        # Simulate only
        self.var_test = tk.BooleanVar()
        self.test_checkbox = ttk.Checkbutton(
            options_frame, text="Simulate only (no writes)",
            variable=self.var_test
        )
        self.test_checkbox.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ToolTip(self.test_checkbox, msg="Analyze images but don't write changes to disk.")

        # Darktable-compatible filenames
        self.var_prefer_exact_filenames = tk.BooleanVar()
        self.prefer_exact_filenames_checkbox = ttk.Checkbutton(
            options_frame, text="Darktable-compatible filenames",
            variable=self.var_prefer_exact_filenames
        )
        self.prefer_exact_filenames_checkbox.grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        ToolTip(self.prefer_exact_filenames_checkbox, msg=(
            "Create PICT0001.JPG.XMP instead of PICT0001.XMP"
        ))

    def create_feature_checkboxes(self):
        """Create feature toggle checkboxes for Tag/IQ/Aesthetics."""
        features_frame = ttk.LabelFrame(self.root, text="Features", padding=5)
        features_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        # Tagging (RAM+)
        self.var_tagging = tk.BooleanVar(value=True)
        self.tag_checkbox = ttk.Checkbutton(
            features_frame, text="🏷️  Content Tagging (RAM+)",
            variable=self.var_tagging
        )
        self.tag_checkbox.grid(row=0, column=0, padx=10, pady=4, sticky=tk.W)
        ToolTip(self.tag_checkbox, msg=(
            "AI content tagging — identifies objects, scenes, and concepts.\n"
            "Writes tags like: st|landscape, st|mushroom, st|sky\n"
            "Uses Recognize Anything Model (3.2 GB download on first run)."
        ))

        # Image Quality (BRISQUE)
        self.var_iq = tk.BooleanVar(value=False)
        self.iq_checkbox = ttk.Checkbutton(
            features_frame, text="📐  Image Quality (BRISQUE)",
            variable=self.var_iq
        )
        self.iq_checkbox.grid(row=0, column=1, padx=10, pady=4, sticky=tk.W)
        ToolTip(self.iq_checkbox, msg=(
            "Technical quality assessment — detects blur, noise, and distortion.\n"
            "Writes tags like: iq|low, iq|medium, iq|high\n"
            "Pure math — no model download needed. Very fast."
        ))

        # Aesthetics (NIMA)
        self.var_aes = tk.BooleanVar(value=False)
        self.aes_checkbox = ttk.Checkbutton(
            features_frame, text="🎨  Aesthetic Score (NIMA)",
            variable=self.var_aes
        )
        self.aes_checkbox.grid(row=0, column=2, padx=10, pady=4, sticky=tk.W)
        ToolTip(self.aes_checkbox, msg=(
            "Aesthetic assessment — rates composition, appeal, and visual quality.\n"
            "Writes tags like: aes|low, aes|medium, aes|high\n"
            "Also writes xmp:Rating stars (1-5) for darktable.\n"
            "Uses MobileNetV2 backbone (no extra download)."
        ))

    def create_scoring_options(self):
        """Create scoring configuration options."""
        scoring_frame = ttk.LabelFrame(self.root, text="Scoring", padding=5)
        scoring_frame.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        # Number of bins
        ttk.Label(scoring_frame, text="Quality bins:").grid(
            row=0, column=0, padx=5, pady=2, sticky=tk.W
        )
        self.var_bins = tk.IntVar(value=3)
        bins_3 = ttk.Radiobutton(scoring_frame, text="3 (low / medium / high)",
                                  variable=self.var_bins, value=3)
        bins_3.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        bins_5 = ttk.Radiobutton(scoring_frame, text="5 (poor → excellent)",
                                  variable=self.var_bins, value=5)
        bins_5.grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)

        # Write stars
        self.var_write_stars = tk.BooleanVar(value=True)
        stars_checkbox = ttk.Checkbutton(
            scoring_frame, text="Write aesthetic score as star rating (xmp:Rating)",
            variable=self.var_write_stars
        )
        stars_checkbox.grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky=tk.W)
        ToolTip(stars_checkbox, msg=(
            "Maps the aesthetic score to 1-5 stars in the standard XMP Rating field.\n"
            "Darktable shows these as star ratings in lighttable for quick culling."
        ))

    def create_buttons(self):
        """Create action buttons."""
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=5, column=0, columnspan=4, pady=10)

        self.run_button = ttk.Button(
            btn_frame, text="▶  Run STAG+", command=self.run_tagger
        )
        self.run_button.grid(row=0, column=0, padx=10)

        self.cancel_button = ttk.Button(
            btn_frame, text="⏹  Cancel", command=self.cancel_tagger
        )
        self.cancel_button.grid(row=0, column=1, padx=10)
        self.cancel_button.config(state='disabled')

    def create_output_area(self):
        """Create the output text area."""
        ttk.Label(self.root, text="Output:").grid(
            row=6, column=0, columnspan=4, padx=5, pady=(10, 0), sticky=tk.W
        )

        text_frame = ttk.Frame(self.root)
        text_frame.grid(row=7, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        self.text_output = tk.Text(text_frame, height=18, wrap="word")
        self.text_output.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(text_frame, command=self.text_output.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.text_output['yscrollcommand'] = scrollbar.set

        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

    def create_branding(self):
        """Create branding elements."""
        brand_frame = ttk.Frame(self.root)
        brand_frame.grid(row=8, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        ttk.Label(brand_frame, text=f"STAG+ v{VERSION}").pack(side=tk.LEFT, padx=5)
        ttk.Label(brand_frame, text="•").pack(side=tk.LEFT)
        ttk.Label(brand_frame, text=f"Fork of DIVISIO STAG").pack(side=tk.LEFT, padx=5)

        link = ttk.Label(brand_frame, text="Original project", foreground="blue", cursor="hand2")
        link.pack(side=tk.LEFT, padx=5)
        link.bind("<Button-1>", lambda e: self.open_webpage("https://github.com/DIVISIO-AI/stag"))

    def run_tagger(self):
        """Start the processing in a separate thread."""
        # Validate at least one feature is enabled
        if not self.var_tagging.get() and not self.var_iq.get() and not self.var_aes.get():
            messagebox.showwarning("No features selected",
                                   "Please enable at least one feature (Tagging, IQ, or Aesthetics).")
            return

        imagedir = self.entry_imagedir.get()
        if not imagedir or not os.path.isdir(imagedir):
            messagebox.showwarning("No directory", "Please select a valid image directory.")
            return

        self.stop_event.clear()
        self.update_ui_state(running=True)

        prefix = self.entry_prefix.get() or self.DEFAULT_PREFIX
        force = not self.var_skip.get()
        test = self.var_test.get()
        prefer_exact_filenames = self.var_prefer_exact_filenames.get()

        enable_tagging = self.var_tagging.get()
        enable_iq = self.var_iq.get()
        enable_aes = self.var_aes.get()
        n_bins = self.var_bins.get()
        write_stars = self.var_write_stars.get()

        threading.Thread(
            target=self.run_tagger_thread,
            args=(imagedir, prefix, force, test, prefer_exact_filenames,
                  enable_tagging, enable_iq, enable_aes, n_bins, write_stars)
        ).start()

    def run_tagger_thread(self, imagedir, prefix, force, test, prefer_exact_filenames,
                          enable_tagging, enable_iq, enable_aes, n_bins, write_stars):
        """Run the tagger in a separate thread."""
        sys.stdout = TextRedirector(self.text_output, "stdout")
        sys.stderr = TextRedirector(self.text_output, "stderr")

        print("Starting STAG+...")

        # Download RAM+ model if tagging enabled
        pretrained = None
        if enable_tagging:
            dl_dir = os.path.join(
                huggingface_hub.constants.HF_HUB_CACHE,
                "models--xinyu1205--recognize-anything-plus-model"
            )
            if not os.path.isdir(dl_dir):
                self.show_startup_alert()
                print("First run — downloading the RAM+ model file.")
                print("This is only done once.")

            try:
                pretrained = hf_hub_download(
                    repo_id=self.MODEL_REPO_ID,
                    filename=self.MODEL_FILENAME
                )
            except Exception as e:
                print(f"Error downloading model: {e}")
                self.root.after(0, lambda: self.update_ui_state(running=False))
                return

        try:
            tagger = SKTagger(
                model_path=pretrained or "",
                image_size=384,
                force_tagging=force,
                test_mode=test,
                prefer_exact_filenames=prefer_exact_filenames,
                tag_prefix=prefix,
                enable_tagging=enable_tagging,
                enable_iq=enable_iq,
                enable_aes=enable_aes,
                n_bins=n_bins,
                write_stars=write_stars,
            )

            if not self.stop_event.is_set():
                tagger.enter_dir(imagedir, self.stop_event)

            print("\nSTAG+ has finished. Have a nice day.")
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            self.root.after(0, lambda: self.update_ui_state(running=False))

    def cancel_tagger(self):
        """Cancel the running process."""
        print("Cancelling...")
        self.stop_event.set()

    def browse_directory(self):
        """Open a file dialog to select an image directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.entry_imagedir.delete(0, tk.END)
            self.entry_imagedir.insert(0, directory)

    def update_ui_state(self, running):
        """Update the UI state based on whether processing is running."""
        all_inputs = [
            self.entry_imagedir, self.entry_prefix, self.browse_button,
            self.run_button, self.force_checkbox, self.test_checkbox,
            self.prefer_exact_filenames_checkbox,
            self.tag_checkbox, self.iq_checkbox, self.aes_checkbox,
        ]
        if running:
            for widget in all_inputs:
                widget.config(state='disabled')
            self.cancel_button.config(state='normal')
        else:
            for widget in all_inputs:
                widget.config(state='normal')
            self.cancel_button.config(state='disabled')

    def open_webpage(self, url):
        """Open a URL in the default web browser."""
        webbrowser.open_new(url)

    def resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and PyInstaller."""
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def show_startup_alert(self):
        """Show a message box about downloading the model on first run."""
        messagebox.showinfo(
            "Welcome to STAG+",
            "STAG+ needs to download the Recognize Anything model from HuggingFace "
            "(~3.2 GB). This only happens once. Subsequent runs will start instantly."
        )


def main():
    """Main entry point."""
    root = tk.Tk()
    app = StagPlusGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
