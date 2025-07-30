import sys
import argparse
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import napari
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QMessageBox
)
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
 
def ridge_analysis(ridge, retchor):
    """
    Calculate mean and peak thickness from ridge and retchor arrays.
    Assumes retchor shape: (Z, Y, X) and ridge shape matches (Z, X).
    """
    if retchor.ndim != 3:
        raise ValueError(f"Expected 3D retchor array, got shape {retchor.shape}")
   
    # half_end = int(retchor.shape[1]/2)
    # retchor = retchor[:,half_end:,:]
 
    rdm = (retchor == 1).sum(axis=1)  # shape: (Z, X)
 
 
    if ridge.shape != rdm.shape:
        raise ValueError(f"Shape mismatch: ridge {ridge.shape} vs thickness map {rdm.shape}")
 
    thickness_vals = rdm[ridge == 4]
    if thickness_vals.size == 0:
        print("[WARN] No overlapping ridge found. Returning NaN.")
        return float('nan'), float('nan')
 
    return thickness_vals.mean(), thickness_vals.max()
 
def write_to_excel(retchor_path: Path, mean_t: float, peak_t: float,
                   excel_file: Path = Path('ridge_analysis_results.xlsx')):
    name = retchor_path.name
    suffix = '_ret_chor_seg.npy'
    if name.endswith(suffix):
        label = name[:-len(suffix)]
    else:
        label = name
 
    row = pd.DataFrame([{
        'Filename': label,
        'Mean': mean_t,
        'Peak': peak_t
    }])
 
    if not excel_file.exists():
        row.to_excel(excel_file, index=False)
    else:
        wb = load_workbook(excel_file)
        ws = wb.active
        ws.append([label, mean_t, peak_t])
        wb.save(excel_file)
 
def run_cli(ridge_path: Path, retchor_path: Path):
    if ridge_path.suffix.lower() == '.npy':
        ridge = np.load(ridge_path)
    elif ridge_path.suffix.lower() == '.mat':
        mat = loadmat(str(ridge_path))
        ridge = np.squeeze(mat.get('ridge'))
    else:
        print(f"Unsupported ridge format: {ridge_path.suffix}")
        sys.exit(1)
 
    retchor = np.load(retchor_path)
    mean_t, peak_t = ridge_analysis(ridge, retchor)
    print(f"{ridge_path.name} vs {retchor_path.name} → Mean: {mean_t:.2f}, Peak: {peak_t:.2f}")
    write_to_excel(retchor_path, mean_t, peak_t)
    print(f"Results written to '{Path.cwd() / 'ridge_analysis_results.xlsx'}'")
 
def collect_pairs(ridge_paths, retchor_paths):
    ret_map = {}
    for p in retchor_paths:
        if "_processed_" in p.name:
            prefix = p.name.split("_processed_")[0] + "_processed_"
            if prefix not in ret_map:
                ret_map[prefix] = p
            else:
                print(f"[WARN] Duplicate retchor prefix '{prefix}': {ret_map[prefix]} and {p}. Using the first.")
        else:
            print(f"[WARN] RetChor file '{p.name}' missing '_processed_'. Skipping.")
 
    pairs = []
    for r in ridge_paths:
        if "_processed_" in r.name:
            prefix = r.name.split("_processed_")[0] + "_processed_"
            if prefix in ret_map:
                pairs.append((r, ret_map[prefix]))
            else:
                print(f"[WARN] No matching RetChor file for Ridge '{r.name}' (prefix='{prefix}').")
        else:
            print(f"[WARN] Ridge file '{r.name}' missing '_processed_'. Skipping.")
 
    return pairs
 
def run_batch(ridge_dir: Path, retchor_dir: Path, viewer: napari.Viewer = None):
    ridge_paths = list(ridge_dir.rglob("*_en_face_ridge_labels.npy")) + \
                  list(ridge_dir.rglob("*.mat"))
    retchor_paths = list(retchor_dir.rglob("*_ret_chor_seg.npy"))
 
    pairs = collect_pairs(ridge_paths, retchor_paths)
 
    if not pairs:
        print("No matching ridge/retchor pairs found. Exiting.")
        return
 
    print(f"Found {len(pairs)} matched pairs. Processing...\n")
    for ridge_file, retchor_file in pairs:
        if ridge_file.suffix.lower() == ".npy":
            ridge_mask = np.load(ridge_file)
        else:
            mat = loadmat(str(ridge_file))
            ridge_mask = np.squeeze(mat.get("ridge"))
 
        retchor_mask = np.load(retchor_file)
 
        # DEBUG: Print shapes
        print(f"[DEBUG] retchor shape: {retchor_mask.shape}, ridge shape: {ridge_mask.shape}")
 
        mean_t, peak_t = ridge_analysis(ridge_mask, retchor_mask)
        print(f"{ridge_file.name} vs {retchor_file.name} → Mean: {mean_t:.2f}, Peak: {peak_t:.2f}")
 
        if viewer is not None:
            pts = np.array([[0, 0]])
            viewer.add_points(
                pts,
                text=[f"{ridge_file.name}\nMean: {mean_t:.2f}\nPeak: {peak_t:.2f}"],
                size=0,
                name=f"Batch: {ridge_file.stem}"
            )
 
        write_to_excel(retchor_file, mean_t, peak_t)
 
    print(f"\nBatch complete. Results appended to '{Path.cwd() / 'ridge_analysis_results.xlsx'}'.")
 
class BatchRidgeAnalysisWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.setObjectName('BatchRidgeAnalysisWidget')
 
        layout = QVBoxLayout()
        self.load_ridge_dir_btn = QPushButton("Load Ridge Folder")
        layout.addWidget(self.load_ridge_dir_btn)
 
        self.ridge_dir_lbl = QLabel("Ridge folder: —")
        layout.addWidget(self.ridge_dir_lbl)
 
        self.load_retchor_dir_btn = QPushButton("Load RetChor Folder")
        layout.addWidget(self.load_retchor_dir_btn)
 
        self.retchor_dir_lbl = QLabel("RetChor folder: —")
        layout.addWidget(self.retchor_dir_lbl)
 
        self.run_batch_btn = QPushButton("Run Batch Analysis")
        layout.addWidget(self.run_batch_btn)
 
        self.status_lbl = QLabel("Status: Waiting for folders...")
        layout.addWidget(self.status_lbl)
 
        self.setLayout(layout)
 
        self.ridge_dir_path = None
        self.retchor_dir_path = None
 
        self.load_ridge_dir_btn.clicked.connect(self._pick_ridge_dir)
        self.load_retchor_dir_btn.clicked.connect(self._pick_retchor_dir)
        self.run_batch_btn.clicked.connect(self._run_batch)
 
    def _pick_ridge_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Ridge Folder", str(Path.cwd()))
        if folder:
            self.ridge_dir_path = Path(folder)
            self.ridge_dir_lbl.setText(f"Ridge folder: {self.ridge_dir_path.name}")
            self._update_status()
 
    def _pick_retchor_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select RetChor Folder", str(Path.cwd()))
        if folder:
            self.retchor_dir_path = Path(folder)
            self.retchor_dir_lbl.setText(f"RetChor folder: {self.retchor_dir_path.name}")
            self._update_status()
 
    def _update_status(self):
        if self.ridge_dir_path and self.retchor_dir_path:
            self.status_lbl.setText("Status: Ready to run batch.")
        elif self.ridge_dir_path:
            self.status_lbl.setText("Status: Ridge folder loaded, awaiting RetChor folder.")
        elif self.retchor_dir_path:
            self.status_lbl.setText("Status: RetChor folder loaded, awaiting Ridge folder.")
        else:
            self.status_lbl.setText("Status: Waiting for folders...")
 
    def _run_batch(self):
        if not self.ridge_dir_path or not self.retchor_dir_path:
            QMessageBox.warning(self, "Missing Folder", "Please load both Ridge and RetChor folders first.")
            return
 
        self.run_batch_btn.setEnabled(False)
        self.status_lbl.setText("Status: Running batch analysis...")
        run_batch(self.ridge_dir_path, self.retchor_dir_path, viewer=self.viewer)
        self.run_batch_btn.setEnabled(True)
        self.status_lbl.setText("Status: Batch complete! Check console & Excel.")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ridge/retchor thickness analysis (single or batch).")
    parser.add_argument("ridge_file", nargs="?", help="(Optional) Path to a single ridge file (.npy or .mat).")
    parser.add_argument("retchor_file", nargs="?", help="(Optional) Path to a single retchor file (.npy).")
    parser.add_argument("--ridge_dir", type=str, help="(Optional) Path to folder containing ridge masks.")
    parser.add_argument("--retchor_dir", type=str, help="(Optional) Path to folder containing retchor masks.")
    args = parser.parse_args()
 
    if args.ridge_dir and args.retchor_dir:
        ridge_dir = Path(args.ridge_dir)
        retchor_dir = Path(args.retchor_dir)
        if not ridge_dir.is_dir() or not retchor_dir.is_dir():
            print("ERROR: One of the provided batch paths is not a directory.")
            sys.exit(1)
        run_batch(ridge_dir, retchor_dir)
        sys.exit(0)
    elif args.ridge_file and args.retchor_file:
        run_cli(Path(args.ridge_file), Path(args.retchor_file))
        sys.exit(0)
    else:
        viewer = napari.Viewer()
        widget = BatchRidgeAnalysisWidget(viewer)
        viewer.window.add_dock_widget(widget, name="Ridge Batch Analysis", area="right")
        napari.run()