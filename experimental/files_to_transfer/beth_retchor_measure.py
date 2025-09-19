import sys
import argparse
from typing import Literal

import numpy as np
import polars as pl
import pandas as pd
from openpyxl import load_workbook
import napari
from numpy.typing import ArrayLike
from magicgui import magicgui
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QMessageBox,
)
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt


def ridge_analysis(
    ridge: ArrayLike,
    retchor: ArrayLike,
    scan_angle: float = 106.0,
    imaging_range: float = 6.0,
    refractive_index: float = 1.33,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    incedence_correction: bool = True,
    micrometer_output: bool = False,
    display_in_napari: bool = False,
    verbose: bool = True,
    debug: bool = False,
) -> tuple[float, float]:
    """Calculate mean and peak thickness from ridge and retchor segmentation labels.

    Calculates retinal thickness in pixels (optionally microns) using Bscan retinal slab segmentations and en face ridge segmentations.

    Args:
        ridge: Label array generated from en face UWF-OCT scan that highlighs the location of the fibrovascular ridge with  shape(Fast Axis, Slow Axis)
        retchor: Label array containing retinal and choroidal slabs generated from Bscans of UWF-OCT scan with shape(Fast Axis,Axial,Slow Axis)
        scan_angle: UWF-OCT field of view in degrees SCAN_ANGLE ASSUMES USING OLD CAMERA (circa 2023-2024); ADJUST WITH DIFFERENT DEVICE
        imaging_range: UWF-OCT axial depth in mm
        refractive_index: Refreactive index for calculating refraction within the eye (valid for contact handheld device)
        mode: Option to determine the center for correcting thickness measurements due to the non-orthogonal inciden of Ascans in the periphery assuming a spherical model of the eye
        incedence_correction: Flag for activating Ascan incedence angle correction
        micrometer_output: Falg for converting output to micrometers
        dispaly_in_napari: Dispaly selected retinal depth maps and other visual debugging informatino in napari viewer
        verbose: Print thickness information
        debug: Print statements helpful for debuging
    Returns:
        Tuple containing the arithmetic mean of the ridge thickness as well as the maximum measured thickness of the fibrovascular ridge
    Raises:
        ValueError if retchor is not 3-Dimensional
        ValueError if ridge shape does not match rdm shape

    Assumes retchor shape: (Z, Y, X) and ridge shape matches (Z, X).
    SCANWIDTH ASSUMES USING OLD CAMERA (circa 2023-2024); ADJUST WITH DIFFERENT DEVICE
    ALSO ASSUMES CENTER OF IMAGE IS CLOSE TO OPTICAL AXIS; NOT VAILD FOR NONTEMPORAL OR CENTRAL IMAGES
    """
    if retchor.ndim != 3:
        raise ValueError(f"Expected 3D retchor array, got shape {retchor.shape}")

    # half_end = int(retchor.shape[1]/2)
    # retchor = retchor[:,half_end:,:]

    retina_mask = (retchor == 1).astype(np.float64)
    rdm = retina_mask.sum(axis=1)  # shape: (Z, X)

    if ridge.shape != rdm.shape:
        raise ValueError(
            f"Shape mismatch: ridge {ridge.shape} vs thickness map {rdm.shape}"
        )

    thickness_vals = rdm[(ridge == 4) & (rdm != 0)]

    if debug:
        print(f"\n\nrdm type: {rdm.dtype}\nthickness type: {thickness_vals.dtype}\n\n")

    if verbose or debug:
        print(
            f"Raw thickness mean:\n{thickness_vals.mean()}\nRaw thickness max: {thickness_vals.max()}\n"
        )

    if incedence_correction:
        x, y = tuple([*rdm.shape[-2:]])

        match mode:
            case "fovea":
                print(
                    "Foveal center is not implemented yet performing calculation using scan center."
                )
                center_x = x // 2
                center_y = y // 2
            case "center":
                center_x = x // 2
                center_y = y // 2
            case "optic disk":
                print(
                    "Optic disk center is not implemented yet performing calculation using scan center."
                )
                center_x = x // 2
                center_y = y // 2

        scan_angle_from_center = (
            scan_angle // 2
        )  # TODO update this in future to account for differences
        min_scan_angle = 90 - scan_angle_from_center
        max_scan_angle = 180 - scan_angle_from_center

        if debug:
            print(f"center of image: {center_x, center_y}")
            print(f"\n\nx,y shapes: {x, y}\n\n")

        thetax, thetay = np.mgrid[
            0 - center_x : x - center_x, 0 - center_y : y - center_y
        ]
        # print(f"\n\ncenter_x,center_y: {center_x,center_y}\n\n")
        if debug:
            print(f"\n\ntheta_x,theta_y: {thetax, thetay},{thetax.shape, thetay.shape}\n\n")

        # x_conv = np.linspace(0-scan_angle_from_center,scan_angle-scan_angle_from_center,num=x)
        # y_conv = np.linspace(0-scan_angle_from_center,scan_angle-scan_angle_from_center,num=y)
        x_conv = np.linspace(-min_scan_angle, min_scan_angle, num=x)
        y_conv = np.linspace(-min_scan_angle, min_scan_angle, num=y)

        if debug:
            print(f"conversion shapes: {x_conv.shape},{y_conv.shape}\n\n")

        x_degree = np.repeat(x_conv[:, None], y, axis=1)
        y_degree = np.repeat(y_conv[None, :], x, axis=0)

        x_rad = x_degree / (2 * np.pi) / 4
        y_rad = y_degree / (2 * np.pi) / 4
        # x_degree = np.tile(x_conv[:,None],(1,y))
        # y_degree = np.tile(y_conv,(x,1))

        # y_degree = np.repeat(y_conv[:,None],y,axis=1)
        # print(f"\n\n\nthetax:{thetax[0],thetax[0].shape}\n\n")
        # print(f"x_conv: {x_conv},{x_conv.shape}\n\n")
        # print(f"y_conv: {y_conv},{y_conv.shape}\n\n")

        if debug:
            print(f"x_degree: {x_degree}\n{x_degree.shape}\n")
            print(f"y_degree: {y_degree}\n{y_degree.shape}\n")

        if debug:
            print(f"x_rad: {x_rad}\n{x_rad.shape}\n")
            print(f"y_rad: {y_rad}\n{y_rad.shape}\n")

        factor = np.cos(np.sqrt(x_rad**2 + y_rad**2))

        if debug:
            print(f"factor:{factor}\nfactor shape: {factor.shape}\n\n")
            print(f"factor at center:{factor[(center_x, center_y)]}\n\n")

        min_factor = np.unravel_index(factor.argmin(), factor.shape)
        max_factor = np.unravel_index(factor.argmax(), factor.shape)

        if debug:
            print(f"factor min: {factor.min(), min_factor}\n\n")
            print(f"factor max: {factor.max(), max_factor}\n\n")

        # thickness_vals_type = thickness_vals.dtype
        # thickness_vals = factor*thickness_vals.astype(thickness_vals_type)

        rdm_type = rdm.dtype
        adjusted_rdm = factor * rdm.astype(rdm_type)
        adjusted_rdm = np.clip(adjusted_rdm,a_min=0,a_max=None)

        corrected_thickness_vals = adjusted_rdm[(ridge == 4) & (adjusted_rdm != 0)]
        
        if verbose or debug:
            print(
                f"Corrected thickness mean:\n{corrected_thickness_vals.mean()}\nCorrected thickness max: {corrected_thickness_vals.max()}\n"
            )

    if thickness_vals.size == 0:
        print("[WARN] No overlapping ridge found. Returning NaN.")
        return float("nan"), float("nan")

    if incedence_correction:
        thickness_vals = corrected_thickness_vals
    if micrometer_output:
        conv_factor = imaging_range / retchor.shape[1] * 1000 / refractive_index # imaging range in mm / ascan len in pixels * um/mm * refractive index ratio = um/pixel
        thickness_vals = conv_factor * thickness_vals # um/pixel * pixels = um

        if verbose or debug:
            print(
                f"Micrometer thickness mean:\n{thickness_vals.mean()}\nMicrometer thickness max: {thickness_vals.max()}\n"
            )

    if display_in_napari:
        viewer = napari.Viewer()
        viewer.add_image(rdm)
        if incedence_correction:
            viewer.add_image(x_rad)
            viewer.add_image(y_rad)
            viewer.add_image(factor)
            viewer.add_image(adjusted_rdm)

    return thickness_vals.mean(), thickness_vals.max()


def write_to_csv(
    retchor_path: Path,
    mean_t: float,
    peak_t: float,
    output_file: Path = Path("ridge_analysis_results.csv"),
    display_dataframe: bool = False,
    write_output: bool = True,
):
    name = retchor_path.name
    suffix = "_ret_chor_seg.npy"
    if name.endswith(suffix):
        label = name[: -len(suffix)]
    else:
        label = name

    current_thickness_df = pl.DataFrame(
        {"Filename": label, "Mean": mean_t, "Peak": peak_t}
    )

    if output_file.exists():
        prior_thickness_df = pl.read_csv(output_file)
        current_thickness_df = pl.concat(
            [prior_thickness_df, current_thickness_df], how="vertical"
        )
    else:
        output_file.parent.mkdir(parents=True,exist_ok=True)

    if display_dataframe:
        with pl.Config(tbl_cols=-1, tbl_rows=-1, set_tbl_width_chars=2000):
            print(current_thickness_df)

    if write_output:
        current_thickness_df.write_csv(output_file)


def write_to_excel(
    retchor_path: Path,
    mean_t: float,
    peak_t: float,
    excel_file: Path = Path("ridge_analysis_results.xlsx"),
):
    name = retchor_path.name
    suffix = "_ret_chor_seg.npy"
    if name.endswith(suffix):
        label = name[: -len(suffix)]
    else:
        label = name

    row = pd.DataFrame([{"Filename": label, "Mean": mean_t, "Peak": peak_t}])

    if not excel_file.exists():
        row.to_excel(excel_file, index=False)
    else:
        wb = load_workbook(excel_file)
        ws = wb.active
        ws.append([label, mean_t, peak_t])
        wb.save(excel_file)


def run_cli(ridge_path: Path, retchor_path: Path):
    if ridge_path.suffix.lower() == ".npy":
        ridge = np.load(ridge_path)
    elif ridge_path.suffix.lower() == ".mat":
        mat = loadmat(str(ridge_path))
        ridge = np.squeeze(mat.get("ridge"))
    else:
        print(f"Unsupported ridge format: {ridge_path.suffix}")
        sys.exit(1)

    retchor = np.load(retchor_path)
    mean_t, peak_t = ridge_analysis(ridge, retchor)
    print(
        f"{ridge_path.name} vs {retchor_path.name} → Mean: {mean_t:.2f}, Peak: {peak_t:.2f}"
    )
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
                print(
                    f"[WARN] Duplicate retchor prefix '{prefix}': {ret_map[prefix]} and {p}. Using the first."
                )
        else:
            print(f"[WARN] RetChor file '{p.name}' missing '_processed_'. Skipping.")

    pairs = []
    for r in ridge_paths:
        if "_processed_" in r.name:
            prefix = r.name.split("_processed_")[0] + "_processed_"
            if prefix in ret_map:
                pairs.append((r, ret_map[prefix]))
            else:
                print(
                    f"[WARN] No matching RetChor file for Ridge '{r.name}' (prefix='{prefix}')."
                )
        else:
            print(f"[WARN] Ridge file '{r.name}' missing '_processed_'. Skipping.")

    return pairs


def run_batch(
    ridge_dir: Path,
    retchor_dir: Path,
    output: Literal[".xlsx", ".csv", "none"] = ".xlsx",
    output_dir_path=Path("ridge_analysis_output"),
    display_dataframe: bool = False,
    scan_angle: float = 106.0,
    imaging_range: float = 6.0,
    refractive_index: float = 1.33,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    incedence_correction: bool = True,
    micrometer_output: bool = False,
    display_in_napari: bool = False,
    verbose: bool = True,
    debug: bool = False,
    viewer: napari.Viewer = None,
):
    # ridge_paths = list(ridge_dir.rglob("*_en_face_ridge_labels.npy")) + \
    #               list(ridge_dir.rglob("*.mat"))
    # retchor_paths = list(retchor_dir.rglob("*_ret_chor_seg.npy"))
    ridge_paths = list(ridge_dir.rglob("*.npy")) + list(ridge_dir.rglob("*.mat"))
    retchor_paths = list(retchor_dir.rglob("*.npy"))

    pairs = collect_pairs(ridge_paths, retchor_paths)

    if not pairs:
        print("No matching ridge/retchor pairs found. Exiting.")
        return

    print(f"Found {len(pairs)} matched pairs. Processing...\n")
    for ridge_file, retchor_file in pairs:
        if ridge_file.suffix.lower() == ".npy":
            ridge_label = np.load(ridge_file)
        else:
            mat = loadmat(str(ridge_file))
            ridge_label = np.squeeze(mat.get("ridge"))

        retchor_labels = np.load(retchor_file)

        # DEBUG: Print shapes
        if debug:
            print(
                f"[DEBUG] retchor shape: {retchor_labels.shape}, ridge shape: {ridge_label.shape}"
            )

        mean_t, peak_t = ridge_analysis(
            ridge_label,
            retchor_labels,
            scan_angle=scan_angle,
            imaging_range=imaging_range,
            refractive_index=refractive_index,
            mode=mode,
            incedence_correction=incedence_correction,
            micrometer_output=micrometer_output,
            display_in_napari=display_in_napari,
            verbose=verbose,
            debug=debug,
        )

        print(
            f"{ridge_file.name} vs {retchor_file.name} → Mean: {mean_t:.2f}, Peak: {peak_t:.2f}"
        )

        if viewer is not None:
            pts = np.array([[0, 0]])
            viewer.add_points(
                pts,
                text=[f"{ridge_file.name}\nMean: {mean_t:.2f}\nPeak: {peak_t:.2f}"],
                size=0,
                name=f"Batch: {ridge_file.stem}",
            )

        match output:
            case ".xlsx":
                write_to_excel(
                    retchor_file,
                    mean_t,
                    peak_t,
                    excel_file=output_dir_path / "ridge_analysis_results.xlsx",
                )
            case ".csv":
                write_to_csv(
                    retchor_file,
                    mean_t,
                    peak_t,
                    output_file=output_dir_path / "ridge_analysis_results.csv",
                    display_dataframe=display_dataframe,
                    write_output=True,
                )
            case "none":
                    write_to_csv(
                    retchor_file,
                    mean_t,
                    peak_t,
                    output_file=output_dir_path / "ridge_analysis_results.csv",
                    display_dataframe=display_dataframe,
                    write_output=False,
                )

    # TODO fix this as it is incorrect
    print(
        f"\nBatch complete. Results appended to '{Path.cwd() / 'ridge_analysis_results.xlsx'}'."
    )


# class BatchRidgeAnalysisWidget(QWidget):
#     def __init__(self, viewer: napari.Viewer):
#         super().__init__()
#         self.viewer = viewer
#         self.setObjectName("BatchRidgeAnalysisWidget")

#         layout = QVBoxLayout()
#         self.load_ridge_dir_btn = QPushButton("Load Ridge Folder")
#         layout.addWidget(self.load_ridge_dir_btn)

#         self.ridge_dir_lbl = QLabel("Ridge folder: —")
#         layout.addWidget(self.ridge_dir_lbl)

#         self.load_retchor_dir_btn = QPushButton("Load RetChor Folder")
#         layout.addWidget(self.load_retchor_dir_btn)

#         self.retchor_dir_lbl = QLabel("RetChor folder: —")
#         layout.addWidget(self.retchor_dir_lbl)

#         self.run_batch_btn = QPushButton("Run Batch Analysis")
#         layout.addWidget(self.run_batch_btn)

#         self.status_lbl = QLabel("Status: Waiting for folders...")
#         layout.addWidget(self.status_lbl)

#         self.setLayout(layout)

#         self.ridge_dir_path = None
#         self.retchor_dir_path = None

#         self.load_ridge_dir_btn.clicked.connect(self._pick_ridge_dir)
#         self.load_retchor_dir_btn.clicked.connect(self._pick_retchor_dir)
#         self.run_batch_btn.clicked.connect(self._run_batch)

#     def _pick_ridge_dir(self):
#         folder = QFileDialog.getExistingDirectory(
#             self, "Select Ridge Folder", str(Path.cwd())
#         )
#         if folder:
#             self.ridge_dir_path = Path(folder)
#             self.ridge_dir_lbl.setText(f"Ridge folder: {self.ridge_dir_path.name}")
#             self._update_status()

#     def _pick_retchor_dir(self):
#         folder = QFileDialog.getExistingDirectory(
#             self, "Select RetChor Folder", str(Path.cwd())
#         )
#         if folder:
#             self.retchor_dir_path = Path(folder)
#             self.retchor_dir_lbl.setText(
#                 f"RetChor folder: {self.retchor_dir_path.name}"
#             )
#             self._update_status()

#     def _update_status(self):
#         if self.ridge_dir_path and self.retchor_dir_path:
#             self.status_lbl.setText("Status: Ready to run batch.")
#         elif self.ridge_dir_path:
#             self.status_lbl.setText(
#                 "Status: Ridge folder loaded, awaiting RetChor folder."
#             )
#         elif self.retchor_dir_path:
#             self.status_lbl.setText(
#                 "Status: RetChor folder loaded, awaiting Ridge folder."
#             )
#         else:
#             self.status_lbl.setText("Status: Waiting for folders...")

#     def _run_batch(self):
#         if not self.ridge_dir_path or not self.retchor_dir_path:
#             QMessageBox.warning(
#                 self,
#                 "Missing Folder",
#                 "Please load both Ridge and RetChor folders first.",
#             )
#             return

#         self.run_batch_btn.setEnabled(False)
#         self.status_lbl.setText("Status: Running batch analysis...")
#         run_batch(self.ridge_dir_path, self.retchor_dir_path, viewer=self.viewer)
#         self.run_batch_btn.setEnabled(True)
#         self.status_lbl.setText("Status: Batch complete! Check console & Excel.")


@magicgui(
    ridge_dir={"label": "Path to folder containing ridge masks.", "mode": "d"},
    retchor_dir={"label": "Path to folder containing retchor masks", "mode": "d"},
    output_dir_path={"label": "Path to output results", "mode": "d"},
    call_button="Run Batch Analysis",
)
def generate_enface_with_labels(
    ridge_dir: Path = Path(r"F:\Beth_RetChor_Stuff\old_calc\ridge"),
    retchor_dir: Path = Path(r"F:\Beth_RetChor_Stuff\old_calc\retchor"),
    output_dir_path: Path = Path(r"F:\Beth_RetChor_Stuff\output"),
    output: Literal[".xlsx", ".csv", "none"] = ".xlsx",
    display_dataframe: bool = True,
    scan_angle: float = 106.0,
    imaging_range: float = 6.0,
    refractive_index: float = 1.33,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    incedence_correction: bool = True,
    micrometer_output: bool = False,
    display_in_napari: bool = False,
    verbose: bool = True,
    debug: bool = False,
):
    run_batch(
        ridge_dir,
        retchor_dir,
        output=output,
        output_dir_path=output_dir_path,
        display_dataframe=display_dataframe,
        scan_angle=scan_angle,
        imaging_range=imaging_range,
        refractive_index=refractive_index,
        mode=mode,
        incedence_correction=incedence_correction,
        micrometer_output=micrometer_output,
        display_in_napari=display_in_napari,
        verbose=verbose,
        debug=debug,
    )


if __name__ == "__main__":
    generate_enface_with_labels.show(run=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run ridge/retchor thickness analysis (single or batch).")
#     parser.add_argument("ridge_file", nargs="?", help="(Optional) Path to a single ridge file (.npy or .mat).")
#     parser.add_argument("retchor_file", nargs="?", help="(Optional) Path to a single retchor file (.npy).")
#     parser.add_argument("--ridge_dir", type=str, help="(Optional) Path to folder containing ridge masks.")
#     parser.add_argument("--retchor_dir", type=str, help="(Optional) Path to folder containing retchor masks.")
#     args = parser.parse_args()

#     if args.ridge_dir and args.retchor_dir:
#         ridge_dir = Path(args.ridge_dir)
#         retchor_dir = Path(args.retchor_dir)
#         if not ridge_dir.is_dir() or not retchor_dir.is_dir():
#             print("ERROR: One of the provided batch paths is not a directory.")
#             sys.exit(1)
#         run_batch(ridge_dir, retchor_dir)
#         sys.exit(0)
#     elif args.ridge_file and args.retchor_file:
#         run_cli(Path(args.ridge_file), Path(args.retchor_file))
#         sys.exit(0)
#     else:
#         viewer = napari.Viewer()
#         widget = BatchRidgeAnalysisWidget(viewer)
#         viewer.window.add_dock_widget(widget, name="Ridge Batch Analysis", area="right")
#         napari.run()
