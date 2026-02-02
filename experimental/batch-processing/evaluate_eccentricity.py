""" """

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List

from kan import KAN
from magicgui.widgets import (
    create_widget,
    Container,
    FileEdit,
    FloatSpinBox,
    LineEdit,
    PushButton,
    CheckBox,
)
from magicgui import magicgui
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import napari

# from napari.utils.notifications import show_info
import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import normaltest, probplot
import torch
from tqdm import tqdm

from experimental_utils import (
    aggregate_per_mask,
    denormalize_from_range,
    equidistant_pixel_error_by_degree,
    equidistant_mask_by_degree,
    find_spatial_outliers,
    grab_center_ellipsoid_optimized,
    grab_center_window,
    kan_regress_and_plot,
    linear_regress_and_plot,
    normalize_to_range,
    visualize_cleaned_map_comparison,
    visualize_map_quad_dashboard,
    visualize_maximizable_quad_dashboard,
    visualize_single_map,
)
from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_registration import CurvCorrectSettings
from napari_cool_tools_registration._fitting_funcs import (
    extract_surfaces_and_curve_correct_coordinates,
    generate_angle_of_incidence_map,
)
from curve_correct_utils import (
    # CurvCorrectSettings,
    equidistant_loss,
    generate_noisy_ellipsoid_sample_data,
    get_incidence_angle_torch,
    get_pixel_spacing_and_padding,
)


@dataclass
class CurvCorrectSettings:
    pivot_point: float = 19.2
    imaging_range: float = 12.0
    reference_motor_position: float = 85.0
    imaging_motor_position: float = 85.0
    imaging_motor_position_delta: float = 0.0
    refractive_index: float = 1.33
    scan_angle: float = 100


def update_save_dataframe(
    output_dataframe_filepath: Path,
    output_dataframe: pl.DataFrame,
    verbose: bool = False,
):
    """"""
    if output_dataframe_filepath.exists():
        existing_dataframe = pl.read_csv(output_dataframe_filepath)
        # output_dataframe = pl.concat([existing_dataframe,output_dataframe])
        output_dataframe = existing_dataframe.vstack(output_dataframe).unique(
            maintain_order=True
        )
    if verbose:
        print(output_dataframe)
    output_dataframe.write_csv(output_dataframe_filepath)


def gen_binned_df(incidence_df):
    """"""

    binned_incidence_df = (
        incidence_df.with_columns(
            [
                ((pl.col("angles") * 2).round() / 2).alias("angles_rounded"),
                ((pl.col("percent_error") * 2).round() / 2).alias(
                    "percent_error_rounded"
                ),
            ]
        )
        .with_columns(
            pl.col("angles_rounded").cut(breaks=list(range(0, 91, 5))).alias("bin")
        )
        .group_by("bin")
        .agg(
            [
                pl.col("angles_rounded").mean().alias("mean_angle"),
                pl.col("percent_error_rounded").mean().alias("mean_percent_error"),
            ]
        )
    )

    qbinned_incidence_df = (
        incidence_df.with_columns(
            [
                # pl.col("angles").qcut(50).alias("quantile_bin")
                pl.col("angles")
                .qcut(
                    100,
                    labels=[f"Q_{idx}" for idx in np.arange(0, 100)],
                    allow_duplicates=True,
                )
                .alias("quantile_bin")
            ]
        )
        .group_by("quantile_bin")
        .agg(
            [
                pl.col("angles").mean().alias("mean_angles"),
                pl.col("percent_error").mean().alias("mean_percent_error"),
                (
                    pl.col("percent_error").std()
                    / pl.col("percent_error").count().sqrt()
                ).alias("sem"),
            ]
        )
        .sort("mean_angles")
    )

    return binned_incidence_df, qbinned_incidence_df


class FileGeneratorWidget(Container):
    """A magicgui widget to generate and step through files in a directory."""

    def __init__(self):
        super().__init__()

        # Widgets for user input
        self.clean_labels_directory_input = FileEdit(
            label="Choose a Clean Labels directory",
            mode="d",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\_quick_test_for_figure"),
            # value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
            # value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.topology_directory_input = FileEdit(
            label="Choose a Topology map directory",
            mode="d",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output"),
            # value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.fovea_file_input = FileEdit(
            label="Choose a data mask directory",
            mode="r",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"E:\_Beth_Thickness_Calculations\FoveDiscandLPCApoints.csv"),
        )
        self.output_directory_input = FileEdit(
            label="Choose an output directory",
            mode="d",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output"),
            # value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.output_file_name = LineEdit(
            label="Choose a output directory",
            value="aggregate_error_plot.png",
            # value="ridge_heights.csv"
        )
        self.clean_labels_extension_input = LineEdit(
            label="Clean Labels File extension", value="*_clean.npz"
        )
        self.save_aggregate_data = CheckBox(
            label="Save aggregate eccentricity data",
            value=False,
        )
        self.show_aggregate_data = CheckBox(
            label="Show aggregate eccentricity data",
            value=False,
        )

        self.generate_button = PushButton(text="Create File Generator")

        # Widget to display the current file
        self.next_file_button = PushButton(text="Next File")

        # Widget to process all files
        self.process_all_files_button = PushButton(text="Process All Files")

        # Widget to calculate aggregate data
        self.calculate_aggregate_data = PushButton(text="Calculate aggregate data")

        # Layout the widgets
        self.extend(
            [
                self.clean_labels_directory_input,
                self.topology_directory_input,
                self.fovea_file_input,
                self.output_directory_input,
                self.output_file_name,
                self.clean_labels_extension_input,
                self.generate_button,
                self.next_file_button,
                self.process_all_files_button,
                self.save_aggregate_data,
                self.show_aggregate_data,
                self.calculate_aggregate_data,
            ]
        )

        # fovea dataframe
        self.fovea_df = pl.read_csv(self.fovea_file_input.value)

        # collect values across images
        self.eccentricity_error_values = []

        # curve correction settings
        self.cc_settings = CurvCorrectSettings(
            imaging_range=12.0,  # 6., #12.,
            imaging_motor_position_delta=6.0,  # 0., #6.,
        )

        # State for the file generator
        self.file_generator = None

        # Connect button signals to methods
        self.generate_button.clicked.connect(self._create_generator)
        # self.generate_button.clicked.connect(self._generate_mask_list)
        self.next_file_button.clicked.connect(self._next_file)
        self.process_all_files_button.clicked.connect(self._process_all_files)
        self.calculate_aggregate_data.clicked.connect(
            self._aggregate_eccentricity_error_values
        )

    def _aggregate_eccentricity_error_values(self):
        """"""
        self.show_plot = self.show_aggregate_data.value
        self.save_plot = self.save_aggregate_data.value

        aligned_eccentricity_values = [
            arr
            for arr in self.eccentricity_error_values
            if arr.shape[0] == (self.cc_settings.scan_angle // 2) - 6
        ]
        print([arr.shape[0] for arr in self.eccentricity_error_values])
        print([arr.shape[0] for arr in aligned_eccentricity_values])
        # print(eccentricity_error_values)

        # all_sampled_pixels = np.stack(eccentricity_error_values,axis=0)
        all_sampled_pixels = np.stack(aligned_eccentricity_values, axis=0)
        all_sampled_pixels = all_sampled_pixels.mean(axis=0)

        if self.show_plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                np.arange(len(all_sampled_pixels)),
                all_sampled_pixels,
                marker="o",
                linestyle="-",
                color="b",
            )

            ax.set_xlabel("Eccentricity (degrees)")
            ax.set_ylabel("Precent Error")
            ax.set_title("Aggregate Percent Error vs. Ececentricity(Degrees)")

            formatter = mtick.PercentFormatter(xmax=100.0, decimals=1)
            ax.yaxis.set_major_formatter(formatter)

            ax.grid(True, linestyle="--", alpha=0.6)
            plt.show()

        if self.save_plot:
            plt.savefig(self.output_directory_input.value / self.output_file_name.value)

    def _create_generator(self):
        """Creates a generator for files with the specified extension."""
        directory = self.clean_labels_directory_input.value
        extension = self.clean_labels_extension_input.value
        self.extension = extension

        # output file path
        self.output_file_path = (
            self.output_directory_input.value / self.output_file_name.value
        )

        self.save_plot = self.save_aggregate_data.value
        self.show_plot = self.show_aggregate_data.value

        if not directory or not Path(directory).is_dir():
            print("Error: Invalid directory.")
            return

        # Create a generator for the files
        self.file_generator = (
            f for f in Path(directory).rglob(extension) if f.is_file()
        )
        print(f"Generator created for '{extension}' files in {directory}.")

        self.file_list = list(
            f for f in Path(directory).rglob(extension) if f.is_file()
        )
        print(
            f"List containing {len(self.file_list)} entries created for '{extension}' files in {directory}."
        )

    def _next_file(self):
        """Advances the generator and displays the next file."""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            next_file = next(self.file_generator)
            print(str(next_file))

            self._process_file(next_file)

            print("File Processed.")
            # TODO add viewer back
            # viewer.add_labels(label_data,name=next_file.stem) #,name=metadata["name"],properties=metadata["properties"])
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

    def _process_all_files(self):
        """Processes all files in the generator"""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            file_path_pbar = tqdm(self.file_generator)
            for file_path in file_path_pbar:
                file_path_pbar.set_description(f"Processing: {file_path}")

                self._process_file(file_path=file_path)

            print("End of files. Processing complete.")
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

    def _process_file(self, file_path):
        """"""

        output_dir = self.output_directory_input.value

        verbose = False  # True # TODO make this an option on the widget
        save_plots = False
        show_plots = True
        scan_angle_pad = -6  # 0 #6
        save_df = False
        display_in_viewer = True

        extension = Path(self.extension).stem.replace("*", "")

        # get id
        id = file_path.stem.replace(extension, "")
        id = id.replace("_ret_chor_seg", "")  # TODO replace hard coding

        # Check for fovea data
        # fovea_data_exists = self.fovea_df.select(
        #     pl.col("Filename").str.contains(id).any()
        # ).item()

        fovea_center_df = self.fovea_df.filter(
            pl.col("Filename").str.contains(id)
        ).select("Fovea_Y", "Fovea_X")

        if not fovea_center_df.is_empty():
            fovea_center = fovea_center_df.to_numpy().squeeze()

            if verbose:
                print(f"Fovea center: {fovea_center}")
        else:
            if verbose:
                print("Missing fovea position...skipping.")
            return

        # load data
        if file_path.suffix == ".npz":
            data, attributes, layer_type = npz_file_reader(
                file_path,
                return_layer=True,
                verbose=False,  # verbose=True
            )[0]
        elif file_path.suffix == ".npy":
            data = np.load(file_path)

        # isolate deired data
        retina = data == 1

        if "metadata" in attributes:
            if "motor_position" in attributes["metadata"]:
                imaging_motor_position = attributes["metadata"]["motor_position"]
                self.cc_settings.imaging_motor_position = imaging_motor_position / 1000
            else:
                if verbose:
                    print("Missing motor position...skipping.")
                return
        else:
            if verbose:
                print("Missing metadata...skipping.")
            return

        # get pixel spacing and padding for the volume
        pixel_spacing, padding_pixel = get_pixel_spacing_and_padding(
            cc_settings=self.cc_settings, axial_data_shape=data.shape[1], verbose=False
        )

        # get surface points and curve correct there positions
        if retina.sum() > 0:
            (
                cc_micro_thickness_map,
                cc_pixel_thickness_map,
                curv_ret_points,
                curv_rpe_points,
                retina_points,
                rpe_points,
                cc_retina_nn_points,
                raw_micro_thickness_map,
                raw_pixel_thickness_map,
            ) = extract_surfaces_and_curve_correct_coordinates(
                retina,
                pixel_spacing=pixel_spacing,
                padding_pixel=padding_pixel,
                refractive_index=self.cc_settings.refractive_index,
            )
        else:
            return

        curv_points_of_interest = curv_rpe_points
        points_of_interest = rpe_points

        nearest_neighbor_dir = curv_ret_points - cc_retina_nn_points

        nn_angles_of_incidence = get_incidence_angle_torch(
            torch.as_tensor(curv_points_of_interest, dtype=torch.bfloat16),
            torch.as_tensor(nearest_neighbor_dir, dtype=torch.bfloat16),
            use_degrees=True,
        )

        nn_angle_of_incidence_map = generate_angle_of_incidence_map(
            points_of_interest, nn_angles_of_incidence, (840, 800)
        )

        epsilon = np.finfo(float).eps
        absolute_diffence_map = np.abs(raw_micro_thickness_map - cc_micro_thickness_map)
        # absolute_diffence_map = np.abs(raw_micron_thickness_map-thickness_map)
        micron_error_map = (
            absolute_diffence_map / (cc_micro_thickness_map + epsilon)
        ).astype("float32") * 100
        # micron_error_map = (absolute_diffence_map/(thickness_map + epsilon)).astype("float32") * 100

        if verbose:
            print(
                f"micron diff argmin: {np.unravel_index(raw_micro_thickness_map.argmin(), cc_micro_thickness_map.shape)}\n"
            )

        # outlier cleanup using interquartile range
        outlier_threshold = (
            3.0  # 1.5 # 1.5 identifies outliers while 3.0 identifies extreme outliers
        )

        q1, q3 = np.percentile(micron_error_map, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (outlier_threshold * iqr)
        upper_bound = q3 + (outlier_threshold * iqr)

        outliers = micron_error_map > upper_bound
        micron_error_map_rem_outliers = micron_error_map.copy()
        micron_error_map_rem_outliers[outliers] = 0

        # spatial outlier calculation
        # mask = np.zeros_like(retina,dtype=bool)
        # mask[tuple(retina_points.T)] = True
        # outliers,_,_ = find_spatial_outliers(retina[None,None,:,:,:],kernel_size=3,std_threshold=3.) TODO experiment with this function with surface voxels only
        # outliers,_,_ = find_spatial_outliers(data[None,None,:,:,:],kernel_size=3,std_threshold=3.)

        fovea_mask_by_degree = equidistant_mask_by_degree(
            micron_error_map.shape,
            center=fovea_center,
            scan_angle=self.cc_settings.scan_angle,
            scan_angle_pad=0, #scan_angle_pad,
        )

        error_by_degree_from_fovea = aggregate_per_mask(
            micron_error_map,
            # micron_error_map_rem_outliers,
            multi_mask=fovea_mask_by_degree,
        )

        central_mask_by_degree = equidistant_mask_by_degree(
            micron_error_map.shape,
            center=None,
            scan_angle=self.cc_settings.scan_angle,
            scan_angle_pad=scan_angle_pad,            
        )

        # validity_mask = np.stack(validity_mask).sum(axis=0)
        validity_mask = np.zeros_like(cc_micro_thickness_map,dtype="uint8")
        for i, mask in enumerate(central_mask_by_degree):
            validity_mask = validity_mask + mask * (i + 1)
        validity_mask = validity_mask > 0

        mask_rings2 = np.zeros_like(cc_micro_thickness_map, dtype="uint8")
        for i, mask in enumerate(fovea_mask_by_degree):
            mask_rings2 = mask_rings2 + mask * (i + 1)

        valid_mask2 = mask_rings2 > 0

        sampled_pixels, sampled_masks = equidistant_pixel_error_by_degree(
            micron_error_map,
            scan_angle=self.cc_settings.scan_angle,
            scan_angle_pad=scan_angle_pad,
            # micron_error_map, scan_angle=self.cc_settings.scan_angle, scan_angle_pad=-6
            # micron_error_map_rem_outliers, scan_angle=self.cc_settings.scan_angle, scan_angle_pad=-6
        )
        self.eccentricity_error_values.append(np.array(sampled_pixels))

        mask_rings = np.zeros_like(cc_micro_thickness_map, dtype="uint8")
        # mask_rings = np.zeros_like(thickness_map,dtype="uint8")
        for i, mask in enumerate(sampled_masks):
            mask_rings = mask_rings + mask * (i + 1)

        valid_mask = mask_rings > 0

        valid_thickness_map = cc_micro_thickness_map * valid_mask
        # valid_thickness_map = thickness_map*valid_mask
        valid_raw_micron_thickness_map = raw_micro_thickness_map * valid_mask
        # valid_raw_micron_thickness_map = raw_micron_thickness_map*valid_mask
        valid_absolute_diffence_map = absolute_diffence_map * valid_mask
        valid_micron_error_map = micron_error_map * valid_mask
        valid_micron_error_map_rem_outliers = micron_error_map_rem_outliers * valid_mask

        valid_nn_angle_of_incidence_map = nn_angle_of_incidence_map * valid_mask

        # make dataframe prep and plot
        center_values = grab_center_window(micron_error_map,window_shape=3,return_numpy=True)
        center_values2 = grab_center_ellipsoid_optimized(micron_error_map,neighbor_count=1,return_numpy=True)
        min_val,mean_val = center_values.min(),center_values.mean()
        min_val2,mean_val2 = center_values2.min(),center_values.mean()
        center_df = error_by_degree_from_fovea[0].min()
        scan_angle_df = (
            pl.DataFrame(
                {
                    # "angles": (np.arange(len(sampled_pixels)) + 1).astype(np.int64),
                    # "percent_error": np.asarray(sampled_pixels).astype(np.float64),
                    "angles": (np.arange(len(error_by_degree_from_fovea)) + 1).astype(np.int64),
                    "percent_error": np.asarray(error_by_degree_from_fovea).astype(np.float64),
                }
            )
            .with_columns(id=pl.lit(id))
            .select("id", "angles", "percent_error")
        )

        incidence_df = (
            pl.DataFrame(
                {
                    "angles": nn_angle_of_incidence_map.flatten().astype(np.float64),
                    "percent_error": micron_error_map.flatten().astype(np.float64),
                }
            )
            .with_columns(id=pl.lit(id))
            .select("id", "angles", "percent_error")
        )

        scan_angle_output_filepath = (
            self.output_directory_input.value / "scan_angle_error_df.csv"
        )
        incidence_angle_output_filepath = (
            self.output_directory_input.value / "incidence_angle_error_df.csv"
        )

        if save_df:
            update_save_dataframe(
                output_dataframe_filepath=scan_angle_output_filepath,
                output_dataframe=scan_angle_df,
                verbose=False,
            )
            update_save_dataframe(
                output_dataframe_filepath=incidence_angle_output_filepath,
                output_dataframe=incidence_df,
                verbose=False,
            )

        # make dataframe prep and plot
        valid_incidence_df = pl.DataFrame(
            {
                "angles": valid_nn_angle_of_incidence_map.flatten(),
                "percent_error": valid_micron_error_map.flatten(),
            }
        )

        valid_incidence_df = (
            valid_incidence_df.with_columns(
                # pl.col("angles").cut(breaks=np.arange(0, 90)).alias("degree_bin"),
                pl.col("angles")
                .qcut(
                    100,
                    labels=[f"Q{idx}" for idx in np.arange(1, 101)],
                    allow_duplicates=True,
                )
                .alias("degree_bin"),
            )
            .group_by("degree_bin")
            .agg(pl.col("percent_error").mean().alias("mean_percent_error"))
            .sort("degree_bin")
        )

        incidence_df2 = pl.DataFrame(
            {
                "angles": (nn_angle_of_incidence_map * valid_mask).flatten(),
                "percent_error": (micron_error_map * valid_mask).flatten(),
            }
        )

        binned_incidence_df, qbinned_incidence_df = gen_binned_df(incidence_df)
        binned_incidence_df2, qbinned_incidence_df2 = gen_binned_df(incidence_df2)

        if verbose:
            print(f"binned_incidence_df:\n{binned_incidence_df}")
            print(f"qbinned_incidence_df:\n{qbinned_incidence_df}")

            print(f"valid_incidence_df\n{valid_incidence_df}")

            print(f"binned_incidence_df2\n{binned_incidence_df2}")
            print(f"qbinned_incidence_df2\n{qbinned_incidence_df2}")

        if save_plots or show_plots:
            # q_angle = incidence_df["angles"].to_numpy()
            # q_error = incidence_df['percent_error'].to_numpy()

            # linear_regress_and_plot(
            #     (
            #         q_angle,
            #         q_error,
            #     ),
            #     ("Incidence Angle", r"% Error"),
            #     None, #sem_data=qbinned_incidence_df["sem"].to_numpy(),
            #     title=r"% Error vs Incidence Angle with Standard Error",
            # )

            # q_angle_means = qbinned_incidence_df["mean_angles"].to_numpy()
            # q_error_means = qbinned_incidence_df["mean_percent_error"].to_numpy()

            # q_angle_means2 = qbinned_incidence_df2["mean_angles"].to_numpy()
            # q_error_means2 = qbinned_incidence_df2["mean_percent_error"].to_numpy()

            # linear_regress_and_plot(
            #     (
            #         q_angle_means,
            #         q_error_means,
            #     ),
            #     ("Incidence Angle by Quantile", r"% Error by Quantile"),
            #     None, #sem_data=qbinned_incidence_df["sem"].to_numpy(),
            #     title=r"Quantile-Binned % Error vs Incidence Angle with Standard Error",
            # )

            # r_squared,residuals =  kan_regress_and_plot(
            #     (
            #         q_angle_means,
            #         q_error_means,
            #     ),
            #     ("Incidence Angle by Quantile", r"% Error by Quantile"),
            #     title=r"Quantile-Binned % Error vs Incidence Angle with Standard Error",
            #     generate_symbolic_formula=False,
            # )

            # linear_regress_and_plot(
            #     (
            #         q_angle_means2,
            #         q_error_means2,
            #     ),
            #     ("Incidence Angle by Quantile", r"% Error by Quantile"),
            #     None, #sem_data=qbinned_incidence_df["sem"].to_numpy(),
            #     title=r"Quantile-Binned % Error vs Incidence Angle with Standard Error",
            # )

            # r_squared,residuals =  kan_regress_and_plot(
            #     (
            #         q_angle_means2,
            #         q_error_means2,
            #     ),
            #     ("Incidence Angle by Quantile 2", r"% Error by Quantile"),
            #     title=r"Quantile-Binned % Error vs Incidence Angle with Standard Error",
            #     generate_symbolic_formula=False,
            # )

            plt.figure(figsize=(10, 6))
            # fig2, ax2 = plt.subplots(figsize=(10, 6))
            plt.plot(
                np.arange(len(sampled_pixels)),
                sampled_pixels,
                marker="o",
                linestyle="-",
                color="b",
            )

            plt.xlabel("Scan Angle (Degrees)")
            plt.ylabel("Precent Error")
            plt.title(f"{id} Percent Error vs. Scan Angle (Degrees)")

            formatter = mtick.PercentFormatter(xmax=100.0, decimals=1)
            plt.gca().yaxis.set_major_formatter(formatter)

            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()

            output_plot_dir = output_dir / "Temp_Thickness_Output"
            # output_plot_dir = Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
            output_plot_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 6))
            # fig3, ax3 = plt.subplots(figsize=(10, 6))
            plt.plot(
                np.arange(len(valid_incidence_df)),
                valid_incidence_df["mean_percent_error"],
                marker="o",
                linestyle="-",
                color="b",
            )

            plt.xlabel("Angle of Incidence by Quantile (degrees)")
            plt.ylabel("Mean Precent Error by Quantile")
            plt.title(
                f"{id} Percent Error vs. Angle of Incidence by Quantile (degrees)"
            )

            formatter = mtick.PercentFormatter(xmax=100.0, decimals=1)
            plt.gca().yaxis.set_major_formatter(formatter)

            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()

            visualize_single_map(valid_thickness_map, title="Thickness")
            visualize_single_map(valid_raw_micron_thickness_map, title="Raw Thickness")
            visualize_single_map(
                valid_absolute_diffence_map, title="Absolute Difference in Thickness"
            )
            visualize_single_map(
                valid_micron_error_map_rem_outliers, title="Percent Error"
            )
            visualize_single_map(valid_micron_error_map, title="Percent Error")
            visualize_map_quad_dashboard(
                valid_thickness_map,
                valid_raw_micron_thickness_map,
                valid_absolute_diffence_map,
                # valid_micron_error_map,
                valid_micron_error_map_rem_outliers,
            )

        if save_plots:
            plt.savefig(output_plot_dir / f"{id}_error_plot.png")

        if show_plots:
            plt.show()

        ##################################################################################

        if display_in_viewer:
            viewer.add_labels(retina, name=f"{id}_retina")
            # viewer.add_labels(outliers*10,name=f"{id}_outliers")

            # viewer.add_image(raw_micron_thickness_map,name=f"{id}_raw_pixel_thickness_map")

            # viewer.add_image(thickness_map,name=f"{id}_pixel_thickness_map")

            viewer.add_image(micron_error_map)
            viewer.add_image(nn_angle_of_incidence_map)
            viewer.add_image(micron_error_map_rem_outliers)
            viewer.add_image(
                cc_micro_thickness_map, name=f"{id}_cc_pixel_thickness_map"
            )
            viewer.add_image(
                raw_micro_thickness_map, name=f"{id}_raw_pixel_thickness_map"
            )
            viewer.add_labels(mask_rings2, name=f"{id}_mask_ring2_{i}",visible=False)
            # viewer.add_labels(mask_rings, name=f"{id}_mask_ring_{i}",visible=False)
            # viewer.add_labels(~valid_mask, name=f"{id}_invalid_mask_{i}",visible=False)
            viewer.add_labels(~validity_mask, name=f"{id}_invalid_mask_{i}",visible=False)
            viewer.add_points(
                fovea_center,
                size=10,
                border_color="yellow",
                face_color="red",
                name="foveal_center",
            )


if __name__ == "__main__":
    # Create the widget and show it
    viewer = napari.Viewer()
    my_widget = FileGeneratorWidget()
    viewer.window.add_dock_widget(my_widget)
    viewer.show()
    napari.run()
    # my_widget.show(run=True)
