""" """

from pathlib import Path

from kornia.morphology import erosion
from magicgui import magicgui
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import napari
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from tqdm import tqdm
import torch

from napari_cool_tools_registration._fitting_funcs import scan_angle_fit_func

from evaluate_eccentricity_funcs import (
    apply_shell_filter,
    calculate_label_statistics_from_summary,
    compute_shell_statistics,
    distance_to_nonzero,
    generate_calculation_mask,
    generate_indexed_map,
    generate_shell_indices,
    project_to_image,
    generate_shell_mask,
    perform_hybrid_label_analysis,
    plot_label_comparison_with_intervals,
    plot_label_distribution_with_ribbon,
    plot_label_ribbon,
)

def percent_error_calculations(
    cc_thickness_map: np.ndarray | torch.Tensor,
    raw_thickness_map: np.ndarray | torch.Tensor,
    incidence_map: np.ndarray | torch.Tensor,
    id_number: str,
    fovea_pt: tuple([int, int]),
    desired_offset: int = 10,
    viewer: napari.Viewer | None = None,
    display_in_napari: bool = False,
    display_graphs: bool = False,
):
    """"""
    cc_thickness_map = torch.as_tensor(cc_thickness_map, dtype=torch.float32)
    raw_thickness_map = torch.as_tensor(raw_thickness_map, dtype=torch.float32)
    incidence_map = torch.as_tensor(incidence_map, dtype=torch.float32)
    epsilon = torch.finfo(float).eps

    absolute_difference_map = torch.abs(cc_thickness_map - raw_thickness_map)

    # outlier cleanup using interquartile range
    outlier_threshold = (
        3.0  # 1.5 # 1.5 identifies outliers while 3.0 identifies extreme outliers
    )
    # remove extreme high outliers from absolute difference map
    q1, q3 = torch.nanquantile(absolute_difference_map, torch.tensor([0.25, 0.75]))
    iqr = q3 - q1
    lower_bound = q1 - (outlier_threshold * iqr)
    upper_bound = q3 + (outlier_threshold * iqr)

    outliers = absolute_difference_map > upper_bound

    # absolute_difference_map[outliers] = torch.nan

    micron_error_map = (
        absolute_difference_map / (torch.abs(cc_thickness_map) + epsilon)
    ).to(torch.float32) * 100

    # calculate by angle of incidence
    incidence_df = pl.DataFrame(
        {
            "incidence": incidence_map.ravel(),
            "percent_error": micron_error_map.ravel(),
        }
    )

    threshold = 100  # 58 #100 #500 #1000

    error_by_incidence_df = (
        incidence_df.group_by("incidence")
        .agg(
            [
                pl.col("percent_error").mean().alias("mean_error"),
                pl.len().alias("count"),  # Add a count column
            ]
        )
        .filter(pl.col("count") >= threshold)
    )

    analysis_output = perform_hybrid_label_analysis(
        input_dataset=incidence_df,
        value_column_name="percent_error",
        label_column_name="incidence",
        minimum_sample_threshold=threshold,
        confidence_level=0.95,
    )

    if analysis_output is not None:
        comparison_table, population_baseline, icc = analysis_output
    else:
        print(f"Label analysis failed.")
        return None

    if display_graphs:
        print(f"Intraclass Correlation Coefficient Angle of Incidence Between Laser and Tissue: {icc}")

        plot_label_distribution_with_ribbon(
            comparison_table=comparison_table,
            global_mean_reference=population_baseline,
            label_column_name="incidence",
            plot_title=r"Average % Error vs UWF-OCT Angle of Incedence between Laser and RPE",
            x_axis_label="Angle of Incedence (Degrees)",
            y_axis_label=r"% Error",
            shrunken_mean_legend_label="Shrunken Mean",
            raw_mean_legend_label=r"Original % Error Mean",
            confidence_interval_legend_label="95% Confidence Interval",
            baseline_legend_label="Baseline Mean",
            show_population_baseline=True,
        )

    # drop count column
    error_by_incidence_df.drop("count")

    # calculate by scan angle or angle from fovea
    # center_pt = fovea_pt
    center_pt = (micron_error_map.size(0) // 2, micron_error_map.size(1) // 2)

    shell_indices = generate_shell_indices(
        micron_error_map.shape,
        center=center_pt,
        device="cpu",
    )

    f_shell_indices = generate_shell_indices(
        micron_error_map.shape,
        center=fovea_pt,
        device="cpu",
    )
    
    calc_mask,input_mask,shell_mask,long_axis = generate_calculation_mask(
        shell_indices=shell_indices,
        cc_thickness_map=cc_thickness_map,
        center_pt=center_pt,
        desired_offset=desired_offset,
    )

    f_calc_mask,f_input_mask,f_shell_mask,f_long_axis = generate_calculation_mask(
        shell_indices=f_shell_indices,
        cc_thickness_map=cc_thickness_map,
        center_pt=fovea_pt,
        desired_offset=desired_offset,
    )    

    # calculate statistics for unmasked data
    mean, std, count = compute_shell_statistics(
        micron_error_map, shell_indices=shell_indices, mask=calc_mask
    )

    f_mean, f_std, f_count = compute_shell_statistics(
        micron_error_map, shell_indices=f_shell_indices, mask=f_calc_mask
    )   

    # project statistics to heatmap
    mean_heat = project_to_image(statistics=mean, shell_indices=shell_indices)
    std_heat = project_to_image(statistics=std, shell_indices=shell_indices)
    count_heat = project_to_image(statistics=count, shell_indices=shell_indices)

    f_mean_heat = project_to_image(statistics=f_mean, shell_indices=f_shell_indices)
    f_std_heat = project_to_image(statistics=f_std, shell_indices=f_shell_indices)
    f_count_heat = project_to_image(statistics=f_count, shell_indices=f_shell_indices)

    # visualize shells
    shell_map = generate_indexed_map(shell_indices)

    f_shell_map = generate_indexed_map(f_shell_indices)

    # map angles to nonlinear imperical angle function
    shell_to_degree_ratio = 50 / (
        long_axis // 2
    )  # 420 shells along the long axis = 50 ish degrees

    # cald angle ranges
    angle_range = np.arange(len(mean)) * shell_to_degree_ratio
    angle_range_mask = angle_range <= 50
    # if verbose:
    #     print(angle_range[angle_range_mask].max())

    f_angle_range = np.arange(len(f_mean)) * shell_to_degree_ratio
    f_angle_range_mask = f_angle_range <= 50

    # if verbose:
    #     print(f_angle_range[f_angle_range_mask].max())

    # generate dataframes
    scan_angle_df = pl.DataFrame({
        "scan_angle": angle_range[angle_range_mask],
        "mean_percent_error": mean[angle_range_mask],
        "percent_error_std": std[angle_range_mask],
        "percent_error_counts": count[angle_range_mask],
    })

    f_scan_angle_df = pl.DataFrame({
        "scan_angle": f_angle_range[f_angle_range_mask],
        "mean_percent_error": f_mean[f_angle_range_mask],
        "percent_error_std": f_std[f_angle_range_mask],
        "percent_error_counts": f_count[f_angle_range_mask],
    })

    # calculate labels statistcs
    statistics_df, global_intercept, icc = calculate_label_statistics_from_summary(
        summary_dataset=scan_angle_df,
        independent_variable="scan_angle",
        dependent_variable_mean="mean_percent_error",
        dependent_variable_standard_deviation="percent_error_std",
        sample_count_column="percent_error_counts",
        confidence_level=0.95
    )

    f_statistics_df, f_global_intercept, f_icc = calculate_label_statistics_from_summary(
        summary_dataset=f_scan_angle_df,
        independent_variable="scan_angle",
        dependent_variable_mean="mean_percent_error",
        dependent_variable_standard_deviation="percent_error_std",
        sample_count_column="percent_error_counts",
        confidence_level=0.95
    )

    if display_graphs:
        print(f"Intraclass Correlation Coefficient Device Scan Angle: {icc}")
        print(f"Intraclass Correlation Coefficient Fovea Centered Scan Angle: {icc}")
    
        # plot scan angle data
        plot_label_ribbon(
            statistics_dataframe=statistics_df,
            global_intercept=global_intercept,
            independent_variable="scan_angle",
            dependent_variable_mean="mean_percent_error",
            plot_title=r"Average % Error vs UWF-OCT Laser Scan Angle",
            x_axis_label="Scan Angle (Degrees)",
            y_axis_label=r"% Error",
            ribbon_legend_label="95% Confidence Interval",
            mean_legend_label="Mean % Error",
            show_baseline=True,
        )

        plot_label_ribbon(
            statistics_dataframe=f_statistics_df,
            global_intercept=f_global_intercept,
            independent_variable="scan_angle",
            dependent_variable_mean="mean_percent_error",
            plot_title=r"Average % Error vs UWF-OCT Laser Scan Angle Relative to Fovea",
            x_axis_label="Scan Angle (Degrees)",
            y_axis_label=r"% Error",
            ribbon_legend_label="95% Confidence Interval",
            mean_legend_label="Mean % Error",
            show_baseline=True,
        )

    if display_in_napari:
        # view in napari
        # viewer = napari.Viewer(show=False)

        viewer.add_image(raw_thickness_map, visible=False)
        viewer.add_image(cc_thickness_map, visible=False)
        viewer.add_image(absolute_difference_map, visible=False)
        viewer.add_image(count_heat, visible=True)
        viewer.add_image(std_heat, visible=True)
        viewer.add_image(mean_heat, visible=True)
        viewer.add_image(f_count_heat, visible=True)
        viewer.add_image(f_std_heat, visible=True)
        viewer.add_image(f_mean_heat, visible=True)
        viewer.add_image(micron_error_map, visible=True)
        viewer.add_image(incidence_map, visible=True)
        viewer.add_labels(f_shell_map, opacity=0.4, visible=False)
        viewer.add_labels(shell_map, opacity=0.4, visible=False)
        viewer.add_labels(
            (input_mask.numpy() * 2).astype("uint8"),
            name=f"{id_number}_input_mask",
            opacity=0.4,
            visible=False,
        )
        viewer.add_labels(
            (calc_mask.numpy() * 6).astype("uint8"),
            name=f"{id_number}_calc_mask",
            opacity=0.4,
            visible=True,
        )
        viewer.add_labels(
            (shell_mask.numpy() * 9).astype("uint8"),
            name=f"{id_number}_shell_mask",
            opacity=0.4,
            visible=False,
        )
        viewer.add_labels(
            (f_input_mask.numpy() * 2).astype("uint8"),
            name=f"{id_number}_f_input_mask",
            opacity=0.4,
            visible=False,
        )
        viewer.add_labels(
            (f_calc_mask.numpy() * 6).astype("uint8"),
            name=f"{id_number}_f_calc_mask",
            opacity=0.4,
            visible=True,
        )
        viewer.add_labels(
            (f_shell_mask.numpy() * 9).astype("uint8"),
            name=f"{id_number}_f_shell_mask",
            opacity=0.4,
            visible=False,
        )
        viewer.add_points(
            center_pt, size=10, border_color="cyan", face_color="red", visible=True
        )
        viewer.add_points(
            fovea_pt, size=10, border_color="red", face_color="cyan", visible=True
        )

        # viewer.show()
        # napari.run()

    return error_by_incidence_df, f_scan_angle_df, scan_angle_df

###########################################################################################################

@magicgui(
    fovea_point_df_path={"label": "Fold Directory", "mode": "r"},
    measurement_map_dir={"label": "Fold Directory", "mode": "d"},
    # npz_label_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Batch %Error Calculations",
)
def batch_clean_labels(
    fovea_point_df_path: Path = Path(
        r"E:\_Beth_Thickness_Calculations\Foveapoints01.30.26.csv"
        # r"E:\_Beth_Thickness_Calculations\FoveDiscandLPCApoints.csv"
    ),
    measurement_map_dir: Path = Path(
        # r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\All ridge project Outputs\Measurement_Topo_Maps"
        r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\Ridge Height Output\Clean_Labels_Topo_Maps\Measurement_Maps"
    ),
    # npz_label_dir: Path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output"),
    # output_dir: Path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\All ridge project Outputs\Measurement_DataFrames"),
    output_dir: Path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\Ridge Height Output\Measurement_DataFrames"),
    # # npz_label_dir: Path = Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2"),
    # # output_dir: Path = Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2"),
    # ret_chor_suffix:str = "ret_chor_seg",
    # clean_ret_chor_suffix:str = "clean",
    # topological_map_suffix:str = "topo_map",
    # ret_chor_label_map:tuple[str,str] = ["retina","choroid"],
    # ret_chor_label_values:tuple[int,int] = [1,2],
    # use_accelerator:bool = True,
    # voxel_threshold:int = 100_000,
    # window_along_axis:int = 80,
    # window_offsets:tuple[int,int] = (0,0),
    # save_clean_labels:bool=True,
    # save_topological_map:bool=True,
    # overwrite:bool=False,
    # scan_for_new_files:bool=False,
    curve_corrected_mirco_topo_search_term: str = "cc_micro", #"cc_nn_micro", # cc_micro
    nearest_neighbor_incidence_search_term: str = "nn_incidence",
    raw_micro_topo_search_term: str = "raw_micro",
    calculation_mask_offset: int = 10,
    save_dataframes: bool = False,
    display_graphs: bool = False,
    display_in_napari: bool = False,
    verbose: bool = False,
):
    # Load DataFrame and extract ids and fovea point data
    fovea_point_df = pl.read_csv(fovea_point_df_path)
    fovea_point_df = fovea_point_df.with_columns(
        id_number=pl.col("Image #"),
    ).select("id_number", "Fovea_Y", "Fovea_X")

    # fovea_point_df = fovea_point_df.with_columns(
    #     id_number=pl.col("Filename")
    #     .str.replace("UNPs_", "")
    #     .str.replace("_processed_ret_chor_seg.npy", "")
    # ).select("id_number", "Fovea_Y", "Fovea_X")

    # Test case
    # fovea_point_df = fovea_point_df.filter(
    #     # pl.col("id_number") == "08810999-2023_06_14-13_25_22",
    #     pl.col("id_number") == "08810999-2023_06_28-14_34_41",
    # )

    if verbose:
        print(fovea_point_df)
    id_list = fovea_point_df["id_number"].to_list()

    # Load required maps for calculations
    measurement_map_list = list(measurement_map_dir.glob("*retina*_map.npy"))

    # Create Viewer If Necessary
    if display_in_napari:
        viewer = napari.Viewer(show=False)
    else:
        viewer = None

    # Loop through ids
    id_pbar = tqdm(id_list[:])
    for id_number in id_pbar:
        id_pbar.set_description(f"Processing id:{id_number}")
        id_pbar.update()

        fovea_point = (
            fovea_point_df.filter(pl.col("id_number") == id_number)
            .select("Fovea_Y", "Fovea_X")
            .row(0)
        )

        # extract maps by id
        paths_of_interest = [
            path for path in measurement_map_list if id_number in str(path)
        ]
        # if maps exist process
        if len(paths_of_interest) > 0:
            cc_thickness_path = [
                path for path in paths_of_interest if curve_corrected_mirco_topo_search_term in str(path)
            ][0]
            incidence_path = [
                path for path in paths_of_interest if nearest_neighbor_incidence_search_term in str(path)
            ][0]
            raw_thickness_path = [
                path for path in paths_of_interest if raw_micro_topo_search_term in str(path)
            ][0]

            cc_thickness_map = np.load(cc_thickness_path)
            incidence_map = np.load(incidence_path)
            raw_thickness_map = np.load(raw_thickness_path)

            # verify that maps are of compatible size
            if not (incidence_map.shape == cc_thickness_map.shape == raw_thickness_map.shape):
                print(f"Map Shapes do not match!!") # TODO replace with logging
                continue

            calculation_output = percent_error_calculations(
                cc_thickness_map,
                raw_thickness_map,
                incidence_map,
                id_number=id_number,
                fovea_pt=fovea_point,
                desired_offset=calculation_mask_offset,
                viewer=viewer,
                display_in_napari=display_in_napari,
                display_graphs=display_graphs,
            )

            if calculation_output is not None:
                error_by_incidence_df, f_scan_angle_df, scan_angle_df = calculation_output
            else:
                print(f"The percent error calculation failed skipping.")
                continue

            if save_dataframes:
                df_map = ["angle_of_incidence","scan_angle_fovea","scan_angle_device"]
                for idx, df in enumerate([error_by_incidence_df, f_scan_angle_df, scan_angle_df]):
                    
                    # add id column
                    df = df.with_columns(pl.lit(id_number).alias("id"))

                    # save unique sub-file inside hive sturcture
                    df_output_dir = output_dir / f"id={id_number}"
                    df_output_dir.mkdir(exist_ok=True)
                    filename = f"chunk_{df_map[idx]}.parquet"
                    
                    file_path = df_output_dir / filename
                    df.write_parquet(file_path)

    if display_in_napari:
        viewer.show()
        napari.run()

def main():
    batch_clean_labels.native.setWindowTitle(r"Batch %Error Calculations")
    batch_clean_labels.show(run=True)


if __name__ == "__main__":
    main()
