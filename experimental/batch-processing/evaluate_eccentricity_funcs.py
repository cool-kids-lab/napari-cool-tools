"""
"""

from kornia.morphology import erosion
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import torch

def generate_shell_indices(
    dimensions: tuple[int, int], 
    center: tuple[int, int] | None = None,
    device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Generates a 2D map of ellipsoid shell indices with a customizable center.

    Args:
        dimensions: A tuple of (height, width) for the pixel grid.
        center: A tuple of (y, x) for the center of the shells. 
                If None, the geometric center of the grid is used.
        device: The PyTorch device for the output tensor.

    Returns:
        A 2D LongTensor where each pixel value is its shell index.
    """
    height, width = dimensions
    
    # Set center to geometric middle if not provided
    if center is None:
        center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    else:
        center_y, center_x = float(center[0]), float(center[1])
    
    # Generate coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device), 
        torch.arange(width, device=device), 
        indexing='ij'
    )

    # Determine scaling based on the Major Axis (longest dimension)
    if height >= width:
        aspect_ratio = width / height
        distances = torch.sqrt((y_grid - center_y)**2 + ((x_grid - center_x) / aspect_ratio)**2)
    else:
        aspect_ratio = height / width
        distances = torch.sqrt(((y_grid - center_y) / aspect_ratio)**2 + (x_grid - center_x)**2)

    return distances.to(torch.long)

# OG Works
# def generate_shell_indices(
#     dimensions: tuple[int, int], 
#     device: str | torch.device = "cpu"
# ) -> torch.Tensor:
#     """
#     Generates a 2D map of ellipsoid shell indices based on the image aspect ratio.

#     The shells are approximately 1 pixel thick along the major axis. The major axis 
#     is determined by the larger of the two dimensions, while the minor axis is 
#     scaled by the ratio of the shorter dimension to the longer one.

#     Args:
#         dimensions: A tuple of (height, width) defining the pixel grid.
#         device: The PyTorch device (e.g., "cpu", "cuda") for the output tensor.

#     Returns:
#         A 2D LongTensor of shape (height, width) where each pixel value 
#         represents its assigned shell index.
#     """
#     height, width = dimensions
    
#     # Calculate the center of the grid
#     center_y = (height - 1) / 2.0
#     center_x = (width - 1) / 2.0
    
#     # Create coordinate grids for every pixel
#     y_grid, x_grid = torch.meshgrid(
#         torch.arange(height, device=device), 
#         torch.arange(width, device=device), 
#         indexing='ij'
#     )

#     # Calculate distances based on the Major Axis (longest dimension)
#     if height >= width:
#         # Height is the major axis; scale width by the aspect ratio
#         aspect_ratio = width / height
#         distances = torch.sqrt((y_grid - center_y)**2 + ((x_grid - center_x) / aspect_ratio)**2)
#     else:
#         # Width is the major axis; scale height by the aspect ratio
#         aspect_ratio = height / width
#         distances = torch.sqrt(((y_grid - center_y) / aspect_ratio)**2 + (x_grid - center_x)**2)

#     # Cast to Long to treat distance steps as discrete shell bins
#     return distances.to(torch.long)

def compute_shell_statistics(
    data: torch.Tensor, 
    shell_indices: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes shell statistics using minlength to ensure the output 1D tensors
    always match the total number of possible shells in shell_indices.
    """
    # 1. Determine the absolute maximum number of bins from the original map
    # This prevents IndexErrors during projection.
    total_bins = int(shell_indices.max().item() + 1)
    
    flat_data = data.reshape(-1).float()
    flat_indices = shell_indices.reshape(-1)
    
    # 2. Apply mask to filter data, but NOT to change the category space
    if mask is not None:
        flat_mask = mask.reshape(-1)
        flat_data = flat_data[flat_mask]
        flat_indices = flat_indices[flat_mask]

    # Handle case where mask excludes everything
    if flat_indices.numel() == 0:
        nan_stats = torch.full((total_bins,), float('nan'), device=data.device)
        return nan_stats, nan_stats, torch.zeros((total_bins,), device=data.device)

    # 3. Use minlength to force bincount to produce a tensor of size 'total_bins'
    pixel_counts = torch.bincount(flat_indices, minlength=total_bins).float()
    value_sums = torch.bincount(flat_indices, weights=flat_data, minlength=total_bins)
    squared_sums = torch.bincount(flat_indices, weights=flat_data**2, minlength=total_bins)
    
    epsilon = torch.finfo(data.dtype).eps
    
    # 4. Calculate means and handle shells with zero valid pixels
    means = value_sums / (pixel_counts + epsilon)
    means[pixel_counts == 0] = float('nan')
    
    # Calculate std dev: sqrt( E[X^2] - (E[X])^2 )
    variances = (squared_sums / (pixel_counts + epsilon)) - (means ** 2)
    standard_deviations = torch.sqrt(torch.clamp(variances, min=0.0))
    standard_deviations[pixel_counts == 0] = float('nan')
    
    return means, standard_deviations, pixel_counts

def project_to_image(statistics: torch.Tensor, shell_indices: torch.Tensor) -> torch.Tensor:
    """
    Projects 1D shell results back into a 2D spatial representation.
    Includes a safety check for empty statistics to prevent IndexErrors.
    """
    # Check if statistics tensor is empty (size 0)
    if statistics.numel() == 0:
        # Return a 2D tensor of NaNs matching the grid shape
        return torch.full(
            shell_indices.shape, 
            fill_value=float('nan'), 
            device=shell_indices.device
        )
        
    return statistics[shell_indices]

def generate_indexed_map(
    shell_indices: torch.Tensor, 
    dtype: torch.dtype = torch.uint16
) -> torch.Tensor:
    """
    Creates a map where pixel = index + 1, cast to the specified integer type.
    
    Args:
        shell_indices: The 2D index map.
        dtype: The target integer torch.dtype (e.g., torch.uint8, torch.int16, torch.int32).
               If uint8 is chosen, values exceeding 254 are clamped to 255 [1].
    """
    offset_indices = shell_indices + 1
    
    if dtype == torch.uint8:
        # Clamp to prevent wrap-around overflow in 8-bit space [2]
        return torch.clamp(offset_indices, 0, 255).to(torch.uint8)
    
    return offset_indices.to(dtype)

def apply_shell_filter(
    statistics: torch.Tensor, 
    counts: torch.Tensor, 
    minimum_pixels: int | None = None, 
    shell_range: tuple[int, int] | None = None
) -> torch.Tensor:
    """
    Applies filters to shell statistics, setting invalid shells to NaN.

    Args:
        statistics: The 1D shell-wise statistics (e.g., means or standard deviations).
        counts: The 1D tensor containing the number of pixels per shell.
        minimum_pixels: Shells with fewer pixels than this value are set to NaN.
        shell_range: A tuple of (min_index, max_index) to keep. 
                     Others are set to NaN.
    """
    filtered_output = statistics.clone()
    
    if minimum_pixels is not None:
        filtered_output[counts < minimum_pixels] = float('nan')
        
    if shell_range is not None:
        min_shell, max_shell = shell_range
        indices = torch.arange(len(statistics), device=statistics.device)
        out_of_range_mask = (indices < min_shell) | (indices > max_shell)
        filtered_output[out_of_range_mask] = float('nan')
        
    return filtered_output

def generate_shell_mask(
    shell_indices: torch.Tensor, 
    shell_range: tuple[int, int]
) -> torch.Tensor:
    """
    Creates a boolean 2D mask representing specific ellipsoid shells.

    Args:
        shell_indices: The 2D LongTensor map generated by generate_shell_indices.
        shell_range: A tuple of (min_index, max_index) inclusive.

    Returns:
        A boolean tensor of shape (height, width) where pixels within 
        the range are True.
    """
    min_index, max_index = shell_range
    
    # Efficient element-wise comparison
    mask = (shell_indices >= min_index) & (shell_indices <= max_index)
    
    return mask

def generate_calculation_mask(shell_indices,cc_thickness_map,center_pt,desired_offset:int=10):
        """"""
        # get total number of shells
        number_of_shells = len(shell_indices)

        # get longer axis
        long_axis = max(cc_thickness_map.shape)

        # calculate distance to edge of laser range
        # shells_to_edge = number_of_shells - long_axis // 2
        # desired_offset = desired_offset
        # offset = desired_offset - 2  # offset relative to the edge of the laser range

        kernel_size = 2 * abs(desired_offset) + 1
        kernel = torch.ones(kernel_size, kernel_size)

        # generate data masks
        # input data mask
        input_mask = cc_thickness_map > 0

        # input_mask_coords = input_mask.nonzero().float()

        #
        pixels_to_edges = distance_to_nonzero(input_mask, center_pt)
        shells_to_farthest_point = round(pixels_to_edges.max().item())

        # mask outer 5 shells
        shell_mask = generate_shell_mask(shell_indices, (0, shells_to_farthest_point))
        # calculation mask
        calc_mask = input_mask & shell_mask
        calc_mask = (
            erosion(calc_mask[None, None, :, :].to(torch.long), kernel=kernel,border_type="constant",border_value=0.0)
            .to(torch.bool)
            .squeeze()
        )

        return calc_mask,input_mask,shell_mask,long_axis

def distance_to_nonzero(mask: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    """
    Calculates the Euclidean distance between all nonzero pixels in a mask 
    and a specific target coordinate (y, x).
    
    Args:
        mask: 2D boolean or integer tensor.
        target: A tuple of (height_idx, width_idx).
        
    Returns:
        A 1D tensor of distances for each nonzero pixel.
    """
    # Get N x 2 tensor of [y, x] coordinates
    coords = torch.nonzero(mask).float()
    
    if coords.numel() == 0:
        return torch.empty(0, device=mask.device)
    
    # Convert target to tensor for broadcasting: [1, 2]
    target_tensor = torch.tensor(target, dtype=torch.float32, device=mask.device)
    
    # Calculate Euclidean distance: sqrt(sum of squared differences)
    # subtraction results in (N, 2), sum(dim=1) results in (N,)
    distances = torch.sqrt(torch.sum((coords - target_tensor)**2, dim=1))
    
    return distances

##########################################################################################################################

def perform_hybrid_label_analysis(
    input_dataset, 
    value_column_name="value", 
    label_column_name="label", 
    minimum_sample_threshold=30,
    confidence_level=0.95
):
    """
    Performs hybrid analysis and returns the comparison table, 
    the population baseline, and the Intraclass Correlation (ICC).
    """
    
    # 1. Stage One: Precision-Based Filtering (Polars)
    filtered_dataset = input_dataset.filter(
        pl.col(value_column_name).count().over(label_column_name) >= minimum_sample_threshold
    )

    # with pl.Config(tbl_rows=-1):
    #     print(filtered_dataset)
    
    # 2. Stage Two: Mixed-Effects Modeling
    pandas_conversion = filtered_dataset.to_pandas()
    formula_string = f"{value_column_name} ~ 1"
    
    mixed_effects_model = smf.mixedlm(
        formula_string, 
        pandas_conversion, 
        groups=pandas_conversion[label_column_name]
    )

    model_results = None
    try:
        model_results = mixed_effects_model.fit(method=["bfgs","lbfgs","powel","nm"],full_output=True)

        if hasattr(model_results, 'converged') and not model_results.converged:
            print("Optimization failed to converge.")
            return None
        
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Fit failed due to numerical instability: {e}")
        return None
    
    # 3. Extract Variances and Calculate ICC
    # Label Variance (Between-group variance)
    label_variance = float(model_results.cov_re.iloc[0])
    # Residual Variance (Within-group noise)
    residual_variance = model_results.scale
    
    intraclass_correlation = label_variance / (label_variance + residual_variance)
    
    # 4. Extract Model Parameters for Estimates
    global_intercept = model_results.params['Intercept']
    standard_error_intercept = model_results.bse['Intercept']
    random_effect_deviations = model_results.random_effects
    random_effect_covariances = model_results.random_effects_cov
    
    z_score = 1.96 if confidence_level == 0.95 else 2.58
    
    analysis_results_list = []
    for label_identity, deviation_series in random_effect_deviations.items():
        random_deviation_value = float(deviation_series.iloc[0])
        shrunken_mean = random_deviation_value + global_intercept
        
        # Extract specific label variance from the covariance dictionary
        group_specific_variance = float(random_effect_covariances[label_identity].iloc[0])
        prediction_standard_error = np.sqrt(group_specific_variance + (standard_error_intercept**2))
        
        analysis_results_list.append({
            label_column_name: label_identity,
            "shrunken_mean": shrunken_mean,
            "lower_confidence_bound": shrunken_mean - (z_score * prediction_standard_error),
            "upper_confidence_bound": shrunken_mean + (z_score * prediction_standard_error)
        })
    
    # 5. Join Results with Original Raw Statistics
    raw_statistics = filtered_dataset.group_by(label_column_name).agg([
        pl.col(value_column_name).mean().alias("raw_mean"),
        pl.len().alias("sample_count")
    ])
    
    final_comparison_table = raw_statistics.join(
        pl.DataFrame(analysis_results_list), 
        on=label_column_name
    )
    
    return final_comparison_table, global_intercept, intraclass_correlation

def plot_label_comparison_with_intervals(comparison_table, global_mean_reference):
    """
    Visualizes the top 25 labels affected by shrinkage and their confidence intervals.
    """
    # Identify labels with the most significant shrinkage impact for visualization
    visual_table = comparison_table.with_columns(
        shrinkage_impact = (pl.col("raw_mean") - pl.col("shrunken_mean")).abs()
    ).sort("shrinkage_impact", descending=True).head(25)
    
    plt.figure(figsize=(12, 10))
    vertical_indices = np.arange(len(visual_table))
    
    # Plot Mixed-Effect Estimates with Confidence Intervals
    plt.errorbar(
        visual_table["shrunken_mean"], 
        vertical_indices, 
        xerr=[
            visual_table["shrunken_mean"] - visual_table["lower_confidence_bound"],
            visual_table["upper_confidence_bound"] - visual_table["shrunken_mean"]
        ],
        fmt='o', color='dodgerblue', label='Shrunken Estimate (95% CI)', capsize=4, elinewidth=1.5
    )
    
    # Plot Raw Means (Red X marks)
    plt.scatter(
        visual_table["raw_mean"], 
        vertical_indices, 
        color='crimson', marker='x', label='Original Raw Mean', s=60, zorder=3
    )
    
    # Formatting and Styling
    plt.axvline(global_mean_reference, color='black', linestyle='--', alpha=0.7, label='Global Population Mean')
    plt.yticks(vertical_indices, visual_table.get_column(visual_table.columns[0]))
    plt.xlabel("Mean Value")
    plt.ylabel("Population Label")
    plt.title("Hybrid Analysis: Comparing Raw vs. Shrunken Population Labels")
    plt.legend(loc="upper right")
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_label_distribution_with_ribbon(
    comparison_table, 
    global_mean_reference, 
    label_column_name="label",
    plot_title="Population Label Distribution",
    x_axis_label="Label Identity",
    y_axis_label="Dependent Variable Value",
    show_population_baseline=True,
    number_of_ticks=20,
    shrunken_mean_legend_label="Shrunken Mean",
    raw_mean_legend_label="Original Raw Mean",
    confidence_interval_legend_label="95% Confidence Interval",
    baseline_legend_label="Population Mean"
):
    """
    Plots labels on the x-axis (sorted by label name) and their shrunken means on the y-axis, 
    with a shaded ribbon and customizable legend/tick parameters.
    """
    # 1. Sort the table by the Label name ascending
    sorted_table = comparison_table.sort(label_column_name)
    
    # 2. Extract plot components as numpy arrays
    label_indices = np.arange(len(sorted_table))
    shrunken_means = sorted_table["shrunken_mean"].to_numpy()
    lower_bounds = sorted_table["lower_confidence_bound"].to_numpy()
    upper_bounds = sorted_table["upper_confidence_bound"].to_numpy()
    raw_means = sorted_table["raw_mean"].to_numpy()
    label_names = sorted_table.get_column(label_column_name).to_numpy()
    
    plt.figure(figsize=(16, 8))
    
    # 3. Plot the Confidence Interval Ribbon (Shaded Area)
    plt.fill_between(
        label_indices, 
        lower_bounds, 
        upper_bounds, 
        color='dodgerblue', 
        alpha=0.25, 
        label=confidence_interval_legend_label
    )
    
    # 4. Plot the Shrunken Mean (Trend line)
    plt.plot(
        label_indices, 
        shrunken_means, 
        color='dodgerblue', 
        linewidth=1.5, 
        marker='o', 
        markersize=3, 
        label=shrunken_mean_legend_label
    )
    
    # 5. Plot Raw Means (Red dots)
    plt.scatter(
        label_indices, 
        raw_means, 
        color='crimson', 
        s=15, 
        alpha=0.5, 
        label=raw_mean_legend_label
    )
    
    # 6. Optional Population Baseline
    if show_population_baseline:
        plt.axhline(
            global_mean_reference, 
            color='black', 
            linestyle='--', 
            alpha=0.8, 
            label=baseline_legend_label
        )
    
    # 7. Custom Labels and Formatting
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    
    # 8. Dynamic XTicks Logic
    total_labels = len(sorted_table)
    tick_step = max(1, total_labels // number_of_ticks)
    tick_indices = np.arange(0, total_labels, step=tick_step)
    
    plt.xticks(
        tick_indices, 
        label_names[tick_indices], 
        rotation=45, 
        ha='right', 
        fontsize=9
    )
    
    plt.legend(loc="best")
    plt.grid(axis='both', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()

def calculate_label_statistics_from_summary(
    summary_dataset, 
    independent_variable="label", 
    dependent_variable_mean="mean", 
    dependent_variable_standard_deviation="standard_deviation",
    sample_count_column="sample_count",
    confidence_level=0.95
):
    """
    Accepts pre-aggregated summary data to calculate confidence intervals,
    the global population intercept, and the Intraclass Correlation (ICC).
    """
    # 1. Calculate Confidence Intervals in Polars
    z_score = 1.96 if confidence_level == 0.95 else 2.58
    
    label_statistics = summary_dataset.with_columns([
        (pl.col(dependent_variable_standard_deviation) / pl.col(sample_count_column).sqrt()).alias("standard_error")
    ]).with_columns([
        (pl.col(dependent_variable_mean) - (z_score * pl.col("standard_error"))).alias("lower_bound"),
        (pl.col(dependent_variable_mean) + (z_score * pl.col("standard_error"))).alias("upper_bound")
    ])

    # 2. Fit Mixed-Effects Model for ICC
    # Note: Statsmodels needs raw data for an accurate ICC. 
    # If raw data is unavailable, we estimate the Intercept as a weighted mean.
    pandas_conversion = label_statistics.to_pandas()
    
    # Calculate Weighted Global Intercept
    total_samples = pandas_conversion[sample_count_column].sum()
    global_intercept = (
        (pandas_conversion[dependent_variable_mean] * pandas_conversion[sample_count_column]).sum() 
        / total_samples
    )

    # Calculate Variances for ICC
    # Within-group variance (Residual) is the pooled variance of the labels
    residual_variance = (
        (pandas_conversion[dependent_variable_standard_deviation]**2 * (pandas_conversion[sample_count_column] - 1)).sum() 
        / (total_samples - len(pandas_conversion))
    )
    
    # Between-group variance (Label Variance)
    label_variance = pandas_conversion[dependent_variable_mean].var()
    
    intraclass_correlation = label_variance / (label_variance + residual_variance)

    return label_statistics, global_intercept, intraclass_correlation

def plot_label_ribbon(
    statistics_dataframe, 
    global_intercept, 
    independent_variable="label",
    dependent_variable_mean="mean",
    ci_lower_variable="lower_bound",
    ci_upper_variable="upper_bound",
    plot_title="Label Mean Distribution",
    x_axis_label="Independent Variable",
    y_axis_label="Dependent Variable Mean",
    ribbon_legend_label="95% Confidence Interval",
    mean_legend_label="Group Mean",
    show_baseline=True,
    number_of_ticks=20  # User parameter to control X-axis density
):
    """
    Generates a ribbon plot with user-controlled X-axis tick density.
    """
    # 1. Sort alphabetically/numerically by label
    sorted_table = statistics_dataframe.sort(independent_variable)
    
    # 2. Extract plot components as numpy arrays
    indices = np.arange(len(sorted_table))
    means = sorted_table[dependent_variable_mean].to_numpy()
    lower_bounds = sorted_table[ci_lower_variable].to_numpy()
    upper_bounds = sorted_table[ci_upper_variable].to_numpy() # Adjusted to match your CI column naming
    label_names = sorted_table.get_column(independent_variable).to_numpy()

    plt.figure(figsize=(16, 8))
    
    # 3. Plot the Confidence Ribbon (Shaded Area)
    plt.fill_between(
        indices, lower_bounds, upper_bounds, 
        color='dodgerblue', alpha=0.3, label=ribbon_legend_label
    )
    
    # 4. Plot the Mean Trend
    plt.plot(
        indices, means, 
        color='dodgerblue', linewidth=1.5, marker='o', markersize=3, label=mean_legend_label
    )

    # 5. Optional Population Baseline
    if show_baseline:
        plt.axhline(
            global_intercept, color='black', linestyle='--', 
            alpha=0.7, label=f'Global Baseline ({global_intercept:.2f})'
        )

    # 6. Labels and Title
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    
    # 7. Dynamic XTicks Logic
    # Calculate the step size based on the total number of labels and requested ticks
    total_labels = len(sorted_table)
    tick_step = max(1, total_labels // number_of_ticks)
    tick_indices = np.arange(0, total_labels, step=tick_step)
    
    plt.xticks(
        tick_indices, 
        label_names[tick_indices], 
        rotation=45, 
        ha='right', 
        fontsize=9
    )
    
    plt.legend(loc="best")
    plt.grid(axis='both', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()