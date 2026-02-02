"""
"""

from pathlib import Path

import polars as pl
import scipy.stats as stats

from evaluate_eccentricity_funcs import plot_label_ribbon

def calculate_global_stats(lf: pl.LazyFrame, group_col: str, target_col: str) -> pl.DataFrame:
    """
    Calculates weighted grand mean, std, and 95% CI while ignoring poisonous NaN values.
    """
    # 1. Standard Error of the Mean (SEM) calculation
    # Using only non-null, non-nan rows for count
    sem = pl.col(target_col).std() / pl.col(target_col).count().sqrt()
    
    return (
        lf
        # 2. FIX: Explicitly drop NaNs before aggregating to prevent poisonous output
        .drop_nans(target_col)
        .group_by(group_col)
        .agg([
            pl.col(target_col).mean().alias("grand_mean"),
            pl.col(target_col).std().alias("global_std"),
            (pl.col(target_col).mean() - (1.96 * sem)).alias("ci_lower"),
            (pl.col(target_col).mean() + (1.96 * sem)).alias("ci_upper"),
            pl.col(target_col).count().alias("n_total")
        ])
        .sort(group_col)
        .collect(engine="streaming")
    )

# def calc_weighted_stats(
#     lf: pl.LazyFrame, 
#     label_col: str, 
#     val_col: str, 
#     weight_col: str, 
#     confidence: float = 0.95
# ) -> pl.LazyFrame:
#     # Map confidence to Z-score
#     z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
#     z = z_map.get(confidence, 1.96)

#     # Define common expressions for readability
#     val = pl.col(val_col)
#     w = pl.col(weight_col)
    
#     # 1. Manual weighted mean: sum(val * weight) / sum(weight)
#     w_mean_expr = (val * w).sum() / w.sum()

#     return (
#         lf.drop_nans(val_col)
#         .group_by(label_col).agg([
#             w_mean_expr.alias("w_mean"),
#             # Weighted Variance formula: sum(w * (x - x_wavg)^2) / sum(w) * (N / (N-1))
#             (
#                 ((w * (val - w_mean_expr).pow(2)).sum() / w.sum())
#                 * (pl.len() / (pl.len() - 1))
#             ).alias("w_var"),
#             w.sum().alias("sum_w")
#         ])
#         .with_columns([
#             pl.col("w_var").sqrt().alias("w_std"),
#             (pl.col("w_var").sqrt() / pl.col("sum_w").sqrt()).alias("w_sem")
#         ])
#         .with_columns([
#             (pl.col("w_mean") - (z * pl.col("w_sem"))).alias("ci_low"),
#             (pl.col("w_mean") + (z * pl.col("w_sem"))).alias("ci_high")
#         ])
#         .drop(["w_var", "sum_w"])
#     )

def calculate_weighted_stats(
    lf: pl.LazyFrame, 
    label_col: str,
    val_col: str, 
    weight_col: str, 
    confidence: float = 0.95
) -> pl.DataFrame:
    # 1. Critical values
    z_crit = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    v, w = pl.col(val_col), pl.col(weight_col)
    
    # 2. Reusable weighted mean expression
    w_mean_expr = (v * w).sum() / w.sum()

    return (
        lf.drop_nans()
        .group_by(label_col)
        .agg([
            w_mean_expr.alias("w_mean"),
            w.sum().alias("sum_weight"),
            pl.len().alias("n"),
            # Weighted Unbiased Variance
            (
                ((w * (v - w_mean_expr).pow(2)).sum() / w.sum()) 
                * (pl.len() / (pl.len() - 1))
            ).alias("w_var")
        ])
        .with_columns([
            pl.col("w_var").sqrt().alias("w_std"),
            # SEM = w_std / sqrt(sum_weight)
            (pl.col("w_var").sqrt() / pl.col("sum_weight").sqrt()).alias("w_sem")
        ])
        .with_columns([
            # Cohen's d: Practical significance (Mean / StdDev)
            (pl.col("w_mean") / pl.col("w_std")).alias("cohens_d"),
            
            # Z-score: Statistical strength (Mean / SEM)
            (pl.col("w_mean") / pl.col("w_sem")).alias("z_score"),
            
            # Confidence Intervals
            (pl.col("w_mean") - (z_crit * pl.col("w_sem"))).alias("ci_low"),
            (pl.col("w_mean") + (z_crit * pl.col("w_sem"))).alias("ci_high")
        ])
        .with_columns([
            # Formatted P-Value: Handles extreme significance without rounding to zero
            pl.col("z_score").abs().map_elements(
                lambda z: "< 1e-300" if z > 37 else f"{2 * stats.norm.sf(z):.4e}", 
                return_dtype=pl.String
            ).alias("p_value_text")
        ])
        .drop("w_var")
        .collect()
    )

# def calc_overall_weighted_stats(
#     lf: pl.LazyFrame, 
#     val_col: str, 
#     weight_col: str, 
#     confidence: float = 0.95
# ) -> pl.LazyFrame:
#     """
#     Calculates overall weighted mean, std, sem, and CI for the entire LazyFrame.
#     """
#     # Z-score mapping for common confidence levels
#     z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
#     z = z_map.get(confidence, 1.96)

#     # 1. Expressions for manual weighted calculation
#     # Polars Expr.mean() does not yet support a 'weights' argument
#     val = pl.col(val_col)
#     w = pl.col(weight_col)
    
#     # Weighted Mean: sum(value * weight) / sum(weight)
#     w_mean_expr = (val * w).sum() / w.sum()

#     return (
#         lf.drop_nans()
#         .select([
#             w_mean_expr.alias("w_mean"),
#             # Weighted Variance (Unbiased): [sum(w * (x - μ_w)^2) / sum(w)] * [N / (N-1)]
#             (
#                 ((w * (val - w_mean_expr).pow(2)).sum() / w.sum())
#                 * (pl.len() / (pl.len() - 1))
#             ).alias("w_var"),
#             w.sum().alias("sum_w")
#         ])
#         .select([
#             pl.col("w_mean"),
#             pl.col("w_var").sqrt().alias("w_std"),
#             # SEM: sqrt(weighted_variance) / sqrt(sum_of_weights)
#             (pl.col("w_var").sqrt() / pl.col("sum_w").sqrt()).alias("w_sem"),
#         ])
#         .with_columns([
#             # Confidence Intervals: mean +/- (Z * SEM)
#             (pl.col("w_mean") - (z * pl.col("w_sem"))).alias("ci_low"),
#             (pl.col("w_mean") + (z * pl.col("w_sem"))).alias("ci_high")
#         ])
#     )

def calc_overall_weighted_stats(
    lf: pl.LazyFrame, 
    val_col: str, 
    weight_col: str, 
    confidence: float = 0.95
) -> pl.DataFrame:
    # 1. Map confidence to Z-score
    z_crit = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

    # 2. Clean and Define Expressions
    lf_clean = lf.drop_nans().select([val_col, weight_col])
    v = pl.col(val_col)
    w = pl.col(weight_col)
    
    # Weighted Mean: sum(v*w) / sum(w)
    w_mean_expr = (v * w).sum() / w.sum()

    # 3. Aggregate Core Values
    res_df = (
        lf_clean.select([
            w_mean_expr.alias("w_mean"),
            w.sum().alias("sum_weight"),
            pl.len().alias("n"),
            # Weighted Unbiased Variance
            (
                ((w * (v - w_mean_expr).pow(2)).sum() / w.sum()) 
                * (pl.len() / (pl.len() - 1))
            ).alias("w_var")
        ])
        .with_columns([
            pl.col("w_var").sqrt().alias("w_std"),
            # SEM = w_std / sqrt(sum_weight)
            (pl.col("w_var").sqrt() / pl.col("sum_weight").sqrt()).alias("w_sem")
        ])
        .collect()
    )

    # 4. Extract for Final Calculations
    # We use row(0) for cleaner scalar access
    row = res_df.row(0, named=True)
    m, std, sem = row["w_mean"], row["w_std"], row["w_sem"]

    # Z-score and Cohen's d
    z_score = m / sem if sem > 0 else 0
    cohens_d = m / std if std > 0 else 0

    # Precise P-Value (String Capped)
    # Z > 37 leads to p-values smaller than float64 limits (~1e-308)
    abs_z = abs(z_score)
    p_val_raw = 2 * stats.norm.sf(abs_z)
    p_val_text = "< 1e-300" if abs_z > 37 else f"{p_val_raw:.4e}"

    # 5. Return Results
    return pl.DataFrame({
        "w_mean": [m],
        "w_std": [std],
        "w_sem": [sem],
        "ci_low": [m - (z_crit * sem)],
        "ci_high": [m + (z_crit * sem)],
        "z_score": [z_score],
        "cohens_d": [cohens_d],
        "p_value": [p_val_text],
        "n": [row["n"]],
        "sum_weight": [row["sum_weight"]]
    })

def main():
    # input_path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\All ridge project Outputs\Measurement_DataFrames")
    input_path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\Ridge Height Output\Measurement_DataFrames")
    # input_path = Path(r"E:\_Measurement_DataFrames_Test")

    lazy_incidence_df = pl.scan_parquet(
        input_path / "**" / "*chunk_angle_of_incidence.parquet",
        hive_partitioning=True
    )
    lazy_scan_angle_df = pl.scan_parquet(
        input_path / "**" / "*chunk_scan_angle_device.parquet",
        hive_partitioning=True
    )
    lazy_fovea_df = pl.scan_parquet(
        input_path / "**" / "*chunk_scan_angle_fovea.parquet",
        hive_partitioning=True
    )

    # pass

    fovea_stats_df = calculate_global_stats(lf=lazy_fovea_df,group_col="scan_angle",target_col="mean_percent_error")
    incidence_stats_df = calculate_global_stats(lf=lazy_incidence_df,group_col="incidence",target_col="mean_error")
    scan_angle_stats_df = calculate_global_stats(lf=lazy_scan_angle_df,group_col="scan_angle",target_col="mean_percent_error")

    # pass
    unique_scans = lazy_fovea_df.select(pl.col("id").unique())
    unique_patients = lazy_fovea_df.with_columns(unique = pl.col("id").str.split_exact("-",1).struct.field("field_0")).select(pl.col("unique").unique())

    # testing = calculate_weighted_stats(lazy_fovea_df,"scan_angle","mean_percent_error","percent_error_counts",0.95)
    testing = calculate_weighted_stats(lazy_fovea_df,"id","mean_percent_error","percent_error_counts",0.95)
    testing2 = calc_overall_weighted_stats(lazy_fovea_df, val_col="mean_percent_error",weight_col="percent_error_counts",confidence=0.95)
    testing3 = calc_overall_weighted_stats(lazy_scan_angle_df, val_col="mean_percent_error",weight_col="percent_error_counts",confidence=0.95)
    testing4 = calc_overall_weighted_stats(lazy_incidence_df, val_col="mean_error",weight_col="count",confidence=0.95)

    plot_label_ribbon(
        fovea_stats_df,
        global_intercept=0,
        independent_variable="scan_angle",
        dependent_variable_mean="grand_mean",
        ci_lower_variable="ci_lower",
        ci_upper_variable="ci_upper",
        plot_title=r"Average % Error Retinal Thickness vs UWF-OCT Laser Scan Angle Relative to Fovea",
        x_axis_label="Scan Angle (Degrees)",
        y_axis_label=r"% Error",
        ribbon_legend_label="95% Confidence Interval",
        mean_legend_label="Mean % Error",
        show_baseline=False,
    )

    plot_label_ribbon(
        scan_angle_stats_df,
        global_intercept=0,
        independent_variable="scan_angle",
        dependent_variable_mean="grand_mean",
        ci_lower_variable="ci_lower",
        ci_upper_variable="ci_upper",
        plot_title=r"Average % Error Retinal Thickness vs UWF-OCT Laser Scan Angle",
        x_axis_label="Scan Angle (Degrees)",
        y_axis_label=r"% Error",
        ribbon_legend_label="95% Confidence Interval",
        mean_legend_label="Mean % Error",
        show_baseline=False,
    )

    plot_label_ribbon(
        incidence_stats_df,
        global_intercept=0,
        independent_variable="incidence",
        dependent_variable_mean="grand_mean",
        ci_lower_variable="ci_lower",
        ci_upper_variable="ci_upper",
        plot_title=r"Average % Error vs UWF-OCT Angle of Incedence between Laser and RPE",
        x_axis_label="Angle of Incedence (Degrees)",
        y_axis_label=r"% Error",
        ribbon_legend_label="95% Confidence Interval",
        mean_legend_label="Mean % Error",
        show_baseline=False,
    )


if __name__ == "__main__":
    main()