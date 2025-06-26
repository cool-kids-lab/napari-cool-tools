from datetime import datetime as dt
from pathlib import Path

import polars as pl

in_path = Path(r"D:\JJ\Projects\Segmentation_Paper\Data\SusanDataOCT.xlsx")

susan_df = pl.read_excel(in_path)

susan_df = susan_df.select(
    pl.col(
        [
            "Subject ID",
            "MRN",
            "BW",
            "gestationalAgeWeeks",
            "gestationalAgeDays",
            "PMARaw",
            "Examiner",
            "Session Date",
            "Session Zone",
            "Session Stage",
            "Session Plus",
            "VSS",
            "Session Category",
            "eye",
        ]
    )
)

susan_df = susan_df.with_columns(
    pl.col("Subject ID").fill_null(strategy="forward"),
)


bscan_data_df = susan_df.filter(
    (pl.col("MRN").is_in(["08364574", "08906231", "08977490", "08369591"]))
)

bscan_data_df = susan_df.filter(
    (pl.col("MRN") == "08135530") & (pl.col("Session Date") == dt(2021, 12, 7))
    | (pl.col("MRN") == "08364574") & (pl.col("Session Date") == dt(2022, 12, 6))
    | (pl.col("MRN") == "08906231") & (pl.col("Session Date") == dt(2023, 12, 27))
    | (pl.col("MRN") == "08977490") & (pl.col("Session Date") == dt(2024, 5, 15))
    | (pl.col("MRN") == "08369591") & (pl.col("Session Date") == dt(2023, 1, 18))
).filter(pl.col("eye") == "od")


plus_df = susan_df.filter(
    (pl.col("Session Plus") == "Plus") & (pl.col("Session Date") > dt(2022, 12, 31))
)

plus_used_df = plus_df.filter(
    (
        pl.col("Subject ID")
        .is_in(
            [
                "OHSU-0522",
                "OHSU-0458",
                "OHSU-0468",
                "OHSU-0475",
                "OHSU-0492",
                "OHSU-0519",
            ]
        )
        .not_()
    )
)

plus_comp_df = susan_df.filter(
    (pl.col("Subject ID") == "OHSU-0464") & (pl.col("Session Date") == dt(2023, 2, 21))
    | (pl.col("Subject ID") == "OHSU-0507")
    & (pl.col("Session Date") == dt(2023, 11, 29))
    | (pl.col("Subject ID") == "OHSU-0518")
    & (pl.col("Session Date") == dt(2024, 2, 14))
    | (pl.col("Subject ID") == "OHSU-0556")
    & (pl.col("Session Date") == dt(2024, 6, 12))
).filter(pl.col("Session Plus") != "Plus")

with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200):
    print(plus_df)
    print(plus_used_df)
    print(plus_comp_df)

entries_df = susan_df.filter(
    ((pl.col("MRN") == "08793507") & (pl.col("Session Date") == dt(2023, 4, 5)))
    | ((pl.col("MRN") == "08328440") & (pl.col("Session Date") == dt(2022, 11, 8)))
    | ((pl.col("MRN") == "08329638") & (pl.col("Session Date") == dt(2022, 12, 6)))
    | ((pl.col("MRN") == "08340598") & (pl.col("Session Date") == dt(2022, 12, 28)))
    | ((pl.col("MRN") == "08376325") & (pl.col("Session Date") == dt(2023, 1, 18)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08376325") & (pl.col("Session Date") == dt(2023, 2, 21)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08364574") & (pl.col("Session Date") == dt(2022, 12, 21)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08364574") & (pl.col("Session Date") == dt(2022, 12, 6)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08363808") & (pl.col("Session Date") == dt(2023, 3, 7)))
    | ((pl.col("MRN") == "08378531") & (pl.col("Session Date") == dt(2023, 2, 8)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08378531") & (pl.col("Session Date") == dt(2023, 2, 21)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08369591") & (pl.col("Session Date") == dt(2023, 3, 7)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08369591") & (pl.col("Session Date") == dt(2023, 1, 18)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08380539") & (pl.col("Session Date") == dt(2023, 2, 15)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08380539") & (pl.col("Session Date") == dt(2022, 1, 25)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08380540") & (pl.col("Session Date") == dt(2023, 3, 1)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08380540") & (pl.col("Session Date") == dt(2022, 1, 25)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08385684") & (pl.col("Session Date") == dt(2023, 2, 8)))
    | ((pl.col("MRN") == "08385686") & (pl.col("Session Date") == dt(2023, 2, 21)))
    | ((pl.col("MRN") == "08385687") & (pl.col("Session Date") == dt(2023, 2, 8)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08385687") & (pl.col("Session Date") == dt(2023, 2, 21)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08379999") & (pl.col("Session Date") == dt(2023, 3, 21)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08379999") & (pl.col("Session Date") == dt(2023, 3, 27)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08380001") & (pl.col("Session Date") == dt(2023, 3, 7)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08380001") & (pl.col("Session Date") == dt(2023, 3, 21)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08763927") & (pl.col("Session Date") == dt(2023, 4, 5)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08763927") & (pl.col("Session Date") == dt(2023, 2, 21)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08772162") & (pl.col("Session Date") == dt(2023, 2, 8)))
    | ((pl.col("MRN") == "08779879") & (pl.col("Session Date") == dt(2023, 2, 21)))
    & (pl.col("eye") == "od")
    | ((pl.col("MRN") == "08779879") & (pl.col("Session Date") == dt(2023, 3, 1)))
    & (pl.col("eye") == "os")
    | ((pl.col("MRN") == "08768871") & (pl.col("Session Date") == dt(2023, 2, 15)))
)

entries_df = entries_df.filter(
    (
        pl.col("Subject ID")
        .is_in(
            [
                "OHSU-0452",
                "OHSU-0455",
                "OHSU-0456",
                "OHSU-0457",
                "OHSU-0458",
                "OHSU-0462",
                "OHSU-0464",
            ]
        )
        .not_()
    )
)  # .filter(
# ((pl.col("MRN") == "08380540") & (pl.col("eye") == "od")).not_()
# )

susan_df.filter((pl.col("Session Plus") == "Plus"))

with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200):
    print(entries_df)

enface_project_df_unfiltered = pl.concat(
    [entries_df, plus_used_df, plus_comp_df], how="vertical"
)

enface_project_df_unfiltered = enface_project_df_unfiltered.with_columns(
    (pl.col("gestationalAgeWeeks") + (pl.col("gestationalAgeDays") / 7.0)).alias(
        "GARaw"
    )
)  # .select("gestationalAgeWeeks","gestationalAgeDays","GARaw")

enface_project_df = enface_project_df_unfiltered.filter(
    (pl.col("Subject ID").is_in(["OHSU-0465", "OHSU-0514", "OHSU-0460"]).not_())
).sort("Subject ID")

enface_project_out_df = enface_project_df.select(
    pl.all().exclude(
        "MRN",
        "gestationalAgeWeeks",
        "gestationalAgeDays",
        "Session Date",
        "Examiner",
        "Session Category",
        "VSS",
    )
).sort("Subject ID")

bscan_project_out_df = (
    bscan_data_df.with_columns(
        GARaw=pl.col("gestationalAgeWeeks") + (pl.col("gestationalAgeDays") / 7.0)
    )
    .select(
        pl.all().exclude(
            "MRN",
            "gestationalAgeWeeks",
            "gestationalAgeDays",
            "Session Date",
            "Examiner",
            "Session Category",
            "VSS",
        )
    )
    .sort("Subject ID")
)

# enface_project_out_df = enface_project_df.select(
#    "Subject ID","BW","GARaw","PMARaw",
# )


def generate_seg_paper_stats(df, decimals: int = 3):
    roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}

    if df.height > 1:
        stats = df.select(
            # Birth_Weight=pl.format("{} ± {}",pl.col("BW").mean(),pl.col("BW").std()),
            # pl.when(pl.len() > 1).then(
            #    pl.format("{} ± {}",pl.col("BW").mean().round(decimals),pl.col("BW").std().round(decimals)).alias("Birth Weight (g)"),
            # ).when(pl.len() == 1).then(
            #    pl.format("{} ± {}",pl.col("BW").round(decimals),pl.col("BW").round(decimals)).alias("Birth Weight (g)"),
            # ).otherwise(
            #    pl.format("{} ± {}",pl.lit(None),pl.lit(None)).alias("Birth Weight (g)"),
            # ),
            pl.format(
                "{} ± {}",
                pl.col("BW").mean().round(decimals),
                pl.col("BW").std().round(decimals),
            ).alias("Birth Weight (g)"),
            pl.format(
                "{} ± {}",
                pl.col("GARaw").mean().round(decimals),
                pl.col("GARaw").std().round(decimals),
            ).alias("Gestational Age (days)"),
            pl.format(
                "{} ± {}",
                pl.col("PMARaw").mean().round(decimals),
                pl.col("PMARaw").std().round(decimals),
            ).alias("Postmenstrual Age (days)"),
            pl.concat_str(
                pl.col("Session Zone")
                .str.replace_all(" ", "")
                .replace(roman_to_int)
                .str.to_integer()
                .mean()
                .round(decimals)
            ).alias("Average Zone"),
            Infants=pl.col("Subject ID").n_unique(),
            Eyes=pl.len(),
            Eyes2=pl.lit(len(df)),
            # Eyes=pl.lit(len(df)//2),
        )
    elif df.height == 1:
        stats = df.select(
            pl.format(
                "{} ± {}", pl.col("BW").round(decimals), pl.lit(0.0).round()
            ).alias("Birth Weight (g)"),
            pl.format(
                "{} ± {}", pl.col("GARaw").round(decimals), pl.lit(0.0).round(decimals)
            ).alias("Gestational Age (days)"),
            pl.format(
                "{} ± {}", pl.col("PMARaw").round(decimals), pl.lit(0.0).round(decimals)
            ).alias("Postmenstrual Age (days)"),
            pl.concat_str(
                (
                    pl.col("Session Zone")
                    .str.replace_all(" ", "")
                    .replace(roman_to_int)
                    .str.to_integer()
                    * 1.0
                )
                .mean()
                .round(decimals)
            ).alias("Average Zone"),
            Infants=pl.col("Subject ID").n_unique(),
            Eyes=pl.len(),
            Eyes2=pl.lit(len(df)),
            # Eyes=pl.lit(len(df)//2),
        )
    elif df.height < 1:
        stats = df.select(
            pl.format("{} ± {}", pl.lit(None), pl.lit(None)).alias("Birth Weight (g)"),
            pl.format("{} ± {}", pl.lit(None), pl.lit(None)).alias(
                "Gestational Age (days)"
            ),
            pl.format("{} ± {}", pl.lit(None), pl.lit(None)).alias(
                "Postmenstrual Age (days)"
            ),
            pl.concat_str(pl.lit(None)).alias("Average Zone"),
            Infants=pl.col("Subject ID").n_unique(),
            Eyes=pl.len(),
            Eyes2=pl.lit(len(df)),
            # Eyes=pl.lit(len(df)//2),
        )
    zone = (
        df.select(
            Infants=pl.lit(len(df)),
            Zone=pl.concat_str(
                pl.col("Session Zone")
                .str.replace_all(" ", "")
                .replace(roman_to_int)
                .str.to_integer()
                .value_counts()
            ),
        )
        .group_by("Infants")
        .agg("Zone")
        .select(
            # pl.col("Average_Zone"), #.list.sort())
            # pl.concat_str(pl.col("Average_Zone"))
            pl.col("Zone")
            .list.sort()
            .list.join("- ")
            .str.replace_all("\{", "zone ")
            .str.replace_all("\}", ")")
            .str.replace_all(",", " (")
            .str.replace_all("-", ",")
        )
    )

    stats_out = pl.concat([stats, zone], how="horizontal").select(
        "Birth Weight (g)",
        "Gestational Age (days)",
        "Postmenstrual Age (days)",
        "Zone",
        "Average Zone",
        "Infants",
        "Eyes",
    )
    return stats_out


def generate_tabular_data(df, decimals: int = 3):
    """ """

    no_plus = df.filter(pl.col("Session Plus") == "Normal")
    pre_plus = df.filter(pl.col("Session Plus") == "Pre-Plus")
    plus = df.filter(pl.col("Session Plus") == "Plus")
    stage_0 = df.filter(pl.col("Session Stage") == 0)
    stage_1 = df.filter(pl.col("Session Stage") == 1)
    stage_2 = df.filter(pl.col("Session Stage") == 2)
    stage_3 = df.filter(pl.col("Session Stage") == 3)

    df_rows = [df, stage_0, stage_1, stage_2, stage_3, no_plus, pre_plus, plus]

    stats_for_table = []

    for df_row in df_rows:
        print(df_row)
        stats_for_table.append(generate_seg_paper_stats(df_row, decimals=decimals))

    stats_for_table_df = pl.concat(stats_for_table, how="vertical").transpose(
        include_header=True,
        header_name="Demographics",
        column_names=[
            "All",
            "Stage 0",
            "Stage 1",
            "Stage 2",
            "Stage 3",
            "Normal",
            "Pre-Plus",
            "Plus",
        ],
    )
    return stats_for_table_df


no_plus = enface_project_out_df.filter(pl.col("Session Plus") == "Normal")
pre_plus = enface_project_out_df.filter(pl.col("Session Plus") == "Pre-Plus")
plus = enface_project_out_df.filter(pl.col("Session Plus") == "Plus")
stage_0 = enface_project_out_df.filter(pl.col("Session Stage") == 0)
stage_1 = enface_project_out_df.filter(pl.col("Session Stage") == 1)
stage_2 = enface_project_out_df.filter(pl.col("Session Stage") == 2)
stage_3 = enface_project_out_df.filter(pl.col("Session Stage") == 3)

all_stats = generate_seg_paper_stats(enface_project_out_df, decimals=2)
no_plus_stats = generate_seg_paper_stats(no_plus, decimals=2)
pre_plus_stats = generate_seg_paper_stats(pre_plus, decimals=2)
plus_stats = generate_seg_paper_stats(plus, decimals=2)
stage_0_stats = generate_seg_paper_stats(stage_0, decimals=2)
stage_1_stats = generate_seg_paper_stats(stage_1, decimals=2)
stage_2_stats = generate_seg_paper_stats(stage_2, decimals=2)
stage_3_stats = generate_seg_paper_stats(stage_3, decimals=2)

stats_for_table = [
    all_stats,
    stage_0_stats,
    stage_1_stats,
    stage_2_stats,
    stage_3_stats,
    no_plus_stats,
    pre_plus_stats,
    plus_stats,
]
# roman_to_int = {"I":1,"II":2,"III":3,"IV":4,"V":5}
stats_for_table_df = pl.concat(stats_for_table, how="vertical").transpose(
    include_header=True,
    header_name="Demographics",
    column_names=[
        "All",
        "Stage 0",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Normal",
        "Pre-Plus",
        "Plus",
    ],
)

en_face_stats_for_table_df = generate_tabular_data(enface_project_out_df, decimals=2)
bscan_stats_for_table_df = generate_tabular_data(bscan_project_out_df, decimals=2)

with pl.Config(tbl_cols=-1):
    for stats in stats_for_table:
        print(stats)
    print("stats_for_table_df", stats_for_table_df)
    print("en_face_stats_for_table_df", en_face_stats_for_table_df)
    print("bscan_stats_for_table_df", bscan_stats_for_table_df)

# stats_for_table_df.write_csv(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\enface_tabular_data.csv"))
# stats_for_table_df.write_excel(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\enface_tabular_data.xlsx"))

en_face_stats_for_table_df.write_csv(
    Path(r"D:\JJ\Projects\Segmentation_Paper\Data\enface_tabular_data.csv")
)
en_face_stats_for_table_df.write_excel(
    Path(r"D:\JJ\Projects\Segmentation_Paper\Data\enface_tabular_data.xlsx")
)

bscan_stats_for_table_df.write_csv(
    Path(r"D:\JJ\Projects\Segmentation_Paper\Data\bscan_tabular_data.csv")
)
bscan_stats_for_table_df.write_excel(
    Path(r"D:\JJ\Projects\Segmentation_Paper\Data\bscan_tabular_data.xlsx")
)


# all_stats = enface_project_out_df.select(
#    Birth_Weight=pl.format("{} ± {}",pl.col("BW").mean(),pl.col("BW").std()),
#    Gestational_Age=pl.format("{} ± {}",pl.col("GARaw").mean(),pl.col("GARaw").std()),
#    Postmenstrual_Age=pl.format("{} ± {}",pl.col("PMARaw").mean(),pl.col("PMARaw").std()),
#    Average_Zone=pl.concat_str(pl.col("Session Zone").str.replace_all(" ","").replace(roman_to_int).str.to_integer().mean()),
#    Infants=pl.lit(len(enface_project_out_df)),
#    Eyes=pl.lit(len(enface_project_out_df)//2),
# )
# zone = enface_project_out_df.select(
#    Infants=pl.lit(len(enface_project_out_df)),
#    Zone=pl.concat_str(pl.col("Session Zone").str.replace_all(" ","").replace(roman_to_int).str.to_integer().value_counts()),
# ).group_by("Infants").agg("Zone").select(
#    #pl.col("Average_Zone"), #.list.sort())
#    #pl.concat_str(pl.col("Average_Zone"))
#    pl.col("Zone").list.sort().list.join("- ").str.replace_all("\{","zone ").str.replace_all("\}","").str.replace_all(",",": ").str.replace_all("-",",")
# )
#
# all_stats = pl.concat([all_stats,zone],how="horizontal")
#
# plus_stats = plus.select(
#    Birth_Weight=pl.format("{} ± {}",pl.col("BW").mean(),pl.col("BW").std()),
#    Gestational_Age=pl.format("{} ± {}",pl.col("GARaw").mean(),pl.col("GARaw").std()),
#    Postmenstrual_Age=pl.format("{} ± {}",pl.col("PMARaw").mean(),pl.col("PMARaw").std()),
#    Average_Zone=pl.concat_str(pl.col("Session Zone").str.replace_all(" ","").replace(roman_to_int).str.to_integer().mean()),
#    Infants=pl.lit(len(plus)),
#    Eyes=pl.lit(len(plus)//2),
# )
# zone = plus.select(
#    Infants=pl.lit(len(plus)),
#    Zone=pl.concat_str(pl.col("Session Zone").str.replace_all(" ","").replace(roman_to_int).str.to_integer().value_counts()),
# ).group_by("Infants").agg("Zone").select(
#    #pl.col("Average_Zone"), #.list.sort())
#    #pl.concat_str(pl.col("Average_Zone"))
#    pl.col("Zone").list.sort().list.join("- ").str.replace_all("\{","zone ").str.replace_all("\}","").str.replace_all(",",": ").str.replace_all("-",",")
# )
#
# plus_stats = pl.concat([plus_stats,zone],how="horizontal")


with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200):
    print(enface_project_df)

with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200):
    print("enface_project_out_df", enface_project_out_df)

enface_plus_counts = enface_project_df.select(
    pl.col("Session Plus").value_counts(sort=True)
).unnest("Session Plus")

enface_zone_counts = enface_project_df.select(
    pl.col("Session Zone").value_counts(sort=True)
).unnest("Session Zone")

enface_stage_counts = enface_project_df.select(
    pl.col("Session Stage").value_counts(sort=True)
).unnest("Session Stage")

enface_means = enface_project_out_df.select(pl.mean("BW", "GARaw", "PMARaw"))

enface_stds = enface_project_out_df.select(
    pl.std("BW"), pl.std("GARaw"), pl.std("PMARaw")
)

with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200):
    print("enface_plus_counts", enface_plus_counts)
    print("enface_zone_counts", enface_zone_counts)
    print("enface_stage_counts", enface_stage_counts)
    print("enface_means", enface_means)
    print("enface_stds", enface_stds)


# enface_project_df.write_csv(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_FULL_Revised_JJ.csv"))
# enface_project_out_df.write_csv(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_Revised_JJ.csv"))
