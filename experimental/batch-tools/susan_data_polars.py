from pathlib import Path
import datetime
from datetime import datetime as dt

import polars as pl

in_path = Path(r"D:\JJ\Projects\Segmentation_Paper\Data\SusanDataOCT.xlsx")

susan_df = pl.read_excel(in_path)

susan_df = susan_df.select(
    pl.col(["Subject ID","MRN","BW","gestationalAgeWeeks","gestationalAgeDays","PMARaw","Examiner","Session Date","Session Zone","Session Stage","Session Plus","VSS","Session Category","eye"])
)

susan_df = susan_df.with_columns(
    pl.col("Subject ID").fill_null(strategy="forward"),
)

plus_df = susan_df.filter(
    (pl.col("Session Plus") == "Plus") & (pl.col("Session Date") > dt(2022,12,31))
)

plus_used_df = plus_df.filter(
    (pl.col("Subject ID").is_in(["OHSU-0522","OHSU-0458","OHSU-0468","OHSU-0475","OHSU-0492","OHSU-0519"]).not_())
)

plus_comp_df = susan_df.filter(
    (pl.col("Subject ID") == "OHSU-0464") & (pl.col("Session Date") == dt(2023,2,21)) |
    (pl.col("Subject ID") == "OHSU-0507") & (pl.col("Session Date") == dt(2023,11,29)) |
    (pl.col("Subject ID") == "OHSU-0518") & (pl.col("Session Date") == dt(2024,2,14)) |
    (pl.col("Subject ID") == "OHSU-0556") & (pl.col("Session Date") == dt(2024,6,12))
).filter(
    pl.col("Session Plus") != "Plus"
)

with pl.Config(tbl_cols=-1,tbl_rows=-1, tbl_width_chars=200):
    print(plus_df)
    print(plus_used_df)
    print(plus_comp_df)

entries_df = susan_df.filter(
    ((pl.col("MRN") == '08793507') & (pl.col("Session Date") == dt(2023,4,5))) |
    ((pl.col("MRN") == '08328440') & (pl.col("Session Date") == dt(2022,11,8))) |
    ((pl.col("MRN") == '08329638') & (pl.col("Session Date") == dt(2022,12,6))) |
    ((pl.col("MRN") == '08340598') & (pl.col("Session Date") == dt(2022,12,28))) |
    ((pl.col("MRN") == '08376325') & (pl.col("Session Date") == dt(2023,1,18))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08376325') & (pl.col("Session Date") == dt(2023,2,21))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08364574') & (pl.col("Session Date") == dt(2022,12,21))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08364574') & (pl.col("Session Date") == dt(2022,12,6))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08363808') & (pl.col("Session Date") == dt(2023,3,7))) |
    ((pl.col("MRN") == '08378531') & (pl.col("Session Date") == dt(2023,2,8))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08378531') & (pl.col("Session Date") == dt(2023,2,21))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08369591') & (pl.col("Session Date") == dt(2023,3,7))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08369591') & (pl.col("Session Date") == dt(2023,1,18))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08380539') & (pl.col("Session Date") == dt(2023,2,15))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08380539') & (pl.col("Session Date") == dt(2022,1,25))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08380540') & (pl.col("Session Date") == dt(2023,3,1))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08380540') & (pl.col("Session Date") == dt(2022,1,25))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08385684') & (pl.col("Session Date") == dt(2023,2,8))) |
    ((pl.col("MRN") == '08385686') & (pl.col("Session Date") == dt(2023,2,21))) |
    ((pl.col("MRN") == '08385687') & (pl.col("Session Date") == dt(2023,2,8))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08385687') & (pl.col("Session Date") == dt(2023,2,21))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08379999') & (pl.col("Session Date") == dt(2023,3,21))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08379999') & (pl.col("Session Date") == dt(2023,3,27))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08380001') & (pl.col("Session Date") == dt(2023,3,7))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08380001') & (pl.col("Session Date") == dt(2023,3,21))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08763927') & (pl.col("Session Date") == dt(2023,4,5))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08763927') & (pl.col("Session Date") == dt(2023,2,21))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08772162') & (pl.col("Session Date") == dt(2023,2,8))) |
    ((pl.col("MRN") == '08779879') & (pl.col("Session Date") == dt(2023,2,21))) & (pl.col("eye") == "od") |
    ((pl.col("MRN") == '08779879') & (pl.col("Session Date") == dt(2023,3,1))) & (pl.col("eye") == "os") |
    ((pl.col("MRN") == '08768871') & (pl.col("Session Date") == dt(2023,2,15)))
)

entries_df = entries_df.filter(
    (pl.col("Subject ID").is_in(["OHSU-0452","OHSU-0455","OHSU-0456","OHSU-0457","OHSU-0458","OHSU-0462","OHSU-0464"]).not_())
)#.filter(
    #((pl.col("MRN") == "08380540") & (pl.col("eye") == "od")).not_()
#)

susan_df.filter(
    (pl.col("Session Plus") == "Plus")
)

with pl.Config(tbl_cols=-1,tbl_rows=-1, tbl_width_chars=200):
    print(entries_df)

enface_project_df_unfiltered = pl.concat([entries_df,plus_used_df,plus_comp_df],how="vertical")

enface_project_df_unfiltered = enface_project_df_unfiltered.with_columns(
    (pl.col("gestationalAgeWeeks") + (pl.col("gestationalAgeDays")/7.0)).alias("GARaw")
)#.select("gestationalAgeWeeks","gestationalAgeDays","GARaw")

enface_project_df = enface_project_df_unfiltered.filter(
    (pl.col("Subject ID").is_in(["OHSU-0465","OHSU-0514","OHSU-0460"]).not_())
).sort("Subject ID")

enface_project_out_df = enface_project_df.select(
    pl.all().exclude("MRN","Session Date","Examiner","Session Category","eye","VSS")
).sort("Subject ID")

enface_project_out_df = enface_project_df.select(
    "Subject ID","BW","GARaw","PMARaw"
)

with pl.Config(tbl_cols=-1,tbl_rows=-1, tbl_width_chars=200):
    print(enface_project_df)

with pl.Config(tbl_cols=-1,tbl_rows=-1, tbl_width_chars=200):
    print(enface_project_out_df)

enface_plus_counts = enface_project_df.select(
    pl.col("Session Plus").value_counts(sort=True)
).unnest("Session Plus")

enface_zone_counts = enface_project_df.select(
    pl.col("Session Zone").value_counts(sort=True)
).unnest("Session Zone")

enface_stage_counts = enface_project_df.select(
    pl.col("Session Stage").value_counts(sort=True)
).unnest("Session Stage")

enface_means = enface_project_out_df.select(
    pl.mean("BW","GARaw","PMARaw")
)

enface_stds = enface_project_out_df.select(
    pl.std("BW"),
    pl.std("GARaw"),
    pl.std("PMARaw")
)   

with pl.Config(tbl_cols=-1,tbl_rows=-1, tbl_width_chars=200):
    print(enface_plus_counts)
    print(enface_zone_counts)
    print(enface_stage_counts)
    print(enface_means)
    print(enface_stds)


enface_project_df.write_csv(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_FULL_Revised_JJ.csv"))
enface_project_out_df.write_csv(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_Revised_JJ.csv"))
enface_project_out_df.write_excel(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_Revised_JJ.xlsx"))


enface_plus_counts.write_excel(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_plus_counts_JJ.xlsx"))
enface_zone_counts.write_excel(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_zone_counts_JJ.xlsx"))
enface_stage_counts.write_excel(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_stage_counts_JJ.xlsx"))
enface_means.write_excel(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_Enface_Data_Revised_means_JJ.xlsx"))
enface_stds.write_excel(Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Enface_Data_Enface_Data_Revised_stds.xlsx"))
