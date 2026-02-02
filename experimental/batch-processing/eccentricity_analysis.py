""""""


from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import polars as pl

input_path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output\DataFrames\incidence_angle_error_df.csv")
print("Started")
def process_batch(df:pl.DataFrame):
    """"""
    print(df)
    return df

incidence_angle_df = pl.read_csv(input_path)
incidence_angle_df = incidence_angle_df.cast({pl.Float64: pl.Float32})
incidence_angle_df = incidence_angle_df.to_pandas()
print(incidence_angle_df)

# 1. Create a canvas (resolution of the output image)
cvs = ds.Canvas(plot_width=1000, plot_height=1000)

# 2. Aggregate the points into the canvas grid
agg = cvs.points(incidence_angle_df, 'angles', 'percent_error')

# 3. Shade the aggregated data into an image with a colormap
img = tf.shade(agg, cmap=cc.fire)

# 4. Display or save
img

# incidence_lf = pl.scan_csv(input_path)
# incidence_lf = pl.read_csv_batched(input_path,batch_size=1000)
# for i, batch_df in enumerate(incidence_lf.collect_batches()):
#     print(f"---- Processing Batch {i} ----")
#     print(batch_df)
# incidence_lf.map_batches(process_batch,streamable=True)

print("Finished")
