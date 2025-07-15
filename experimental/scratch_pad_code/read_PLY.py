import pyvista as pv
from pyvista import examples
from pathlib import Path

filename = Path(r"D:\JJ\Projects\RT_Registration\Data\Test_Data\PLY_data\13_09_20_hand_tgt.ply")#examples.download_lobster(load=False)
#Path(filename).name
#'lobster.ply'
reader = pv.get_reader(filename)
mesh = reader.read()
mesh = mesh.scale([1/.10,1/.10,1/.10])
mesh.plot()