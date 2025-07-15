import pyvista as pv
pl = pv.Plotter()
actor = pl.add_mesh(pv.Sphere())
widget = pl.add_affine_transform_widget(actor)
pl.show()