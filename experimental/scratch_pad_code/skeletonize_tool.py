


import numpy as np
import torch
import napari
from jj_nn_framework.skeletonize import Skeletonize

viewer = napari.Viewer(show=False)
data = np.arange(0,255)

data_t = torch.tensor(data.copy().astype(np.float32)).cuda()
data_t = data_t[None,None,:,:]
skele = Skeletonize()
skele.cuda()
skele_t = skele(data_t)
skele_out = skele_t.detach().cpu().squeeze().numpy()
viewer.add_labels(skele_out.astype(np.uint8),name=f"skele")