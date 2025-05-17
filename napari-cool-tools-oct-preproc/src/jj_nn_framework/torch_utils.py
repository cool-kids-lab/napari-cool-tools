"""
Pytorch utility functions
"""
import torch

def torch_interp(x,xp,yp,device='cpu'):
    """
    Prototype function to emulate numpy.interp
    """
    # find indicies where X would be inserted in xp
    indicies = torch.searchsorted(xp,x) #,side='right')

    # add sentinel value repeating final value of yp this behavior can be expanded
    y_cap = torch.tensor([yp[-1]]).to(device)
    y = torch.cat((yp,y_cap))

    # get prior indicies wherever the previous indx is not zero
    prev_indicies = torch.where(indicies > 0, indicies-1, 0)
    clipped_indicies = torch.where(indicies > len(xp)-1,len(xp)-1,indicies) # clip indicies beyond domain of xp

    # use indicies and prev indicies to determine range of xp
    mins = xp[prev_indicies]
    maxs = xp[clipped_indicies]
    vals = x

    # caculate weights for values of yp normalized to 0-1 range
    w = (vals-mins)/(maxs-mins)
    w = torch.nan_to_num(w,nan=0) # address nan in case max and min == 0

    # perform linear interpolation between values of the data
    out = torch.lerp(yp[prev_indicies].float(),yp[clipped_indicies].float(),w.float()) # ensure vaules are proper dtype for lerp


    return out