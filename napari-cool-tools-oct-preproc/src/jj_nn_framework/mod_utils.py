import math
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

def check_rand_sample_per_crop(x,y,h_idxs,w_idxs,device='cpu',verbose=False):
    ''''''
    
    x_samp,y_samp = len(x),len(y) # of images and labels
    num_h,num_w = len(h_idxs),len(w_idxs) # indicies for sub-images
    img_idx = int(x_samp/(num_h*num_w)) # idx number which indicates new image
    
    if verbose:
        print(f"images: {x_samp}, height indicies: {num_h} , width indicies {num_w}, new image step: {img_idx}")

    samp_idxs = torch.arange(0,x_samp,img_idx,device=device) # tensor of indicies for each new image
    
    #print(f"samp_idxs:\n{samp_idxs.tolist()}\n")
    
    num_sub_img = samp_idxs[1] # number of sub-images per image
    sq_dim_samp = math.ceil(math.sqrt(num_sub_img)) # calculation of closest complete squareroot to represent 1 of each sample
    sq_dim_img =  math.ceil(math.sqrt(num_h*num_w)) # calculation of closest complete squareroot to represent 1 of each image
    
    # alternate display does not convert to 3-channel but is slower than using torchvison make_grid
    
    #plt.figure(figsize=(16.,16.))
    #for i in range(num_sub_img):
    #    plt.subplot(int(num_sub_img / sq_dim +1),sq_dim,i+1)
    #    plt.imshow(x[:samp_idxs[1]][i].detach().to('cpu').permute(1,2,0))
    
    grid_x = make_grid(x[:samp_idxs[1]],sq_dim_samp) # torchvision make grid limited to first minibatch with closest squareroot rows
    grid_y = make_grid(y[:samp_idxs[1]],sq_dim_samp) # torchvision make grid limited to first minibatch with closest squareroot rows
    grid_x2 = make_grid(x[:samp_idxs[1]*num_h*num_w:samp_idxs[1]],num_w) #sq_dim_img) # torchvision make grid sampling from single image
    grid_y2 = make_grid(y[:samp_idxs[1]*num_h*num_w:samp_idxs[1]],num_w) #sq_dim_img) # torchvision make grid sampling from single image
    
    plt.figure(figsize=(16.,16.))
    plt.subplot(221)
    plt.imshow(grid_x.detach().to('cpu').permute(1,2,0))  # display random crop for all images in minibatch
    plt.subplot(222)
    #plt.imshow(grid_y.detach().to('cpu').to(torch.uint8).permute(1,2,0)) # values are 0-255 conveted to float for processing
    plt.imshow(grid_y.detach().to('cpu').permute(1,2,0))
    plt.subplot(223)
    plt.imshow(grid_x2.detach().to('cpu').permute(1,2,0))  # display random crop for all images in minibatch
    plt.subplot(224)
    #plt.imshow(grid_y.detach().to('cpu').to(torch.uint8).permute(1,2,0)) # values are 0-255 conveted to float for processing
    plt.imshow(grid_y2.detach().to('cpu').permute(1,2,0))
    plt.show()


def get_crops(x,y,kh,kw,h_idxs,w_idxs,shuffle=True,device='cpu',verbose=False):
    ''''''
    num_h = len(h_idxs)
    num_w = len(w_idxs)
    coord_idx = torch.arange(0,num_h*num_w,device=device)
    coord_idx = coord_idx.view(num_h,num_w)
    
    if shuffle:
        idx_order = torch.randperm(num_h*num_w,device=device)
    else:
        idx_order = torch.arange(num_h*num_w,device=device)
        
    if verbose:    
        print(f"coord_idx:\n{coord_idx}\n")
        print(f"idx_order:\n{idx_order}\n")
    
    #idxs = []
    crops_x = []
    crops_y = []
    for i in idx_order:
        h_i,w_i = (coord_idx == i).nonzero().squeeze()
        h,w = h_idxs[h_i],w_idxs[w_i]
        #idxs.append((h.item(),w.item()))
        crops_x.append(TF.crop(x,h,w,kh,kw))
        crops_y.append(TF.crop(y,h,w,kh,kw))
        
    crops_x = torch.cat(crops_x,dim=0)
    crops_y = torch.cat(crops_y,dim=0)
    
    return crops_x,crops_y