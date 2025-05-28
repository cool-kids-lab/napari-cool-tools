from torchvision.transforms.v2 import InterpolationMode

def unpad_and_resize(img,target_size=(992,800),interpolation:InterpolationMode=InterpolationMode.BILINEAR):
    """"""
    import torch
    from torchvision.transforms.v2.functional import resize
    img_h, img_w = img.shape[1:]
    width_pad = int((img_w - target_size[1])/2)
    height_pad = int((img_h - target_size[0])/2)
    
    if width_pad > 0:
        img = img[:,:,width_pad:-width_pad]
    if height_pad > 0:
        img = img[:,height_pad:-height_pad,:]
        
    resized = resize(torch.tensor(img.copy()),size=target_size,interpolation=interpolation).numpy()
        
    
    return resized