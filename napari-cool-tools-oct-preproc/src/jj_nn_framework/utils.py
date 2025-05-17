# code based on code from https://www.learnpytorch.io/05_pytorch_going_modular/
"""
Contains various utility functions for PyTorch model training and saving.
"""
import os
import json
from pathlib import Path

from ruamel.yaml import YAML
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

def get_config(config_dir,file_pre ="",verbose=False,silent=False):
    ''''''
    
    config_path = Path(os.path.abspath(config_dir))
    
    if verbose:
        if config_path.is_dir():
            print(f"\nConfig directory exists @:\n{config_path}\n")
        else:
            print(f"\nDid not find config directory @:\n{config_path}\ncreating directory...\n")
            config_path.mkdir(parents=True, exist_ok=True)

    if file_pre != "" :
        config_file_path = config_path / f"{file_pre}_config.json"
    else:
        config_file_path = config_path / "config.json"
        #print(config_file_path.is_file())
    
    if config_file_path.is_file():
        if not silent:
            print(f"\nConfig file exists retrieving {config_file_path.name} from:\n{config_file_path.parent}\n")
        with open(config_file_path,"r") as config_file:
            config_data = json.load(config_file)
            if verbose:
                print(config_data)
            else:
                pass
            return config_data
    else:
        print(f"\nCould not open config file does not exist @:\n{config_file_path}\n")
        

def save_config(config_dir,config_data,file_pre="",verbose=False,silent=False,as_yaml:bool=False):
    ''''''
    
    config_path = Path(os.path.abspath(config_dir))
    
    if as_yaml:
        yaml = YAML()
        ext = "yaml"
    else:
        ext = "json"
    
    if verbose:
        if config_path.is_dir():
            print(f"\nConfig directory exists @:\n{config_path}\n")
        else:
            print(f"\nDid not find config directory @:\n{config_path}\ncreating directory...\n")
            config_path.mkdir(parents=True, exist_ok=True)

    if file_pre != "":    
        config_file_path = config_path / f"{file_pre}_config.{ext}"
        #print(config_file_path.is_file())
    else:
        config_file_path = config_path / f"config.{ext}"
    
    if config_file_path.is_file():
        if not silent:
            print(f"\nConfig file exists...modifying {config_file_path.name} @:\n{config_file_path.parent}\n")
        if as_yaml:
            with open(config_file_path,"w") as config_file:
                yaml.dump(config_data,config_file)
        else:
            with open(config_file_path,"w") as config_file:
                json.dump(config_data,config_file,indent=4)
    else:
        if not silent:
            print(f"\nConfig file does not exist...creating {config_file_path.name} @:\n{config_file_path.parent}\n")
        if as_yaml:
            with open(config_file_path,"w") as config_file:
                yaml.dump(config_data,config_file)
        else:
            with open(config_file_path,"w") as config_file:
                json.dump(config_data,config_file,indent=4)
    pass

def reactivate_checkpoint(chkpt_file_dir, model, optimizer):
    ''''''
    chkpt_path = Path(chkpt_file_dir)
    
    try:
        assert(chkpt_path.is_file()) and (chkpt_path.suffix == '.pt')
    except: 
        print(f"Path: {chkpt_path} is not a file of type '.pt'")
    else:
        
        sess_chkpt_path = chkpt_path.parent
        sess_name = sess_chkpt_path.stem
        
        chkpt = torch.load(chkpt_path)
        model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        
        if 'early_stop' in chkpt.keys():
            estc = chkpt['early_stop']
        else:
            estc = 0
        
        out = {
            'chkpt_path':sess_chkpt_path,
            'sess_name':sess_name,
            'epoch': chkpt['epoch'],
            'train_loss':chkpt['train_loss'],
            'train_acc':chkpt['train_acc'],
            'val_loss':chkpt['val_loss'],
            'val_acc':chkpt['val_acc'],
            'optim_val_loss': chkpt['optim_val_loss'],
            'optim_train_loss': chkpt['optim_train_loss'],
            'best_val_acc': chkpt["best_val_acc"],
            'best_train_acc': chkpt["best_train_acc"],
            'early_stop': estc,
            'model':model,
            'optimizer':optimizer
        }        
        
        return out

def model_out_to_probs(net_out,pred=None):
    '''
    Generates predictions and corresponding probabilities from a trained
    networks output from a batch of images
    '''
    
    if pred == None:
        pred = net_out.argmax(-1)
    else:
        pass
    
    idx = torch.arange(len(net_out))
    prob = net_out[idx,pred]
    return pred,prob


def plot_mnist_preds(mod_out,images,labels,preds=None):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "model_out_to_probs" function.
    '''
    
    # get predictions and probabilities
    if preds == None:
        preds,probs = model_out_to_probs(mod_out)
    else:
        _,probs = model_out_to_probs(mod_out,preds)
        
    # plot the images in the batch, along with predicted and true labels
    
    labels_int = labels.argmax(-1)
    num_images = len(images)
    
    row_div = (num_images/4)/6
    col_div = (num_images/4)/9
    
    
    fig = plt.figure(figsize=(int((num_images/4)/row_div), int((num_images/4)/col_div)))
    for i in range(num_images):
        
        ax = fig.add_subplot(4,int(num_images/4),i+1,xticks=[],yticks=[])
        matplotlib_imshow(images[i],one_channel=True)
        ax.set_title(f"pred:{preds[i]}\nprob:{probs[i]:.4f}\nlabel({labels_int[i]})",
                    color=("green" if preds[i]==labels_int[i] else "red"))
        
        
    
    return fig

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.to("cpu").numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))