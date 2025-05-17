"""
Functions for loading data .
"""
import gc
import os
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict,List
from numpy.typing import NDArray
from pathlib import Path
from skimage import io
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from jj_nn_framework.utils import get_config

class TrainingSplits():
    ''''''
    def __init__(self,X_train,y_train,X_test,y_test,X_val=None,y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        if X_val != None and y_val != None:
            self.X_val = X_val
            self.y_val = y_val
    
    def __str__(self):
        ''''''
        out_str = "Dataset containing:\n"
        for i,key in enumerate(vars(self).keys()):
            if (i-1)%2 == 0:
                x = f"{key}: {vars(self)[key].shape}\n"
            else:
                x = f"{key}: {vars(self)[key].shape}, "
            out_str = out_str + x
        return out_str
            

def load_mnist_img_label_splits(data_dir, val_split=0.0):
    ''''''
    
    dataset_tuple = load_mnist_datasets(data_dir)
    
    training = dataset_tuple[0]
    testing = dataset_tuple[1]
    
    X_train,y_train = training.data,training.targets
    X_test,y_test = testing.data,testing.targets
    
    if val_split == 0:
        return TrainingSplits(X_train,y_train,X_test,y_test)
    elif val_split >= 1.0 or val_split < 0:
        print("val_split must be >= 0 and < 1 as it indicates the percentage split\
 between testing and validation sets\n")
    else:
        samples = len(X_test)
        val_samples = int(val_split * samples)
        test_samples = samples - val_samples
        X_val,y_val = X_test[test_samples:],y_test[test_samples:]
        X_test,y_test = X_test[:test_samples],y_test[:test_samples]
        
        return TrainingSplits(X_train,y_train,X_test,y_test,X_val,y_val)
    

def load_mnist_datasets(data_dir):
    ''''''
    
    data_path = Path(os.path.abspath(data_dir))
    try:
        assert data_path.is_dir()
    except:
        print(f"\nNo valid data_path directory @:\n{data_path}\n")
    else:
        #return data_path
        
        # Datasets
        MNIST_training = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        MNIST_testing = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        
        return MNIST_training,MNIST_testing
    
def create_dataloaders(
    config_dir: str,
    data_dir: str,
    val_split = 0,
):
    ''''''
    
    # get configuration
    config_data = get_config(config_dir)
    dl_config = config_data["dataloader_params"]
    batch_sz = dl_config["batch_sz"]
    num_workers = 0 #dl_config["num_workers"]
    shuffle = bool(dl_config["shuffle"])
    
    # define data loader parameters
    # parameters
    params = {
        'batch_size':batch_sz,
        'shuffle':shuffle,
        'num_workers':num_workers,
        'pin_memory':True
    }
    
    # get dataset splits
    splits = load_mnist_img_label_splits(data_dir,val_split)
    
    datasets = datasets_from_splits(splits)
    
    
    dataloaders = {}
    
    for key,dataset in datasets.items():
        print(f"key:{key}\ndataset:\n{dataset}\n")
        dataloaders[key] = DataLoader(dataset,**params)
    
            
    return dataloaders

def datasets_from_splits(splits):
    ''''''
    datasets = {}
    # preprocess image data
    for key in vars(splits).keys():
        if "X" in key:
            y_key = key.replace("X","y")
            X_raw = vars(splits)[key]
            y_raw = vars(splits)[y_key]
            #vars(splits)[key] = (vars(splits)[key]/256).to(torch.float32)
            vars(splits)[key],vars(splits)[y_key] = preproc_uint8_norm_1hot(X_raw,y_raw)
            X_out,y_out = vars(splits)[key],vars(splits)[y_key]
            
            if "train" in key:
                o_key = "training"
            elif "test" in key:
                o_key = "testing"
            elif "val" in key:
                o_key = "validation"
            else:
                o_key = "misc"
            
            datasets[o_key] = DatasetImgLabel(X_out,y_out)
            
            
    return datasets
    

class DatasetImgLabel(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_data,labels):
        '''Initialization'''
        
        try:
            assert(img_data != None and labels != None)
        except:
            print(f"Image or Labels data is missing !!")
        else:
            self.img_data = img_data
            self.labels = labels
        
        
    def __len__(self):
        '''Returns total number of samples'''
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''Generates a single sample of the data'''        
        
        # get data and label        
        return self.img_data[idx].unsqueeze(0),self.labels[idx]
    
    def __str__(self):
        return f"Image, Label Dataset: {type(self.img_data).__name__}{list(self.img_data.shape)}({self.img_data.dtype}), \
{type(self.labels).__name__}{list(self.labels.shape)}({self.labels.dtype})"
    
    def __repr__(self):
        return f"DatasetImgLabel(img_data={type(self.img_data).__name__},labels={type(self.labels).__name__})"
    
def preproc_uint8_norm_1hot(X_vec,y_scalar):
    ''''''
    
    X_out = X_vec/256
    num_class = y_scalar.max() + 1
    num_samples = y_scalar.shape[0]
    y_out = torch.zeros(num_samples,num_class,dtype=torch.uint8)
    y_range = torch.arange(0,num_samples,dtype=torch.int64)
    y_out[y_range,y_scalar[y_range]] = 1
    
    return X_out,y_out


def gen_files(img_paths,label_paths,num_samples,data_path,file_name,file_dir):
        ''''''
        names = []
        images = []
        labels = []

        for i in range(num_samples):

            img_name = img_paths[i].as_posix()
            lbl_name = label_paths[i].as_posix()
            img_type = img_paths[i].suffixes
            lbl_type = label_paths[i].suffixes
            names.append({'img_name':img_name, 'lbl_name':lbl_name})

            try:
                assert(len(img_type) == 1 and len(lbl_type) == 1)
            except:
                print(f"Cannot open multipart file extension/s {img_type if len(img_type)>1 else ''} and/or {lbl_type if len(lbl_type)>1 else ''} are not supported\n")
            else:
                img_type = img_type[0]
                lbl_type = lbl_type[0]

            if img_type == '.npy':
                print("inside im_type == '.npy'")
                img = np.load(img_paths[i])
            elif img_type == '.png':
                img = io.imread(img_paths[i])
            else:
                print(f"File type {img_type} is not supported\n")

            if lbl_type == '.npy':
                label = np.load(label_paths[i])
            elif lbl_type == '.png':
                label = io.imread(label_paths[i])
            else:
                print(f"File type {img_type} is not supported\n")
            
            images.append(img)
            labels.append(label)

        images = torch.from_numpy(np.array(images))
        labels = torch.from_numpy(np.array(labels))

        save_path = data_path/file_dir

        os.makedirs(save_path,exist_ok=True)

        torch.save(
            {
                'file_paths':names,
                'images':images,
                'masks':labels
            }, save_path / f"{file_name}.pt"
        )
        
        print(f"{file_name} was saved to:\n{save_path}\nContains indexed file paths, images and labels.\n")

        return names,images,labels
        
        ''''''
        img_path = Path(img_dir)
        label_path = Path(label_dir)
        data_path = img_path.parent


def gen_pt_files(img_dir,label_dir,file_name:str='image_label_data',file_dir = 'unified_Pytorch_data'):
    ''''''
    img_path = Path(img_dir)
    label_path = Path(label_dir)
    data_path = img_path.parent
        
    try:
        assert(img_path.is_dir() and label_path.is_dir())
    except:
        print(f"Image or Labels data is missing !!")
    else:
        imgs = [x for x in img_path.iterdir() if x.is_file()]
        labels = [y for y in label_path.iterdir() if y.is_file()]
        num_imgs = len(imgs)
        num_labels = len(labels)
        try:
            assert(num_imgs == num_labels)
        except:
            print(f"Number of images and labels do not match !!\n"
                f"There are {num_imgs} images and {num_labels} labels.\n")
        
        else:
            out = gen_files(imgs,labels,num_imgs,data_path,file_name=file_name,file_dir=file_dir)
            return out

def load_img_lbl_tensors(data_dir,device='cpu'):
    ''''''
    
    data_path = Path(data_dir)

    try:
        assert(data_path.is_file())
        assert(data_path.suffix == '.pt')
    except:
        print(f"Image or Labels data are missing !!")
    else:
        
        data = torch.load(data_path)
        if "file_paths" in data.keys():
            paths = data["file_paths"]
        else:
            pass
        
        images = data["images"]
        labels = data["masks"] # change this to conditionally accept "masks" or "labels"

        num_imgs = len(images)
        num_labels = len(labels)

        try:
            assert(num_imgs == num_labels)
        except:
            print(f"Number of images and labels do not match !!\n"
                  f"There are {num_imgs} images and {num_labels} labels.\n")
        else:
            #print(f"There are {num_imgs} images and {num_labels} labels per chunk.\n")
            return(images.to(device),labels.to(device))


class RetCamTensorDataset(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, images,labels,transform=None,device='cpu'):
        '''Initialization'''
        
        try:
            assert(len(images) == len(labels))
        except:
            print(f"Number of images and labels do not match !!\n"
                  f"There are {len(images)} images and {len(labels)} labels.\n")
            return(images,labels)
        
        else:
            self.images = images
            self.labels = labels
            self.num_samples = len(images)
            self.transform = transform
            self.device = device
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples
    
    def __getitem__(self, idx):
        '''Generates a single sample of the data'''
        
        image = self.images[idx] / 256 #### comment something here
        label = self.labels[idx]
            
        if self.transform:
            data = self.transform((image,label))
            image = data[0]
            label = data[1]
        else:
            return image,label
    
    def __str__(self):
        str_out = (f"RetCamDataset:\nContains {self.num_samples} images with labels\nimages:\n{self.images.shape}\n"
                   f"labels:\n{self.labels.shape}\n")
        return str_out
    
    def __repr__(self):
        return f"RetCamDataset(images={torch.Tensor.__name__},labels={torch.Tensor.__name__})"
    
    def generate_split_datasets(self,per_tr=.8,per_t=.1,per_v=.1,possible_splits=20):
        ''''''
        num_train = round(self.num_samples*per_tr)
        num_test = round(self.num_samples*per_t)
        num_val = round(self.num_samples*per_v)
        
        print(f"train:{num_train}, test:{num_test}, val:{num_val} = {self.num_samples}?"
              f"{(num_train+num_test+num_val)==self.num_samples}")
        
        manual_seed = torch.randint(possible_splits,(1,)).item()
        torch.manual_seed(manual_seed)
        idxs = torch.randperm(self.num_samples)
        
        train_idx = idxs[:num_train]
        test_idx = idxs[num_train:num_train+num_test]
        val_idx = idxs[num_train+num_test:]
        
        print(f"train_idx:{train_idx.shape},test_idx{test_idx.shape},val_idx{val_idx.shape}\n")
        
        train_imgs = self.images[train_idx]
        train_lbls = self.labels[train_idx]
        
        train_dset = RetCamTensorDataset(train_imgs,train_lbls,transform=self.transform,device=self.device)
        train_dl = DataLoader(train_dset)
        
        training = {
            'type':'training',
            'manual_seed':manual_seed,
            'dataset':train_dset
        }
        
        test_imgs = self.images[test_idx]
        test_lbls = self.labels[test_idx]
        
        test_dset = RetCamTensorDataset(test_imgs,test_lbls,transform=self.transform,device=self.device)
        test_dl = DataLoader(test_dset)
        
        testing = {
            'type':'testing',
            'manual_seed':manual_seed,
            'dataset':test_dset
        }
        
        val_imgs = self.images[val_idx]
        val_lbls = self.labels[val_idx]
        
        val_dset = RetCamTensorDataset(val_imgs,val_lbls,transform=self.transform,device=self.device)
        val_dl = DataLoader(val_dset)
        
        validation = {
            'type':'validation',
            'manual_seed':manual_seed,
            'dataset':val_dset
        }
        
        return {'train':training,'test':testing,'val':validation}
    
    def check_samples(self):
        ''''''
        
        if self.num_samples < 8:
            idxs = torch.tensor([0])
        else:
            #idxs = torch.randint(0,self.num_samples,(16,))
            perm = torch.randperm(self.num_samples)
            idxs = perm[:16] 
            #idxs = torch.tensor((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),dtype=torch.uint8) # comment out to return to random

        fig = plt.figure(figsize=(16,12))
        fig.suptitle('Sanity Check for Image/Label Data')

        for i,idx in enumerate(idxs):
            image,label = self[idx]
            #print(sample,i)
            if i % 2 == 0:
                fig.add_subplot(4, 4, i+1)
                plt.tight_layout()
                plt.title(f'Sample {i} @\n idx: {idx}\n{image.shape[1]}x{image.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(image):
                    plt.imshow(image.detach().to('cpu'))
                else:
                    pass

                fig.add_subplot(4, 4, i+2)
                plt.tight_layout()
                plt.title(f'Label {i} @\n idx: {idx}\n{label.shape[1]}x{label.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(label):
                    plt.imshow(label.detach().to('cpu'))
                else:
                    pass

        plt.show()


class BasicTensorDataset(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, images,labels,transform=None,device='cpu'):
        '''Initialization'''
        
        try:
            assert(len(images) == len(labels))
        except:
            print(f"Number of images and labels do not match !!\n"
                  f"There are {len(images)} images and {len(labels)} labels.\n")
            return(images,labels)
        
        else:
            self.images = images
            self.labels = labels
            self.num_samples = len(images)
            self.transform = transform
            self.device = device
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples
    
    def __getitem__(self, idx):
        '''Generates a single sample of the data'''
        
        image = self.images[idx] 
        label = self.labels[idx]
            
        if self.transform:
            data = self.transform((image,label))
            image = data[0]
            label = data[1]
        else:
            return image,label

    def check_samples(self):
        ''''''
        
        if self.num_samples < 8:
            idxs = torch.tensor([0])
        else:
            #idxs = torch.randint(0,self.num_samples,(16,))
            perm = torch.randperm(self.num_samples)
            idxs = perm[:16] 
            #idxs = torch.tensor((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),dtype=torch.uint8) # comment out to return to random

        fig = plt.figure(figsize=(16,12))
        fig.suptitle('Sanity Check for Image/Label Data')

        for i,idx in enumerate(idxs):
            image,label = self[idx]
            #print(sample,i)
            if i % 2 == 0:
                fig.add_subplot(4, 4, i+1)
                plt.tight_layout()
                plt.title(f'Sample {i} @\n idx: {idx}\n{image.shape[1]}x{image.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(image):
                    plt.imshow(image.detach().permute(1,2,0).to('cpu'))
                else:
                    pass

                fig.add_subplot(4, 4, i+2)
                plt.tight_layout()
                plt.title(f'Label {i} @\n idx: {idx}\n{label.shape[1]}x{label.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(label):
                    plt.imshow(label.detach().permute(1,2,0).to('cpu'))
                else:
                    pass

        plt.show()

class LoadTensorDataset(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_lbl_tensor_dir,transform=None,preprocessing=None,device='cpu'):
        '''Initialization'''

        img_lbl_tensor_path = Path(img_lbl_tensor_dir)
        
        try:
            assert(img_lbl_tensor_path.is_dir())
        except:
            print(f"{img_lbl_tensor_path} is not a valid path!!\n")
        
        else:
            batches = list(img_lbl_tensor_path.glob('*.pt'))

            image,label = load_img_lbl_tensors(batches[0],device='cpu')

            self.num_samples = len(image)
            self.batches = batches
            self.num_batches = len(batches)
            self.transform = transform
            self.preprocessing = preprocessing
            self.device = device
            del image
            del label
            #images,labels = load_img_lbl_tensors(img_lbl_tensor_path,device=device)
            #self.images = images
            #self.labels = labels
            #self.num_samples = len(images)
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_batches

    def __getitem__(self, idx):
        '''Generates a single sample of the data'''
        
        img_lbl_tensor_f = self.batches[idx]
        img_lbl_tensor_path = Path(img_lbl_tensor_f)
        image,label = load_img_lbl_tensors(img_lbl_tensor_path,device=self.device)
            
        if self.transform:
            print("Transforming data.\n")
            image,label = self.transform((image,label))
            #print(image.shape)
            self.num_samples = len(image)
        else:
            pass

        if self.preprocessing:
            print("Preprocessing data.\n")
            image,label = self.preprocessing((image,label))    
            
        return image,label

    def check_samples(self):
        ''''''

        image_batch,label_batch = self[0]
        #print(image_batch.shape,label_batch.shape)

        print(f"num_samples: {self.num_samples}\n")
        if self.num_samples < 8:
            idxs = torch.tensor([0])
        else:
            #idxs = torch.randint(0,self.num_samples,(16,))
            perm = torch.randperm(self.num_samples)
            idxs = perm[:16] 
            #idxs = torch.tensor((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),dtype=torch.uint8) # comment out to return to random

        fig = plt.figure(figsize=(16,12))
        fig.suptitle('Sanity Check for Image/Label Data')

        for i,idx in enumerate(idxs):
            image,label = image_batch[idx],label_batch[idx]

            if image.dim() == 2:
                im_h_dim,im_w_dim,im_ch = 0,1,1
            elif image.dim() == 3:
                im_h_dim,im_w_dim,im_ch = 1,2,image.size()[0]
            else:
                pass # error checking here or elsewhere

            if label.dim() == 2:
                lb_h_dim,lb_w_dim,lb_ch = 0,1,1
            elif label.dim() == 3:
                lb_h_dim,lb_w_dim,lb_ch = 1,2,label.size()[0]
            else:
                pass # error checking here or elsewhere

            #print(sample,i)
            if i % 2 == 0:
                fig.add_subplot(4, 4, i+1)
                plt.tight_layout()
                plt.title(f'Sample {i} @\n idx: {idx}\n{image.shape[im_h_dim]}x{image.shape[im_w_dim]}x{im_ch}')
                plt.axis('off')
                if torch.is_tensor(image):
                    if len(image.size()) == 3:
                        plt.imshow(image.detach().to('cpu').permute(1,2,0))
                    elif len(image.size()) == 2:
                        plt.imshow(image.detach().to('cpu'))
                    else:
                        pass # probably need error checking here or earlier
                else:
                    pass

                fig.add_subplot(4, 4, i+2)
                plt.tight_layout()
                plt.title(f'Label {i} @\n idx: {idx}\n{label.shape[lb_h_dim]}x{label.shape[lb_w_dim]}x{lb_ch}')
                plt.axis('off')
                if torch.is_tensor(label):
                    if len(label.size()) == 3:
                        plt.imshow(label.detach().to('cpu').permute(1,2,0))
                    elif len(label.size()) == 2:
                        plt.imshow(label.detach().to('cpu'))
                    else:
                        pass # probably need error checking here or earlier
                else:
                    pass

        plt.show()

class LoadTensorDataset2(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_lbl_tensor_dir,transform=None,preprocessing=None,device='cpu'):
        '''Initialization'''

        img_lbl_tensor_path = Path(img_lbl_tensor_dir)
        
        try:
            assert(img_lbl_tensor_path.is_dir())
        except:
            print(f"{img_lbl_tensor_path} is not a valid path!!\n")
        
        else:
            batches = list(img_lbl_tensor_path.glob('*.pt'))

            print(f"\nLoading Tensor Data Chunk {0}/{len(batches)}\n")

            img_data,lbl_data = load_img_lbl_tensors(batches[0],device=device)

            if transform:
                print("\nTransforming data.\n")
                img_data,lbl_data = transform((img_data,lbl_data))

            if preprocessing:
                print("\nPreprocessing data.\n")
                img_data,lbl_data = preprocessing((img_data,lbl_data))


            self.curr_data = (img_data,lbl_data)
            self.curr_batch = 0

            self.num_samples = len(img_data)*len(batches)
            self.batches = batches
            self.samp_per_batch = len(img_data)
            self.num_batches = len(batches)
            self.transform = transform
            self.preprocessing = preprocessing
            self.device = device
            #images,labels = load_img_lbl_tensors(img_lbl_tensor_path,device=device)
            #self.images = images
            #self.labels = labels
            #self.num_samples = len(images)
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples

    def __getitem__(self, abs_idx):
        '''Generates a single sample of the data'''
        
        batch = math.floor(abs_idx/self.samp_per_batch)
        idx = abs_idx - (batch * self.samp_per_batch)

        #print(
        #    f"abs_idx: {abs_idx}\n"
        #    f"batch: {batch}\n"
        #    f"idx: {idx}\n"
        #)

        if self.curr_batch == batch:
            pass
        else:
            print(f"\nLoading Tensor Data Chunk {batch+1}/{self.num_batches}\n")
            #print(
            #    f"Current memory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            del self.curr_data
            gc.collect
            torch.cuda.empty_cache()
            self.curr_batch = batch
            self.curr_data = None

            img_lbl_tensor_f = self.batches[batch]
            img_lbl_tensor_path = Path(img_lbl_tensor_f)

            #print(
            #    f"Prior to loading next batch\nMemory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            img_data,lbl_data = load_img_lbl_tensors(img_lbl_tensor_path,device=self.device)
            
            if self.transform:
                print("\nTransforming data.\n")
                img_data,lbl_data = self.transform((img_data,lbl_data))

            if self.preprocessing:
                print("\nPreprocessing data.\n")
                img_data,lbl_data = self.preprocessing((img_data,lbl_data))  

            self.curr_data = (img_data,lbl_data)

        image,label = self.curr_data[0][idx],self.curr_data[1][idx]
            
        return image,label

class LoadTensorDataset2v2(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, batches,transform=None,preprocessing=None,device='cpu'):
        '''Initialization'''

        print(f"\nLoading Tensor Data Chunk {0}/{len(batches)}\n")

        img_data,lbl_data = load_img_lbl_tensors(batches[0],device=device)

        if transform:
            print("\nTransforming data.\n")
            img_data,lbl_data = transform((img_data,lbl_data))

        if preprocessing:
            print("\nPreprocessing data.\n")
            img_data,lbl_data = preprocessing((img_data,lbl_data))


        self.curr_data = (img_data,lbl_data)
        self.curr_batch = 0

        self.num_samples = len(img_data)*len(batches)
        self.batches = batches
        self.samp_per_batch = len(img_data)
        self.num_batches = len(batches)
        self.transform = transform
        self.preprocessing = preprocessing
        self.device = device
        #images,labels = load_img_lbl_tensors(img_lbl_tensor_path,device=device)
        #self.images = images
        #self.labels = labels
        #self.num_samples = len(images)
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples

    def __getitem__(self, abs_idx):
        '''Generates a single sample of the data'''
        
        batch = math.floor(abs_idx/self.samp_per_batch)
        idx = abs_idx - (batch * self.samp_per_batch)

        #print(
        #    f"abs_idx: {abs_idx}\n"
        #    f"batch: {batch}\n"
        #    f"idx: {idx}\n"
        #)

        if self.curr_batch == batch:
            pass
        else:
            print(f"\nLoading Tensor Data Chunk {batch+1}/{self.num_batches}\n")
            #print(
            #    f"Current memory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            del self.curr_data
            gc.collect
            torch.cuda.empty_cache()
            self.curr_batch = batch
            self.curr_data = None

            img_lbl_tensor_f = self.batches[batch]
            img_lbl_tensor_path = Path(img_lbl_tensor_f)

            #print(
            #    f"Prior to loading next batch\nMemory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            img_data,lbl_data = load_img_lbl_tensors(img_lbl_tensor_path,device=self.device)
            
            if self.transform:
                print("\nTransforming data.\n")
                img_data,lbl_data = self.transform((img_data,lbl_data))

            if self.preprocessing:
                print("\nPreprocessing data.\n")
                img_data,lbl_data = self.preprocessing((img_data,lbl_data))  

            self.curr_data = (img_data,lbl_data)

        image,label = self.curr_data[0][idx],self.curr_data[1][idx]
            
        return image,label
    
class LoadTensorDataset3(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_lbl_tensor:Dict,chunk_size:int=1,transform=None,preprocessing=None,
                 img_key:str='images',lbl_key:str='masks',device='cpu',verbose:bool=False,debug:bool=False):
        '''Initialization'''
        
        keys = list(img_lbl_tensor.keys())
        vals = list(img_lbl_tensor.values())


        img_data,lbl_data = img_lbl_tensor[img_key],img_lbl_tensor[lbl_key]

        '''
        if transform:
            print("\nTransforming data.\n")
            img_data,lbl_data = transform((img_data,lbl_data))

        if preprocessing:
            print("\nPreprocessing data.\n")
            img_data,lbl_data = preprocessing((img_data,lbl_data))
        '''

        try:
            assert(len(img_data) == len(lbl_data))
        except:
            print(f"Number of images {len(img_data)} != Number of labels/masks {len(lbl_data)}!!\n")
            raise SystemExit()

        num_samples = len(img_data)
        if num_samples % chunk_size == 0:
            chunk_idxs = int(num_samples / chunk_size)
            final_chunk_size = chunk_size
        else:
            chunk_idxs = int(num_samples / chunk_size) + 1
            final_chunk_size = num_samples % chunk_size

        init_chunk_idx = 0
        start = init_chunk_idx*chunk_size
        end = start+chunk_size

        self.keys = keys
        self.vals = vals
        self.img_key = img_key
        self.lbl_key = lbl_key
        self.data = (img_data,lbl_data)
        self.image_shape = img_data.shape
        self.label_shape = lbl_data.shape
        self.curr_data = (img_data[start:end].to(device),lbl_data[start:end].to(device))
        self.init_chunk_idx = init_chunk_idx
        self.curr_chunk_idx = None
        self.num_samples = num_samples
        self.chunk_idxs = chunk_idxs
        self.chunk_size = chunk_size
        self.final_chunk_size = final_chunk_size
        self.transform = transform
        self.preprocessing = preprocessing
        self.device = device
        self.verbose = verbose
        self.debug = debug

        if debug:
            print(f"current image data shape before preprocessing or transformation: {self.curr_data[0].shape}\n")
            print(f"current label data shape before preprocessing or transformation: {self.curr_data[1].shape}\n")
            print(f"image at index 0 shape: {self.curr_data[0][0].shape}\n")
            print(f"label at index 0 shape: {self.curr_data[1][0].shape}\n")
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples
    
    def __str__(self):
        str_out = (f"LoadTensorDataset:\nContains {self.num_samples} images with labels\nimages: {self.image_shape}\n"
                   f"labels: {self.label_shape}\n{self.chunk_size} samples are loaded at a time and the final chunk loads {self.final_chunk_size} samples\n")
        return str_out
    
    def __repr__(self):
        return f"RetCamDataset(images={torch.Tensor.__name__},labels={torch.Tensor.__name__})"

    def __getitem__(self, abs_idx):
        '''Generates a single sample of the data'''
        
        chunk = math.floor(abs_idx/self.chunk_size)
        idx = abs_idx - (chunk * self.chunk_size)

        if self.debug:
            print(
                f"abs_idx: {abs_idx}\n"
                f"chunk: {chunk}\n"
                f"idx: {idx}\n"
            )

        if self.init_chunk_idx == chunk and self.curr_chunk_idx == None:
            self.curr_chunk_idx = chunk

            #'''
            if self.transform:
                if self.verbose or self.debug:
                    print("\nTransforming data.\n")
                self.curr_data = self.transform(self.curr_data)

            if self.preprocessing:
                if self.verbose or self.debug:
                    print("\nPreprocessing data.\n")
                self.curr_data = self.preprocessing(self.curr_data)
            #'''
            
        elif self.curr_chunk_idx == chunk:
            pass
        else:
            if self.verbose or self.debug:
                print(f"\nLoading Tensor Data Chunk {chunk+1}/{self.chunk_idxs}\n")
            #print(
            #    f"Current memory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            del self.curr_data
            gc.collect
            torch.cuda.empty_cache()
            self.curr_chunk_idx = chunk
            self.curr_data = None

            start = chunk * self.chunk_size
            if chunk != (self.chunk_idxs - 1):
                end = start + self.chunk_size
            else:
                end = start + self.final_chunk_size

            #print(
            #    f"Prior to loading next batch\nMemory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            img_data,lbl_data = (self.data[0][start:end].to(self.device),self.data[1][start:end].to(self.device))

            if self.debug:
                print(f"\nChunk Size: {len(img_data)}\n")
                print(f"img_data shape: {img_data.shape}\n") 

            self.curr_data = (img_data,lbl_data)

            #'''
            if self.transform:
                if self.verbose or self.debug:
                    print("\nTransforming data.\n")
                self.curr_data = self.transform(self.curr_data)

            if self.preprocessing:
                if self.verbose or self.debug:
                    print("\nPreprocessing data.\n")
                self.curr_data = self.preprocessing(self.curr_data)
            #'''

        image,label = self.curr_data[0][idx],self.curr_data[1][idx]

        '''
        if self.transform:
            if self.verbose or self.debug:
                print("\nTransforming data.\n")
            image,label = self.transform((image,label))

        if self.preprocessing:
            if self.verbose or self.debug:
                print("\nPreprocessing data.\n")
            image,label = self.preprocessing((image,label))
        '''

        if self.debug:
            print(f"image shape post preprocessing and transformation: {image.shape}\n")
            
        return image,label
    
    '''
    def __del__(self):
        """"""
        del self.curr_data, self.data, self.preprocessing, self.transform
        gc.collect
        torch.cuda.empty_cache()
    '''
    
    def create_subdataset(self,start:int,stop:int,step:int=1):
        """"""
        keys = self.keys
        vals = self.vals
        #new_vals = []
        new_dict = {}

        if self.verbose or self.debug:
            print(f"keys:\n{keys} of type: {type(keys)}\nlen: {len(keys)}\ndir:\n{dir(keys)}\n")
            print(f"vals:\n{len(vals)} of type: {type(vals)}\nlen: {len(vals)}\ndir:\n{dir(vals)}\n")

        for i,key in enumerate(keys):
            new_dict[key] = vals[i][start:stop:step]

        new_dataset = LoadTensorDataset3(new_dict,
                                         self.chunk_size,
                                         self.transform,self.preprocessing,
                                         self.img_key,self.lbl_key,
                                         self.device,
                                         self.verbose,self.debug)
        
        #if self.verbose or self.debug:
        print(f"Creating new subdataset:\n{new_dataset}\n")

        return new_dataset
    
    def create_subdataset2(self,indicies:List[int]):
        """"""
        keys = self.keys
        vals = self.vals
        #new_vals = []
        new_dict = {}

        if self.verbose or self.debug:
            print(f"keys:\n{keys} of type: {type(keys)}\nlen: {len(keys)}\ndir:\n{dir(keys)}\n")
            print(f"vals:\n{len(vals)} of type: {type(vals)}\nlen: {len(vals)}\ndir:\n{dir(vals)}\n")

        for i,key in enumerate(keys):
            new_dict[key] = vals[i][indicies]

        new_dataset = LoadTensorDataset3(new_dict,
                                         self.chunk_size,
                                         self.transform,self.preprocessing,
                                         self.img_key,self.lbl_key,
                                         self.device,
                                         self.verbose,self.debug)
        
        #if self.verbose or self.debug:
        print(f"Creating new subdataset:\n{new_dataset}\n")

        return new_dataset
    

class LoadNumpyData(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img:NDArray,chunk_size:int=1,transform=None,preprocessing=None,
                 device='cpu',verbose:bool=False,debug:bool=False):
        '''Initialization'''


        num_samples = len(img)
        if num_samples % chunk_size == 0:
            chunk_idxs = int(num_samples / chunk_size)
            final_chunk_size = chunk_size
        else:
            chunk_idxs = int(num_samples / chunk_size) + 1
            final_chunk_size = num_samples % chunk_size

        init_chunk_idx = 0
        start = init_chunk_idx*chunk_size
        end = start+chunk_size

        img_tensor = torch.tensor(img.copy())

        self.data = img_tensor
        self.image_shape = img.shape
        self.curr_data = img_tensor[start:end].to(device)
        self.init_chunk_idx = init_chunk_idx
        self.curr_chunk_idx = None
        self.num_samples = num_samples
        self.chunk_idxs = chunk_idxs
        self.chunk_size = chunk_size
        self.final_chunk_size = final_chunk_size
        self.transform = transform
        self.preprocessing = preprocessing
        self.device = device
        self.verbose = verbose
        self.debug = debug

        if debug:
            print(f"current image data shape before preprocessing or transformation: {self.curr_data[0].shape}\n")
            print(f"image at index 0 shape: {self.curr_data[0][0].shape}\n")
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples
    
    def __str__(self):
        str_out = (f"LoadTensorDataset:\nContains {self.num_samples} images with labels\nimages: {self.image_shape}\n"
                   f"{self.chunk_size} samples are loaded at a time and the final chunk loads {self.final_chunk_size} samples\n")
        return str_out
    
    def __repr__(self):
        return f"RetCamDataset(images={torch.Tensor.__name__}"

    def __getitem__(self, abs_idx):
        '''Generates a single sample of the data'''
        
        chunk = math.floor(abs_idx/self.chunk_size)
        idx = abs_idx - (chunk * self.chunk_size)

        if self.debug:
            print(
                f"abs_idx: {abs_idx}\n"
                f"chunk: {chunk}\n"
                f"idx: {idx}\n"
            )

        if self.init_chunk_idx == chunk and self.curr_chunk_idx == None:
            self.curr_chunk_idx = chunk

            #'''
            if self.transform:
                if self.verbose or self.debug:
                    print("\nTransforming data.\n")
                self.curr_data = self.transform((self.curr_data,))[0]

            if self.preprocessing:
                if self.verbose or self.debug:
                    print("\nPreprocessing data.\n")
                self.curr_data = self.preprocessing((self.curr_data,))[0]
            #'''
            
        elif self.curr_chunk_idx == chunk:
            pass
        else:
            if self.verbose or self.debug:
                print(f"\nLoading Tensor Data Chunk {chunk+1}/{self.chunk_idxs}\n")
            #print(
            #    f"Current memory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            del self.curr_data
            gc.collect
            torch.cuda.empty_cache()
            self.curr_chunk_idx = chunk
            self.curr_data = None

            start = chunk * self.chunk_size
            if chunk != (self.chunk_idxs - 1):
                end = start + self.chunk_size
            else:
                end = start + self.final_chunk_size

            #print(
            #    f"Prior to loading next batch\nMemory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            img_data = (self.data[start:end].to(self.device))

            if self.debug:
                print(f"\nChunk Size: {len(img_data)}\n")
                print(f"img_data shape: {img_data.shape}\n") 

            self.curr_data = img_data

            #'''
            if self.transform:
                if self.verbose or self.debug:
                    print("\nTransforming data.\n")
                self.curr_data = self.transform((self.curr_data,))[0]

            if self.preprocessing:
                if self.verbose or self.debug:
                    print("\nPreprocessing data.\n")
                self.curr_data = self.preprocessing((self.curr_data))[0]
            #'''

        image = self.curr_data[idx]

        '''
        if self.transform:
            if self.verbose or self.debug:
                print("\nTransforming data.\n")
            image,label = self.transform((image,label))

        if self.preprocessing:
            if self.verbose or self.debug:
                print("\nPreprocessing data.\n")
            image,label = self.preprocessing((image,label))
        '''

        if self.debug:
            print(f"image shape post preprocessing and transformation: {image.shape}\n")
            
        return image
    
def rand_row_grouping_flip(curr_img_data_arr,curr_lbl_data_arr):
    """
    """

    curr_img_data = curr_img_data_arr
    curr_lbl_data = curr_lbl_data_arr

    rand_o = torch.randint(0,curr_img_data.ndim,(1,)).item()
    #print(f"rand_o: {rand_o}\n")

    orig_h = curr_img_data.shape[rand_o]
    org_shape = curr_img_data.shape
    curr_h = curr_img_data.shape[0]

    start = int((orig_h - curr_h)/2)
    
    # Reorient and trim excess for new orientation
    if rand_o == 1:
        curr_img_data = curr_img_data.permute(1,0,2)[start:start+curr_h]
        curr_lbl_data = curr_lbl_data.permute(1,0,2)[start:start+curr_h]
    elif rand_o == 2:
        curr_img_data = curr_img_data.permute(2,0,1)[start:start+curr_h]
        curr_lbl_data = curr_lbl_data.permute(2,0,1)[start:start+curr_h]
    else:
        pass
        #curr_img_data = curr_img_data
        #curr_lbl_data = curr_lbl_data
    
    #curr_img_data = curr_img_data_arr.swapaxes(0,rand_o)
    #curr_lbl_data = curr_lbl_data_arr.swapaxes(0,rand_o)

    #print(f" curr_img_data vs orig shape: {curr_img_data.shape[rand_o]}, {org_shape}")

    #'''
    if orig_h > curr_img_data.shape[1]:

        #curr_img_data,curr_lbl_data = rand_row_grouping_flip(curr_img_data,curr_lbl_data)

        pre_rep_h = curr_img_data.shape[1]
        # replicate to fill dimension
        replicas = int(orig_h/curr_img_data.shape[1])
        curr_img_data = curr_img_data.repeat(1,replicas,1)
        curr_lbl_data = curr_lbl_data.repeat(1,replicas,1)

        # flip to vary data and fill crops
        num_rows = curr_img_data.shape[1]
        #rows_to_flip = int(num_rows/2) # flip 50%
        #print(f"rows: {num_rows}, random indicies: {random_indicies}\n")
        #random_indicies = torch.randperm(num_rows)[:rows_to_flip]
        #curr_img_data[:,random_indicies,:] = torch.flip(curr_img_data[:,random_indicies,:],dims=[2])
        #curr_lbl_data[:,random_indicies,:] = torch.flip(curr_lbl_data[:,random_indicies,:],dims=[2])

        group_size = pre_rep_h # size of groupings
        rows_to_flip2 = int(group_size/2) # how many groups to flip
        #start_is = torch.randperm(orig_h)[:rows_to_flip2]
        #end_is = start_is + group_size
        
        # generate start indicies
        start_indices = torch.randint(0, num_rows - group_size + 1, (rows_to_flip2,))

        # create mask
        flip_mask = torch.zeros(num_rows, dtype=torch.bool)
        indicies = (start_indices[:,None] + torch.arange(group_size)).flatten()
        flip_mask[indicies] = True

        #f flip row groups
        curr_img_data[:,flip_mask,:] = torch.flip(curr_img_data[:,flip_mask,:],dims=[2])
        curr_lbl_data[:,flip_mask,:] = torch.flip(curr_lbl_data[:,flip_mask,:],dims=[2])
        
        #start_i = torch.randint(0,num_rows - group_size+1, (1,)).item()
        #print(f"start_i: {start_i}\n")
        #print(f"start_i: {start_is}, {len(start_is)}\n")
        #curr_img_data[:,start_i:start_i+group_size,:] = torch.flip(curr_img_data[:,start_i:start_i+group_size,:],dims=[2])
        #curr_img_data[:,start_is:end_is,:] = torch.flip(curr_img_data[:,start_is:end_is,:],dims=[2])
        #'''
        #print(f"This happened")

    #print(f"This happened")

    return curr_img_data,curr_lbl_data


def random_tensor_from_list(num_values:int,list_of_values):
    """"""
    random_indicies = torch.randint(0,len(list_of_values),(num_values,))
    return torch.tensor([list_of_values[i] for i in random_indicies])


class LoadVariationData(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_lbl_tensor:Dict,chunk_size:int=1,transform=None,preprocessing=None,
                 img_key:str='images',lbl_key:str='masks',device='cpu',is_tensor:bool=False,indicies_to_use=None,rand_orientation:bool=False,verbose:bool=False,debug:bool=False):
        '''Initialization'''
        
        keys = list(img_lbl_tensor.keys())
        vals = list(img_lbl_tensor.values())


        img_data,lbl_data = img_lbl_tensor[img_key],img_lbl_tensor[lbl_key].squeeze()


        print(f"\nInitialization image shape, lablel shape: {img_data.shape}, {lbl_data.shape}\n")

        if not is_tensor:
            img_data = torch.tensor(img_data.copy())
            lbl_data = torch.tensor(lbl_data.copy())
        else:
            pass

        '''
        if transform:
            print("\nTransforming data.\n")
            img_data,lbl_data = transform((img_data,lbl_data))

        if preprocessing:
            print("\nPreprocessing data.\n")
            img_data,lbl_data = preprocessing((img_data,lbl_data))
        '''
        if img_data.shape[-3] != lbl_data.shape[-3]:
            raise RuntimeError(f"Image and Label dimension index -3 do not match\nImage: {img_data.shape}\nLabel: {lbl_data.shape}\n")
        #try:
        #    assert(img_data.shape[1] == len(lbl_data))
        #except:
        #    print(f"Number of images {len(img_data)} != Number of labels/masks {len(lbl_data)}!!\n")
        #    raise SystemExit()

        num_samples = img_data.shape[1]
        num_variations = len(img_data)
        if num_samples % chunk_size == 0:
            chunk_idxs = int(num_samples / chunk_size)
            final_chunk_size = chunk_size
        else:
            chunk_idxs = int(num_samples / chunk_size) + 1
            final_chunk_size = num_samples % chunk_size

        init_chunk_idx = 0
        start = init_chunk_idx*chunk_size
        end = start+chunk_size

        if debug:
            print(f"Full image data before chunk window selection: {img_data.shape}\n")

        # select variation from 0 dimension of image data resulting in NHW current image
        if indicies_to_use == None:
            x = torch.randint((num_variations-1),(chunk_size,))  # select random variation
        else:
            x = random_tensor_from_list(chunk_size,indicies_to_use)
            #x = torch.full((chunk_size,),4)
        y = torch.arange(chunk_size)        # indicies for chunk
        window = img_data[:,start:end]      # select chunk with all possible variations 
        curr_img_data = window[x,y]              # select specific random variation and full chunk for that index
        curr_lbl_data = lbl_data[start:end]      # select acompanying labels

        if rand_orientation:
            curr_img_data,curr_lbl_data = rand_row_grouping_flip(curr_img_data,curr_lbl_data)

        # a single 2D image is indexed each iteration this should be expanded to match NCHW format
        #curr_img_data = curr_img_data.unsqueeze(0).unsqueeze(0)
        #curr_lbl_data = curr_lbl_data.unsqueeze(0).unsqueeze(0)

        if debug:
            print(f"curr_img_data shape: {curr_img_data.shape}\n")

        self.keys = keys
        self.vals = vals
        self.img_key = img_key
        self.lbl_key = lbl_key
        self.data = (img_data,lbl_data)
        self.image_shape = img_data.shape
        self.label_shape = lbl_data.shape
        self.curr_data = (curr_img_data.to(device),curr_lbl_data.to(device)) #(img_data[0,start:end].to(device),lbl_data[start:end].to(device))
        self.init_chunk_idx = init_chunk_idx
        self.curr_chunk_idx = None
        self.num_samples = num_samples
        self.num_variations = num_variations
        self.chunk_idxs = chunk_idxs
        self.chunk_size = chunk_size
        self.final_chunk_size = final_chunk_size
        self.transform = transform
        self.preprocessing = preprocessing
        self.indicies_to_use = indicies_to_use
        self.rand_orientation = rand_orientation # modification for random orientation
        self.device = device
        self.verbose = verbose
        self.debug = debug

        if debug:
            print(f"current image data shape before preprocessing or transformation: {self.curr_data[0].shape}\n")
            print(f"image output shape at index 0: {self.curr_data[0][0].shape}\n")
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples
    
    def __str__(self):
        str_out = (f"LoadTensorDataset:\nContains {self.num_samples} images with labels\nimages: {self.image_shape}\n"
                   f"labels: {self.label_shape}\n{self.chunk_size} samples are loaded at a time and the final chunk loads {self.final_chunk_size} samples\n")
        return str_out
    
    def __repr__(self):
        return f"RetCamDataset(images={torch.Tensor.__name__},labels={torch.Tensor.__name__})"

    def __getitem__(self, abs_idx):
        '''Generates a single sample of the data'''
        
        chunk_idx = math.floor(abs_idx/self.chunk_size)
        idx = abs_idx - (chunk_idx * self.chunk_size)

        if self.debug:
            print(
                f"abs_idx: {abs_idx}\n"
                f"chunk: {chunk_idx}\n"
                f"idx: {idx}\n"
            )

        if self.init_chunk_idx == chunk_idx and self.curr_chunk_idx == None:
            self.curr_chunk_idx = chunk_idx

            #'''
            if self.debug:
                print("\nInitial Chunk load and processing\n")

            if self.preprocessing:
                if self.verbose or self.debug:
                    print("\nPreprocessing data.\n")
                self.curr_data = self.preprocessing(self.curr_data)

            if self.transform:
                if self.verbose or self.debug:
                    print("\nTransforming data.\n")
                self.curr_data = self.transform(self.curr_data)
            #'''
            
        elif self.curr_chunk_idx == chunk_idx:
            pass
        else:
            if self.verbose or self.debug:
                print(f"\nLoading Tensor Data Chunk {chunk_idx+1}/{self.chunk_idxs}\n")
            #print(
            #    f"Current memory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            del self.curr_data
            gc.collect
            torch.cuda.empty_cache()
            self.curr_chunk_idx = chunk_idx
            self.curr_data = None

            start = chunk_idx * self.chunk_size
            if chunk_idx != (self.chunk_idxs - 1):
                end = start + self.chunk_size
                if self.indicies_to_use == None:
                    x = torch.randint((self.num_variations-1),(self.chunk_size,))
                else:
                    x = random_tensor_from_list(self.chunk_size,self.indicies_to_use)
                y = torch.arange(self.chunk_size)
            else:
                end = start + self.final_chunk_size
                if self.indicies_to_use == None:
                    x = torch.randint((self.num_variations-1),(self.final_chunk_size,))
                else:
                    x = random_tensor_from_list(self.final_chunk_size,self.indicies_to_use)
                y = torch.arange(self.final_chunk_size)

            #print(
            #    f"Prior to loading next batch\nMemory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)
            window = self.data[0][:,start:end]
            #x = torch.randint(5,(self.chunk_size,))
            #y = torch.arange(self.chunk_size)
            curr_img_data = window[x,y]
            curr_lbl_data = self.data[1][start:end] # modification for random orientation

            if self.rand_orientation:
                curr_img_data,curr_lbl_data = rand_row_grouping_flip(curr_img_data,curr_lbl_data) # modification for random orientation

            # a single 2D image is indexed each iteration this should be expanded to match NCHW format
            #curr_img_data = curr_img_data.unsqueeze(0).unsqueeze(0)
            #curr_lbl_data = curr_lbl_data.unsqueeze(0).unsqueeze(0)

            if self.debug:
                print(f"new curr_img_data shape: {curr_img_data.shape}\n")

            img_data = curr_img_data.to(self.device)
            lbl_data = curr_lbl_data.to(self.device) # modification for random orientation
            #lbl_data = self.data[1][start:end].to(self.device)

            #img_data,lbl_data = (self.data[0][0,start:end].to(self.device),self.data[1][start:end].to(self.device))

            if self.debug:
                print(f"\nChunk Size: {len(img_data)}\n")
                print(f"img_data shape: {img_data.shape}\n") 

            self.curr_data = (img_data,lbl_data)

            #'''
            if self.preprocessing:
                if self.verbose or self.debug:
                    print("\nPreprocessing data.\n")
                self.curr_data = self.preprocessing(self.curr_data)

            if self.transform:
                if self.verbose or self.debug:
                    print("\nTransforming data.\n")
                self.curr_data = self.transform(self.curr_data)
                
                # Experiment change dataset size on the fly
                #self.num_samples = self.num_samples*2
                #self.chunk_idxs = self.chunk_idxs*2
                #self.final_chunk_size = self.final_chunk_size*2
            #'''
        if self.debug:
            print(f"__getitem__ debug current_chunk shape: {self.curr_data[0].shape}\n")

        image,label = self.curr_data[0][idx],self.curr_data[1][idx]

        '''
        if self.transform:
            if self.verbose or self.debug:
                print("\nTransforming data.\n")
            image,label = self.transform((image,label))

        if self.preprocessing:
            if self.verbose or self.debug:
                print("\nPreprocessing data.\n")
            image,label = self.preprocessing((image,label))
        '''

        # a single sample is indexed each iteration and should return the format CHW per sample, number of batches N indicates the number of iterations
        #image = image.unsqueeze(0)
        #label = label.unsqueeze(0)

        # conditonals for when transforms add channels
        if image.ndim == 3:
            pass
        elif image.ndim == 4:
            image = image.squeeze()
        elif image.ndim == 2:
            image = image.unsqueeze(0)
        else:
            pass # put error here

        if label.ndim == 3:
            pass
        elif label.ndim == 4:
            label = label.squeeze()
        elif label.ndim == 2:
            label = label.unsqueeze(0)
        else:
            pass # put error here

        if self.debug:
            print(f"image output shape at idx {idx}: {image.shape}\n")
            print(f"label output shape at idx {idx}: {label.shape}\n")
            
        return image,label
    
    '''
    def __del__(self):
        """"""
        del self.curr_data, self.data, self.preprocessing, self.transform
        gc.collect
        torch.cuda.empty_cache()
    '''
    
    def create_subdataset(self,start:int,stop:int,step:int=1):
        """"""
        keys = self.keys
        vals = self.vals
        #new_vals = []
        new_dict = {}

        if self.verbose or self.debug:
            print(f"keys:\n{keys} of type: {type(keys)}\nlen: {len(keys)}\ndir:\n{dir(keys)}\n")
            print(f"vals:\n{len(vals)} of type: {type(vals)}\nlen: {len(vals)}\ndir:\n{dir(vals)}\n")

        for i,key in enumerate(keys):
            new_dict[key] = vals[i][start:stop:step]

        new_dataset = LoadTensorDataset3(new_dict,
                                         self.chunk_size,
                                         self.transform,self.preprocessing,
                                         self.img_key,self.lbl_key,
                                         self.device,
                                         self.verbose,self.debug)
        
        #if self.verbose or self.debug:
        print(f"Creating new subdataset:\n{new_dataset}\n")

        return new_dataset
    
    def create_subdataset2(self,indicies:List[int]):
        """"""
        keys = self.keys
        vals = self.vals
        #new_vals = []
        new_dict = {}

        if self.verbose or self.debug:
            print(f"keys:\n{keys} of type: {type(keys)}\nlen: {len(keys)}\ndir:\n{dir(keys)}\n")
            print(f"vals:\n{len(vals)} of type: {type(vals)}\nlen: {len(vals)}\ndir:\n{dir(vals)}\n")

        for i,key in enumerate(keys):
            new_dict[key] = vals[i][indicies]

        new_dataset = LoadTensorDataset3(new_dict,
                                         self.chunk_size,
                                         self.transform,self.preprocessing,
                                         self.img_key,self.lbl_key,
                                         self.device,
                                         self.verbose,self.debug)
        
        #if self.verbose or self.debug:
        print(f"Creating new subdataset:\n{new_dataset}\n")

        return new_dataset