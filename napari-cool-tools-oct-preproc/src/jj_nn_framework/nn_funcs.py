import torch
import torch.nn.functional as F

def jj_cce(y_pred,truth_1hot,reduction='mean',verbose=False):
    ''''''
    
    if y_pred.dim() == 1 and truth_1hot.dim() == y_pred.dim():
        #y_pred = y_pred.unsqueeze(0)
        pass
        
    
    if verbose:
        print(f"y_pred.dim type: {y_pred.dim()}")
    
    b_idx = torch.arange(truth_1hot.shape[0])
    
    if verbose:
        print(f"b_idx: {b_idx}\n")
        
    y_log = torch.log(y_pred)
    if verbose:
        print(f"y_log shape:{y_log.shape}\ny_log:\n{y_log}\n")
        
    if y_pred.dim() >= 2 and truth_1hot.dim() == y_pred.dim():
        y_inter = y_log[b_idx]*truth_1hot[b_idx]
    else:
        y_inter = y_log*truth_1hot
        
    if verbose:
        print(f"intermediate result shape: {y_inter.shape}\nintermediate result:\n{y_inter}\n")
    loss = y_inter.sum(-1) * -1
    
    if reduction == 'none' or reduction == None:
        out = loss
    elif reduction == 'mean':
        out = loss.mean(-1)
    elif reduction == 'sum':
        out = loss.sum(-1)
    else:
        print(f"Invalid reduction argument {reduction}\nValid values include 'mean', 'sum', 'none', or {None}\n")

    return out

def jj_cce_prototype(y_pred,truth_1hot,verbose=False):
    ''''''
    
    y_log = torch.log(y_pred)
    if verbose:
        print(f"y_log:\n{y_log}")
    y_inter = y_log*truth_1hot
    if verbose:
        print(f"intermediate result:\n{y_inter}")
    loss = y_inter.sum() * -1
    
    return loss

def dice_metric4(y_pred,y_true,return_loss=True):

    print(f"y_pred: {y_pred.shape},y_true{y_true.shape}")

    y_pred = torch.where(y_pred>.9,1,0).to(torch.int64)
    y_pred = y_pred.permute(0,3,1,2)
    y_true = y_true.permute(0,3,1,2)

    print(f"y_pred: {y_pred.shape},y_true{y_true.shape}")    

    loss = y_true - y_pred

    # create the labels one hot tensor
    #target_one_hot = figure out for multiclass this is single class


    

    return loss

def dice_metric2(y_pred,y_true,return_loss=True):
    
    batch_size = y_pred.shape[0]
    
    # commented out to verify equation these parts required for actual prediction and truth vectors
    y_pred = torch.where(y_pred>.9,1,0).to(torch.int64) #.squeeze(0).detach()
    y_true = y_true.permute(0,2,3,1)
    
    intersect = y_true.logical_and(y_pred).nonzero() #.sum()
    truth = y_true.nonzero() #.sum()

    print(f"inter:{intersect.shape}, truth: {truth.shape}")

    dice_coe = 2 * (intersect/(truth + intersect))
    
    # account for batch size
    dice_coe = dice_coe / batch_size
    
    if return_loss:
        dice_loss = (1 - dice_coe)
        print(f"dice_coe:{dice_coe},{type(dice_coe)}")
        print(f"dice_loss:{dice_loss},{type(dice_loss)}")
        return(dice_coe, dice_loss)
    else:
        return(dice_coe,)


def dice_metric(y_pred,y_true,return_loss=True):
    
    y_pred = torch.where(y_pred>.9,255,0).to(torch.int64).squeeze(0).detach()
    y_true = y_true.permute(1,2,0)
    
    intersect = y_true.logical_and(y_pred).nonzero().shape[0]
    truth = y_true.nonzero().shape[0]
    dice_coe = 2 * (intersect/(truth + intersect))
    
    if return_loss:
        dice_loss = 1 - dice_coe
        return(dice_coe,dice_loss)
    else:
        return(dice_coe,)


def get_class_and_cnts(y):
    ''''''
    y = y.detach()
    y_flat = y.flatten(start_dim=1)
    y_bool = y_flat.bool()
    y_cnts = y_bool.sum(dim=1)
    y_class = y_cnts.bool().to(torch.int64)
    return y_class,y_cnts