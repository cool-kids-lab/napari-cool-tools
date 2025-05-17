import torch
import torch.nn.functional as F
from torch import nn
from jj_nn_framework.nn_funcs import jj_cce, dice_metric4

# custom losses
class JJ_CCE(nn.Module):
    ''''''
    def __init__(self,reduction='mean'):
        super(JJ_CCE, self).__init__()
        self.jj_cce = jj_cce
        self.reduction = reduction

    def forward(self, preds,targets):
        return self.jj_cce(preds,targets,reduction=self.reduction)


class Dice_Loss(nn.Module):
    ''''''
    def __init__(self):
        super(Dice_Loss, self).__init__()
        self.dice_metric4 = dice_metric4

    def forward(self, preds,targets):
        return self.dice_metric4(preds,targets)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice