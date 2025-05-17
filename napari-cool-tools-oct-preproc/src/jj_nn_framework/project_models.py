import torch
import torchvision
import torch.nn as nn
import segmentation_models_pytorch as smp
import lightning as L

class LitUnet(L.LightningModule):
    def __init__(self, train_config, loss_metric, acc_metric,debug:bool=False):
        super().__init__()
        self.model = smp.Unet(encoder_name=train_config['ENCODER'], # smp.UnetPlusPlus(encoder_name=ENCODER,
                    encoder_weights=train_config['ENCODER_WEIGHTS'],
                    encoder_depth=train_config["ENCODER_DEPTH"], #5,
                    decoder_channels=train_config['DECODER_CHANNELS'], #(224,112,56,28,14),#(224,112,56,28,14),(864,432,216,108,54),(16,8,4,2,1),(32,16,8,4,2),(64,32,16,8,4),(128,64,32,16,8),(256,128,64,32,16),(512,256,128,64,32),
                    in_channels=train_config["IN_CHANNELS"], #1, #3,
                    classes=len(train_config['CLASSES']),
                    activation=train_config['ACTIVATION'])
        self.train_config = train_config
        self.loss_metric = loss_metric
        self.acc_metric = acc_metric
        self.num_classes = len(train_config['CLASSES'])
        self.warm_up_iter = train_config["WARMUP_ITER"]
        self.debug = debug
        

    def forward(self,x):
        # lightning module functional use
        pred = self.model(x)

        pred_out = torch.zeros_like(pred[:,0,:,:])
        pred_argmax = pred.argmax(1)
        for i in range(pred.shape[1]):
            pred_out[pred_argmax==i] = i

        pred_out = pred_out / pred_out.max()

        pred_out = pred_out.unsqueeze(1) * 2 #255 #int(255 / (3-1))
        pred_out = pred_out.to(torch.uint8)

        return pred_out, pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x,y = batch
        pred = self.model(x)

        #print(f"x/y shapes: {x.shape}/{y.shape}")

        loss = self.loss_metric(pred,y.long())
        y_one_hot = nn.functional.one_hot(y.long(),num_classes=self.num_classes)
        y_one_hot = y_one_hot.squeeze().permute(0,3,1,2)
        acc = self.acc_metric(pred,y_one_hot) #nn.functional.one_hot(y.long(),num_classes=2)

        lr = self.lr_schedulers().get_last_lr()[0]
        log_vals = {"Train/loss": loss, "Train/acc": acc, "Train/lr":lr}
        self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        #self.log("Train/loss",loss,prog_bar=True,on_epoch=True)
        
        #if self.trainer.current_epoch % 5 == 0:
            #print(f"\n\nStep is: {self.trainer.global_step}!!\n\n")

        return loss
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop
        x,y = batch
        pred = self.model(x)

        val_loss = self.loss_metric(pred,y.long())
        y_one_hot = nn.functional.one_hot(y.long(),num_classes=self.num_classes)
        y_one_hot = y_one_hot.squeeze().permute(0,3,1,2)
        val_acc = self.acc_metric(pred,y_one_hot) #nn.functional.one_hot(y.long(),num_classes=2)

        log_vals = {"Val/loss": val_loss, "Val/acc": val_acc}
        self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        #self.log("Validation/loss",val_loss,prog_bar=True,on_epoch=True)
        return log_vals

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop
        x,y = batch
        pred = self.model(x)

        if len(batch) > 1:
            sample_idx = int(len(batch)/2) - 1
        else:
            sample_idx = 0

        if self.debug:
            print(f"\nx,y shapes are: {x.shape}, {y.shape}\n")
            print(f"\npred shape, y shape: {pred.shape}, {y.shape}\n")
            print(f"\npred min/max,: {pred.min()}/{pred.max()}\n")

        y_one_hot = nn.functional.one_hot(y.long(),num_classes=self.num_classes)
        y_one_hot = y_one_hot.squeeze().permute(0,3,1,2)
        test_acc = self.acc_metric(pred,y_one_hot) #nn.functional.one_hot(y.long(),num_classes=2)
        
        sample_imgs = x
        #print(f"\nsample images shape: {sample_imgs.shape}\n")
        grid = torchvision.utils.make_grid(sample_imgs[:4],nrow=4)
        #self.logger.experiment.add_image('example_images', grid, 0)
        self.logger.experiment.add_image('example_images', sample_imgs[sample_idx], 0)

        #print(f"y min/max: {y.min()}/{y.max()}")

        sample_gts = y*int(255 / (self.num_classes-1)) # uint8 max value of 255 is rescaled to match the maximum possible value of y which = num classes-1
        sample_gts = sample_gts #.unsqueeze(1) # had to add this with the changes) made fis now its gone again

        if self.debug:
            print(f"\nsample_gts, grid shape & type: {sample_gts.shape}, {sample_gts.dtype}, {grid.shape}, {grid.dtype}\n")
            print(f"\nsample_gts min/max: {sample_gts.min()}/{sample_gts.max()}\n")

        #grid = torchvision.utils.make_grid(sample_gts,nrow=4)

        grid = torchvision.utils.make_grid(sample_gts[:4],nrow=4)
        #self.logger.experiment.add_image('example_gt', grid, 0)
        self.logger.experiment.add_image('example_gt', sample_gts[sample_idx], 0)

        pred_out = torch.zeros_like(pred[:,0,:,:])
        pred_argmax = pred.argmax(1)
        for i in range(pred.shape[1]):
            pred_out[pred_argmax==i] = i

        pred_out = pred_out / pred_out.max()

        if self.debug:
            print(f"\npred_out info: {pred_out.min()}/{pred_out.max()}, {pred_out.shape}, {pred_out.dtype}\n")

        sample_preds = pred_out.unsqueeze(1) * 255 #int(255 / (3-1))
        sample_preds = sample_preds.to(torch.uint8)

        if self.debug:
            print(f"\npred_out info: {sample_preds.max()}/{sample_preds.min()}, {sample_preds.shape}, {sample_preds.dtype}\n")

        grid = torchvision.utils.make_grid(sample_preds[:4],nrow=4)
        #self.logger.experiment.add_image('example_preds', grid, 0)
        self.logger.experiment.add_image('example_preds', sample_preds[sample_idx], 0)

        for i in range(pred.shape[1]):
            grid = torchvision.utils.make_grid(pred[:,i,:,:][:4].unsqueeze(1),nrow=4)
            #self.logger.experiment.add_image(f'example_preds class: {i}', grid, 0)
            self.logger.experiment.add_image(f'example_preds class: {i}', pred[sample_idx,i,:,:].unsqueeze(0), 0)

        #log_vals = {"test_acc", test_acc}
        log_vals = {"Test/acc": test_acc}
        self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        #self.log("Test/acc",test_acc,prog_bar=True,on_epoch=True)
        #self.log("Test/acc_std",test_acc,prog_bar=True,on_epoch=True,reduce_fx=torch.std)
        return log_vals #test_acc

    def predict_step(self,batch):
        pred = self.model(batch)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.train_config['LR'])
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1/100,end_factor=1.0,total_iters=self.warm_up_iter,last_epoch=-1)
        return [optimizer],[lr_scheduler]