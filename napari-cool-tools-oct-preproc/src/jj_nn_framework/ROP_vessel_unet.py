""" """

import torch
import torchvision
import segmentation_models_pytorch as smp
import lightning as L

rop_vessel_config = {
    "ENCODER": "efficientnet-b0",
    "ENCODER_WEIGHTS": "imagenet",
    "ENCODER_DEPTH": 5,
    "DECODER_CHANNELS": [224, 112, 56, 28, 14],
    "IN_CHANNELS": 1,
    "CLASSES": ["vessel"],
    "ACTIVATION": "sigmoid",
    "WARM_UP_ITER": 2,
}

class ROPVesselSegUnet(L.LightningModule):
    def __init__(self, train_config, loss_metric, acc_metric):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=train_config[
                "ENCODER"
            ],  # smp.UnetPlusPlus(encoder_name=ENCODER,
            encoder_weights=train_config["ENCODER_WEIGHTS"],
            encoder_depth=train_config["ENCODER_DEPTH"],  # 5,
            decoder_channels=train_config[
                "DECODER_CHANNELS"
            ],  # (224,112,56,28,14),#(224,112,56,28,14),(864,432,216,108,54),(16,8,4,2,1),(32,16,8,4,2),(64,32,16,8,4),(128,64,32,16,8),(256,128,64,32,16),(512,256,128,64,32),
            in_channels=train_config["IN_CHANNELS"],  # 1, #3,
            classes=len(train_config["CLASSES"]),
            activation=train_config["ACTIVATION"],
        )
        self.train_config = train_config
        self.loss_metric = loss_metric
        self.acc_metric = acc_metric
        #self.warm_up_iter = train_config["WANM_UP_ITER"]  # 2
        self.warm_up_iter = train_config["WARM_UP_ITER"]  # 2

    def forward(self, x):
        # lightning module functional use
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y = batch
        pred = self.model(x)

        loss = self.loss_metric(pred, y)
        acc = self.acc_metric(pred, y)  # nn.functional.one_hot(y.long(),num_classes=2)

        log_vals = {"Train/loss": loss, "Train/acc": acc}
        self.log_dict(log_vals, prog_bar=True, on_epoch=True, on_step=False)
        # self.log("Train/loss",loss,prog_bar=True,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop
        x, y = batch
        pred = self.model(x)

        val_loss = self.loss_metric(pred, y)
        val_acc = self.acc_metric(
            pred, y
        )  # nn.functional.one_hot(y.long(),num_classes=2)

        log_vals = {"Val/loss": val_loss, "Val/acc": val_acc}
        self.log_dict(log_vals, prog_bar=True, on_epoch=True, on_step=False)
        # self.log("Validation/loss",val_loss,prog_bar=True,on_epoch=True)
        return log_vals

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop
        x, y = batch
        pred = self.model(x)

        test_acc = self.acc_metric(
            pred, y
        )  # nn.functional.one_hot(y.long(),num_classes=2)

        sample_imgs = x
        grid = torchvision.utils.make_grid(sample_imgs, nrow=4)
        self.logger.experiment.add_image("example_images", grid, 0)

        sample_gts = y
        grid = torchvision.utils.make_grid(sample_gts, nrow=4)
        self.logger.experiment.add_image("example_gt", grid, 0)

        sample_preds = pred
        grid = torchvision.utils.make_grid(sample_preds, nrow=4)
        self.logger.experiment.add_image("example_preds", grid, 0)

        # log_vals = {"test_acc", test_acc}
        log_vals = {"Test/acc": test_acc}
        self.log_dict(log_vals, prog_bar=True, on_epoch=True, on_step=False)
        # self.log("Test/acc",test_acc,prog_bar=True,on_epoch=True)
        # self.log("Test/acc_std",test_acc,prog_bar=True,on_epoch=True,reduce_fx=torch.std)
        return log_vals  # test_acc

    def predict_step(self, batch):
        pred = self.model(batch)
        return pred

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(),lr=self.train_config['LR'])
        # return optimizer

        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config["LR"])
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1 / 100,
            end_factor=1.0,
            total_iters=self.warm_up_iter,
            last_epoch=-1,
        )
        return [optimizer], [lr_scheduler]
