import segmentation_models_pytorch as smp

def create_unet(backbone="efficientnet-b0", classes=2):
    """Factory function for single-channel U-Net."""
    return smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet", 
        in_channels=1, # Single channel intensity
        classes=classes
    )