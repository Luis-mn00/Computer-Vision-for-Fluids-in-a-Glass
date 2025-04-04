import torch.nn as nn
import torchvision.models as models

# U-Net with Pretrained ResNet Encoder
class UNetResNet(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):  
        super(UNetResNet, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) 
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  
        self.enc3 = resnet.layer2  
        self.enc4 = resnet.layer3  

        # Additional decoder block for correct upsampling
        self.dec3 = self._decoder_block(256, 128)  
        self.dec2 = self._decoder_block(128, 64)   
        self.dec1 = self._decoder_block(64, 64)    
        self.dec0 = self._decoder_block(64, 32)  # Additional upsampling step

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)  

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

    def forward(self, x):
        enc1 = self.enc1(x)  
        enc2 = self.enc2(enc1)  
        enc3 = self.enc3(enc2)  
        enc4 = self.enc4(enc3)  

        dec3 = self.dec3(enc4) + enc3  
        dec2 = self.dec2(dec3) + enc2  
        dec1 = self.dec1(dec2) + enc1  
        dec0 = self.dec0(dec1)  # Final upsampling step
        out = self.final(dec0)  

        return out  
    
class UNetResNet_low(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):  
        super(UNetResNet_low, self).__init__()
        
        resnet = models.resnet18(pretrained=pretrained)
        
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Use fewer layers from ResNet
        self.encoder = nn.Sequential(*list(resnet.children())[:-4])  # Stop at layer2 instead of layer3
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  
        self.enc3 = resnet.layer2  # Use only up to layer2 for encoder

        # Adjust decoder blocks accordingly (fewer channels and layers)
        self.dec2 = self._decoder_block(128, 64)  
        self.dec1 = self._decoder_block(64, 64)  
        self.dec0 = self._decoder_block(64, 32)  

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)  

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

    def forward(self, x):
        enc1 = self.enc1(x)  
        enc2 = self.enc2(enc1)  
        enc3 = self.enc3(enc2)  

        dec2 = self.dec2(enc3) + enc2  # Skip connection from enc2
        dec1 = self.dec1(dec2) + enc1  # Skip connection from enc1
        dec0 = self.dec0(dec1)  # Final upsampling step
        out = self.final(dec0)  

        return out