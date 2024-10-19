# V10

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class InnerUNet(nn.Module):
    def __init__(self, in_channels = 128, features = 64): # (128, 32) but as per new H1 we want it to be (256, feat)  for now feat = 64
        #basically i/p here is [1,256,64,64]
        super(InnerUNet, self).__init__()
        self.down1 = Block(128, 64, down=True, act="leaky", use_dropout=False)  #d1 [1, 64, 32, 32]
        self.down2 = Block(64 , 128, down=True, act="leaky", use_dropout=False) #d2 [1, 128, 16, 16]
        self.down3 = Block(128, 256, down=True, act="leaky", use_dropout=False) #d3 [1, 256, 8,  8]
        self.down4 = Block(256, 512, down=True, act="leaky", use_dropout=False) #d4 [1, 512, 4,  4]

        self.bottleneck = nn.Sequential(                                                               #d5 [1, 512, 2,  2]
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(512, 256, down=False, act="relu", use_dropout=False)   #u1 [1, 256,  8,  8] 
        self.up2 = Block(256 + 256, 128, down=False, act="relu", use_dropout=False)   #u2 [1,  128, 16, 16]
        self.up3 = Block(128 + 128, 64, down=False, act="relu", use_dropout=False)       #u3 [1,  64, 32, 32]
        self.up4 = Block(64 + 64 , 256, down=False, act="relu", use_dropout=False)          #u4 [1,  256, 64, 64]
        # self.final_up = nn.ConvTranspose2d(features, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bottleneck = self.bottleneck(d4)
        up1 = self.up1(d4)
        up2 = self.up2(torch.cat([up1, d3], dim=1))
        up3 = self.up3(torch.cat([up2, d2], dim=1))
        up4 = self.up4(torch.cat([up3, d1], dim=1))
        
        return bottleneck, up2, up4, d4

class OuterUNet(nn.Module):
    def __init__(self, in_channels, features=64): # Assumming [bs, 3,512,512] input and features == 64
        super(OuterUNet, self).__init__()
        # self.initial_down = nn.Sequential(
        #     nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
        #     nn.LeakyReLU(0.2),
        # )                                                                                         #d0 [1,  64, 256,256]
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )                                                                                         #d1 [1, 64, 128,128]
        self.down2 = Block(64 , 128, down=True, act="leaky", use_dropout=False) #d2 [1, 256, 64, 64]
        self.down3 = Block(128, 256, down=True, act="leaky", use_dropout=False) #d3 [1, 512, 32, 32]
        self.down4 = Block(256, 512, down=True, act="leaky", use_dropout=False) #d4 [1, 512, 16, 16]
        self.down5 = Block(512, 512, down=True, act="leaky", use_dropout=False) #d5 [1, 512, 8, 8]
        self.down6 = Block(512, 512, down=True, act="leaky", use_dropout=False) #d6 [1, 512, 4, 4]
        self.down7 = nn.Sequential(                                                               #d7 [1, 512, 2, 2]  
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU() 
        )
        self.inner_unet = InnerUNet(128, 64) # (256, 64)
        
        self.up1 = Block(512 + 512 , 512, down=False, act="relu", use_dropout=True)       #u1 [1, 512, 4, 4]
        self.up2 = Block(512 + 512, 512, down=False, act="relu", use_dropout=True)     #u2 [1, 512, 8 ,8] 
        self.up3 = Block(512 + 512, 512, down=False, act="relu", use_dropout=True)    #u3 [1,  512,16 ,16]
        self.up4 = Block(512 + 128, 512, down=False, act="relu", use_dropout=False)   #u4 [1,  512, 32 ,32] 
        self.up5 = Block(512 + 256, 256, down=False, act="relu", use_dropout=False)   #u5 [1,  256, 64 ,64]
        self.up6 = Block(256 + 256, 128, down=False, act="relu", use_dropout=False)   #u6 [1,  128, 128 ,128]
        self.up7 = Block(128 + 64, 3, down=False, act="relu", use_dropout=False)       #u7 [1,   256, 256 ,256]

    def forward(self, x):
        # d0 = self.initial_down(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bottleneck_inner , up2_inner, up4_inner, d4_inner = self.inner_unet(d2)

        up1 = self.up1(torch.cat([d7, bottleneck_inner], dim=1))
        up2 = self.up2(torch.cat([d4_inner ,up1], dim=1))
        up3 = self.up3(torch.cat([up2, d5], dim=1))
        up4 = self.up4(torch.cat([up2_inner, up3], dim=1))
        up5 = self.up5(torch.cat([up4, d3], dim=1))
        up6 = self.up6(torch.cat([up4_inner, up5], dim=1))
        up7 = self.up7(torch.cat([up6, d1], dim=1))
        return up7
    
def test_model():
    x = torch.randn((1, 3, 256, 256))  
    model = OuterUNet(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

test_model()