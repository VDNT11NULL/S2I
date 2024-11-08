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

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class BahdanauAttentionNet(nn.Module):
    def __init__(self, in_channels):
        super(BahdanauAttentionNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1)  # Output scalar score for attention
        )

    def forward(self, x):
        return self.fc(x)

class InnerUNet(nn.Module):
    def __init__(self, in_channels=128, features=64):
        super(InnerUNet, self).__init__()
        self.down1 = Block(128, 64, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(64, 128, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(128, 256, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(256, 512, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(512, 256, down=False, act="relu", use_dropout=False)
        self.up2 = Block(256 + 256, 128, down=False, act="relu", use_dropout=False)
        self.up3 = Block(128 + 128, 64, down=False, act="relu", use_dropout=False)
        self.up4 = Block(64 + 64, 256, down=False, act="relu", use_dropout=False)

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
    def __init__(self, in_channels, features=64):  # Assuming [bs, 3,512,512] input and features == 64
        super(OuterUNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )  
        self.down2 = Block(64 , 128, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(128, 256, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(256, 512, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(512, 512, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(512, 512, down=True, act="leaky", use_dropout=False)
        self.down7 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU() 
        )
        self.inner_unet = InnerUNet(128, 64)
        
        self.up1 = Block(512 + 512 , 512, down=False, act="relu", use_dropout=True)
        self.up2 = Block(512 + 512, 512, down=False, act="relu", use_dropout=True)
        self.up3 = Block(512 + 512, 512, down=False, act="relu", use_dropout=True)
        self.up4 = Block(512, 512, down=False, act="relu", use_dropout=False)  # Adjusted input channels to match
        self.up5 = Block(512 + 256, 256, down=False, act="relu", use_dropout=False)
        self.up6 = Block(256 + 256, 128, down=False, act="relu", use_dropout=False)
        self.up7 = Block(128 + 64, 3, down=False, act="relu", use_dropout=False)

        # Bahdanau attention network for h1 and h2
        self.attention_net = BahdanauAttentionNet(in_channels=1024)
        self.attention_net2 = BahdanauAttentionNet(in_channels=640)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bottleneck_inner, up2_inner, up4_inner, d4_inner = self.inner_unet(d2)

        up1 = self.up1(torch.cat([d7, bottleneck_inner], dim=1))
        up2 = self.up2(torch.cat([d4_inner, up1], dim=1))
        up3 = self.up3(torch.cat([up2, d5], dim=1))

        h1 = torch.cat([up3, d4], dim=1)  #1024
        h2 = torch.cat([up3, up2_inner], dim=1)  #640

        e1 = self.attention_net(h1)
        e2 = self.attention_net(h2)

        attention_weights = torch.softmax(torch.cat([e1, e2], dim=1), dim=1)
        a1, a2 = torch.split(attention_weights, 1, dim=1)

        # Print attention weights a1 and a2
        print("Attention weight a1:", a1)
        print("Attention weight a2:", a2)

        # Compute weighted sum: a1 * d4 + a2 * up2_inner
        weighted_sum = a1 * d4 + a2 * up2_inner

        # Pass weighted sum to up4
        up4 = self.up4(weighted_sum)
        up5 = self.up5(torch.cat([up4, d3], dim=1))
        up6 = self.up6(torch.cat([up4_inner, up5], dim=1))
        up7 = self.up7(torch.cat([up6, d1], dim=1))

        return up7

def test_model():
    x = torch.randn((1, 3, 512, 512))
    model = OuterUNet(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

test_model()
