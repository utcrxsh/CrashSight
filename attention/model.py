import torch
import torch.nn as nn
import torchvision.models as models

class CBAM_Enhanced(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.channel_attention = None
        self.spatial_attention = None
    def forward(self, x):
        self.channel_attention = self.channel_mlp(x)
        chn_attn = self.channel_attention * x
        avg = torch.mean(chn_attn, dim=1, keepdim=True)
        max_ = torch.max(chn_attn, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, max_], dim=1)
        self.spatial_attention = self.spatial_conv(cat)
        return self.spatial_attention * chn_attn

class CNN_TSM_Attn_Enhanced(nn.Module):
    def __init__(self, attention_heads=2):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.cbam = CBAM_Enhanced(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_attn = nn.MultiheadAttention(embed_dim=512, num_heads=attention_heads, batch_first=True)
        self.fc = nn.Linear(512, 1)
        self.temporal_attention_weights = None
    def forward(self, x):
        B, C, T, H, W = x.shape
        feats = []
        for t in range(T):
            ft = self.backbone(x[:, :, t, :, :])
            ft = self.cbam(ft)
            ft = self.pool(ft).squeeze(-1).squeeze(-1)
            feats.append(ft)
        feats = torch.stack(feats, dim=1)
        feats, self.temporal_attention_weights = self.temporal_attn(feats, feats, feats)
        x = torch.mean(feats, dim=1)
        return torch.sigmoid(self.fc(x)).mean() 