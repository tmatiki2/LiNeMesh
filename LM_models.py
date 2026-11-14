import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def agg_feas(x, mask=None, use_variance=False):
    """
    Aggregates features across dimension 0.

    Args:
        x : Tensor
            Shape: (S, B, D)

        mask : Bool tensor or None
            Shape: (B, S)
            True  -> invalid
            False -> valid

        use_variance : bool
            False -> mean aggregation
            True  -> variance aggregation

    Returns:
        Tensor of shape (B, D)
    """

    eps = 1e-6
    if mask is not None:
        valid = (~mask).float().T.unsqueeze(2)
        count = valid.sum(0).clamp(min=eps)
        mean = (x * valid).sum(0) / count
        if use_variance:
            diff2 = ((x - mean.unsqueeze(0)) ** 2) * valid
            var = diff2.sum(0) / count
            return var
        return mean
    else:
        mean = x.mean(0)
        if use_variance:
            var = ((x - mean.unsqueeze(0)) ** 2).mean(0)
            return var
        return mean

class context(nn.Module):
    def __init__(self, embed_dim=512, out_dim=24):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out_dim)
    def forward(self, x_, mask=None):
        x = agg_feas(x_, mask, use_variance=False)
        x = self.fc(x)
        return x

def normalize_uv(uv, H=480, W=640, sz=300, odim=True):
    if odim:
        u = W*uv[..., 0]
        v = H*uv[..., 1]
        u = (sz/W)*u
        v = (sz/H)*v
    else:
        u = uv[..., 0]
        v = uv[..., 1]

    u_norm = 2.0 * u / (sz - 1) - 1.0
    v_norm = 2.0 * v / (sz - 1) - 1.0
    return torch.stack([u_norm, v_norm], dim=-1)

class AResNet50Encoder(nn.Module):
    def __init__(self, out_h, out_w, pretrained=True, big=False):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        if big:
            backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            self.proj = nn.Conv2d(2048, 3, kernel_size=1)
        else:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.proj = nn.Conv2d(512, 3, kernel_size=1)
        
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        feat = F.interpolate(
            feat,
            size=(self.out_h, self.out_w),
            mode="bilinear",
            align_corners=False,
        )
        return feat
class SequenceSelfAttention(nn.Module):
    def __init__(
        self,
        channels=3, # 3, 18, 3
        embed_dim=64,
        num_heads=1,
        dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Linear(channels, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.proj = nn.Linear(embed_dim, channels)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        S, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.reshape(H * W, S, C)
        x = self.embed(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn = self.ffn(x)
        x = self.norm2(x + ffn)
        x = self.proj(x)
        x = x.reshape(H, W, S, C)
        x = x.permute(2, 3, 0, 1)
        return x
class ResNet50FPN2(nn.Module):

    def __init__(
        self,
        out_channels=256,
        pretrained=True,
        resv=False,
        in_channels=6
    ):
        super().__init__()
        if resv:
            if pretrained:
                backbone = resnet34(
                    weights=ResNet34_Weights.IMAGENET1K_V1
                )
            else:
                backbone = resnet34(weights=None)

            backbone_channels = [64, 128, 256, 512]

        else:
            if pretrained:
                backbone = resnet101(
                    weights=ResNet101_Weights.IMAGENET1K_V2
                )
            else:
                backbone = resnet101(weights=None)

            backbone_channels = [256, 512, 1024, 2048]
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            backbone.conv1 = new_conv

        self.backbone = create_feature_extractor(
            backbone,
            return_nodes={
                "layer1": "c2",
                "layer2": "c3",
                "layer3": "c4",
                "layer4": "c5",
            },
        )

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=backbone_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        return fpn_features

class CorrelationSamplerbatch(nn.Module):
    def __init__(self, in_dim=4900, out_dim=512, w=300, h=300):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, 2048, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2048, out_dim, kernel_size=1)
        )

        self.w = w
        self.h = h

    def forward(self, corr, uv):
        """
        Args:
            corr: M x C_in x H x W
                  e.g. M x 4900 x 70 x 70

            uv:   M x T x 2
                  UV coordinates in the original 300x300
                  image coordinate system.

        Returns:
            feat: M x T x out_dim
        """
        corr = self.encoder(corr)

        M, C, H, W = corr.shape
        T = uv.shape[1]

        uv = uv.float()

        grid = uv.clone()

        grid[..., 0] = (
            2.0 * grid[..., 0] / (self.w - 1)
        ) - 1.0

        grid[..., 1] = (
            2.0 * grid[..., 1] / (self.h - 1)
        ) - 1.0
        grid = grid.unsqueeze(2)
        feat = F.grid_sample(
            corr,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        feat = feat.squeeze(-1).transpose(1, 2)

        return feat

class ExtendedResidualBlock2(nn.Module):
    def __init__(self, in_features, out_features, n_layers=5):
        super(ExtendedResidualBlock2, self).__init__()
        self.n_layers = n_layers
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features))
        self.fcn_dict = nn.ModuleDict()
        self.fcn_dict2 = nn.ModuleDict()
        for i in range(n_layers):
            self.fcn_dict[str(i)] = nn.Sequential(nn.Linear(out_features, out_features))
         
    def forward(self, x):
        out = F.tanh(self.fc1(x)) #torch.sin(self.fc1(x)) #
        for i in range(self.n_layers):
            out = self.fcn_dict[str(i)](out)
            if (i+1)%3==0:
                out = out + self.shortcut(x) #F.tanh(out)
        out = out + self.shortcut(x)
        return out

class EfficientNetwork2(nn.Module):
    def __init__(self, inp, out, n_layers=5):
        super(EfficientNetwork2, self).__init__()
        self.input_layer = nn.Linear(inp, 1024)
        self.residual_block1 = ExtendedResidualBlock2(1024, 512, n_layers)
        self.residual_block2 = ExtendedResidualBlock2(512, 256, n_layers)
        self.residual_block3 = ExtendedResidualBlock2(256, 128, n_layers)
        self.residual_block4 = ExtendedResidualBlock2(128, 64, n_layers)
        self.residual_block5 = ExtendedResidualBlock2(64, 32, n_layers)
        self.fc3 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, out)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.fc3(out)
        out = self.output_layer(out)
        return out
class base_network(nn.Module):
    def __init__(self,resv=False):
        super().__init__()

        self.out_ch = 256
        self.out_h = 70
        self.out_w = 70
        self.resv = resv
        self.model3 = EfficientNetwork2(1536, 10, 2)

        self.model_fpn = ResNet50FPN2(
            out_channels=self.out_ch,
            resv=resv,
            in_channels=3
        )

        self.cor_model = CorrelationSamplerbatch(
            in_dim=self.out_h * self.out_h,
            out_dim=512
        )

        self.model2 = context(
            embed_dim=1536,
            out_dim=1536)

    def forward(
        self,
        fimg_enc, #imgs_f,#
        imgts,
        uvts,
        viz_mat=None,
        xyzs=None,
        rshp=False,
        UV_feas=None
    ):

        M = imgts.shape[0]
        imgts = imgts.float().to(device, non_blocking=True)
        imgts_70 = F.interpolate(
            imgts,
            size=(self.out_h, self.out_w),
            mode="bilinear",
            align_corners=False
        )

        imgts_flat = imgts_70.flatten(2)
        corr = torch.matmul(
            imgts_flat.transpose(1, 2),   # M x 4900 x 3
            fimg_enc.expand(M, -1, -1)    # M x 3 x 4900
        )
        corr = corr.reshape(M,self.out_h,self.out_w,self.out_h,self.out_w)
        corr = corr.permute(0, 3, 4, 1, 2)
        corr = corr.reshape(M,self.out_h * self.out_w,self.out_h,self.out_w)

        uv = torch.as_tensor(np.stack(uvts),dtype=torch.float32,device=device)
        T = uv.shape[1]
        corr_feas = self.cor_model(corr,uv)

        uv_i = normalize_uv(uv,sz=300,odim=False)
        grid = uv_i.unsqueeze(2)
        if UV_feas is None:
            pyramid = self.model_fpn(imgts)
            multi_scale_feats = []
            for lvl in pyramid.values():
                sampled = F.grid_sample(lvl,grid,mode="bilinear",align_corners=True)
                sampled = sampled.squeeze(-1)
                sampled = sampled.transpose(1, 2)
                multi_scale_feats.append(sampled)
            UV_feas = torch.cat(multi_scale_feats,dim=-1)

        comb_feas = torch.cat([UV_feas, corr_feas],dim=-1)
        if rshp:
            Nb = T // 9
            comb_feas = comb_feas[:, :Nb * 9]
            comb_feas = comb_feas.reshape(M,Nb,9,-1).mean(dim=2)

        if viz_mat is not None:
            mask = (viz_mat == 0).to(device)
        else:
            mask = None

        fess2 = self.model2(comb_feas,mask)
        pred_cls = self.model3(fess2)

        return pred_cls

class group_occup_sdf(nn.Module):
    def __init__(self):
        super(group_occup_sdf, self).__init__()
        self.out_h, self.out_w = 70, 70
        self.enc_model = AResNet50Encoder(out_h=self.out_h, out_w=self.out_w, big=False)
        self.attnm = SequenceSelfAttention() 
        self.basenet = base_network()
    def forward(self, all_inps_, viz_mat_=None):
        preds_cls = []
        for kk1 in range(len(all_inps_)):
            imgts_, uvts = all_inps_[kk1]
            imgts, imgs_f = imgts_
            fimg_enc = self.attnm(self.enc_model(imgs_f.float().to(device))).mean(0).unsqueeze(0)
            fimg_enc = fimg_enc.reshape(1, 3, self.out_h * self.out_w)
            pred_cls_i = self.basenet(fimg_enc,imgts,uvts,viz_mat_[kk1],rshp=True)
            preds_cls.append(pred_cls_i) 
        preds_cls_ = torch.vstack(preds_cls)
        return preds_cls_
