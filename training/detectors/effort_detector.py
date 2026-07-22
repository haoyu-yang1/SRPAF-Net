import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)

class SafeLargeSelectiveKernel(nn.Module):
    """
    Baseline-safe LSK
    只学习一个空间残差 delta，不再直接 identity * attention 强乘。
    初始时接近恒等映射，避免拉低 baseline。
    """
    def __init__(self, dim):
        super(SafeLargeSelectiveKernel, self).__init__()

        hidden_dim = dim // 2

        self.conv0 = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=False
        )

        self.conv_spatial = nn.Conv2d(
            dim, dim, kernel_size=7, stride=1,
            padding=9, groups=dim, dilation=3, bias=False
        )

        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)

        self.conv_squeeze = nn.Conv2d(
            2, 2, kernel_size=7, padding=3, bias=False
        )

        self.delta_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        # 初始很小，保证一开始几乎等于 baseline
        self.lsk_scale = nn.Parameter(torch.ones(1) * 1.5e-2)

    def forward(self, x):
        identity = x

        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)

        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)

        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        feat = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]

        delta = self.delta_proj(feat)

        # 去掉残差的全局偏移，避免整体改变 CLIP 特征分布
        delta = delta - delta.mean(dim=(2, 3), keepdim=True)

        scale = torch.clamp(self.lsk_scale, min=0.0, max=0.12)

        out = identity + scale * delta

        return out

class PhaseAmpFreqAttentionBlock(nn.Module):
    """
    Baseline-safe PAFA
    由 PAFA 生成频域残差 freq_delta，再用 gate 控制是否加入。
    不再输出 [0,1] mask 去直接放大原始特征。
    """
    def __init__(self, dim=1024, h=16, w=16):
        super(PhaseAmpFreqAttentionBlock, self).__init__()

        self.amp_filter = nn.Parameter(
            torch.randn(1, dim, h, w // 2 + 1, dtype=torch.float32) * 1e-3
        )

        self.pha_filter = nn.Parameter(
            torch.randn(1, dim, h, w // 2 + 1, dtype=torch.float32) * 1e-3
)
        self.freq_perturb_scale = nn.Parameter(torch.ones(1) * 0.08)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 16, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 16, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 初始很小，PAFA 一开始接近恒等映射
        self.pafa_scale = nn.Parameter(torch.ones(1) * 8e-3)

    def forward(self, x):
        B, C, H, W = x.shape
        dtype = x.dtype
        x_float = x.float()

        x_fft = torch.fft.rfft2(x_float, norm='ortho')

        amp = torch.abs(x_fft + 1e-8)
        pha = torch.angle(x_fft + 1e-8)

        amp_filter = self.amp_filter
        pha_filter = self.pha_filter

        if amp_filter.shape[-2:] != amp.shape[-2:]:
            amp_filter = F.interpolate(
                amp_filter,
                size=amp.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            pha_filter = F.interpolate(
                pha_filter,
                size=pha.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        perturb_scale = torch.clamp(
            self.freq_perturb_scale, min=0.02, max=0.15
        )

        amp_filtered = amp * (1.0 + perturb_scale * torch.tanh(amp_filter))
        pha_filtered = pha + perturb_scale * torch.tanh(pha_filter)

        real_part = amp_filtered * torch.cos(pha_filtered)
        imag_part = amp_filtered * torch.sin(pha_filtered)
        x_fft_filtered = torch.complex(real_part, imag_part)

        freq_feat = torch.fft.irfft2(
            x_fft_filtered,
            s=(H, W),
            norm='ortho'
        )

        # PAFA 的核心：频域扰动残差
        freq_delta = freq_feat - x_float

        # 去掉全局偏移，避免整体漂移
        freq_delta = freq_delta - freq_delta.mean(dim=(2, 3), keepdim=True)

        # 用原始特征 + 频域残差幅值生成 gate
        gate_input = torch.cat([x_float, torch.abs(freq_delta)], dim=1)

        spatial_gate = self.spatial_gate(gate_input)
        channel_gate = self.channel_gate(gate_input)

        # 空间为主，通道为辅，避免 PAFA 过拟合某些通道
        gate = spatial_gate * (0.80 + 0.20 * channel_gate)

        pafa_scale = torch.clamp(self.pafa_scale, min=0.0, max=0.10)

        out = x_float + pafa_scale * gate * freq_delta

        return out.to(dtype=dtype), gate.to(dtype=dtype)

@DETECTOR.register_module(module_name='effort_freq')
class EffortFreqDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortFreqDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)

        self.lsk_block = SafeLargeSelectiveKernel(dim=1024)
        self.freq_block = PhaseAmpFreqAttentionBlock(dim=1024, h=16, w=16)

        self.use_lsk = True  # 是否使用 LSK 模块，默认为 False，可以通过 config 设置为 True
        self.use_pa_fgsa = True

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_norm = nn.LayerNorm(1024)

        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.enhance_logit_scale = nn.Parameter(torch.ones(1) * 0.35)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # Download model
        # https://huggingface.co/openai/clip-vit-large-patch14
        
        # mean: [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]
        
        # ViT-L/14 224*224
        clip_model = CLIPModel.from_pretrained("/home/haoyu/DeepfakeBench/preprocessing/clip-vit-large-patch14")

        # Apply SVD to self_attn layers only
        # ViT-L/14 224*224: 1024-1
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024-1)

        for name, param in clip_model.vision_model.named_parameters():
            print('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in clip_model.vision_model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in clip_model.vision_model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        out = self.backbone(data_dict['image'])
        # CLIP ViT-L/14 输入 224×224 时：
    # last_hidden_state: [B, 257, 1024]
    # 257 = 1 个 CLS token + 256 个 patch token
        tokens = out['last_hidden_state']
        patch_tokens = tokens[:, 1:, :]  # [B, 256, 1024]
        B, N, C = patch_tokens.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, f"Patch token number {N} cannot reshape to square map"
        feat_map = patch_tokens.transpose(1, 2).contiguous().reshape(B, C, H, W)
        if not hasattr(self, "debug_feature_printed"):
            print("tokens:", tokens.shape)
            print("patch_tokens:", patch_tokens.shape)
            print("feat_map:", feat_map.shape)
            self.debug_feature_printed = True
        return feat_map



    def classifier(self, feat_map: torch.Tensor):
        feat_vec = self.pool(feat_map).flatten(1)
        feat_vec = self.feat_norm(feat_vec)
        pred = self.head(feat_vec)
        return pred, feat_vec


    # def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #     label = data_dict['label']
    #     pred = pred_dict['cls']
    #     loss = self.loss_func(pred, label)
        
    #     # Regularization term
    #     lambda_reg = 0.1
    #     orthogonal_losses = []
    #     for module in self.backbone.modules():
    #         if isinstance(module, SVDResidualLinear):
    #             # Apply orthogonal constraints to the U_residual and V_residual matrix
    #             orthogonal_losses.append(module.compute_orthogonal_loss())
        
    #     if orthogonal_losses:
    #         reg_term = sum(orthogonal_losses)
    #         loss += lambda_reg * reg_term
        
    #     loss_dict = {'overall': loss}
    #     return loss_dict

    def compute_weight_loss(self):
        weight_sum_dict = {}
        num_weight_dict = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                if str(weight_curr.size()) not in weight_sum_dict.keys():
                    weight_sum_dict[str(weight_curr.size())] = weight_curr
                    num_weight_dict[str(weight_curr.size())] = 1
                else:
                    weight_sum_dict[str(weight_curr.size())] += weight_curr
                    num_weight_dict[str(weight_curr.size())] += 1
        
        loss2 = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss2 += -torch.mean(S_sum)
        loss2 /= len(weight_sum_dict.keys())
        return loss2

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']  # Tensor of shape [batch_size]
        pred = pred_dict['cls']     # Tensor of shape [batch_size, num_classes]

        # Compute overall loss using all samples
        loss_main = self.loss_func(pred, label)
        loss_base = self.loss_func(pred_dict['cls_base'], label)
        loss_enhanced = self.loss_func(pred_dict['cls_enhanced'], label)

        loss = loss_main + 0.10 * loss_base + 0.45 * loss_enhanced

        # Create masks for real and fake classes
        mask_real = label == 0  # Boolean tensor
        mask_fake = label == 1  # Boolean tensor

        # Compute loss for real class
        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            # No real samples in batch
            loss_real = torch.tensor(0.0, device=pred.device)

        # Compute loss for fake class
        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            # No fake samples in batch
            loss_fake = torch.tensor(0.0, device=pred.device)
        

        # loss2 = self.compute_weight_loss()
        # overall_loss = loss + loss2

        # Return a dictionary with all losses
        loss_dict = {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
            # 'erank_loss': loss2
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict



    def forward(self, data_dict: dict, inference=False) -> dict:
        feat_spatial = self.features(data_dict)  # [B, 1024, 16, 16]

        base_feat = feat_spatial

        # SRS：空间响应选择与增强
        if self.use_lsk:
            feat_lsk = self.lsk_block(base_feat)
        else:
            feat_lsk = base_feat

        # PAFA：对SRS增强后的特征进行频域建模
        if self.use_pa_fgsa:
            feat_pafa, attention_mask = self.freq_block(feat_lsk)
        else:
            attention_mask = None
            feat_pafa = feat_lsk

        feat_after_lsk = feat_lsk

        lsk_res = feat_lsk - base_feat
        pafa_res = feat_pafa - feat_lsk

        feat_fused = base_feat + 1.3 * lsk_res + 0.45 * pafa_res

        if not hasattr(self, "debug_pa_printed"):
            print("feat_spatial:", feat_spatial.shape)
            print("feat_after_lsk:", feat_after_lsk.shape)

            if attention_mask is not None:
                print("attention_mask:", attention_mask.shape)
            else:
                print("attention_mask: None")

            print("feat_fused:", feat_fused.shape)
            self.debug_pa_printed = True

        pred_base, feat_vec_base = self.classifier(base_feat)
        pred_enhanced, feat_vec = self.classifier(feat_fused)

        base_prob = torch.softmax(pred_base.detach(), dim=1)
        base_conf = torch.max(base_prob, dim=1, keepdim=True)[0]

        hard_weight = 1.0 - base_conf
        hard_weight = torch.clamp(hard_weight * 2.5, min=0.25, max=1.00)

        eta = torch.clamp(self.enhance_logit_scale, min=0.10, max=0.65)

        pred = pred_base + eta * hard_weight * (pred_enhanced - pred_base)

        prob = torch.softmax(pred, dim=1)[:, 1]

        pred_dict = {
            'cls': pred,
            'cls_base': pred_base,
            'cls_enhanced': pred_enhanced,
            'prob': prob,
            'feat': feat_vec
        }

        return pred_dict

      

# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # print("residual_weight的形状：{}".format(residual_weight.shape)) [1024,1024]
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
            # print("weight的形状：{}".format(weight.shape)) [1024,1024]
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main
        

        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self):
        # According to the properties of orthogonal matrices: A^TA = I
        UUT_residual = self.U_residual @ self.U_residual.t()
        VVT_residual = self.V_residual @ self.V_residual.t()
        
        # Construct an identity matrix
        UUT_residual_identity = torch.eye(UUT_residual.size(0), device=UUT_residual.device)
        VVT_residual_identity = torch.eye(VVT_residual.size(0), device=VVT_residual.device)
        
        # Frobenius norm
        loss = 0.5 * torch.norm(UUT_residual - UUT_residual_identity, p='fro') + 0.5 * torch.norm(VVT_residual - VVT_residual_identity, p='fro')
        
        return loss
        

# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            # S_residual is trainable
            new_module.S_residual = nn.Parameter(S_residual.clone())
            # U_residual and V_residual are also trainable
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
        else:
            # If no residual components, set placeholders
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

        return new_module
    else:
        return module
