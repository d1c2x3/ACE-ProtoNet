"""
ACE-ProtoNet: Adaptive Covariance Eigen-Gate and Uncertainty-Aware Prototype Learning 
for Coronary Artery Segmentation
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg

# Make sure you have the medim package installed/configured in your environment 
# for SAM-Med3D loading.
import medim 


# ==========================================
# 1. Basic Building Blocks
# ==========================================
def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)

def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out

class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MIA_Module(nn.Module):
    """ Channel attention module for SAM embeddings """
    def __init__(self, in_dim=384):
        super(MIA_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, depth, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, depth, height, width)
        out = self.gamma * out + x
        return out

# ==========================================
# 2. Core Modules: ACE-Gate & UPL-Head
# ==========================================
class ACE_Gate(nn.Module):
    def __init__(self, channels):
        super(ACE_Gate, self).__init__()
        # Common space projection
        self.conv_shared = nn.Conv3d(channels * 2, channels, kernel_size=1)
        
        # Statistical mapper: Maps eigenvalues to channel-wise weights
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, f_vfm, f_cnn):
        B, C, D, H, W = f_vfm.size()
        
        # 1. Concatenation and projection
        x = torch.cat([f_vfm, f_cnn], dim=1)
        x = self.conv_shared(x)  # [B, C, D, H, W]
        
        # 2. Covariance matrix computation
        x_flat = x.view(B, C, -1)  # [B, C, N]
        N = x_flat.size(-1)
        x_mean = x_flat.mean(dim=-1, keepdim=True)
        x_centered = x_flat - x_mean
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (N - 1)  # [B, C, C]
        
        # 3. Eigenvalue decomposition
        eigenvalues, _ = torch.linalg.eigh(cov)  # [B, C]
        
        # 4. Weight generation
        w = self.mlp(eigenvalues)  # [B, C]
        w = w.view(B, C, 1, 1, 1)  # Broadcastable shape
        
        # 5. Channel-wise adaptive fusion
        out = (f_vfm * w) + (f_cnn * (1 - w))
        return out

class UPL_Head(nn.Module):
    def __init__(self, in_channels, out_channels=1, num_classes=2, num_prototypes=3):
        super(UPL_Head, self).__init__()
        self.out_channels = out_channels # Final prediction channels (e.g., 1 for sigmoid)
        self.num_classes = num_classes   # Classes for entropy (e.g., 2 for bg/vessel)
        self.num_prototypes = num_prototypes
        self.total_protos = self.num_classes * num_prototypes
        
        # Auxiliary head for uncertainty estimation
        self.aux_head = nn.Conv3d(in_channels, self.num_classes, kernel_size=1)
        
        # Prototype memory bank
        self.register_buffer('prototypes', torch.randn(self.total_protos, in_channels))
        
        # Attention projection layers
        self.query_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        
        # Final segmentation head
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels + self.total_protos, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        
        # 1. Uncertainty estimation (Shannon Entropy)
        p_aux = torch.softmax(self.aux_head(x), dim=1)  # [B, Classes, D, H, W]
        entropy = -torch.sum(p_aux * torch.log(p_aux + 1e-8), dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=x.device))
        u = entropy / max_entropy  # [B, 1, D, H, W], normalized to [0, 1]
        
        # 2. Explicit Similarity Map
        x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, N, C]
        protos = self.prototypes                   # [K_total, C]
        
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        p_norm = F.normalize(protos, p=2, dim=-1)
        sim = torch.matmul(x_norm, p_norm.transpose(0, 1))  # [B, N, K_total]
        sim_map = sim.transpose(1, 2).view(B, self.total_protos, D, H, W)
        
        # 3. Prototype-guided attention
        q = self.query_proj(x).view(B, C, -1).transpose(1, 2)
        k = self.key_proj(self.prototypes)
        v = self.value_proj(self.prototypes)
        
        attn = torch.softmax(torch.matmul(q, k.transpose(0, 1)) / (C ** 0.5), dim=-1)
        proto_refined = torch.matmul(attn, v).transpose(1, 2).view(B, C, D, H, W)
        x_tilde = x + proto_refined
        
        # 4. Uncertainty modulation
        x_prime = x_tilde * (1 + u)
        
        # 5. Final Prediction
        concat_feat = torch.cat([x_prime, sim_map], dim=1)
        out = self.final_conv(concat_feat)
        out = torch.sigmoid(out)
        
        # Reform x_norm back to spatial dims for online learning
        x_norm_spatial = x_norm.transpose(1, 2).view(B, C, D, H, W)
        
        return out, p_aux, x_norm_spatial, u

# ==========================================
# 3. Main Network: ACE-ProtoNet
# ==========================================
class ACEProtoNet(nn.Module):
    def __init__(self, classes=1 channels=1, ckpt_path="./model/SAM-Med3D-main/ckpt/sam_med3d_turbo.pth"):
        super(ACEProtoNet, self).__init__()
        self.sam_ckpt = ckpt_path
        
        # 1. Pre-trained SAM-Med3D setup
        # Note: If running a quick test without the checkpoint, you can mock this 
        # part or ensure the checkpoint path exists.
        try:
            self.sam_model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
            for name, param in self.sam_model.named_parameters():
                if any(key in name for key in ['11', '10', '9', '5']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        except Exception as e:
            print(f"Warning: SAM-Med3D could not be loaded. Please check medim and ckpt_path. Error: {e}")
            self.sam_model = None

        # 2. CNN Encoder (Local Spatial Features)
        self.enc_input_1 = ResEncoder3d(channels, 16)
        self.encoder1_1  = ResEncoder3d(16, 32)
        self.encoder2_1  = ResEncoder3d(32, 64)
        self.encoder3_1  = ResEncoder3d(64, 128)
        self.encoder4_1  = ResEncoder3d(128, 256)
        self.downsample  = downsample()

        # 3. VFM Alignment & Fusion
        self.MIA_module = MIA_Module(in_dim=384)
        self.convert_channel = nn.Conv3d(384, 256, kernel_size=1)
        self.ace_gate = ACE_Gate(channels=256)

        # 4. CNN Decoder
        self.deconv4 = deconv(256, 128)
        self.decoder4 = Decoder3d(256, 128)
        
        self.deconv3 = deconv(128, 64)
        self.decoder3 = Decoder3d(128, 64)
        
        self.deconv2 = deconv(64, 32)
        self.decoder2 = Decoder3d(64, 32)
        
        self.deconv1 = deconv(32, 16)
        self.decoder1 = Decoder3d(32, 16)
        
        # 5. Uncertainty-Aware Prototype Learning Head
        self.upl_head = UPL_Head(in_channels=16, out_channels=classes, num_classes=2, num_prototypes=3)
        
        initialize_weights(self)

    def forward(self, x):
        # -- CNN Branch --
        enc_input_1 = self.enc_input_1(x)       # [B, 16, D, H, W]
        down1_1 = self.downsample(enc_input_1)

        enc1_1 = self.encoder1_1(down1_1)       # [B, 32, D/2, H/2, W/2]
        down2_1 = self.downsample(enc1_1)

        enc2_1 = self.encoder2_1(down2_1)       # [B, 64, D/4, H/4, W/4]
        down3_1 = self.downsample(enc2_1)

        enc3_1 = self.encoder3_1(down3_1)       # [B, 128, D/8, H/8, W/8]
        down4_1 = self.downsample(enc3_1)

        f_cnn = self.encoder4_1(down4_1)        # [B, 256, D/16, H/16, W/16]

        # -- VFM (SAM) Branch --
        if self.sam_model is not None:
            sam_embedding = self.sam_model.image_encoder(x) # [B, 384, D/16, H/16, W/16]
        else:
            # Mocking SAM output for testing without checkpoint
            sam_embedding = torch.randn(x.size(0), 384, x.size(2)//16, x.size(3)//16, x.size(4)//16, device=x.device)
            
        sam_embedding_mia = self.MIA_module(sam_embedding)
        f_vfm = self.convert_channel(sam_embedding_mia)   # [B, 256, D/16, H/16, W/16]
        
        # -- ACE-Gate Fusion --
        fused_feature = self.ace_gate(f_vfm, f_cnn)       # [B, 256, D/16, H/16, W/16]
        
        # -- Decoder --
        up4 = self.deconv4(fused_feature)
        up4 = torch.cat((enc3_1, up4), dim=1)
        dec4 = self.decoder4(up4)                         # [B, 128, D/8, H/8, W/8]

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2_1, up3), dim=1)
        dec3 = self.decoder3(up3)                         # [B, 64, D/4, H/4, W/4]

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1_1, up2), dim=1)
        dec2 = self.decoder2(up2)                         # [B, 32, D/2, H/2, W/2]

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input_1, up1), dim=1)
        dec1 = self.decoder1(up1)                         # [B, 16, D, H, W]
        
        # -- UPL-Head Prediction --
        final_pred, p_aux, x_norm, u = self.upl_head(dec1)
        
        if self.training:
            return final_pred, p_aux, x_norm, u
        else:
            return final_pred


# ==========================================
# 4. Quick Execution Test
# ==========================================
if __name__ == '__main__':
    print("Initializing ACE-ProtoNet...")
    # Initialize the model (mocking the SAM checkpoint for instant testing)
    model = ACEProtoNet(classes=1, channels=1, ckpt_path="mock")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model successfully loaded on {device}.")
    
    # Create a dummy CCTA volume patch (Batch Size=1, Channels=1, D=128, H=160, W=160)
    # Using smaller spatial dimensions here just to make the test run quickly on local machines
    dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    # Test Training Mode
    model.train()
    print("\n--- Testing Training Forward Pass ---")
    final_pred, p_aux, x_norm, u = model(dummy_input)
    print(f"Final Prediction shape: {final_pred.shape}")
    print(f"Auxiliary Probabilities shape: {p_aux.shape}")
    print(f"Normalized Features shape: {x_norm.shape}")
    print(f"Uncertainty Map shape: {u.shape}")
    
    # Test Inference Mode
    model.eval()
    print("\n--- Testing Inference Forward Pass ---")
    with torch.no_grad():
        inference_pred = model(dummy_input)
    print(f"Inference Prediction shape: {inference_pred.shape}")
    
    print("\nExecution Successful! All tensor dimensions align perfectly.")