import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import os
from functools import partial
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.modules.block import Block
from utils import add_zero_channels
from metrics import nmse2d, pcc2d_torch

project_path = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from scipy.interpolate import Rbf

def spherical_spline_upsample(lr_eeg, lr_pos, hr_pos):
    """
    lr_eeg: (B, input_ch, T)
    lr_pos: (B, input_ch, 3)
    hr_pos: (B, output_ch, 3)
    """
    B, C_in, T = lr_eeg.shape
    hr_eeg = torch.zeros(B, hr_pos.shape[1], T, device=lr_eeg.device)
    
    for b in range(B):
        for t in range(T):
            # RBF interpolation per timestep
            rbf = Rbf(lr_pos[b, :, 0].cpu(), lr_pos[b, :, 1].cpu(), lr_pos[b, :, 2].cpu(), 
                     lr_eeg[b, :, t].cpu().numpy(), function='thin_plate')
            hr_eeg[b, :, t] = torch.tensor(rbf(hr_pos[b, :, 0].cpu(), hr_pos[b, :, 1].cpu(), hr_pos[b, :, 2].cpu()), device=lr_eeg.device)
    
    return hr_eeg

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim, dataset_name="mmi"):
        super(LabelEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim) if dataset_name=="mmi" else nn.Identity()
        self.label_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, labels):
        x = self.embedding(labels)
        x = self.layer_norm(x)
        x = self.label_proj(x)
        return x

class SubPixel1D(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor, ks_size = 3, sr_type="temporal"):

        super(SubPixel1D, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv1d(in_channels, out_channels * upscale_factor, kernel_size=ks_size, padding=ks_size//2)
        if sr_type == "temporal":
            ks_size = 3
            self.act = nn.Tanh()
            self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=ks_size, padding=ks_size//2)
        self.sr_type = sr_type

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)  # [B, out_channels*upscale, T]
        B, C, T = x.size()
        r = self.upscale_factor
        if C % r != 0:
            raise ValueError(f"Channel dimension {C} not divisible by upscale factor {r}")
        if self.sr_type == "temporal":
            # Temporal SR: rearrange time
            x = x.view(B, C // r, r, T)       # [B, out_channels, r, T]
            x = x.permute(0, 1, 3, 2)         # [B, out_channels, T, r]
            x = x.contiguous().view(B, C // r, T * r)  # [B, out_channels, T*r]
            x = self.act(x)
            x = self.final_conv(x)  # Final conv to refine
        return x

# -------------------------------
# Encoder Block
# -------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels, strides, drops):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch[0], out_ch[0], kernels[0], strides[0], padding=kernels[0]//2)
        self.conv2 = nn.Conv1d(in_ch[1], out_ch[1], kernels[1], strides[1], padding=kernels[1]//2)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(drops[0])
        self.dropout2 = nn.Dropout(drops[1])

    def forward(self, x):
        x = self.dropout1(self.act(self.conv1(x)))
        x = self.dropout2(self.act(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernels, strides, drops):
        super(DecoderBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_ch[0], out_ch[0], kernels[0], strides[0], padding=kernels[0]//2, output_padding=strides[0]-1)
        self.deconv2 = nn.ConvTranspose1d(in_ch[1], out_ch[1], kernels[1], strides[1], padding=kernels[1]//2, output_padding=strides[1]-1)
        self.act = nn.SiLU()
        if drops[1] is None:
            drops[1] = 0.0
        self.dropout1 = nn.Dropout(drops[0])
        self.dropout2 = nn.Dropout(drops[1])

    def forward(self, x):
        x = self.dropout1(self.act(self.deconv1(x)))
        x = self.dropout2(self.act(self.deconv2(x)))
        return x

class SpatialEmbedding(nn.Module):
    def __init__(self, d_model, hidden_dim=32):
        super().__init__()
        self.d_model = d_model  
       
        self.pos_proj = nn.Sequential(  # Pos -> channel embeds
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)  # Fixed 62->64 adapt
        )
        self.channel_adapt = nn.Linear(d_model, 64)  # Adapt src ch to mamba_dim seq

    def forward(self, positions=None):
        pos = positions # Assume positions is (B,C,3)
        pos_emb = self.pos_proj(pos.reshape(-1,3)).reshape(pos.shape[0], pos.shape[1], -1)  # (B,C,64)
        if pos_emb.shape[1] != 64:
            seq_emb = self.channel_adapt(pos_emb.mean(-1))  # (B,C) -> (B,64) via Linear
        else:
            seq_emb = pos_emb.mean(-1)  # (B,64)
        return seq_emb.unsqueeze(-1)  # (B,64,1) — expand to L later

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):  # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class ConvSpatialEmbedding(nn.Module):
    def __init__(self, d_model=64, hidden_dim=32, kernel_size=3):  # d_model=64 for Mamba
        super().__init__()
        # CoordConv: Append x,y,z as extra channels
        self.conv = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size, padding="same"),  # in_ch=3
            nn.SiLU(),
            nn.Conv1d(hidden_dim, d_model, 1),  # To d_model
            #nn.Dropout(0.2)
        )
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, positions=None):
        pos = positions  # Assume positions is (B,C,3)
        pos_flat = pos.transpose(1,2)  # (B,3,C)
        conv_out = self.conv(pos_flat)  # Assume (B,64,C)
        #conv_out = self.norm(conv_out.transpose(1,2)).transpose(1,2)  # (B,C,64)
        pooled = conv_out.mean(dim=-1)  # Pool electrodes → (B,64)
        return pooled.unsqueeze(-1)  # (B,64,1) ✓ — now expands to L=1600

class TemporalEmbedding(nn.Module):

    def __init__(self, time_dim, d_model):
        
        super().__init__()
        self.time_dim = time_dim
        self.d_model = d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.time_proj = nn.Conv1d(self.time_dim, self.d_model, kernel_size=1)

    def _timestep_embedding(self, t, dim):

        # t can be scalar () or 1-D (B,)
        if t.dim() == 0:
            t = t.unsqueeze(0)          # -> (1,)
        # now t is (B,)

        device = t.device
        half_dim = dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)  # (half_dim,)

        t = t.float()                   # (B,)
        emb = t[:, None] * freqs[None, :]   # (B, half_dim)

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, dim)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):  # t: (B, d_model, T)
        
        temb = self._timestep_embedding(t, self.time_dim)  # [B, time_dim]
        temb = self.time_mlp(temb).unsqueeze(-1) # (B,64,1)
        temb = self.time_proj(temb)  # (B, d_model, 1)
        return temb

class SignalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim, sr_type="temporal", kernel_size=5):
        super().__init__()
        groups = 2 if out_dim == 62 else 4
        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, out_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.GroupNorm(groups, out_dim) if sr_type == "temporal" else nn.BatchNorm1d(out_dim),
                #nn.BatchNorm1d(out_dim) #no batchnorm, problems with temporal sr 
                #nn.LeakyReLU(), #bad for conditioning
                #nn.Dropout(0.3),
        )
    def forward(self, x):
        return self.encoder(x)

# -------------------------------
# DiBiMa_SubPixelRes Model
# -------------------------------

class DiBiMa_nn(nn.Module):
    def __init__(self, target_channels, ref_position=None, num_channels=64, fs_lr=80, fs_hr=160, seconds=10,
                 residual_global=True, use_mamba=True, use_diffusion=False, use_positional_encoding=False, use_electrode_embedding=False,
                 residual_internal=True, use_subpixel=True, sr_type="temporal", n_mamba_blocks = 1, n_mamba_layers=1, mamba_dim=64, mamba_d_state=16,
                 mamba_version=1, merge_type='concat', use_lr_conditioning=False, use_label=False, num_classes=None, dataset_name="mmi", multiplier = 2):
        super(DiBiMa_nn, self).__init__()

        self.target_channels = target_channels
        self.num_channels = num_channels
        self.target_length = fs_hr * seconds
        self.sr_type = sr_type

        self.fs_lr = fs_lr
        self.fs_hr = fs_hr
        self.seconds = seconds
        self.hr_len = fs_hr * seconds
        self.lr_len = fs_lr * seconds

        self.use_mamba = use_mamba
        self.n_mamba_layers = n_mamba_layers    
        self.mamba_dim = mamba_dim
        self.mamba_d_state = mamba_d_state
        self.mamba_version = mamba_version
        self.n_mamba_blocks = n_mamba_blocks
        self.ref_position = ref_position
        self.merge_type = merge_type

        if self.use_mamba:
            enc_in = int(self.mamba_dim/2)
            enc_out = self.mamba_dim
        else:   
            enc_in = self.num_channels
            enc_out = self.target_channels

        self.use_diffusion = use_diffusion
        self.use_positional_encoding = use_positional_encoding
        self.use_electrode_embedding = use_electrode_embedding
        self.use_lr_conditioning = use_lr_conditioning
        self.use_label = use_label
        self.num_classes = num_classes 
        self.dataset_name = dataset_name
        self.multiplier = multiplier

        if self.use_diffusion:
            self.time_mlp = TemporalEmbedding(self.num_channels, enc_out)
            if self.use_positional_encoding:        
                self.pos_enc = PositionalEncoding(d_model=enc_out)
            if self.use_electrode_embedding:
                #self.spatial_emb = SpatialEmbedding(enc_out)
                self.spatial_emb = ConvSpatialEmbedding(hidden_dim=self.num_channels, d_model=enc_out)
            if self.use_lr_conditioning:
                if self.sr_type == "spatial":
                    lr_condition_channels = self.num_channels
                else:
                    lr_condition_channels = self.target_channels
                self.lr_condition_encoder = SignalEncoder(lr_condition_channels, enc_out, sr_type=self.sr_type, kernel_size=3)
            if self.use_label:
                print(f"Using label conditioning with {self.num_classes} classes.")
                self.label_emb = LabelEmbedding(self.num_classes, enc_out, dataset_name=self.dataset_name)
            self.encoder = SignalEncoder(self.target_channels, enc_out, sr_type=self.sr_type, kernel_size=3)
        else:
            self.encoder = SignalEncoder(self.num_channels, enc_out, sr_type=self.sr_type, kernel_size=3)
            
        self.residual_global = residual_global
        self.residual_internal = residual_internal
        self.use_subpixel = use_subpixel

        # --- Bottleneck (residual internal) ---
        if self.use_mamba:
            print("Using Bi-Mamba bottleneck with residual internal")
            self.bottleneck = BidirectionalMamba(d_model=self.mamba_dim,
                                                     d_state=self.mamba_d_state,
                                                     n_mamba_blocks=self.n_mamba_blocks,
                                                     n_layers=self.n_mamba_layers,
                                                     mamba_version=self.mamba_version,
                                                     merge_type=self.merge_type)
        else:
            print("Using simple bottleneck with residual internal")
            self.bottleneck = nn.Sequential(
                    nn.Conv1d(enc_out, enc_out, kernel_size=1, padding=0),
                    nn.BatchNorm1d(enc_out),
                    nn.SiLU(),
                    nn.Dropout(0.3)
            )

        # --- Decoder ---
        if self.use_subpixel:
            ks = 3
            pad = 1
            if self.use_mamba:
                if self.merge_type == "concat":
                    conv_d = nn.Conv1d(enc_out*2*self.n_mamba_layers, enc_out, kernel_size=ks, padding=pad)
                else:
                    conv_d = nn.Conv1d(enc_out, enc_out, kernel_size=ks, padding=pad)
                group = 2 if enc_out == 62 else 4
                batch_n = nn.BatchNorm1d(enc_out) if self.sr_type == "spatial" else nn.GroupNorm(group, enc_out)
                sub_pixel_in = enc_out
            else:
                group = 2 if enc_out == 62 else 4
                conv_d = nn.Conv1d(enc_out, enc_out, kernel_size=ks, padding=pad)
                batch_n = nn.BatchNorm1d(enc_out) if self.sr_type == "spatial" else nn.GroupNorm(group, enc_out)
                sub_pixel_in = enc_out

            if self.sr_type == "spatial":
                print(f"Using spatial SR: {num_channels} channels to {self.target_channels}")
                upscale = 1 #int(self.target_channels // num_channels)  # spatial: channels change, length fixed
            else:
                print(f"Using temporal SR: {self.lr_len} to {self.hr_len} length")
                if self.use_diffusion:
                    upscale = 1 # Diffusion model outputs full HR length directly
                else:
                    upscale = int(self.hr_len // self.lr_len)

            self.decoder_sr = nn.Sequential(
                conv_d,
                batch_n
            )

            self.subpixel = SubPixel1D(
                in_channels=sub_pixel_in,
                out_channels=self.target_channels,
                upscale_factor=upscale,
                ks_size=3 if dataset_name=="mmi" else 1,
                sr_type=self.sr_type
            )

        else:
            if not self.use_mamba:
                self.decoder_sr = DecoderBlock(
                    [enc_out, enc_in], [enc_in, self.target_channels],
                    [30, 30], [5, 2], [None, None]
                )
            else:
                self.decoder_sr = DecoderBlock(
                    [enc_out*2*self.n_mamba_layers, enc_out], [enc_out, self.target_channels],
                    [30, 30], [5, 2], [None, None]
                )

    def encoder_forward(self, x, lr = None, pos=None, t=None, label=None, debug=False):
        """
        x: [B, C_in, L_in]
        t: [B] or None
        """
        if debug:
            print(f"Input shape: {x.shape}")

        if not self.use_diffusion:
            x = self.encoder(x)
            residual_premamba = x.clone() if self.residual_internal else None
        
        else:
        
            if t is None:
                raise ValueError("Time embedding t must be provided for diffusion mode.")                
            # Conditioning mode
            conds = []
            x = self.encoder(x)
            temb = self.time_mlp(t)
            temb = temb.expand(-1, x.size(1), x.size(-1))
            conds.append(temb)
            if self.use_lr_conditioning and lr is not None:
                lr_patch = self.lr_condition_encoder(lr)
                if self.sr_type == "temporal": lr_patch = F.interpolate(lr_patch, x.size(-1), mode='linear')
                conds.append(lr_patch)
            else:
                if self.use_lr_conditioning:
                    print("Warning: LR conditioning enabled but no LR input provided.")
            if self.use_electrode_embedding and pos is not None:
                se = self.spatial_emb(pos).expand(-1, -1, x.size(-1))
                #se = F.normalize(se, dim=1)
                conds.append(se)
            else:
                if self.use_electrode_embedding:
                    print("Warning: Electrode embedding enabled but no position input provided.")
            if self.use_label and label is not None:
                #print(label)
                la = self.label_emb(label.long()).unsqueeze(-1).expand(-1, -1, x.size(-1))
                #la = F.normalize(la, dim=1)
                conds.append(la)
            else:
                if self.use_label:
                    print("Warning: Label conditioning enabled but no label input provided.")

            #for cond in conds: 
            #    print(min(cond.flatten()), max(cond.flatten()), cond.shape)
            #    print(f"Cond mean: {cond.mean().item():.4f}, std: {cond.std().item():.4f}")
            
            if conds: x = x + sum(conds)
            residual_premamba = x.clone() if self.residual_internal else None

        if self.use_mamba:
            x_mamba = x.transpose(1, 2)
            x_mamba_out = self.bottleneck(x_mamba, debug=debug)
            x = x_mamba_out.transpose(1, 2)
        else:
            x = self.bottleneck(x)

        return x, residual_premamba

    def decoder_forward(self, z, residual_premamba, lr=None, t=None, pos = None, label=None, debug=False):
        
        sr = self.decoder_sr(z)  # (B, C_out, L_out)

        if self.residual_internal and residual_premamba is not None:
            if residual_premamba.shape != sr.shape:
                raise ValueError(f"Residual shape {residual_premamba.shape} does not match decoder output shape {sr.shape}")
            sr += residual_premamba
        
        if self.use_subpixel:
            sr = self.subpixel(sr)
            #if not self.use_mamba:
            #    print(f"Decoder output shape after subpixel: {sr.shape}")

        if self.residual_global and lr is not None:
            if self.sr_type == "spatial":
                lr_up = add_zero_channels(lr, self.target_channels, dataset_name=self.dataset_name, multiplier=self.multiplier)
            else:
                lr_up = F.interpolate(lr, size=self.hr_len, mode='linear', align_corners=False)

            #print(lr_up.shape, sr.shape)
            sr += lr_up #+ residual_premamba if residual_premamba is not None else lr_up
            
        return sr

    # ---------- regression SR path ----------
    def forward_regression(self, lr, hr=None, debug=False, return_latent = False):
        """
        lr: low-res input  (B, C_lr, L_lr)
        hr: high-res target (B, C_hr, L_hr), optional (needed for loss)
        returns:
            sr, loss, log_dict
        """

        # encode
        if debug:
            print(f"Input LR shape: {lr.shape}")
        z, residual_premamba = self.encoder_forward(lr, debug=debug)          # (B, 256, L_enc)
        if debug:
            print(f"Encoded tokens shape: {z.shape}")
            
        # decode SR
        pred = self.decoder_forward(z, residual_premamba=residual_premamba, debug=debug)   # (B, C_hr, L_hr)
        if debug:
            print(f"Final SR shape: {pred.shape}")
        
        if return_latent:
            return pred, z
        return pred

    # ---------- diffusion (DDPM) path ----------
    def forward_diffusion(self, x_t_hr, t, lr=None, label=None, pos=None, debug=False, return_latent = False):
        """
        x_t_hr: noised HR, (B, C_HR, L_HR)
        lr: clean (or lightly noised) LR EEG, (B, C_LR, L_LR)
        returns: pred_noise_hr, (B, C_HR, L_HR)
        """
        # x_t_hr: (B, C, L_HR), lr: (B, C, L_LR)
        if debug:
            print(f"Diffusion forward shapes - x_t_hr: {x_t_hr.shape}, lr: {lr.shape}, t: {t.shape}, pos: {pos.shape if pos is not None else 'None'}")
        
        model_input = x_t_hr
        z, residual_premamba = self.encoder_forward(model_input, lr=lr, t=t, pos=pos, label=label, debug=debug)
        pred = self.decoder_forward(z, residual_premamba=residual_premamba, lr = lr, t=t, pos=pos, label=label, debug=debug)
        if return_latent:
            return pred, z
        return pred

    # ---------- unified forward ----------
    def forward(self, *args, **kwargs):
        """
        If use_diffusion = False:
            expect forward(lr, hr=None) -> SR
        If use_diffusion = True:
            expect forward(x_t_hr, t, pos, cond_lr) -> pred_hr_noise_or_sample
        """
        if self.use_diffusion:
            return self.forward_diffusion(*args, **kwargs)
        else:
            return self.forward_regression(*args, **kwargs)

class DiBiMa(pl.LightningModule):

    def __init__(self, model, dataset_name="mmi", multiplier = 2, epochs=10, learning_rate=0.0001, loss_fn=nn.MSELoss(), debug=False, plot=False):

        super().__init__()
        self.model = model
        self.criterion = loss_fn
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []
        self.val_nmse = []
        self.val_pcc = []
        self.debug = debug
        self.plot = plot
        self.dataset_name = dataset_name
        self.multiplier = multiplier
        self.epochs = epochs
        
        self.lr_to_plot = None
        self.hr_to_plot = None
        self.pred_to_plot = None

        if self.plot:
            #input vs output vs condition
            self.fig_inout = plt.figure(figsize=(12, 6))
            self.ax_inout = self.fig_inout.add_subplot(1, 1, 1)
            self.ax_inout.set_title("GeneratedHR - TargetHR - LRCondition")
            self.ax_inout.set_xlabel("Time")
            self.ax_inout.set_ylabel("Amplitude")

    def forward(self, x):
        return self.model(x, debug=self.debug)

    def compute_loss(self, sr_recon, hr_target):
        if self.criterion.__class__.__name__ == 'ReconstructionLoss':
            loss = self.criterion(sr_recon, hr_target, self.model)
        else:
            loss = self.criterion(sr_recon, hr_target)
        return loss
    
    def on_fit_end(self):
        """Close GUI after final validation"""
        if hasattr(self, 'fig_inout') and self.fig_inout is not None:
            plt.close(self.fig_inout)
            self.fig_inout = None
            print("Plot window closed.")

    def training_step(self, batch, batch_idx):
        lr_input, hr_target, _, _ = batch
        sr_recon = self(lr_input)
        loss = self.compute_loss(sr_recon, hr_target)
        if self.criterion.__class__.__name__ == "EEGSuperResolutionLoss":
            loss = loss[0]
        self.log('train_loss', loss, prog_bar=True)
        self.train_losses.append(loss.cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        lr_input, hr_target, _, _ = batch
        sr_recon = self(lr_input)
        if self.plot and batch_idx == 0:
            self.lr_to_plot = lr_input[0:1].squeeze().cpu()#.numpy()
            self.hr_to_plot = hr_target[0:1].squeeze().cpu()#.numpy()
            self.pred_to_plot = sr_recon[0:1].squeeze().cpu()#.numpy()

        hr_target_flat = hr_target.flatten()
        sr_recon_flat = sr_recon.flatten()
        self.val_nmse.append(nmse2d(hr_target_flat, sr_recon_flat))
        self.val_pcc.append(pcc2d_torch(hr_target_flat, sr_recon_flat))
        loss = self.compute_loss(sr_recon, hr_target)
        if self.criterion.__class__.__name__ == "EEGSuperResolutionLoss":
            loss = loss[0]
        self.val_losses.append(loss.cpu().item())
        return loss
        
    def on_train_epoch_end(self):
        avg_loss = np.mean(self.train_losses)
        self.log('avg_train_loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        return super().on_train_epoch_end()
    
    def on_train_epoch_start(self):
        self.model.train()
        self.train()
        self.train_losses = []
        return super().on_train_epoch_start()

    def on_validation_epoch_end(self):

        mean_val_loss = np.mean(self.val_losses)
        self.log("avg_val_loss", mean_val_loss, prog_bar=True, on_epoch=True)
        mean_val_pcc = np.mean(self.val_pcc)
        self.log("avg_val_pcc", mean_val_pcc, prog_bar=True, on_epoch=True)
        mean_val_nmse = np.mean(self.val_nmse)
        self.log("avg_val_nmse", mean_val_nmse, prog_bar=True, on_epoch=True)

        self.val_nmse = []
        self.val_losses = []
        self.val_pcc = []
        
        if self.plot and self.lr_to_plot is not None and self.hr_to_plot is not None and self.pred_to_plot is not None:
            
            #print(self.lr_to_plot.shape, self.hr_to_plot.shape, self.pred_to_plot.shape)
            lr = self.lr_to_plot 
            hr = self.hr_to_plot
            output = self.pred_to_plot

            if self.model.sr_type == "spatial":
                lr_up = add_zero_channels(lr, target_channels=hr.shape[0], dataset_name=self.model.dataset_name, multiplier=self.model.multiplier)
            else:
                lr_up = F.interpolate(lr.unsqueeze(0), size=hr.shape[-1], mode='linear', align_corners=False).squeeze(0)

            lr_up_ch = lr_up.mean(dim=0)
            hr_ch = hr.mean(dim=0)
            output_ch = output.mean(dim=0)
            
            self.ax_inout.clear()
            self.ax_inout.plot(lr_up_ch, label="LRCondition")
            self.ax_inout.plot(hr_ch, label="TargetHR")
            self.ax_inout.plot(output_ch, label="GeneratedHR")
            self.ax_inout.legend()
            self.fig_inout.canvas.draw()     
            self.fig_inout.canvas.flush_events()  # non-blocking update
            plt.pause(0.001)  # minimal delay

        return super().on_validation_epoch_end()

    def on_validation_epoch_start(self):
        self.model.eval()
        self.eval()
        self.val_losses = []
        self.val_nmse = []
        self.val_pcc = []
        return super().on_validation_epoch_start()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-6 if self.model.sr_type == "spatial" else 1e-4
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                        'scheduler': lr_scheduler,
                        'monitor': 'avg_val_loss', 
                        'interval': 'epoch',
                        'frequency': 1
            }
        }

class BidirectionalMamba(nn.Module):

    def __init__(self, d_model=256, d_state=16, d_conv=3, expand=2, n_layers=1, n_mamba_blocks = 1, mamba_version=1, merge_type = "concat", device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_mamba_blocks = n_mamba_blocks
        self.merge_type = merge_type

        self.forward_layers = nn.ModuleList([])
        self.backward_layers = nn.ModuleList([])

        self.forward_convs = nn.ModuleList([])
        self.backward_convs = nn.ModuleList([])

        self.device = device
    
        print("Multiple Bi-Mamba layers with convolutional residual connections are used.")
        self.residual_convs = nn.ModuleList([])
            
        in_channels = d_model
        for i in range(self.n_layers):
            if self.merge_type == "concat":
                out_channels = in_channels * 2
            else:
                out_channels = in_channels
            self.residual_convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ).to(self.device)
            )
            in_channels = out_channels

        for i in range(n_layers):

            if mamba_version == 1:
                mamba_instance = Mamba
            else:
                mamba_instance = Mamba2

            fwd_blocks = nn.ModuleList([])
            bwd_blocks = nn.ModuleList([])
            fwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1).to(self.device)
            bwd_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1).to(self.device)
            
            if i > 0:
                if self.merge_type == "concat":
                    d_model *= 2

            for j in range(n_mamba_blocks):
                fwd_blocks.append(
                        Block(
                            d_model,
                            mixer_cls = partial(mamba_instance, layer_idx = j, d_state=d_state, d_conv=d_conv, expand=expand),
                            mlp_cls=partial(nn.Linear, d_model, d_model),
                            norm_cls = partial(RMSNorm, eps=1e-5),
                            fused_add_norm = False
                        ).to(self.device)
                    )
                
                bwd_blocks.append(
                    Block(
                        d_model,
                        mixer_cls = partial(mamba_instance, layer_idx = j, d_state=d_state, d_conv=d_conv, expand=expand),
                        mlp_cls=partial(nn.Linear, d_model, d_model),
                        norm_cls = partial(RMSNorm, eps=1e-5),
                        fused_add_norm = False
                    ).to(self.device)
                )

            self.apply(partial(_init_weights, n_layer=i))

            self.forward_layers.append(fwd_blocks)
            self.backward_layers.append(bwd_blocks)
            self.forward_convs.append(fwd_conv)    
            self.backward_convs.append(bwd_conv)

    def _bimamba_layer(self, x, fwd_blocks, bwd_blocks, layer_idx, debug=False):
        
        # === FORWARD PASS ===
        for_residual = None
        forward_f = x.clone()
        for block in fwd_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        forward_f = self.forward_convs[layer_idx](forward_f.transpose(1,2)).transpose(1,2)

        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if bwd_blocks is not None:
            back_residual = None
            backward_f = torch.flip(x, [1])
            for block in bwd_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            backward_f = self.backward_convs[layer_idx](backward_f.transpose(1,2)).transpose(1,2)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            if self.merge_type == "concat":
                residual = torch.cat([residual, back_residual], -1)
            else:
                residual += back_residual
        
        return residual

    def forward(self, x, debug=False):  # [B, L, D]
        # Optional: outermost residual
        # residual_outer = x.clone()
        
        for i in range(self.n_layers):
            # Clone input to THIS layer (will be residual for NEXT layer)
            residual_inner = x.clone()
            if debug:
                print(f"Layer {i} input/residual for next: {residual_inner.shape}")
            
            # Apply Bi-Mamba layer
            x = self._bimamba_layer(
                x,
                self.forward_layers[i],
                self.backward_layers[i],
                layer_idx=i,
                debug=debug
            )
            """
            # connect PREV layer output (=this input) to this output                
            residual_proj = self.residual_convs[i](residual_inner.transpose(1,2)).transpose(1,2)
            if debug:
                print(f"Before add layer {i}: x={x.shape}, residual={residual_proj.shape}")
            
            x += residual_proj         
            if debug:
                print(f"After residual add layer {i}: {x.shape}")
            """
            x += residual_inner  # direct residual add
        if debug:
            print(f"Layer {i} output: {x.shape}")
        
        # Optional: x += residual_outer
        return x

import matplotlib.pyplot as plt

class DiBiMa_Diff(pl.LightningModule):
    def __init__(
        self, 
        model,
        train_scheduler,
        val_scheduler,
        criterion = nn.MSELoss(),
        epochs = 0,
        learning_rate=1e-4,
        predict_type="sample",  # "epsilon" or "sample"
        use_diffusion=True,
        debug=False,
        plot=False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['criterion', 'model'])
        
        # Core model
        self.model = model
        self.model.use_diffusion = use_diffusion
        self.use_diffusion = use_diffusion
        self.epochs = epochs

        # Criterion
        self.criterion = criterion

        # Prediction type
        self.predict_type = predict_type
        
        self.train_scheduler = train_scheduler
        self.val_scheduler = val_scheduler
        
        self.learning_rate = learning_rate
        
        self.train_losses = []
        self.val_losses = []
        self.val_losses_one_step = []
        self.val_pcc = []
        self.val_nmse = []

        self.lr_to_plot = None
        self.pred_to_plot_one_step = None
        self.noisy_lr_to_plot = None
        self.hr_to_plot = None
        self.pred_to_plot = None

        #print(f"Initialized with prediction type: {self.predict_type}")
        #print(f"Scheduler prediction type: {scheduler_prediction_type}")

        self.debug = debug
        self.plot = plot
        self.fig_inout = None
        self.ax_inout = None
                    
    def forward(self, *args, **kwargs):
        """
        If self.model.use_diffusion == False:
            expect forward(lr) -> SR 
        If self.model.use_diffusion == True:
            expect forward(x_t_hr, t, lr=None, label=None, pos=None) -> pred (noise or sample) but outputs always SR sample 
        """
        if self.model.use_diffusion:
            return self.model.forward_diffusion(*args, debug=self.debug, **kwargs)
        else:
            return self.model.forward_regression(*args, debug=self.debug, **kwargs)
        
    def compute_loss(self, output, target):
        if self.criterion.__class__.__name__ == 'ReconstructionLoss':
            return self.criterion(output, target, self.model)
        else:
            return self.criterion(output, target)

    def on_fit_end(self):
        """Close GUI after final validation"""
        if hasattr(self, 'fig_inout') and self.fig_inout is not None:
            plt.close(self.fig_inout)
            self.fig_inout = None
            print("Plot window closed.")

    def training_step(self, batch, batch_idx):
        lr, hr_gt, pos, label = batch
        bs = lr.size(0)
        device = hr_gt.device
        
        t = torch.randint(0, self.train_scheduler.num_train_timesteps, (bs,), device=device)
        noise = torch.randn_like(hr_gt, device=device)
        x_t = self.train_scheduler.add_noise(hr_gt, noise, t)
        model_out = self(x_t, t, lr=lr, pos=pos, label=label)
    
        if self.predict_type in ["epsilon", "v_prediction"]:  
            loss = self.criterion(model_out, noise)
        elif self.predict_type == "sample":
            loss = self.criterion(model_out, hr_gt)  # DIRECT compare!
        if self.criterion.__class__.__name__ == "EEGSuperResolutionLoss":
            loss = loss[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):

        lr, hr_gt, pos, label = batch
        bs = lr.size(0)
        device = self.device
        
        noise = torch.randn_like(hr_gt, device=device)
        
        self.eval()
        with torch.no_grad():
            if self.predict_type in ["epsilon", "v_prediction"]:
                xt = noise
                for ts in self.val_scheduler.timesteps:
                    pred_noise = self(xt, ts.to(device), lr=lr, pos=pos, label=label)
                    xt = self.val_scheduler.step(pred_noise, ts, xt).prev_sample
    
            elif self.predict_type == "sample":
                
                # 1. Start from pure Gaussian noise
                xt = torch.randn_like(hr_gt, device=device)
                #2 . Random timestep t for one-step generation
                t = torch.randint(0, self.train_scheduler.num_train_timesteps, (bs,), device=device)
                # Add noise corresponding to timestep t
                xt = self.val_scheduler.add_noise(hr_gt, xt, t)
                # 3. Direct Prediction (Model maps [Noise + LR] -> HR directly)
                xt = self(xt, t, lr=lr, pos=pos, label=label)
                
                # Now test one-step generation
                hr_one_step = self.sample_from_lr(lr, pos=pos, label=label)
                one_step_loss = self.criterion(hr_one_step, hr_gt)
                if self.criterion.__class__.__name__ == "EEGSuperResolutionLoss":
                    one_step_loss = one_step_loss[0]
                    
                hr_gt_flat = hr_gt.flatten() 
                hr_one_step_flat = hr_one_step.flatten() 
                sr_output = hr_one_step_flat if self.model.use_lr_conditioning else xt.flatten()
                pcc_one_step = pcc2d_torch(hr_gt_flat, sr_output)
                nmse_one_step = nmse2d(hr_gt_flat, sr_output)
                self.val_pcc.append(pcc_one_step)
                self.val_nmse.append(nmse_one_step)

                #This does the multiple-step sample prediction
                """
                elif self.predict_type == "sample":
                xt = noise  # Start from pure noise
                for ts in self.val_scheduler.timesteps:
                    xt = self(xt, ts.to(device), lr=lr, pos=pos, label=label)  # Direct prediction at each step
                """    
        val_loss = self.criterion(xt, hr_gt)
        if self.criterion.__class__.__name__ == "EEGSuperResolutionLoss":
            val_loss = val_loss[0]
        self.val_losses.append(val_loss.item())
        self.val_losses_one_step.append(one_step_loss.item())
        
        self.lr_to_plot = lr[0:1].squeeze().cpu()
        self.pred_to_plot_one_step = hr_one_step[0:1].squeeze().cpu()
        self.hr_to_plot = hr_gt[0:1].squeeze().cpu()
        self.pred_to_plot = xt[0:1].squeeze().cpu()
        return val_loss

    def on_train_epoch_start(self):
        
        self.model.train()
        self.train()
        self.train_losses = []
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        
        self.model.eval()
        self.eval()
        self.val_losses = []
        self.val_losses_one_step = []
        return super().on_validation_epoch_start()
        
    def on_train_epoch_end(self):

        self.log("avg_train_loss", np.mean(self.train_losses), prog_bar=True, on_epoch=True)
        sch = self.lr_schedulers()
        if sch:
            self.log("lr", sch.get_last_lr()[0], prog_bar=True)
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        
        mean_val_loss = np.mean(self.val_losses)
        self.log("avg_val_loss", mean_val_loss, prog_bar=True, on_epoch=True)
        mean_val_loss_one_step = np.mean(self.val_losses_one_step)
        self.log("avg_val_loss_one_step", mean_val_loss_one_step, prog_bar=True, on_epoch=True)
        mean_val_pcc = np.mean(self.val_pcc)
        self.log("avg_val_pcc", mean_val_pcc, prog_bar=True, on_epoch=True)
        mean_val_nmse = np.mean(self.val_nmse)
        self.log("avg_val_nmse", mean_val_nmse, prog_bar=True, on_epoch=True)   

        self.val_losses = []
        self.val_losses_one_step = []
        self.val_pcc = []
        self.val_nmse = []

        lr = self.lr_to_plot
        hr = self.hr_to_plot
        output = self.pred_to_plot
        output_one_step = self.pred_to_plot_one_step
        if not self.plot:
            return super().on_validation_epoch_end()
        else:
            if lr is None or hr is None or output is None:
                print("No signals stored for plotting.")
            else:
                #print("Debugging mode - plotting results for the first sample in the batch")
                #print(f"Plotting shapes - LR: {lr.shape}, HR: {hr.shape}, Output: {output.shape}")
                
                # Plot signals with proper labels
                if self.model.sr_type == "spatial":
                    lr_up = add_zero_channels(lr, target_channels=hr.shape[0], dataset_name=self.model.dataset_name, multiplier=self.model.multiplier)
                else:
                    lr_up = F.interpolate(lr.unsqueeze(0), size=hr.shape[-1], mode='linear', align_corners=False).squeeze(0)

                lr_up_ch = lr_up.mean(dim=0)
                hr_ch = hr.mean(dim=0)
                output_ch = output.mean(dim=0)
                output_one_step_ch = output_one_step.mean(dim=0)

                if self.plot:
                    
                    if self.fig_inout is None:
                        self.fig_inout = plt.figure(figsize=(12, 6))
                        self.ax_inout = self.fig_inout.add_subplot(1, 1, 1)
                    
                    # Clear & redraw
                    self.ax_inout.clear()
                    self.ax_inout.set_title(f"Epoch {self.current_epoch} | MSE: {mean_val_loss:.4f}")
                    self.ax_inout.plot(lr_up_ch, 'r-', label="LR", alpha=0.7, linewidth=1)
                    self.ax_inout.plot(hr_ch, 'g-', label="HR GT", linewidth=2)
                    self.ax_inout.plot(output_ch, 'b-', label="Generated from Noise", linewidth=2)
                    self.ax_inout.plot(output_one_step_ch, 'm--', label="One-Step Generated from Noised Upsampled LR", linewidth=2)
                    self.ax_inout.legend()
                    self.ax_inout.grid(True, alpha=0.3)
                    
                    self.fig_inout.canvas.draw()
                    self.fig_inout.canvas.flush_events()  # non-blocking update
                    plt.pause(0.001)  # minimal delay

            return super().on_validation_epoch_end()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        lr, hr_gt, pos, label = batch
        bs = lr.size(0)
        device = self.device
        
        noise = torch.randn_like(hr_gt, device=device)

        self.eval()
        if self.predict_type == "sample":
            # 1. Start from pure Gaussian noise
            xt = noise.clone()  # (B, C, L_HR)
                
            # 2. Use the maximum timestep (e.g., 999 for a 1000-step scheduler)
            # This tells the model "the input is fully noisy, please remove ALL noise"
            max_t = self.train_scheduler.config.num_train_timesteps - 1
            t = torch.full((bs,), max_t, device=device, dtype=torch.long)
              
            # 3. Direct Prediction (Model maps [Noise + LR] -> HR directly)
            xt = self(xt, t, lr=lr, pos=pos, label=label)        
        
        elif self.predict_type in ["epsilon", "v_prediction"]:
            xt = noise
            for ts in self.val_scheduler.timesteps:
                pred = self(xt, ts.to(device), lr=lr, pos=pos, label=label)
                xt = self.val_scheduler.step(pred, ts, xt).prev_sample       
        return xt       
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-6 if self.model.sr_type == "spatial" else 1e-4
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                        'scheduler': lr_scheduler,
                        'monitor': 'avg_val_loss', 
                        'interval': 'epoch',
                        'frequency': 1
            }
        }
                
    @torch.no_grad()
    def sample(self, lr, pos, label=None, lr_upsampled = None, num_inference_steps=100, return_all_steps=False):
        
        self.eval()
        device = lr.device
    
        # Only needed for the iterative loop; irrelevant for one-step
        self.val_scheduler.set_timesteps(num_inference_steps, device=device)

        batch_size = lr.shape[0]
        target_length = self.model.hr_len
        target_channels = self.model.target_channels
        
        #print(lr.shape, pos.shape if pos is not None else 'None', label.shape if label is not None else 'None')
        if pos is not None:
            if pos.ndim == 1:
                pos = pos.unsqueeze(0).repeat(batch_size, 1) # (B, 3)
            if pos.ndim == 2:
                pos = pos.unsqueeze(1).repeat(1, target_channels, 1)  # (B, C, 3)
            pos = pos.to(device)
        if label is not None and label.ndim == 0:
            label = label.unsqueeze(0).long().to(device)

        # Initialize from pure noise
        # Ensure target_length is an integer (e.g. 1024) or unpack if it's a tuple
        x = torch.randn(batch_size, target_channels, target_length, device=device)

        if return_all_steps:
            samples = []

        if self.predict_type in ["epsilon", "v_prediction"]:
            # Standard diffusion: iterative denoising
            for t in self.val_scheduler.timesteps:
                # Timesteps in the loop are SCALARS.
                # Expand for model input if your model expects (B,)
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                model_output = self(x, t_batch.to(device), lr=lr, pos=pos, label=label)
                
                # step() expects scalar 't' for standard schedulers
                step_output = self.val_scheduler.step(model_output, t, x)
                x = step_output.prev_sample

                if return_all_steps:
                    samples.append(x.detach().cpu())
        
        elif self.predict_type == "sample":
            # -----------------------------------------------------------
            # Consistency / One-Step Model: Direct Prediction
            # -----------------------------------------------------------
            
            # Use the MAXIMUM training timestep (e.g., 999) to indicate "pure noise" input
            max_t = self.train_scheduler.config.num_train_timesteps - 1
            t_batch = torch.full((batch_size,), max_t, device=device, dtype=torch.long)
            
            # Direct map: Noise -> Clean Data
            if lr_upsampled is not None:
                x += lr_upsampled  # Add lr upsampled as initial bias
            x = x.to(device)
            #else only noise             
            x = self(x, t_batch.to(device), lr=lr, pos=pos, label=label)

            if return_all_steps:
                samples.append(x.detach().cpu())

        if return_all_steps:
            return torch.stack(samples, dim=1)
        else:
            return x#.cpu()

    def sample_from_lr(self, lr, pos=None, label=None):

        if self.model.sr_type == "spatial":
            lr_upsampled = add_zero_channels(lr, target_channels=self.model.target_channels, dataset_name=self.model.dataset_name, multiplier=self.model.multiplier)
        else:
            lr_upsampled = torch.nn.functional.interpolate(
                lr,
                size=self.model.hr_len,
                mode='linear',
                align_corners=False
            )

        lr_upsampled = lr_upsampled.to(self.device)
        pos = pos.to(self.device) if pos is not None else None
        label = label.to(self.device) if label is not None else None

        with torch.no_grad():
            sr = self.sample(
                lr=lr,
                pos=pos,
                label=label,
                lr_upsampled=lr_upsampled,
                num_inference_steps=100,
                return_all_steps=False
            )
        return sr
