import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import os
from utils import unmask_channels, add_zero_channels
 
project_path = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Blocco SubPixel 1D
# -------------------------------
class SubPixel1D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixel1D, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv1d(in_channels, out_channels * upscale_factor, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)  # [B, out_channels*upscale, T]
        B, C, T = x.size()
        r = self.upscale_factor
        if C % r != 0:
            raise ValueError(f"Channel dimension {C} not divisible by upscale factor {r}")
        x = x.view(B, C // r, r, T)       # [B, out_channels, r, T]
        x = x.permute(0, 1, 3, 2)         # [B, out_channels, T, r]
        x = x.contiguous().view(B, C // r, T * r)  # [B, out_channels, T*r]
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
    def __init__(self, reference_position, d_model, hidden_dim=128):
        super().__init__()
        if reference_position is not None:
            self.reference_pos = nn.Parameter(reference_position.float(), requires_grad=False)  # (C_ref, 3)
        else:
            self.reference_pos = None
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
    
    def forward(self, x, positions=None):
        # Assume proj ends with Linear(..., 64) -> dim=-1=64 unwanted
        if positions is None:
            proj_out = self.proj(self.reference_pos)  # (C, out_dim)
            se = proj_out.mean(-1, keepdim=True).unsqueeze(0)  # (1, C, 1) -> broadcast B
        else:
            b, c, _ = positions.shape
            proj_out = self.proj(positions.view(-1, 3)).view(b, c, -1)  # (B,C,out_dim)
            se = proj_out.mean(-1, keepdim=True)  # (B, C, 1) — average features per channel
        return se  # Guaranteed (B,C,1);



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

# -------------------------------
# DiBiMa_SubPixelRes Model
# -------------------------------

class DiBiMa_nn(nn.Module):
    def __init__(self, target_channels, ref_position=None, num_channels=64, fs_lr=80, fs_hr=160, seconds=10,
                 residual_global=True, use_mamba=True, use_diffusion=False, use_positional_encoding=False, use_electrode_embedding=False,
                 residual_internal=True, use_subpixel=True, sr_type="temporal", n_mamba_blocks = 1, n_mamba_layers=1, mamba_dim=64, mamba_d_state=16, mamba_version=1):
        super(DiBiMa_nn, self).__init__()

        self.target_channels = target_channels
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

        if self.use_mamba:
            enc_in = int(self.mamba_dim/2)
            enc_out = self.mamba_dim
        else:   
            enc_in = 128
            enc_out = 256

        self.use_diffusion = use_diffusion
        self.use_positional_encoding = use_positional_encoding
        self.use_electrode_embedding = use_electrode_embedding
        d_model = enc_out

        self.time_dim = enc_out
        if self.use_diffusion:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_dim, self.time_dim),
                nn.SiLU(),
                nn.Linear(self.time_dim, self.time_dim),
            )
            if use_positional_encoding:        
                self.pos_enc = PositionalEncoding(d_model=d_model)

            if use_electrode_embedding:
                self.spatial_emb = SpatialEmbedding(self.ref_position, d_model)

        if self.sr_type == "spatial":
            if self.use_diffusion:
                self.num_channels = num_channels + self.target_channels
            else:
                self.num_channels = num_channels
            self.input_length = self.hr_len
        else:
            #if self.use_diffusion:
            self.num_channels = num_channels
            self.input_length = self.lr_len

        self.residual_global = residual_global
        self.residual_internal = residual_internal
        self.use_subpixel = use_subpixel
        
        # --- Encoder ---        
        self.encoder = nn.Sequential(EncoderBlock([self.num_channels, enc_in], [enc_in, enc_out], [3, 3], [1, 1], [0.1, 0.1]))

        # --- Bottleneck (residual internal) ---
        if self.residual_internal:
            if self.use_mamba:
                print("Using Bi-Mamba bottleneck with residual internal")
                self.bottleneck = BidirectionalMamba(d_model=self.mamba_dim,
                                                     d_state=self.mamba_d_state,
                                                     n_mamba_blocks=self.n_mamba_blocks,
                                                     n_layers=self.n_mamba_layers,
                                                     mamba_version=self.mamba_version)
            else:
                print("Using simple bottleneck with residual internal")
                self.bottleneck = nn.Sequential(
                    nn.Conv1d(enc_out, enc_out, kernel_size=1, padding=0),
                    nn.BatchNorm1d(enc_out),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(0.3)
                )

        # --- Decoder ---
        if self.use_subpixel:
            if self.use_mamba:
                conv_d = nn.Conv1d(enc_out*2, enc_out, kernel_size=3, padding=1)
                batch_n = nn.BatchNorm1d(enc_out)
                sub_pixel_in = enc_out
            else:
                conv_d = nn.Conv1d(enc_out, enc_out, kernel_size=3, padding=1)
                batch_n = nn.BatchNorm1d(enc_out)
                sub_pixel_in = enc_out

            if self.sr_type == "spatial":
                print(f"Using spatial SR: {num_channels} channels to {self.target_channels}")
                upscale = 1  # spatial: channels change, length fixed
            else:
                print(f"Using temporal SR: {self.lr_len} to {self.hr_len} length")
                upscale = int(self.hr_len // self.lr_len)

            self.decoder_sr = nn.Sequential(
                conv_d,
                batch_n
            )

            self.subpixel = SubPixel1D(
                in_channels=sub_pixel_in,
                out_channels=self.target_channels,
                upscale_factor=upscale
            )

        else:
            if not self.use_mamba:
                self.decoder_sr = DecoderBlock(
                    [enc_out, enc_in], [enc_in, self.target_channels],
                    [30, 30], [5, 2], [None, None]
                )
            else:
                self.decoder_sr = DecoderBlock(
                    [enc_out*2, enc_out], [enc_out, self.target_channels],
                    [30, 30], [5, 2], [None, None]
                )
        
    def timestep_embedding(self, t, dim):
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

    def encoder_forward(self, x, pos=None, t=None, debug=False):
        """
        x: [B, C_in, L_in]
        t: [B] or None
        """
        if debug:
            print(f"Input shape: {x.shape}")
        x = self.encoder(x)  # [B, 256, L_enc] (or mamba_dim)

        if self.use_diffusion and t is not None:
            # Time embedding
            temb = self.timestep_embedding(t, self.time_dim)  # (B, time_dim)
            temb = self.time_mlp(temb).unsqueeze(-1)  # (B, d_model, 1); broadcast-ready
            
            # Spatial embedding (electrode-aware)
            if self.use_electrode_embedding and pos is not None:
                se = self.spatial_emb(x, pos)  # (B, d_model, L_enc); or mean(1) if global
                se = se.mean(dim=-1, keepdim=True)
            else:
                print("No electrode positional embedding used.")
                se = torch.zeros_like(x)  # No spatial embedding
                
            # Fuse into feature tensor for conditioning (align dims: repeat temb/se to seq_len)
            #print(f"x: {x.shape}, temb shape: {temb.shape}, se shape: {se.shape}")
            
            feat = x + temb + se  # Broadcast add: (B, d_model, L_enc)
            
            if self.use_positional_encoding:
                feat = feat.transpose(1, 2)  # (B, L_enc, d_model)
                feat = self.pos_enc(feat)
                feat = feat.transpose(1, 2)  # Back to (B, d_model, L_enc)
            
            condition = feat.mean(2)  # Pool over length: fixed c (B, d_model)
            if debug: print(f"Condition c shape: {condition.shape}")
            
            # Now inject condition back (global add or concat+proj)
            condition_exp = condition.unsqueeze(-1)  # (B, d_model, 1)
            x = x + condition_exp  # Or use cross-attn if needed

        if self.residual_internal:
            residual_premamba = x.clone()
            if self.use_mamba:
                x_mamba = x.transpose(1, 2)
                x_mamba_out = self.bottleneck(x_mamba, debug=debug)
                x = x_mamba_out.transpose(1, 2)
            else:
                x = self.bottleneck(x)
        else:    
            residual_premamba = None

        return x, residual_premamba

    def decoder_forward(self, z, residual_premamba, lr=None, t=None, debug=False):
        
        sr = self.decoder_sr(z)  # (B, C_out, L_out)

        if self.residual_internal and residual_premamba is not None:
            if residual_premamba.shape != sr.shape:
                raise ValueError(f"Residual shape {residual_premamba.shape} does not match decoder output shape {sr.shape}")
            sr += residual_premamba
        
        if self.use_subpixel:
            sr = self.subpixel(sr)
            #if not self.use_mamba:
            #    print(f"Decoder output shape after subpixel: {sr.shape}")

        if self.residual_global is not None and lr is not None:
            if self.sr_type == "spatial":
                lr_up = add_zero_channels(lr, self.target_channels)
            else:
                lr_up = F.interpolate(lr, size=self.target_length, mode='linear', align_corners=False)
            sr += lr_up
            
        return sr

    # ---------- regression SR path ----------
    def forward_regression(self, lr, hr=None, debug=False):
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

        return pred

    # ---------- diffusion (DDPM) path ----------
    def forward_diffusion(self, x_t_hr, t, lr, pos, debug=False):
        """
        x_t_hr: noised HR, (B, C_HR, L_HR)
        lr: clean (or lightly noised) LR EEG, (B, C_LR, L_LR)
        returns: pred_noise_hr, (B, C_HR, L_HR)
        """
        # x_t_hr: (B, C, L_HR), lr: (B, C, L_LR)
        if debug:
            print(f"Diffusion forward shapes - x_t_hr: {x_t_hr.shape}, lr: {lr.shape}, t: {t.shape}, pos: {pos.shape if pos is not None else 'None'}")
        
        if self.sr_type == "spatial":
            model_input = torch.cat([x_t_hr, lr], dim=1)  # (B, C_HR+C_LR, L)
        else:
            model_input = lr # (B, C, L_LR)

        z, residual_premamba = self.encoder_forward(model_input, t=t, pos=pos, debug=debug)
        pred_noise_hr = self.decoder_forward(z, residual_premamba=residual_premamba, lr = lr, t=t, debug=debug)
        return pred_noise_hr

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

    def __init__(self, model, learning_rate=0.0001, loss_fn=nn.MSELoss(), debug=False):

        super().__init__()
        self.model = model
        self.criterion = loss_fn
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []
        self.debug = debug

    def forward(self, x):
        return self.model(x, debug=self.debug)

    def compute_loss(self, sr_recon, hr_target):
        if self.criterion.__class__.__name__ == 'ReconstructionLoss':
            loss = self.criterion(sr_recon, hr_target, self.model)
        else:
            loss = self.criterion(sr_recon, hr_target)
        return loss
    
    def training_step(self, batch, batch_idx):
        lr_input, hr_target, _ = batch
        sr_recon = self(lr_input)
        loss = self.compute_loss(sr_recon, hr_target)
        self.log('train_loss', loss, prog_bar=True)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        lr_input, hr_target, _ = batch
        sr_recon = self(lr_input)
        loss = self.compute_loss(sr_recon, hr_target)
        self.val_losses.append(loss)
        return loss
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log('avg_train_loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        return super().on_train_epoch_end()
    
    def on_train_epoch_start(self):
        self.model.train()
        self.train_losses = []
        return super().on_train_epoch_start()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True, on_epoch=True)
        return super().on_validation_epoch_end()

    def on_validation_epoch_start(self):
        self.model.eval()
        self.val_losses = []
        return super().on_validation_epoch_start()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.01, min_lr=1e-6)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'avg_train_loss'}

from mamba_ssm import Mamba, Mamba2

class BidirectionalMamba(nn.Module):

    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, n_layers=1, n_mamba_blocks = 1, mamba_version=1, device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_mamba_blocks = n_mamba_blocks

        self.forward_layers = []
        self.backward_layers = []
        self.device = device

        for _ in range(n_layers):
            if mamba_version == 1:
                self.forward_layers.append(
                    nn.ModuleList([
                        Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand
                        ).to(self.device) for _ in range(self.n_mamba_blocks)
                    ])
                )
                self.backward_layers.append(
                    nn.ModuleList([
                        Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand
                        ).to(self.device) for _ in range(self.n_mamba_blocks)
                    ])
                )

            elif mamba_version == 2:
                self.forward_layers.append(
                    nn.ModuleList([ 
                        Mamba2(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand
                        ).to(self.device) for _ in range(self.n_mamba_blocks)
                    ])
                )
                self.backward_layers.append(
                    nn.ModuleList([
                        Mamba2(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand
                        ).to(self.device) for _ in range(self.n_mamba_blocks)
                    ])
                )
            elif mamba_version == 3:
                raise NotImplementedError("Mamba version 3 is not implemented yet.")
            
        self.norm1_layers = []
        self.norm2_layers = []

        for _ in range(n_layers):

            self.norm1_layers.append(nn.RMSNorm(d_model).to(self.device))
            self.norm2_layers.append(nn.RMSNorm(d_model).to(self.device))

        self.norm1_layers = nn.ModuleList(self.norm1_layers)
        self.norm2_layers = nn.ModuleList(self.norm2_layers)
        self.forward_layers = nn.ModuleList(self.forward_layers)
        self.backward_layers = nn.ModuleList(self.backward_layers)


    def _bimamba_layer(self, x, fwd_layer, bwd_layer, norm1, norm2, debug=False):
        
        residual = x.clone()
        x_norm = norm1(x)  # Shared pre-norm
        
        # === FORWARD PASS ===
        mamba_out_forward = x_norm
        for i, layer in enumerate(fwd_layer):
            mamba_out_forward = layer(mamba_out_forward) + x_norm  # FIXED: original x_norm
            if debug: print(f"Fwd {i}: {mamba_out_forward.shape}")
        
        # === BACKWARD PASS ===
        bw_input = torch.flip(x_norm, dims=[1])  # Flip shared norm
        backward_residual = bw_input.clone()
        mamba_out_backward = bw_input
        for i, layer in enumerate(bwd_layer):
            mamba_out_backward = layer(mamba_out_backward) + backward_residual  # FIXED: original backward_residual
            if debug: print(f"Bwd {i}: {mamba_out_backward.shape}")
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Unflip
        
        # === COMBINE ===
        combined = norm2(mamba_out_forward + mamba_out_backward)
        out = torch.cat([combined, residual], dim=-1) #out = combined + residual
        return out

    def forward(self, x, debug=False):  # [B, L, D]
        
        #residual = x.clone()
        for i in range(self.n_layers):
            #residual_inner = x.clone()
            x = self._bimamba_layer(
                x,
                self.forward_layers[i],
                self.backward_layers[i],
                self.norm1_layers[i],
                self.norm2_layers[i],
                debug=debug
            )
            #x = x + residual_inner
            if debug:
                print(f"After Bi-Mamba layer {i}: {x.shape}")
        #x = x + residual
        return x    
        

    
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt

class DiBiMa_Diff(pl.LightningModule):
    def __init__(
        self, 
        model,
        criterion = nn.MSELoss(),
        diffusion_params=None,
        learning_rate=1e-4,
        scheduler_params=None,
        predict_type="epsilon",  # "epsilon" or "sample"
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
        
        # Criterion
        self.criterion = criterion

        # Prediction type
        self.predict_type = predict_type
        
        # Diffusion scheduler - set prediction_type accordingly
        diffusion_params = diffusion_params or {}
        
        # Map predict_type to scheduler prediction_type
        scheduler_prediction_type = predict_type
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_params.get('num_train_timesteps', 1000),
            beta_start=diffusion_params.get('beta_start', 1e-6),
            beta_end=diffusion_params.get('beta_end', 1e-2),
            beta_schedule=diffusion_params.get('beta_schedule', 'squaredcos_cap_v2'),
            prediction_type=diffusion_params.get('prediction_type', scheduler_prediction_type),
            clip_sample=diffusion_params.get('clip_sample', False),
            clip_sample_range=diffusion_params.get('clip_sample_range', None)
        )
        
        self.learning_rate = learning_rate
        self.scheduler_params = scheduler_params or {}
        
        self.train_losses = []
        self.val_losses = []

        self.lr_to_plot = None
        self.noisy_lr_to_plot = None
        self.hr_to_plot = None

        #print(f"Initialized with prediction type: {self.predict_type}")
        #print(f"Scheduler prediction type: {scheduler_prediction_type}")

        self.debug = debug
        self.plot = plot

        if self.debug or self.plot:
 
            #input vs output vs condition
            self.fig_inout = plt.figure(figsize=(12, 6))
            self.ax_inout = self.fig_inout.add_subplot(1, 1, 1)
            self.ax_inout.set_title("GeneratedHR - TargetHR - LRCondition")
            self.ax_inout.set_xlabel("Time")
            self.ax_inout.set_ylabel("Amplitude")
            
            #patches visualization
            """
            self.fig_patches = plt.figure(figsize=(12, 6))
            self.ax_patches = self.fig_patches.add_subplot(1, 1, 1)
            self.ax_patches.set_title("Patches Visualization")
            self.ax_patches.set_xlabel("Patch Index")
            self.ax_patches.set_ylabel("Patch Value")
            """
            
    def forward(self, *args, **kwargs):
        """
        If self.model.use_diffusion == False:
            expect forward(lr) -> SR 
        If self.model.use_diffusion == True:
            expect forward(x_t_hr, t, lr) -> pred (noise or sample)
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
        
    def training_step(self, batch, batch_idx):
        
        lr, hr, pos = batch
        #print(f"Training step batch shapes - LR: {lr.shape}, HR: {hr.shape}, POS: {pos.shape}")
        batch_size = lr.size(0)
                
        # x0_hr: clean HR EEG, shape (B, C_HR, L)
        noise_hr = torch.randn_like(hr)
        t = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=hr.device, dtype=torch.long)

        x_t_hr = self.scheduler.add_noise(hr, noise_hr, t)  # (B, C_HR, L)
        
        # Model prediction
        output = self(x_t_hr, t, lr, pos)  # (B, C_HR, L)

        if output.shape != x_t_hr.shape:    
            raise ValueError(f"Model output shape {output.shape} is different than noisy EEG shape {x_t_hr.shape}.")
        
        # Compute loss based on prediction type
        if self.predict_type == "epsilon":
            # Predict epsilon (noise)
            #print("Computing loss on predicted noise")
            #print(f"Output shape: {output.shape}, Noise shape: {noise.shape}")
            loss = self.compute_loss(output, noise_hr)
        else:
            #print("Computing loss on predicted clean sample")
            #print(f"Output shape: {output.shape}, HR shape: {hr.shape}")
            # Predict x0 (clean sample)
            loss = self.compute_loss(output, hr)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        self.train_losses.append(loss.item())

        if self.debug:
            raise Exception("Debug mode - stopping after one training step")

        del output, noise_hr, t, lr, hr  # Free up memory
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        lr, hr, pos = batch
        batch_size = lr.size(0)

        # Sample timesteps as in training
        t = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch_size,),
            device=hr.device,
            dtype=torch.long
        )

        # Diffuse HR
        noise_hr = torch.randn_like(hr)
        x_t_hr = self.scheduler.add_noise(hr, noise_hr, t)

        # Model prediction (same signature as training)
        output = self(x_t_hr, t, lr, pos)  # (B, C_HR, L')

        # Fix length mismatch on time dimension if needed
        if output.shape != x_t_hr.shape:
            raise ValueError(f"Model output length {output.shape} is shorter than noisy HR length {x_t_hr.shape}.")

        # Store for plotting (use something consistent)
        if batch_idx == 0 and (self.debug or self.plot):
            self.noisy_hr_to_plot = x_t_hr[0].detach().cpu()
            self.lr_to_plot = lr[0].detach().cpu()
            self.hr_to_plot = hr[0].detach().cpu()
            self.pred_to_plot = output[0].detach().cpu()

        # Compute loss based on prediction type
        if self.predict_type == "epsilon":
            val_loss = self.compute_loss(output, noise_hr)  # HR noise
        else:  # "sample"
            val_loss = self.compute_loss(output, hr)        # HR clean sample

        #self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_losses.append(val_loss.item())

        del x_t_hr, output, noise_hr, t, lr, hr
        return val_loss


    def on_train_epoch_start(self):
        self.model.train()
        self.train_losses = []
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self):
        self.log("avg_train_loss", torch.mean(torch.tensor(self.train_losses)), prog_bar=True, on_step=False, on_epoch=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        self.log("avg_val_loss", torch.mean(torch.tensor(self.val_losses)), prog_bar=True, on_step=False, on_epoch=True)
        
        lr = self.lr_to_plot
        hr = self.hr_to_plot
        noisy_lr = self.noisy_lr_to_plot

        if not self.plot:
            return super().on_validation_epoch_end()
        else:
            if lr is None or hr is None or noisy_lr is None:
                print("No signals stored for plotting.")
            else:
                #print("Debugging mode - plotting results for the first sample in the batch")
                
                # Clear previous plot
                self.ax_inout.clear()
                
                # Plot signals with proper labels
                if self.lr_to_plot is not None:
                    self.ax_inout.plot(self.lr_to_plot.detach().cpu().numpy(), 
                                    label="LR Condition", color='green', alpha=0.5, linewidth=1)
                
                self.ax_inout.plot(self.noisy_lr_to_plot.detach().cpu().numpy(), 
                                label="Generated HR", color='blue', alpha=0.5, linewidth=1)
                
                self.ax_inout.plot(self.hr_to_plot.detach().cpu().numpy(), 
                                label="Target HR", color='red', alpha=0.5, linewidth=0.5)
                
                self.ax_inout.set_xlabel('Time Steps')
                self.ax_inout.set_ylabel('Amplitude')
                self.ax_inout.set_title('LR-to-HR Conversion Results')
                self.ax_inout.legend(loc='upper right')
                self.ax_inout.grid(True, alpha=0.3)
                
                # Update the plot
                self.fig_inout.canvas.draw()
                self.fig_inout.canvas.flush_events()
                plt.pause(0.001)  # Critical: allows GUI to update

            return super().on_validation_epoch_end()
    
    def on_validation_epoch_start(self):
        self.model.eval()
        self.val_losses = []
        return super().on_validation_epoch_start()
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        if not self.scheduler_params:
            return optimizer
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_params.get('T_max', 100),
            eta_min=self.scheduler_params.get('eta_min', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
                
    @torch.no_grad()
    def sample(self, lr, pos, num_inference_steps=100, return_all_steps=False):
        
        self.eval()
        device = lr.device

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        batch_size = lr.shape[0]
        target_length = self.model.hr_len
        target_channels = self.model.target_channels

        # 1. initialize x_t_hr (e.g., pure hr noise)
        x = torch.randn(batch_size, target_channels, target_length, device=device)

        if return_all_steps:
            samples = []

        for t in self.scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            #we concatenate lr as condition inside the model forward, no need to do it here
            model_output = self(x, t_batch, lr, pos)  # pass t_batch
            step_output = self.scheduler.step(model_output, t, x)
            x = step_output.prev_sample

            if return_all_steps:
                samples.append(x.detach().cpu())

        if return_all_steps:
            return torch.stack(samples, dim=1)
        else:
            return x
