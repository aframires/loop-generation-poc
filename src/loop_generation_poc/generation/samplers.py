import torch
from loop_generation_poc.models.diffusion import ConditionalDiffusion

@torch.no_grad()
def sample_rectified_flow(
    model: ConditionalDiffusion, 
    text_embeds: torch.Tensor, 
    seq_len: int,
    num_steps: int = 50, 
    cfg_scale: float = 3.0
) -> torch.Tensor:
    """
    Samples from a trained Rectified Flow model using Euler integration and CFG.
    
    Args:
        model: The trained model, in this case ConditionalDiffusion.
        text_embeds: text embeddings of shape [Batch, SeqLen, Dim].
        seq_len: The target sequence length of the generated latents.
        num_steps: Number of Euler integration steps.
        cfg_scale: Classifier-Free Guidance scale.
        
    Returns:
        Generated latents of shape [Batch, seq_len, LatentDim].
    """
    B, S_txt, D_txt = text_embeds.shape
    device = model.device
    
    # 1. Start from pure Gaussian noise (t=0)
    x_t = torch.randn(B, seq_len, model.hparams.latent_dim, device=device)
    
    # 2. Prepare Classifier-Free Guidance (CFG) context
    uncond_embeds = torch.zeros_like(text_embeds)
    combined_context = torch.cat([text_embeds, uncond_embeds], dim=0)
    
    # 3. Euler Integration Loop
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_sched = i * dt
        # Timestep matrix
        t = torch.full((B * 2,), t_sched, device=device)
        
        # Duplicate x_t for CFG batched forward pass
        x_t_combined = torch.cat([x_t, x_t], dim=0)
        
        # Predict velocity vector
        v_pred = model(x_t_combined, t, combined_context)
        
        # Split predictions and apply CFG
        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # Euler step
        x_t = x_t + v_cfg * dt
        
    return x_t

@torch.no_grad()
def sample_v_prediction(
    model: ConditionalDiffusion, 
    text_embeds: torch.Tensor, 
    seq_len: int,
    num_steps: int = 50, 
    cfg_scale: float = 3.0
) -> torch.Tensor:
    """
    Samples from a trained V-Prediction model using deterministic continuous-time DDIM integration and CFG.
    
    Args:
        model: The trained ConditionalDiffusion.
        text_embeds: text embeddings of shape [Batch, SeqLen, Dim].
        seq_len: The target sequence length of the generated latents.
        num_steps: Number of denoising steps.
        cfg_scale: Classifier-Free Guidance scale.
        
    Returns:
        Generated latents of shape [Batch, seq_len, LatentDim].
    """
    B, S_txt, D_txt = text_embeds.shape
    device = model.device
    
    # 1. Start from pure Gaussian noise 
    x_t = torch.randn(B, seq_len, model.hparams.latent_dim, device=device)
    
    # 2. Prepare Classifier-Free Guidance (CFG) context
    uncond_embeds = torch.zeros_like(text_embeds)
    combined_context = torch.cat([text_embeds, uncond_embeds], dim=0)
    
    # 3. Create discrete timesteps stepping backwards from 1.0 down to 0.0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_val = timesteps[i]
        s_val = timesteps[i+1] # The next timestep (closer to 0)
        
        # Timestep matrix
        t = torch.full((B * 2,), t_val, device=device)
        
        # Duplicate x_t for CFG batched forward pass
        x_t_combined = torch.cat([x_t, x_t], dim=0)
        
        # Predict the v-vector
        v_pred = model(x_t_combined, t, combined_context)
        
        # Split predictions and apply CFG
        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # --- DDIM Step for V-Prediction ---
        # Get blending coefficients for current time (t) and next time (s)
        theta_t = t_val * (torch.pi / 2.0)
        alpha_t = torch.cos(theta_t)
        sigma_t = torch.sin(theta_t)
        
        theta_s = s_val * (torch.pi / 2.0)
        alpha_s = torch.cos(theta_s)
        sigma_s = torch.sin(theta_s)
        
        # 1. Reconstruct the data (x_0) and the noise (eps) from x_t and v
        x_0_pred = alpha_t * x_t - sigma_t * v_cfg
        eps_pred = sigma_t * x_t + alpha_t * v_cfg
        
        # 2. Re-blend them for the next timestep (s)
        x_t = alpha_s * x_0_pred + sigma_s * eps_pred
        
    return x_t