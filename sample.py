import argparse
import torch
from pathlib import Path

from loop_generation_poc.models.diffusion import ConditionalDiffusion
from loop_generation_poc.generation.samplers import sample_rectified_flow, sample_v_prediction

def main():
    parser = argparse.ArgumentParser(description="Generate audio latents from a trained checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the PyTorch Lightning .ckpt file")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seq_len", type=int, default=256, help="Target sequence length for the generated audio latents")
    parser.add_argument("--text_seq_len", type=int, default=77, help="Sequence length of the dummy text embeddings")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--output", type=str, default="generated_loop.pt", help="Output file name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Sampling on device: {device}")
    
    if not Path(args.ckpt).exists():
            raise FileNotFoundError(f"Checkpoint not found at {args.ckpt}")
        
    print(f"Loading checkpoint and hyperparameters from {args.ckpt}...")
    model = ConditionalDiffusion.load_for_inference(args.ckpt, map_location=device)

    # Dummy text embeddings representing a prompt
    dummy_prompt_embeds = torch.randn(1, args.text_seq_len, model.hparams.text_dim, device=device)

    objective = getattr(model.hparams, "diffusion_objective", "v_prediction")
    print(f"Model objective is: '{objective}'. Starting generation loop")

    if objective == "rectified_flow":
        latents = sample_rectified_flow(
            model=model,
            text_embeds=dummy_prompt_embeds,
            seq_len=args.seq_len,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale
        )
    elif objective == "v_prediction":
        latents = sample_v_prediction(
            model=model,
            text_embeds=dummy_prompt_embeds,
            seq_len=args.seq_len,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale
        )
    else:
        raise ValueError(f"Unknown objective found in checkpoint: {objective}")

    # Save the output
    torch.save(latents.cpu(), args.output)
    print(f"Saved generated latents of shape {latents.shape} to {args.output}")

if __name__ == "__main__":
    main()