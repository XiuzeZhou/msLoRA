import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F_nn
from typing import List
from scipy.stats import pearsonr
from module import MyLoraLayer, msLoRA, msLoRA_Concat

def analyze_mslora_modality_correlation(model, test_data, device, num_batches=10):
    model.eval()
    metrics = {
        'txt_img': {'cos': [], 'pearson': []},
        'txt_aud': {'cos': [], 'pearson': []},
        'img_aud': {'cos': [], 'pearson': []}
    }

    # cosine similarity
    def get_cos_sim(v1, v2):
        v1 = v1.flatten().to(torch.float32)
        v2 = v2.flatten().to(torch.float32)
        return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()

    # Pearson correlation
    def get_pearson(v1, v2):
        v1_np = v1.flatten().detach().cpu().to(torch.float32).numpy()
        v2_np = v2.flatten().detach().cpu().to(torch.float32).numpy()
        corr, _ = pearsonr(v1_np, v2_np)
        return corr

    print("\n" + "="*60)
    print("correlation analysis (Cosine & Pearson)...")

    with torch.no_grad():
        for i in range(min(num_batches, test_data.total_step)):
            img_ids, aud_ids, input_ids, mask, t_lens, _ = test_data.next_batch()
            input_ids = input_ids.to(device)

            # 1. msLoRA
            if isinstance(model, msLoRA):
                model.set_multimodal_features(img_ids, aud_ids)
                target_layer = None
                for m in model.modules():
                    if m.__class__.__name__ == 'MyLoraLayer' and getattr(m, 'has_multimodal', False):
                        target_layer = m
                        break
                
                if target_layer is None: continue

                x_txt = model.model.get_input_embeddings()(input_ids).mean(dim=1)
                x_txt = torch.nn.functional.normalize(x_txt, p=2, dim=-1)
                x_img = target_layer.x_img * 4.0   # As same in msLoRA definition 
                x_aud = target_layer.x_aud * 8.0 if target_layer.x_aud is not None else None  # As same in msLoRA definition 

            # 2. msLoRA_Concat
            elif isinstance(model, msLoRA_Concat):
                x_txt = model.model.get_input_embeddings()(input_ids).mean(dim=1)
                x_txt = torch.nn.functional.normalize(x_txt, p=2, dim=-1)
                mm_embeds = model._get_multimodal_embeds(img_ids, aud_ids)
                x_img = mm_embeds[:, 0, :]
                x_aud = mm_embeds[:, 1, :] if mm_embeds.shape[1] > 1 else None

            # T vs I
            if x_img is not None:
                metrics['txt_img']['cos'].append(get_cos_sim(x_txt, x_img))
                metrics['txt_img']['pearson'].append(get_pearson(x_txt, x_img))

            if x_aud is not None:
                metrics['txt_aud']['cos'].append(get_cos_sim(x_txt, x_aud))
                metrics['txt_aud']['pearson'].append(get_pearson(x_txt, x_aud))
                metrics['img_aud']['cos'].append(get_cos_sim(x_img, x_aud))
                metrics['img_aud']['pearson'].append(get_pearson(x_img, x_aud))

    print(f"{'Pair':<15} | {'Cosine Sim':<15} | {'Pearson Corr':<15}")
    print("-" * 50)
    for key, val in metrics.items():
        if val['cos']:
            avg_cos = np.mean(val['cos'])
            avg_pea = np.mean(val['pearson'])
            print(f"{key:<15} | {avg_cos:^15.4f} | {avg_pea:^15.4f}")
    print("="*60)

    return metrics

# --- 2. SVD ---

def compute_delta_w(lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float) -> torch.Tensor:
    """Delta W = B @ A * scaling"""
    # LoraA: (r, D_in), LoraB: (D_out, r)
    # Delta W: (D_out, D_in)
    # Delta W = (B^T)^T @ (A^T)^T * scaling = B @ A * scaling
    
    lora_A = lora_A.to(torch.float32)
    lora_B = lora_B.to(torch.float32)
    
    return (lora_B @ lora_A) * scaling

def analyze_svd_of_lora_weights(model, lora_index=5, num_singular_values: int = 50, output_dir: str = './analysis_results'):
    """
    Args:
        model (msLoRA): trained model
        num_singular_values (int): the number of SVD
        output_dir (str): path for saving figures.
    """
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "="*80)
    print("Start conducting SVD analysis...")
    
    results = {}
    
    lora_layers: List[MyLoraLayer] = []
    for name, module in model.named_modules():
        if isinstance(module, MyLoraLayer):
            lora_layers.append((name, module))

    if not lora_layers:
        print("The MyLoraLayer instance was not found, so SVD analysis cannot be performed.")
        return

    # MyLoraLayer weights
    name, lora_layer = lora_layers[lora_index]
    scaling = lora_layer.scaling
    
    # 1. Delta W_t
    delta_w_t = compute_delta_w(lora_layer.lora_A_t.data, lora_layer.lora_B_t.data, scaling)
    
    # 2. Delta W_aud
    scaling_audio = scaling * 8.0
    delta_w_audio = compute_delta_w(lora_layer.lora_A_aud.data, lora_layer.lora_B_aud.data, scaling_audio)

    # 3. Delta W_img
    scaling_img = scaling * 3.0
    delta_w_img = compute_delta_w(lora_layer.lora_A_img.data, lora_layer.lora_B_img.data, scaling_img)
    
    # 4. Delta W_fused
    delta_w_fused = delta_w_t + delta_w_audio + delta_w_img

    matrices_to_analyze = {
        'Text_$\Delta$W': delta_w_t,
        'Audio_$\Delta$W': delta_w_audio,
        'Image_$\Delta$W': delta_w_img,
        'Fused_$\Delta$W': delta_w_fused
    }

    plt.figure(figsize=(14, 6))

    for idx, (label, delta_w) in enumerate(matrices_to_analyze.items()):
        
        # SVD
        s = torch.linalg.svdvals(delta_w).cpu().numpy()
        
        num_to_plot = min(len(s), num_singular_values)
        s_plot = s[:num_to_plot]
        indices = np.arange(1, num_to_plot + 1)
        
        results[label] = {
            'total_rank': delta_w.shape[0],
            f'top_{num_to_plot}_variance_ratio': np.sum(s_plot**2) / np.sum(s**2) if np.sum(s**2) > 0 else 0
        }
        
        print(f"\n--- SVD results ({label}) ---")
        print(f"Total rank (D): {delta_w.shape[0]} | LoRA rank (r): {model.r}")
        print(f"Sum of squares of the first {num_to_plot} singular values as a proportion of the total variance: {results[label][f'top_{num_to_plot}_variance_ratio'] * 100:.2f}%")
        
        # Log-Log SVD
        plt.plot(indices, s_plot, marker='.', linestyle='-', markersize=4, label=label)

    # plot
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Log-Log Singular Value Spectrum of $\\Delta W$ Components in {name}')
    plt.xlabel('Singular Value Index $\\log(k)$')
    plt.ylabel('Singular Value $\\log(\\sigma_k)$')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Cumulative Information Span
    rank_checkpoints = [4, 16, 36]
    labels = ['$r_t=4$', '$r_a/r_i=16$', '$k=\sum r_m=36$']
    colors = ['blue', 'blue', 'blue']

    for rc, label, col in zip(rank_checkpoints, labels, colors):
        # Add vertical dashed lines
        plt.axvline(x=rc, color=col, linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Add text annotations above the line.
        plt.text(rc * 1.05, plt.ylim()[1] * 0.01, label, color=col, 
                fontsize=16, fontweight='bold', rotation=0)

    # Fused_ΔW
    plt.annotate('Information Span Extension', 
                xy=(36, 11), xytext=(25, 1e-1),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=16, color='red')
    
    # Save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'svd_spectrum_analysis.png')
    plt.savefig(plot_path)
    print(f"\nSVD figure saved to: {plot_path}")
    print("="*80)
    
    return results