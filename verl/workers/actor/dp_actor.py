# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
import numpy as np

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.current_global_step = 0  # Track training step for plotting

    def _compute_icepop_mask(self, rollout_log_probs, old_log_probs, response_mask, alpha=0.5, beta=2.0):
        """
        Compute IcePop double-sided masking to filter out noisy gradient updates.

        The masking function M(k) where k = p_train / p_rollout:
        M(k) = { k  if k ∈ [α, β]
               { 0  otherwise

        Args:
            rollout_log_probs: Log probabilities from rollout/inference engine (vLLM), shape [batch_size, response_length]
            old_log_probs: Log probabilities from training engine (FSDP old policy), shape [batch_size, response_length]
            response_mask: Mask for valid (non-padding) tokens, shape [batch_size, response_length]
            alpha: Lower bound for probability ratio (default: 0.5)
            beta: Upper bound for probability ratio (default: 2.0)

        Returns:
            icepop_mask: Binary mask tensor, 1.0 for healthy updates, 0.0 for clipped tokens
            clipping_stats: Dictionary with clipping statistics for logging
        """
        with torch.no_grad():
            # Debug: Log input statistics
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[TIS+IcePop-DEBUG] _compute_icepop_mask called with α={alpha}, β={beta}")
                valid_tokens = (response_mask > 0).sum().item()
                total_tokens = response_mask.numel()
                print(f"[TIS+IcePop-DEBUG] Valid tokens (response_mask > 0): {valid_tokens}/{total_tokens}")
                print(f"[TIS+IcePop-DEBUG] rollout_log_probs: min={rollout_log_probs.min().item():.4f}, "
                      f"max={rollout_log_probs.max().item():.4f}, mean={rollout_log_probs.mean().item():.4f}")
                print(f"[TIS+IcePop-DEBUG] old_log_probs: min={old_log_probs.min().item():.4f}, "
                      f"max={old_log_probs.max().item():.4f}, mean={old_log_probs.mean().item():.4f}")

            # Initialize icepop_mask with ones (all tokens valid by default)
            icepop_mask = torch.ones_like(response_mask, dtype=torch.float32)

            # Only compute probability ratios for valid (non-padding) tokens
            valid_mask = response_mask > 0

            if valid_mask.sum() == 0:
                # No valid tokens, return all-ones mask
                clipping_stats = {
                    'icepop/clipping_ratio': 0.0,
                    'icepop/clipped_tokens': 0,
                    'icepop/total_tokens': 0,  # Only valid (non-padding) tokens
                    'icepop/clipped_lower': 0,
                    'icepop/clipped_upper': 0,
                    'icepop/valid_ratio_mean': 0.0,
                    'icepop/clipped_ratio_mean': 0.0,
                    'icepop/prob_ratio_mean': 0.0,
                    'icepop/prob_ratio_std': 0.0,
                    'icepop/prob_ratio_min': 0.0,
                    'icepop/prob_ratio_max': 0.0,
                }
                return icepop_mask, clipping_stats

            # Extract only valid tokens for probability ratio computation
            rollout_log_probs_valid = rollout_log_probs[valid_mask]
            old_log_probs_valid = old_log_probs[valid_mask]

            # Convert log probabilities to probabilities (only for valid tokens)
            # p_train (old_log_probs) / p_rollout (rollout_log_probs) = exp(log_p_train - log_p_rollout)
            log_ratio = old_log_probs_valid - rollout_log_probs_valid
            prob_ratio = torch.exp(log_ratio)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[TIS+IcePop] prob_ratio (valid tokens only): min={prob_ratio.min().item():.4f}, "
                      f"max={prob_ratio.max().item():.4f}, mean={prob_ratio.mean().item():.4f}, "
                      f"median={prob_ratio.median().item():.4f}")

                # Detailed analysis of probability ratios
                print(f"[TIS+IcePop] Log ratio statistics:")
                print(f"  log_ratio: min={log_ratio.min().item():.4f}, "
                      f"max={log_ratio.max().item():.4f}, mean={log_ratio.mean().item():.4f}")
                print(f"[TIS+IcePop] Percentage of prob_ratio in different ranges (valid tokens only):")
                total = prob_ratio.numel()
                print(f"  ratio < 0.1: {(prob_ratio < 0.1).sum().item()}/{total} ({(prob_ratio < 0.1).sum().item()/total*100:.2f}%)")
                print(f"  0.1 <= ratio < 0.5: {((prob_ratio >= 0.1) & (prob_ratio < 0.5)).sum().item()}/{total} ({((prob_ratio >= 0.1) & (prob_ratio < 0.5)).sum().item()/total*100:.2f}%)")
                print(f"  0.5 <= ratio <= 2.0: {((prob_ratio >= 0.5) & (prob_ratio <= 2.0)).sum().item()}/{total} ({((prob_ratio >= 0.5) & (prob_ratio <= 2.0)).sum().item()/total*100:.2f}%)")
                print(f"  2.0 < ratio < 10.0: {((prob_ratio > 2.0) & (prob_ratio < 10.0)).sum().item()}/{total} ({((prob_ratio > 2.0) & (prob_ratio < 10.0)).sum().item()/total*100:.2f}%)")
                print(f"  ratio >= 10.0: {(prob_ratio >= 10.0).sum().item()}/{total} ({(prob_ratio >= 10.0).sum().item()/total*100:.2f}%)")

            # Apply double-sided clipping: keep only ratios in [alpha, beta] for valid tokens
            # Mask out tokens where:
            # 1. prob_ratio < alpha (training prob << rollout prob - huge divergence)
            # 2. prob_ratio > beta (training prob >> rollout prob - overconfident)
            valid_in_range = ((prob_ratio >= alpha) & (prob_ratio <= beta)).float()

            # Scatter the valid token mask back to the full tensor
            # Padding tokens keep their value of 1.0 (not clipped)
            # Valid tokens get their computed mask value
            icepop_mask[valid_mask] = valid_in_range

            # Compute clipping statistics (only on valid tokens, not padding)
            total_valid_tokens = prob_ratio.numel()  # Only count valid tokens
            clipped_valid_tokens = (valid_in_range == 0.0).sum().item()
            clipping_ratio = clipped_valid_tokens / total_valid_tokens if total_valid_tokens > 0 else 0.0

            # Analyze clipped vs non-clipped tokens (only valid tokens)
            clipped_lower = (prob_ratio < alpha).sum().item()  # Training prob too low
            clipped_upper = (prob_ratio > beta).sum().item()   # Training prob too high

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[TIS+IcePop] IcePop masking results (VALID TOKENS ONLY, excluding padding):")
                print(f"  Total valid tokens: {total_valid_tokens}")
                print(f"  Clipped valid tokens: {clipped_valid_tokens} ({clipping_ratio*100:.2f}%)")
                print(f"  Clipped lower (ratio < {alpha}): {clipped_lower} ({clipped_lower/total_valid_tokens*100:.2f}%)")
                print(f"  Clipped upper (ratio > {beta}): {clipped_upper} ({clipped_upper/total_valid_tokens*100:.2f}%)")
                print(f"  Kept valid tokens: {total_valid_tokens - clipped_valid_tokens} ({(1-clipping_ratio)*100:.2f}%)")

                # Show sample values from clipped lower tokens
                lower_mask_valid = prob_ratio < alpha
                if lower_mask_valid.any():
                    lower_ratios = prob_ratio[lower_mask_valid]
                    lower_old_lp = old_log_probs_valid[lower_mask_valid]
                    lower_rollout_lp = rollout_log_probs_valid[lower_mask_valid]
                    print(f"[TIS+IcePop] Sample of clipped_lower tokens (first 10):")
                    for i in range(min(10, lower_ratios.numel())):
                        print(f"    Token {i}: ratio={lower_ratios[i].item():.6f}, "
                              f"old_lp={lower_old_lp[i].item():.4f}, rollout_lp={lower_rollout_lp[i].item():.4f}")

                # Show sample values from clipped upper tokens
                upper_mask_valid = prob_ratio > beta
                if upper_mask_valid.any():
                    upper_ratios = prob_ratio[upper_mask_valid]
                    upper_old_lp = old_log_probs_valid[upper_mask_valid]
                    upper_rollout_lp = rollout_log_probs_valid[upper_mask_valid]
                    print(f"[TIS+IcePop] Sample of clipped_upper tokens (first 10):")
                    for i in range(min(10, upper_ratios.numel())):
                        print(f"    Token {i}: ratio={upper_ratios[i].item():.6f}, "
                              f"old_lp={upper_old_lp[i].item():.4f}, rollout_lp={upper_rollout_lp[i].item():.4f}")
                else:
                    print(f"[TIS+IcePop] WARNING: No clipped_upper tokens found! This is abnormal.")

            # Calculate mean probability ratios for diagnostics (only on valid tokens)
            kept_mask = valid_in_range == 1.0
            clipped_mask = valid_in_range == 0.0
            valid_ratio_mean = prob_ratio[kept_mask].mean().item() if kept_mask.any() else 0.0
            clipped_ratio_mean = prob_ratio[clipped_mask].mean().item() if clipped_mask.any() else 0.0

            clipping_stats = {
                'icepop/clipping_ratio': clipping_ratio,
                'icepop/clipped_tokens': clipped_valid_tokens,
                'icepop/total_tokens': total_valid_tokens,  # Only valid (non-padding) tokens
                'icepop/clipped_lower': clipped_lower,
                'icepop/clipped_upper': clipped_upper,
                'icepop/valid_ratio_mean': valid_ratio_mean,
                'icepop/clipped_ratio_mean': clipped_ratio_mean,
                'icepop/prob_ratio_mean': prob_ratio.mean().item(),
                'icepop/prob_ratio_std': prob_ratio.std().item(),
                'icepop/prob_ratio_min': prob_ratio.min().item(),
                'icepop/prob_ratio_max': prob_ratio.max().item(),
            }

        return icepop_mask, clipping_stats

    def _plot_logprobs_comparison(self, rollout_log_probs, old_log_probs, response_mask, global_step):
        """
        Generate line charts comparing rollout log probs (vLLM) vs training log probs (FSDP old policy).
        Saves charts to default_local_dir/logprobs_comparison/step_{global_step}/.

        Args:
            rollout_log_probs: Tensor of shape (batch_size, response_length) - vLLM rollout log probs
            old_log_probs: Tensor of shape (batch_size, response_length) - FSDP old policy log probs
            response_mask: Tensor of shape (batch_size, response_length) - mask for valid tokens
            global_step: Current training step for organizing outputs
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for server environments
            import matplotlib.pyplot as plt
            import os
        except ImportError:
            print("[LogProbs-Viz] matplotlib not available, skipping visualization")
            return

        # Only rank 0 generates plots to avoid duplicate work
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except:
            rank = 0

        if rank != 0:
            return

        # Create output directory
        default_local_dir = getattr(self.config, 'default_local_dir', '.')
        print(f"[LogProbs-Viz] Using default_local_dir: {default_local_dir}")

        viz_dir = os.path.join(default_local_dir, "logprobs_comparison", f"step_{global_step:03d}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[LogProbs-Viz] Saving plots to: {viz_dir}")

        # Move tensors to CPU and convert to numpy
        rollout_log_probs_np = rollout_log_probs.detach().cpu().numpy()
        old_log_probs_np = old_log_probs.detach().cpu().numpy()
        response_mask_np = response_mask.detach().cpu().numpy()

        batch_size = rollout_log_probs_np.shape[0]

        # Plot ALL trajectories to get complete view
        print(f"[LogProbs-Viz] Plotting all {batch_size} trajectories")

        for traj_idx in range(batch_size):
            # Filter valid tokens (response_mask > 0)
            valid_mask = response_mask_np[traj_idx] > 0
            valid_positions = np.where(valid_mask)[0]

            print(f"\n[LogProbs-Viz-Filtering] Trajectory {traj_idx}:")
            print(f"  Total tokens (including padding): {len(rollout_log_probs_np[traj_idx])} tokens")
            print(f"  Valid tokens (response_mask>0): {valid_mask.sum()} tokens")

            if len(valid_positions) == 0:
                continue  # Skip empty trajectories

            # Extract valid log probs for this trajectory (for visualization)
            rollout_log_probs_valid = rollout_log_probs_np[traj_idx][valid_mask]
            old_log_probs_valid = old_log_probs_np[traj_idx][valid_mask]

            # Convert to probabilities for proper comparison (as IcePop/TIS does)
            rollout_probs_valid = np.exp(rollout_log_probs_valid)
            old_probs_valid = np.exp(old_log_probs_valid)

            # Print statistics
            print(f"\n{'='*80}")
            print(f"[LogProbs-Viz-DEBUG] Trajectory {traj_idx}: {len(valid_positions)} tokens")
            print(f"{'='*80}")

            # Print PROBABILITY statistics (this is what TIS/IcePop actually uses!)
            print(f"[LogProbs-Viz-DEBUG] Rollout (vLLM) PROBABILITY Statistics:")
            print(f"  Range: [{rollout_probs_valid.min():.6f}, {rollout_probs_valid.max():.6f}]")
            print(f"  Mean: {rollout_probs_valid.mean():.6f}, Std: {rollout_probs_valid.std():.6f}")
            print(f"  Median: {np.median(rollout_probs_valid):.6f}")

            print(f"\n[LogProbs-Viz-DEBUG] Training (FSDP) PROBABILITY Statistics:")
            print(f"  Range: [{old_probs_valid.min():.6f}, {old_probs_valid.max():.6f}]")
            print(f"  Mean: {old_probs_valid.mean():.6f}, Std: {old_probs_valid.std():.6f}")
            print(f"  Median: {np.median(old_probs_valid):.6f}")

            # Print correlations (both log prob and probability space)
            logprob_correlation = np.corrcoef(rollout_log_probs_valid, old_log_probs_valid)[0,1]
            prob_correlation = np.corrcoef(rollout_probs_valid, old_probs_valid)[0,1]
            print(f"\n[LogProbs-Viz-DEBUG] Correlation (Log Prob Space): {logprob_correlation:.6f}")
            print(f"[LogProbs-Viz-DEBUG] Correlation (Probability Space): {prob_correlation:.6f} ← THIS IS WHAT MATTERS!")

            correlation = prob_correlation

            # Print sample token-by-token comparisons (first 20)
            print(f"\n[LogProbs-Viz-DEBUG] Sample Token Comparisons (first 20 tokens):")
            print(f"  {'Pos':<6} {'Rollout_P':<12} {'Old_P':<12} {'Prob_Ratio':<12} {'Rollout_LP':<12} {'Old_LP':<12}")
            print(f"  {'-'*72}")
            for i in range(min(20, len(rollout_probs_valid))):
                pos = valid_positions[i]
                rollout_p = rollout_probs_valid[i]
                old_p = old_probs_valid[i]
                rollout_lp = rollout_log_probs_valid[i]
                old_lp = old_log_probs_valid[i]
                # Compute probability ratio: P_old / P_rollout (for TIS/IcePop)
                prob_ratio = old_p / rollout_p if rollout_p > 0 else float('inf')
                # Mark if ratio is outside IcePop bounds [0.5, 2.0]
                flag = "⚠️" if prob_ratio < 0.5 or prob_ratio > 2.0 else "  "
                print(f"{flag} {pos:<6} {rollout_p:<12.6f} {old_p:<12.6f} {prob_ratio:<12.6f} {rollout_lp:<12.4f} {old_lp:<12.4f}")

            print(f"{'='*80}\n")

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Subplot 1: Overlay comparison - PROBABILITIES (0 to 1)
            ax1.plot(valid_positions, rollout_probs_valid, 'b-', label='Rollout (vLLM)', alpha=0.8, linewidth=2.0, marker='o', markersize=3)
            ax1.plot(valid_positions, old_probs_valid, 'r--', label='Training (FSDP Old Policy)', alpha=0.8, linewidth=2.0, marker='s', markersize=3)

            # Set y-axis to probability range (0 to 1)
            ax1.set_ylim(0, 1)

            ax1.set_xlabel('Token Position', fontsize=12)
            ax1.set_ylabel('Probability (0 to 1)', fontsize=12)
            ax1.set_title(f'Trajectory {traj_idx}: Rollout vs Training Probabilities (Step {global_step})\n' +
                         f'{len(valid_positions)} Tokens (Correlation: {correlation:.4f})',
                         fontsize=13)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.4, linestyle='--')

            # Compute difference for statistics
            diff = rollout_probs_valid - old_probs_valid

            # Subplot 2: Scatter plot for direct comparison
            ax2.scatter(rollout_probs_valid, old_probs_valid, alpha=0.5, s=30, c='purple', edgecolors='black', linewidth=0.5)

            # Add perfect correlation line (y=x)
            ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Correlation (y=x)', alpha=0.7)

            # Add IcePop/TIS bounds
            # Ratio = 0.5 means old = 0.5 * rollout, ratio = 2.0 means old = 2.0 * rollout
            x_vals = np.linspace(0, 1, 100)
            ax2.plot(x_vals, 0.5 * x_vals, 'g--', linewidth=1.5, label='IcePop Lower Bound (ratio=0.5)', alpha=0.5)
            ax2.plot(x_vals, 2.0 * x_vals, 'orange', linestyle='--', linewidth=1.5, label='IcePop Upper Bound (ratio=2.0)', alpha=0.5)

            ax2.set_xlabel('Rollout Probability (vLLM)', fontsize=12)
            ax2.set_ylabel('Training Probability (FSDP)', fontsize=12)
            ax2.set_title(f'Direct Comparison: Rollout vs Training Probabilities\n(Correlation: {correlation:.4f})',
                         fontsize=13)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.4, linestyle='--')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_aspect('equal', adjustable='box')

            # Add summary statistics as text box
            # Compute clipping on valid tokens only
            prob_ratios_valid = old_probs_valid / np.maximum(rollout_probs_valid, 1e-10)
            valid_clipped_lower = (prob_ratios_valid < 0.5).sum()
            valid_clipped_upper = (prob_ratios_valid > 2.0).sum()
            valid_clipped_total = valid_clipped_lower + valid_clipped_upper
            valid_clipping_ratio = valid_clipped_total / len(prob_ratios_valid)

            stats_text = (
                f"Statistics (PROBABILITY SPACE):\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Rollout:   mean={np.mean(rollout_probs_valid):.6f}, std={np.std(rollout_probs_valid):.6f}\n"
                f"           min={np.min(rollout_probs_valid):.6f}, max={np.max(rollout_probs_valid):.6f}\n"
                f"Training:  mean={np.mean(old_probs_valid):.6f}, std={np.std(old_probs_valid):.6f}\n"
                f"           min={np.min(old_probs_valid):.6f}, max={np.max(old_probs_valid):.6f}\n"
                f"Difference: mean={np.mean(diff):.6f}, std={np.std(diff):.6f}\n"
                f"Correlation: {correlation:.4f}\n"
                f"Prob Ratio (valid): mean={np.mean(prob_ratios_valid):.4f}, std={np.std(prob_ratios_valid):.4f}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"IcePop Clipping (valid tokens, α=0.5, β=2.0):\n"
                f"  Lower: {valid_clipped_lower}/{len(prob_ratios_valid)} ({100*valid_clipped_lower/len(prob_ratios_valid):.2f}%)\n"
                f"  Upper: {valid_clipped_upper}/{len(prob_ratios_valid)} ({100*valid_clipped_upper/len(prob_ratios_valid):.2f}%)\n"
                f"  Total: {valid_clipped_total}/{len(prob_ratios_valid)} ({100*valid_clipping_ratio:.2f}%)\n"
                f"Valid Tokens: {len(valid_positions)} / {rollout_log_probs_np.shape[1]}"
            )
            fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), verticalalignment='bottom')

            plt.tight_layout(pad=3.0)

            # Save figure
            output_path = os.path.join(viz_dir, f"traj_{traj_idx:03d}_logprobs_comparison.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"[LogProbs-Viz] Saved trajectory {traj_idx} comparison to {output_path}")

        # Create summary plot showing statistics across all trajectories
        self._plot_batch_summary(rollout_log_probs_np, old_log_probs_np, response_mask_np, viz_dir, global_step)

    def _plot_batch_summary(self, rollout_log_probs_np, old_log_probs_np, response_mask_np, viz_dir, global_step):
        """Generate summary statistics plot for entire batch."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os
        except ImportError:
            return

        batch_size = rollout_log_probs_np.shape[0]

        # Collect per-trajectory statistics
        traj_means_rollout = []
        traj_means_old = []
        traj_diffs = []
        traj_correlations = []

        for traj_idx in range(batch_size):
            valid_mask = response_mask_np[traj_idx] > 0
            if not valid_mask.any():
                continue

            rollout_log_valid = rollout_log_probs_np[traj_idx][valid_mask]
            old_log_valid = old_log_probs_np[traj_idx][valid_mask]

            # Convert to probabilities for comparison (as TIS/IcePop does)
            rollout_valid = np.exp(rollout_log_valid)
            old_valid = np.exp(old_log_valid)

            traj_means_rollout.append(np.mean(rollout_valid))
            traj_means_old.append(np.mean(old_valid))
            traj_diffs.append(np.mean(rollout_valid - old_valid))

            if len(rollout_valid) > 1:  # Need at least 2 points for correlation
                # Compute correlation in PROBABILITY space
                corr = np.corrcoef(rollout_valid, old_valid)[0, 1]
                traj_correlations.append(corr)

        # Create 2x2 subplot grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Scatter of mean probabilities
        ax1.scatter(traj_means_rollout, traj_means_old, alpha=0.6, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Agreement (y=x)')
        ax1.set_xlabel('Rollout Mean Probability')
        ax1.set_ylabel('Training Mean Probability')
        ax1.set_title('Per-Trajectory Mean Probability Comparison')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Histogram of differences
        ax2.hist(traj_diffs, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(traj_diffs), color='g', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(traj_diffs):.6f}')
        ax2.set_xlabel('Mean Difference (Rollout - Training)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Probability Differences')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Correlation distribution
        if traj_correlations:
            ax3.hist(traj_correlations, bins=20, edgecolor='black', alpha=0.7, color='purple')
            ax3.axvline(x=np.mean(traj_correlations), color='r', linestyle='-', linewidth=2,
                       label=f'Mean: {np.mean(traj_correlations):.4f}')
            ax3.set_xlabel('Correlation Coefficient (Probability Space)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Rollout-Train Probability Correlations')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics table
        ax4.axis('off')
        summary_stats = [
            ['Metric', 'Rollout', 'Training', 'Difference'],
            ['Mean', f'{np.mean(traj_means_rollout):.4f}', f'{np.mean(traj_means_old):.4f}',
             f'{np.mean(traj_diffs):.4f}'],
            ['Std', f'{np.std(traj_means_rollout):.4f}', f'{np.std(traj_means_old):.4f}',
             f'{np.std(traj_diffs):.4f}'],
            ['Min', f'{np.min(traj_means_rollout):.4f}', f'{np.min(traj_means_old):.4f}',
             f'{np.min(traj_diffs):.4f}'],
            ['Max', f'{np.max(traj_means_rollout):.4f}', f'{np.max(traj_means_old):.4f}',
             f'{np.max(traj_diffs):.4f}'],
            ['', '', '', ''],
            ['Avg Correlation', '', '', f'{np.mean(traj_correlations) if traj_correlations else "N/A"}'],
            ['Batch Size', '', '', f'{batch_size}'],
            ['Valid Trajectories', '', '', f'{len(traj_means_rollout)}'],
        ]

        table = ax4.table(cellText=summary_stats, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.23, 0.23, 0.24])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax4.set_title(f'Batch Summary Statistics (Step {global_step})',
                     fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save summary figure
        output_path = os.path.join(viz_dir, f"batch_summary_step_{global_step:03d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[LogProbs-Viz] Saved batch summary to {output_path}")

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm) or grad_norm >= self.config.grad_norm_threshold:
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Both TIS and IcePop require rollout_log_probs from inference engine
        if self.config.tis_imp_ratio_cap > 0 or self.config.enable_icepop:
            assert "rollout_log_probs" in data.batch.keys(), (
                "TIS/IcePop requires `actor_rollout_ref.rollout.calculate_log_probs=True`. "
                "This works in both standard vLLM rollout and agent-based (server) mode."
            )
            select_keys.append("rollout_log_probs")
            
        if 'traj_mask' in data.batch:
            select_keys.append('traj_mask')

        # Handle pad steps for agent-based training
        if 'is_pad_step' in data.non_tensor_batch:
            is_pad_step = data.non_tensor_batch["is_pad_step"]
            pad_step_indices = np.where(is_pad_step == True)[0]
            if len(pad_step_indices) > 0:
                data.batch["advantages"][pad_step_indices] = 0

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}

        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # Collect data for plotting (only for first mini-batch of first epoch)
                collected_rollout_log_probs = []
                collected_old_log_probs = []
                collected_response_masks = []

                for micro_batch_idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    if "traj_mask" in model_inputs:
                        response_mask = model_inputs["traj_mask"]
                        print("[TrainingLogs] Using mask schema from traj mask!")
                    else:
                        response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    # IcePop+TIS: Apply double-sided masking and truncated importance sampling
                    icepop_stats = {}
                    tis_stats = {}

                    # IcePop: Filter noisy gradient updates from training-inference mismatch
                    # Uses same rollout_log_probs as TIS (unified source)
                    if self.config.enable_icepop and rollout_log_probs is not None:
                        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                            print(f"[TIS+IcePop-DEBUG] update_policy - epoch={epoch_idx}, batch={batch_idx}, micro_batch={micro_batch_idx}")
                            print(f"[TIS+IcePop-DEBUG] IcePop ENABLED with α={self.config.icepop_alpha}, β={self.config.icepop_beta}")
                            print(f"[TIS+IcePop-DEBUG] rollout_log_probs shape={rollout_log_probs.shape}, "
                                  f"mean={rollout_log_probs.mean().item():.4f}, "
                                  f"non_zero_ratio={(rollout_log_probs != 0.0).sum().item() / rollout_log_probs.numel():.4f}")

                        # Compute IcePop mask using configurable alpha/beta parameters
                        # Pass response_mask to exclude padding tokens from IcePop computation
                        icepop_mask, icepop_stats = self._compute_icepop_mask(
                            rollout_log_probs=rollout_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            alpha=self.config.icepop_alpha,
                            beta=self.config.icepop_beta
                        )

                        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                            print(f"[TIS+IcePop-DEBUG] IcePop clipping: "
                                  f"{icepop_stats['icepop/clipped_tokens']}/{icepop_stats['icepop/total_tokens']} tokens "
                                  f"({icepop_stats['icepop/clipping_ratio']:.4f}), "
                                  f"lower={icepop_stats['icepop/clipped_lower']}, "
                                  f"upper={icepop_stats['icepop/clipped_upper']}, "
                                  f"prob_ratio_mean={icepop_stats['icepop/prob_ratio_mean']:.4f}±{icepop_stats['icepop/prob_ratio_std']:.4f}")

                        # Combine IcePop mask with response_mask (element-wise AND)
                        # Original response_mask filters padding/invalid tokens
                        # IcePop mask filters tokens with excessive probability discrepancy
                        response_mask = response_mask * icepop_mask

                        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                            original_valid_tokens = (model_inputs["response_mask"] > 0).sum().item()
                            new_valid_tokens = (response_mask > 0).sum().item()
                            print(f"[TIS+IcePop-DEBUG] Response mask: {original_valid_tokens} → {new_valid_tokens} valid tokens "
                                  f"(IcePop filtered {original_valid_tokens - new_valid_tokens} tokens)")
                    else:
                        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0 and epoch_idx == 0 and batch_idx == 0 and micro_batch_idx == 0:
                            if not self.config.enable_icepop:
                                print(f"[TIS+IcePop-DEBUG] IcePop DISABLED (enable_icepop=False)")
                            elif rollout_log_probs is None:
                                print(f"[TIS+IcePop-DEBUG] IcePop NOT APPLIED (rollout_log_probs not available)")

                    # TIS: Log TIS status for vanilla DAPO mode
                    if self.config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
                        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0 and epoch_idx == 0 and batch_idx == 0 and micro_batch_idx == 0:
                            print(f"[TIS+IcePop-DEBUG] TIS (Truncated Importance Sampling) ENABLED for vanilla DAPO")
                            print(f"[TIS+IcePop-DEBUG] TIS clip threshold C={self.config.tis_imp_ratio_cap}")
                            print(f"[TIS+IcePop-DEBUG] rollout_log_probs shape={rollout_log_probs.shape}, "
                                  f"mean={rollout_log_probs.mean().item():.4f}")
                    elif torch.distributed.is_initialized() and torch.distributed.get_rank() == 0 and epoch_idx == 0 and batch_idx == 0 and micro_batch_idx == 0:
                        print(f"[TIS+IcePop-DEBUG] TIS DISABLED (tis_imp_ratio_cap={self.config.tis_imp_ratio_cap})")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    # Collect data for plotting from all micro-batches in first mini-batch
                    if epoch_idx == 0 and batch_idx == 0 and rollout_log_probs is not None:
                        collected_rollout_log_probs.append(rollout_log_probs.detach().cpu())
                        collected_old_log_probs.append(old_log_prob.detach().cpu())
                        collected_response_masks.append(response_mask.detach().cpu())

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    # Add IcePop statistics if masking was applied
                    if icepop_stats:
                        micro_batch_metrics.update(icepop_stats)
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

                # After processing all micro-batches, generate plots for first mini-batch
                if epoch_idx == 0 and batch_idx == 0 and len(collected_rollout_log_probs) > 0:
                    # Concatenate all collected micro-batches from this rank
                    local_rollout_log_probs = torch.cat(collected_rollout_log_probs, dim=0)
                    local_old_log_probs = torch.cat(collected_old_log_probs, dim=0)
                    local_response_masks = torch.cat(collected_response_masks, dim=0)

                    print(f"[LogProbs-Viz] Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: "
                          f"Collected {local_rollout_log_probs.shape[0]} trajectories from {len(collected_rollout_log_probs)} micro-batches")

                    # Gather from all ranks to get complete mini-batch view
                    if torch.distributed.is_initialized():
                        world_size = torch.distributed.get_world_size()
                        rank = torch.distributed.get_rank()

                        # Gather all trajectories from all ranks
                        gathered_rollout = [torch.zeros_like(local_rollout_log_probs) for _ in range(world_size)]
                        gathered_old = [torch.zeros_like(local_old_log_probs) for _ in range(world_size)]
                        gathered_masks = [torch.zeros_like(local_response_masks) for _ in range(world_size)]

                        torch.distributed.all_gather(gathered_rollout, local_rollout_log_probs.contiguous())
                        torch.distributed.all_gather(gathered_old, local_old_log_probs.contiguous())
                        torch.distributed.all_gather(gathered_masks, local_response_masks.contiguous())

                        if rank == 0:
                            # Concatenate all gathered data
                            all_rollout_log_probs = torch.cat(gathered_rollout, dim=0)
                            all_old_log_probs = torch.cat(gathered_old, dim=0)
                            all_response_masks = torch.cat(gathered_masks, dim=0)

                            print(f"[LogProbs-Viz] Gathered {all_rollout_log_probs.shape[0]} total trajectories from {world_size} ranks")

                            self._plot_logprobs_comparison(
                                rollout_log_probs=all_rollout_log_probs,
                                old_log_probs=all_old_log_probs,
                                response_mask=all_response_masks,
                                global_step=self.current_global_step
                            )
                    else:
                        # Single process, no gathering needed
                        self._plot_logprobs_comparison(
                            rollout_log_probs=local_rollout_log_probs,
                            old_log_probs=local_old_log_probs,
                            response_mask=local_response_masks,
                            global_step=self.current_global_step
                        )

        self.actor_optimizer.zero_grad()
        self.current_global_step += 1  # Increment step counter for plotting
        return metrics
