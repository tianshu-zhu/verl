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
        self.current_global_step = 0  # Track training step for IcePop plotting

    def _compute_icepop_mask(self, inf_log_probs, old_log_probs, alpha=0.5, beta=2.0):
        """
        Compute IcePop double-sided masking to filter out noisy gradient updates.

        The masking function M(k) where k = p_train / p_infer:
        M(k) = { k  if k ∈ [α, β]
               { 0  otherwise

        Args:
            inf_log_probs: Log probabilities from inference engine (vLLM), shape [batch_size, response_length]
            old_log_probs: Log probabilities from training engine (FSDP old policy), shape [batch_size, response_length]
            alpha: Lower bound for probability ratio (default: 0.5)
            beta: Upper bound for probability ratio (default: 2.0)

        Returns:
            icepop_mask: Binary mask tensor, 1.0 for healthy updates, 0.0 for clipped tokens
            clipping_stats: Dictionary with clipping statistics for logging
        """
        with torch.no_grad():
            # Convert log probabilities to probabilities
            # p_train (old_log_probs) / p_infer (inf_log_probs) = exp(log_p_train - log_p_infer)
            log_ratio = old_log_probs - inf_log_probs
            prob_ratio = torch.exp(log_ratio)

            # Apply double-sided clipping: keep only ratios in [alpha, beta]
            # Mask out tokens where:
            # 1. prob_ratio < alpha (training prob << inference prob - huge divergence)
            # 2. prob_ratio > beta (training prob >> inference prob - overconfident)
            icepop_mask = ((prob_ratio >= alpha) & (prob_ratio <= beta)).float()

            # Compute clipping statistics
            total_tokens = icepop_mask.numel()
            clipped_tokens = (icepop_mask == 0.0).sum().item()
            clipping_ratio = clipped_tokens / total_tokens if total_tokens > 0 else 0.0

            # Analyze clipped vs non-clipped tokens
            clipped_lower = (prob_ratio < alpha).sum().item()  # Training prob too low
            clipped_upper = (prob_ratio > beta).sum().item()   # Training prob too high

            # Calculate mean probability ratios for diagnostics
            valid_ratio_mean = prob_ratio[icepop_mask == 1.0].mean().item() if (icepop_mask == 1.0).any() else 0.0
            clipped_ratio_mean = prob_ratio[icepop_mask == 0.0].mean().item() if (icepop_mask == 0.0).any() else 0.0

            clipping_stats = {
                'icepop/clipping_ratio': clipping_ratio,
                'icepop/clipped_tokens': clipped_tokens,
                'icepop/total_tokens': total_tokens,
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

    def _plot_logprobs_comparison(self, inf_log_probs, old_log_probs, response_mask, global_step):
        """
        Generate line charts comparing inference log probs (vLLM) vs training log probs (FSDP old policy).
        Saves charts to default_local_dir/logprobs_comparison/step_{global_step}/.

        Args:
            inf_log_probs: Tensor of shape (batch_size, response_length) - vLLM inference log probs
            old_log_probs: Tensor of shape (batch_size, response_length) - FSDP old policy log probs
            response_mask: Tensor of shape (batch_size, response_length) - mask for valid tokens
            global_step: Current training step for organizing outputs
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for server environments
            import matplotlib.pyplot as plt
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
        # Get default_local_dir from actor config (passed via ++actor_rollout_ref.actor.default_local_dir in train.sh)
        default_local_dir = getattr(self.config, 'default_local_dir', '.')
        print(f"[LogProbs-Viz] Using default_local_dir: {default_local_dir}")

        viz_dir = os.path.join(default_local_dir, "logprobs_comparison", f"step_{global_step:03d}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[LogProbs-Viz] Saving plots to: {viz_dir}")

        # Move tensors to CPU and convert to numpy
        inf_log_probs_np = inf_log_probs.detach().cpu().numpy()
        old_log_probs_np = old_log_probs.detach().cpu().numpy()
        response_mask_np = response_mask.detach().cpu().numpy()

        batch_size = inf_log_probs_np.shape[0]

        # Plot up to 5 trajectories to avoid overwhelming output
        num_plots = min(batch_size, 5)

        for traj_idx in range(num_plots):
            # Two filtering strategies:
            # 1. STRICT (current): response_mask > 0 AND inf_log_probs != 0 (only assistant tokens)
            # 2. SIMPLE (previous): response_mask > 0 (includes environment tokens)

            # Try STRICT filtering first
            strict_valid_mask = (response_mask_np[traj_idx] > 0) & (inf_log_probs_np[traj_idx] != 0.0)
            # Also try SIMPLE filtering (like previous implementation)
            simple_valid_mask = response_mask_np[traj_idx] > 0

            # Use STRICT by default, but report both
            valid_mask = strict_valid_mask
            valid_positions = np.where(valid_mask)[0]

            print(f"\n[LogProbs-Viz-Filtering] Trajectory {traj_idx}:")
            print(f"  STRICT filtering (response_mask>0 & inf!=0): {strict_valid_mask.sum()} tokens")
            print(f"  SIMPLE filtering (response_mask>0): {simple_valid_mask.sum()} tokens")
            print(f"  Difference: {simple_valid_mask.sum() - strict_valid_mask.sum()} tokens (likely environment)")

            if len(valid_positions) == 0:
                continue  # Skip empty trajectories

            # Extract valid log probs for this trajectory
            inf_log_probs_valid = inf_log_probs_np[traj_idx][valid_mask]
            old_log_probs_valid = old_log_probs_np[traj_idx][valid_mask]

            # CRITICAL: Convert to probabilities for proper comparison (as IcePop does)
            # IcePop compares probability ratios, not log probability ratios
            inf_probs_valid = np.exp(inf_log_probs_valid)
            old_probs_valid = np.exp(old_log_probs_valid)

            # EXTENSIVE DEBUG: Print raw statistics and sample comparisons
            print(f"\n{'='*80}")
            print(f"[LogProbs-Viz-DEBUG] Trajectory {traj_idx}: {len(valid_positions)} model-generated tokens")
            print(f"{'='*80}")

            # Print LOG PROB statistics
            print(f"[LogProbs-Viz-DEBUG] Inference (vLLM) LOG PROB Statistics:")
            print(f"  Range: [{inf_log_probs_valid.min():.6f}, {inf_log_probs_valid.max():.6f}]")
            print(f"  Mean: {inf_log_probs_valid.mean():.6f}, Std: {inf_log_probs_valid.std():.6f}")
            print(f"  Median: {np.median(inf_log_probs_valid):.6f}")
            print(f"  Values > -0.1: {(inf_log_probs_valid > -0.1).sum()} / {len(inf_log_probs_valid)} ({100*(inf_log_probs_valid > -0.1).sum()/len(inf_log_probs_valid):.2f}%)")
            print(f"  Values < -10: {(inf_log_probs_valid < -10).sum()} / {len(inf_log_probs_valid)} ({100*(inf_log_probs_valid < -10).sum()/len(inf_log_probs_valid):.2f}%)")

            print(f"\n[LogProbs-Viz-DEBUG] Training (FSDP) LOG PROB Statistics:")
            print(f"  Range: [{old_log_probs_valid.min():.6f}, {old_log_probs_valid.max():.6f}]")
            print(f"  Mean: {old_log_probs_valid.mean():.6f}, Std: {old_log_probs_valid.std():.6f}")
            print(f"  Median: {np.median(old_log_probs_valid):.6f}")
            print(f"  Values > -0.1: {(old_log_probs_valid > -0.1).sum()} / {len(old_log_probs_valid)} ({100*(old_log_probs_valid > -0.1).sum()/len(old_log_probs_valid):.2f}%)")
            print(f"  Values < -10: {(old_log_probs_valid < -10).sum()} / {len(old_log_probs_valid)} ({100*(old_log_probs_valid < -10).sum()/len(old_log_probs_valid):.2f}%)")

            # Print PROBABILITY statistics (this is what IcePop actually uses!)
            print(f"\n[LogProbs-Viz-DEBUG] Inference (vLLM) PROBABILITY Statistics:")
            print(f"  Range: [{inf_probs_valid.min():.6f}, {inf_probs_valid.max():.6f}]")
            print(f"  Mean: {inf_probs_valid.mean():.6f}, Std: {inf_probs_valid.std():.6f}")
            print(f"  Median: {np.median(inf_probs_valid):.6f}")

            print(f"\n[LogProbs-Viz-DEBUG] Training (FSDP) PROBABILITY Statistics:")
            print(f"  Range: [{old_probs_valid.min():.6f}, {old_probs_valid.max():.6f}]")
            print(f"  Mean: {old_probs_valid.mean():.6f}, Std: {old_probs_valid.std():.6f}")
            print(f"  Median: {np.median(old_probs_valid):.6f}")

            # Print correlations (both log prob and probability space)
            logprob_correlation = np.corrcoef(inf_log_probs_valid, old_log_probs_valid)[0,1]
            prob_correlation = np.corrcoef(inf_probs_valid, old_probs_valid)[0,1]
            print(f"\n[LogProbs-Viz-DEBUG] Correlation (Log Prob Space): {logprob_correlation:.6f}")
            print(f"[LogProbs-Viz-DEBUG] Correlation (Probability Space): {prob_correlation:.6f} ← THIS IS WHAT MATTERS!")

            # Use probability correlation as the main correlation metric
            correlation = prob_correlation

            # Print sample token-by-token comparisons (first 20 and last 20)
            print(f"\n[LogProbs-Viz-DEBUG] Sample Token Comparisons (first 20 tokens):")
            print(f"  {'Pos':<6} {'Inf_Prob':<12} {'Old_Prob':<12} {'Prob_Ratio':<12} {'Inf_LogP':<12} {'Old_LogP':<12}")
            print(f"  {'-'*72}")
            for i in range(min(20, len(inf_probs_valid))):
                pos = valid_positions[i]
                inf_p = inf_probs_valid[i]
                old_p = old_probs_valid[i]
                inf_lp = inf_log_probs_valid[i]
                old_lp = old_log_probs_valid[i]
                # Compute probability ratio: P_old / P_inf (this is what IcePop uses)
                prob_ratio = old_p / inf_p if inf_p > 0 else float('inf')
                # Mark if ratio is outside IcePop bounds [0.5, 2.0]
                flag = "⚠️" if prob_ratio < 0.5 or prob_ratio > 2.0 else "  "
                print(f"{flag} {pos:<6} {inf_p:<12.6f} {old_p:<12.6f} {prob_ratio:<12.6f} {inf_lp:<12.4f} {old_lp:<12.4f}")

            if len(inf_probs_valid) > 40:
                print(f"  ... (skipping middle tokens) ...")
                print(f"\n[LogProbs-Viz-DEBUG] Sample Token Comparisons (last 20 tokens):")
                print(f"  {'Pos':<6} {'Inf_Prob':<12} {'Old_Prob':<12} {'Prob_Ratio':<12} {'Inf_LogP':<12} {'Old_LogP':<12}")
                print(f"  {'-'*72}")
                for i in range(max(0, len(inf_probs_valid)-20), len(inf_probs_valid)):
                    pos = valid_positions[i]
                    inf_p = inf_probs_valid[i]
                    old_p = old_probs_valid[i]
                    inf_lp = inf_log_probs_valid[i]
                    old_lp = old_log_probs_valid[i]
                    prob_ratio = old_p / inf_p if inf_p > 0 else float('inf')
                    flag = "⚠️" if prob_ratio < 0.5 or prob_ratio > 2.0 else "  "
                    print(f"{flag} {pos:<6} {inf_p:<12.6f} {old_p:<12.6f} {prob_ratio:<12.6f} {inf_lp:<12.4f} {old_lp:<12.4f}")

            # CRITICAL: Identify tokens with extreme negative old_log_probs
            extreme_negative_indices = np.where(old_log_probs_valid < -10)[0]
            if len(extreme_negative_indices) > 0:
                print(f"\n[LogProbs-Viz-CRITICAL] Found {len(extreme_negative_indices)} tokens with EXTREME negative old_log_probs (<-10)!")
                print(f"  Positions: {[valid_positions[i] for i in extreme_negative_indices[:10]]}")  # First 10
                print(f"  Old_LogProbs: {[old_log_probs_valid[i] for i in extreme_negative_indices[:10]]}")
                print(f"  Inf_LogProbs: {[inf_log_probs_valid[i] for i in extreme_negative_indices[:10]]}")

                # Check the ORIGINAL tensors to see if these are environment or assistant tokens
                # If inf_log_probs was 0 originally, it's environment token (shouldn't be included!)
                original_inf_values = [inf_log_probs_np[traj_idx][valid_positions[i]] for i in extreme_negative_indices[:10]]
                print(f"  Original Inf_LogProbs (before exp): {original_inf_values}")

                # Check if we're accidentally including environment tokens
                if any(v == 0.0 for v in original_inf_values):
                    print(f"  ⚠️ WARNING: Some extreme negatives have inf_log_prob=0 ORIGINALLY!")
                    print(f"      This suggests environment tokens are being included!")
                else:
                    print(f"  ✓ All extreme negatives have non-zero inf_log_probs")
                    print(f"  This suggests old_log_probs computation issue for ASSISTANT tokens!")

            # Check for potential issues
            print(f"\n[LogProbs-Viz-DEBUG] Potential Issues Check:")

            # Compute probability ratios for analysis
            prob_ratios = old_probs_valid / np.maximum(inf_probs_valid, 1e-10)  # Avoid division by zero

            # Issue 1: IcePop clipping statistics
            clipped_lower = (prob_ratios < 0.5).sum()
            clipped_upper = (prob_ratios > 2.0).sum()
            clipped_total = clipped_lower + clipped_upper
            clipping_ratio = clipped_total / len(prob_ratios)
            print(f"  IcePop would clip: {clipped_total}/{len(prob_ratios)} tokens ({100*clipping_ratio:.2f}%)")
            print(f"    Lower bound (ratio < 0.5): {clipped_lower} tokens")
            print(f"    Upper bound (ratio > 2.0): {clipped_upper} tokens")
            if clipping_ratio > 0.1:
                print(f"  ⚠️  WARNING: High clipping ratio (>{10:.1f}%) - significant mismatch!")

            # Issue 2: Very low probability correlation
            if prob_correlation < 0.5:
                print(f"  ⚠️  WARNING: Low probability correlation ({prob_correlation:.4f})")
                print(f"      Expected high correlation on model-generated tokens!")

            # Issue 3: Check if shifted correlation is higher (alignment issue)
            if len(inf_probs_valid) > 1:
                shifted_corr = np.corrcoef(inf_probs_valid[:-1], old_probs_valid[1:])[0,1]
                if shifted_corr > prob_correlation + 0.1:
                    print(f"  ⚠️  WARNING: Shifted correlation ({shifted_corr:.4f}) > Normal correlation ({prob_correlation:.4f})")
                    print(f"      This suggests a token alignment/indexing issue!")

            # Issue 4: Probability values out of range
            if (inf_probs_valid > 1.0).any() or (old_probs_valid > 1.0).any():
                print(f"  ⚠️  WARNING: Probability values > 1.0 found!")
                print(f"      Inf: {(inf_probs_valid > 1.0).sum()}, Old: {(old_probs_valid > 1.0).sum()}")
                print(f"      This indicates log probs were not properly converted!")

            # Issue 5: All probabilities near 1 (too confident)
            if (inf_probs_valid > 0.99).sum() > len(inf_probs_valid) * 0.9:
                print(f"  ⚠️  WARNING: {(inf_probs_valid > 0.99).sum()} inference probabilities > 0.99")
                print(f"      Model is extremely confident - unusual!")

            print(f"{'='*80}\n")

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Subplot 1: Overlay comparison - PROBABILITIES (0 to 1)
            ax1.plot(valid_positions, inf_probs_valid, 'b-', label='Inference (vLLM)', alpha=0.8, linewidth=2.0, marker='o', markersize=3)
            ax1.plot(valid_positions, old_probs_valid, 'r--', label='Training (FSDP Old Policy)', alpha=0.8, linewidth=2.0, marker='s', markersize=3)

            # Set y-axis to probability range (0 to 1)
            ax1.set_ylim(0, 1)

            # Highlight model-generated token regions
            all_positions = np.arange(inf_log_probs_np.shape[1])
            non_model_mask = (inf_log_probs_np[traj_idx] == 0.0) | (response_mask_np[traj_idx] == 0)
            non_model_positions = all_positions[non_model_mask]

            # Shade non-model-generated regions
            for pos in non_model_positions:
                if pos <= valid_positions[-1]:  # Only shade within plot range
                    ax1.axvspan(pos-0.5, pos+0.5, color='gray', alpha=0.2, label='Non-Model Token' if pos == non_model_positions[0] else '')

            ax1.set_xlabel('Token Position', fontsize=12)
            ax1.set_ylabel('Probability (0 to 1)', fontsize=12)
            ax1.set_title(f'Trajectory {traj_idx}: Inference vs Training Probabilities (Step {global_step})\n' +
                         f'{len(valid_positions)} Model-Generated Tokens (Correlation: {correlation:.4f})',
                         fontsize=13)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.4, linestyle='--')

            # Compute difference for statistics
            diff = inf_probs_valid - old_probs_valid

            # Subplot 2: Scatter plot for direct comparison
            ax2.scatter(inf_probs_valid, old_probs_valid, alpha=0.5, s=30, c='purple', edgecolors='black', linewidth=0.5)

            # Add perfect correlation line (y=x)
            ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Correlation (y=x)', alpha=0.7)

            # Add IcePop bounds
            # Ratio = 0.5 means old = 0.5 * inf, ratio = 2.0 means old = 2.0 * inf
            x_vals = np.linspace(0, 1, 100)
            ax2.plot(x_vals, 0.5 * x_vals, 'g--', linewidth=1.5, label='IcePop Lower Bound (ratio=0.5)', alpha=0.5)
            ax2.plot(x_vals, 2.0 * x_vals, 'orange', linestyle='--', linewidth=1.5, label='IcePop Upper Bound (ratio=2.0)', alpha=0.5)

            ax2.set_xlabel('Inference Probability (vLLM)', fontsize=12)
            ax2.set_ylabel('Training Probability (FSDP)', fontsize=12)
            ax2.set_title(f'Direct Comparison: Inference vs Training Probabilities\n(Correlation: {correlation:.4f})',
                         fontsize=13)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.4, linestyle='--')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_aspect('equal', adjustable='box')

            # Add summary statistics as text box
            stats_text = (
                f"Statistics (Model-Generated Tokens, PROBABILITY SPACE):\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Inference: mean={np.mean(inf_probs_valid):.6f}, std={np.std(inf_probs_valid):.6f}\n"
                f"           min={np.min(inf_probs_valid):.6f}, max={np.max(inf_probs_valid):.6f}\n"
                f"Training:  mean={np.mean(old_probs_valid):.6f}, std={np.std(old_probs_valid):.6f}\n"
                f"           min={np.min(old_probs_valid):.6f}, max={np.max(old_probs_valid):.6f}\n"
                f"Difference: mean={np.mean(diff):.6f}, std={np.std(diff):.6f}\n"
                f"Correlation: {correlation:.4f}\n"
                f"Prob Ratio: mean={np.mean(prob_ratios):.4f}, std={np.std(prob_ratios):.4f}\n"
                f"IcePop Clipping: {clipped_total}/{len(prob_ratios)} ({100*clipping_ratio:.2f}%)\n"
                f"Valid Tokens: {len(valid_positions)} / {inf_log_probs_np.shape[1]}"
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
        self._plot_batch_summary(inf_log_probs_np, old_log_probs_np, response_mask_np, viz_dir, global_step)

    def _plot_batch_summary(self, inf_log_probs_np, old_log_probs_np, response_mask_np, viz_dir, global_step):
        """Generate summary statistics plot for entire batch."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        batch_size = inf_log_probs_np.shape[0]

        # Collect per-trajectory statistics
        traj_means_inf = []
        traj_means_old = []
        traj_diffs = []
        traj_correlations = []

        for traj_idx in range(batch_size):
            # CRITICAL FIX: Only use model-generated tokens (where inf_log_probs != 0)
            valid_mask = (response_mask_np[traj_idx] > 0) & (inf_log_probs_np[traj_idx] != 0.0)
            if not valid_mask.any():
                continue

            inf_log_valid = inf_log_probs_np[traj_idx][valid_mask]
            old_log_valid = old_log_probs_np[traj_idx][valid_mask]

            # Convert to probabilities for comparison (as IcePop does)
            inf_valid = np.exp(inf_log_valid)
            old_valid = np.exp(old_log_valid)

            traj_means_inf.append(np.mean(inf_valid))
            traj_means_old.append(np.mean(old_valid))
            traj_diffs.append(np.mean(inf_valid - old_valid))

            if len(inf_valid) > 1:  # Need at least 2 points for correlation
                # Compute correlation in PROBABILITY space
                corr = np.corrcoef(inf_valid, old_valid)[0, 1]
                traj_correlations.append(corr)

        # Create 2x2 subplot grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Scatter of mean probabilities
        ax1.scatter(traj_means_inf, traj_means_old, alpha=0.6, s=50)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Agreement (y=x)')
        ax1.set_xlabel('Inference Mean Probability')
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
        ax2.set_xlabel('Mean Difference (Inference - Training)')
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
            ax3.set_title('Distribution of Inf-Train Probability Correlations')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics table
        ax4.axis('off')
        summary_stats = [
            ['Metric', 'Inference', 'Training', 'Difference'],
            ['Mean', f'{np.mean(traj_means_inf):.4f}', f'{np.mean(traj_means_old):.4f}',
             f'{np.mean(traj_diffs):.4f}'],
            ['Std', f'{np.std(traj_means_inf):.4f}', f'{np.std(traj_means_old):.4f}',
             f'{np.std(traj_diffs):.4f}'],
            ['Min', f'{np.min(traj_means_inf):.4f}', f'{np.min(traj_means_old):.4f}',
             f'{np.min(traj_diffs):.4f}'],
            ['Max', f'{np.max(traj_means_inf):.4f}', f'{np.max(traj_means_old):.4f}',
             f'{np.max(traj_diffs):.4f}'],
            ['', '', '', ''],
            ['Avg Correlation', '', '', f'{np.mean(traj_correlations) if traj_correlations else "N/A"}'],
            ['Batch Size', '', '', f'{batch_size}'],
            ['Valid Trajectories', '', '', f'{len(traj_means_inf)}'],
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
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")
            
        if 'traj_mask' in data.batch:
            select_keys.append('traj_mask')
        # Add inference log probs from vLLM for IcePop
        if 'inf_log_probs' in data.batch:
            select_keys.append('inf_log_probs')
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[IcePop] dp_actor.py: update_policy - inf_log_probs found in data.batch, shape={data.batch['inf_log_probs'].shape}")

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

                    # IcePop: Apply double-sided masking if inf_log_probs available
                    # Hard-coded hyperparameters based on paper recommendations
                    USE_ICEPOP = True  # Enable/disable IcePop masking
                    ICEPOP_ALPHA = 0.5  # Lower bound for prob ratio (p_train/p_infer)
                    ICEPOP_BETA = 2.0   # Upper bound for prob ratio (p_train/p_infer)

                    icepop_stats = {}

                    if USE_ICEPOP and 'inf_log_probs' in model_inputs:
                        inf_log_probs = model_inputs['inf_log_probs']

                        # Save original response_mask for visualization (before IcePop modification)
                        original_response_mask = response_mask.clone()

                        # Compute IcePop mask by comparing inf_log_probs and old_log_probs
                        # This filters out noisy gradient updates where training/inference diverge
                        icepop_mask, icepop_stats = self._compute_icepop_mask(
                            inf_log_probs=inf_log_probs,
                            old_log_probs=old_log_prob,
                            alpha=ICEPOP_ALPHA,
                            beta=ICEPOP_BETA
                        )

                        # Combine IcePop mask with response_mask (element-wise AND)
                        # response_mask: filters padding/invalid tokens
                        # icepop_mask: filters tokens with excessive prob discrepancy
                        response_mask = response_mask * icepop_mask

                        # Print detailed debug statistics
                        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                            print(f"[IcePop] Step {self.current_global_step} - "
                                  f"α={ICEPOP_ALPHA}, β={ICEPOP_BETA}, "
                                  f"clipping_ratio={icepop_stats['icepop/clipping_ratio']:.4f} "
                                  f"({icepop_stats['icepop/clipped_tokens']}/{icepop_stats['icepop/total_tokens']} tokens), "
                                  f"lower={icepop_stats['icepop/clipped_lower']}, "
                                  f"upper={icepop_stats['icepop/clipped_upper']}, "
                                  f"prob_ratio_mean={icepop_stats['icepop/prob_ratio_mean']:.4f}±{icepop_stats['icepop/prob_ratio_std']:.4f}")

                        # Generate comparison visualization (only once per step on rank 0)
                        # IMPORTANT: Use original_response_mask (before IcePop clipping) for accurate visualization
                        if epoch_idx == 0 and batch_idx == 0 and micro_batch_idx == 0:
                            self._plot_logprobs_comparison(
                                inf_log_probs=inf_log_probs,
                                old_log_probs=old_log_prob,
                                response_mask=original_response_mask,
                                global_step=self.current_global_step
                            )

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
        self.actor_optimizer.zero_grad()
        self.current_global_step += 1  # Increment step counter for IcePop plotting
        return metrics
