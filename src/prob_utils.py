"""
Probabilistic representation and free-energy relevance for Composed Image Retrieval.

Each query / candidate is represented as a diagonal Gaussian N(mu, diag(sigma^2)).
Free energy is defined as the KL divergence:

    F(q, c) = D_KL( q || c )
    s(q, c) = -F(q, c)                     (relevance: larger = more relevant)

This replaces the dot product used by the original CLIP4Cir.

For diagonal Gaussians, D_KL has a closed form:
    D_KL(q || c) = ½ Σ_d [ (μ_q-μ_c)²/σ_c² + σ_q²/σ_c² + log(σ_c²/σ_q²) - 1 ]

The variance head predicts log-variance from frozen CLIP features, clamped to a
narrow range for numerical stability.
"""

import torch
import torch.nn.functional as F
from torch import nn


class VarianceHead(nn.Module):
    """Predicts per-dimension log-variance for a CLIP feature vector."""

    def __init__(self, feature_dim: int, hidden_dim: int = None,
                 log_var_min: float = -3.0, log_var_max: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Start near log_var_min so early training behaves close to the deterministic baseline
        nn.init.normal_(self.net[0].weight, std=feature_dim ** -0.5)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=hidden_dim ** -0.5)
        nn.init.constant_(self.net[2].bias, log_var_min)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        log_var = self.net(features)
        return torch.clamp(log_var, self.log_var_min, self.log_var_max)


class ProbabilisticHead(nn.Module):
    """Wraps the query-side and candidate-side variance heads."""

    def __init__(self, feature_dim: int, hidden_dim: int = None,
                 log_var_min: float = -3.0, log_var_max: float = 0.0):
        super().__init__()
        self.query_var = VarianceHead(feature_dim, hidden_dim, log_var_min, log_var_max)
        self.target_var = VarianceHead(feature_dim, hidden_dim, log_var_min, log_var_max)

    def query_log_var(self, query_features: torch.Tensor) -> torch.Tensor:
        return self.query_var(query_features)

    def target_log_var(self, target_features: torch.Tensor) -> torch.Tensor:
        return self.target_var(target_features)


# ---------------------------------------------------------------------------
#  Free energy via closed-form KL divergence of diagonal Gaussians
# ---------------------------------------------------------------------------

def _kl_diag(query_mu: torch.Tensor, query_log_var: torch.Tensor,
             target_mu: torch.Tensor, target_log_var: torch.Tensor) -> torch.Tensor:
    """
    D_KL( q || c ) for diagonal Gaussians, computed as (N, M) pairwise matrix.

    D_KL = ½ Σ_d [ (μ_q-μ_c)² / σ_c² + σ_q²/σ_c² + log(σ_c²/σ_q²) - 1 ]

    Evaluated WITHOUT materialising the (N, M, D) tensor by expanding the squared
    Mahalanobis term into three separate (N, M) matrix products.
    """
    D = target_mu.shape[-1]

    sigma_q_sq = torch.exp(query_log_var)                    # [N, D]
    sigma_c_sq = torch.exp(target_log_var)                  # [M, D]
    inv_sigma_c_sq = torch.exp(-target_log_var)             # [M, D]

    # --- mu term: Σ_d (μ_q - μ_c)² / σ_c²  ---
    mu_q_sq = query_mu * query_mu                            # [N, D]
    term_a = mu_q_sq @ inv_sigma_c_sq.T                      # Σ μ_q² / σ_c²      [N, M]
    mu_c_over_sig2 = target_mu * inv_sigma_c_sq              # [M, D]
    term_b = torch.sum(target_mu * mu_c_over_sig2, dim=-1)   # Σ μ_c² / σ_c²      [M]
    term_c = query_mu @ mu_c_over_sig2.T                     # Σ μ_q·μ_c / σ_c²   [N, M]
    mu_term = term_a + term_b.unsqueeze(0) - 2.0 * term_c    # [N, M]

    # --- variance terms ---
    ratio_term = sigma_q_sq @ inv_sigma_c_sq.T               # Σ σ_q² / σ_c²      [N, M]
    log_sum_c = torch.sum(target_log_var, dim=-1)            # Σ log σ_c²          [M]
    log_sum_q = torch.sum(query_log_var,   dim=-1)           # Σ log σ_q²          [N]
    log_term = log_sum_c.unsqueeze(0) - log_sum_q.unsqueeze(1)  # Σ log(σ_c²/σ_q²) [N, M]

    per_pair = 0.5 * (mu_term + ratio_term + log_term - D)
    return per_pair


def negative_free_energy(query_mu: torch.Tensor, query_log_var: torch.Tensor,
                         target_mu: torch.Tensor, target_log_var: torch.Tensor,
                         scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    """
    Relevance = - D_KL(q || c), scaled and shifted for the cross-entropy loss.

    :return: [N, M] logits where larger values mean more relevant
    """
    kl = _kl_diag(query_mu, query_log_var, target_mu, target_log_var)
    return -scale * kl + shift


def free_energy_distance(query_mu: torch.Tensor, query_log_var: torch.Tensor,
                         target_mu: torch.Tensor, target_log_var: torch.Tensor) -> torch.Tensor:
    """
    Free energy = D_KL(q || c).  Smaller means more relevant.
    Used for ranking at evaluation time.
    """
    return _kl_diag(query_mu, query_log_var, target_mu, target_log_var)


# ---------------------------------------------------------------------------
#  CoPE distance (from ACL25-CoPE / ClosedFormSampledDistanceLoss)
#    dist(q, c) = ||mu_q - mu_c||^2 + ||sigma_q - sigma_c||^2 + 2D * sigma_bar_q * sigma_bar_c
#  Smaller means more relevant.
# ---------------------------------------------------------------------------

def cope_distance(query_mu: torch.Tensor, query_log_var: torch.Tensor,
                  target_mu: torch.Tensor, target_log_var: torch.Tensor) -> torch.Tensor:
    """
    CoPE probabilistic distance for diagonal Gaussians, [N, M] pairwise matrix.
    Implemented without materialising the (N, M, D) tensor.
    """
    D = target_mu.shape[-1]

    sigma_q = torch.sqrt(torch.exp(query_log_var))                # [N, D]
    sigma_c = torch.sqrt(torch.exp(target_log_var))               # [M, D]

    # --- ||mu_q - mu_c||^2  (Euclidean, expanded via (a-b)^2 = a^2 + b^2 - 2ab) ---
    mu_q_sq = torch.sum(query_mu * query_mu, dim=-1)              # [N]
    mu_c_sq = torch.sum(target_mu * target_mu, dim=-1)            # [M]
    mu_dot = query_mu @ target_mu.T                               # [N, M]
    mu_term = mu_q_sq.unsqueeze(1) + mu_c_sq.unsqueeze(0) - 2.0 * mu_dot   # [N, M]

    # --- ||sigma_q - sigma_c||^2  (per-dim, sum) ---
    sigma_q_sq = torch.sum(sigma_q * sigma_q, dim=-1)             # [N]
    sigma_c_sq = torch.sum(sigma_c * sigma_c, dim=-1)             # [M]
    sigma_dot = sigma_q @ sigma_c.T                               # [N, M]
    sigma_diff_term = sigma_q_sq.unsqueeze(1) + sigma_c_sq.unsqueeze(0) - 2.0 * sigma_dot   # [N, M]

    # --- 2D * sigma_bar_q * sigma_bar_c  ---
    sigma_q_bar = sigma_q.mean(dim=-1)                            # [N]
    sigma_c_bar = sigma_c.mean(dim=-1)                            # [M]
    cross_term = 2.0 * D * sigma_q_bar.unsqueeze(1) * sigma_c_bar.unsqueeze(0)   # [N, M]

    return mu_term + sigma_diff_term + cross_term


def negative_cope_logits(query_mu: torch.Tensor, query_log_var: torch.Tensor,
                         target_mu: torch.Tensor, target_log_var: torch.Tensor,
                         scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    """Relevance = -scale * CoPE distance + shift. Larger means more relevant."""
    d = cope_distance(query_mu, query_log_var, target_mu, target_log_var)
    return -scale * d + shift


# ---------------------------------------------------------------------------
#  Dispatch: map a relevance name to (logits fn for training, distance fn for ranking)
# ---------------------------------------------------------------------------

def get_prob_distances(relevance: str):
    """
    Return (logits_fn, distance_fn) for a probabilistic relevance name.
    Logits fn: (mu_q, lv_q, mu_c, lv_c, scale) -> [N, M] logits (larger = more relevant).
    Distance fn: (mu_q, lv_q, mu_c, lv_c) -> [N, M] distance (smaller = more relevant).
    """
    if relevance == 'cope':
        return negative_cope_logits, cope_distance
    # default: free_energy (KL)
    return negative_free_energy, free_energy_distance


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------

def save_prob_head(path, epoch: int, head: ProbabilisticHead):
    torch.save({"epoch": epoch, "ProbabilisticHead": head.state_dict()}, str(path))


def load_prob_head(path, feature_dim: int, device, hidden_dim: int = None) -> ProbabilisticHead:
    head = ProbabilisticHead(feature_dim, hidden_dim).to(device)
    state = torch.load(path, map_location=device)
    head.load_state_dict(state["ProbabilisticHead"])
    head.eval()
    return head
