"""
CmIR: Causal modality-Invariant Representation learning.

This module implements the model described in
"Learning Invariant Modality Representation for Robust Multimodal Learning
from a Causal Inference Perspective" (ACL 2026).

For every modality m, an encoder g_m disentangles the unimodal feature X_m
into
    - z_inv^m : invariant (causal) representation,
    - z_spu^m : environment-specific (spurious) representation,
a decoder r_m reconstructs X_m from the pair, and a shared predictor h fuses
the invariant representations of all modalities to predict the target Y.

The training objective (Eq. 9) is
    L = L_pred + lambda1 * R_inv + lambda2 * R_dec + lambda3 * R_rec
where
    R_inv : invariance constraint across virtual environments (Eq. 4),
    R_dec : mutual-information / orthogonality constraint (Eq. 5-6),
    R_rec : reconstruction constraint (Eq. 8).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import BertPooler
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)

import global_configs
from global_configs import DEVICE
from modules.transformer import TransformerEncoder

# Reproducibility: make cuDNN deterministic.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


class CmIRModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM,
            global_configs.ACOUSTIC_DIM,
            global_configs.VISUAL_DIM,
        )
        self.config = config
        self.pooler = BertPooler(config)

        # The pretrained language model path is configurable (--model).
        plm_path = getattr(multimodal_config, "model", "microsoft/deberta-v3-base")
        self.model = DebertaV2Model.from_pretrained(plm_path).to(DEVICE)

        # Shared representation dimension d (paper: Shared Dimensionality).
        self.d_l = multimodal_config.share_dim
        self.attn_dropout = multimodal_config.drop_prob
        self.unimodal_dropout = multimodal_config.dropout_unimodal
        self.unimodal_head = multimodal_config.transformer_head
        self.label_dim = 1

        # ---- Unimodal projection layers (Appendix B, Eq. 11-13) ----
        kernel = multimodal_config.kernel_size
        self.proj_a = nn.Conv1d(
            ACOUSTIC_DIM, self.d_l, kernel_size=kernel, stride=1,
            padding=(kernel - 1) // 2, bias=False,
        )
        self.proj_l = nn.Linear(TEXT_DIM, self.d_l, bias=False)
        self.proj_v = nn.Conv1d(
            VISUAL_DIM, self.d_l, kernel_size=kernel, stride=1,
            padding=(kernel - 1) // 2, bias=False,
        )

        # Unimodal transformer encoders for acoustic / visual streams.
        self.transa = self.get_network("a", layers=multimodal_config.transformer_layer)
        self.transv = self.get_network("v", layers=multimodal_config.transformer_layer)

        self.LayerNorm_l = nn.LayerNorm(self.d_l)
        self.LayerNorm_a = nn.LayerNorm(self.d_l)
        self.LayerNorm_v = nn.LayerNorm(self.d_l)

        # ---- Per-modality disentangling encoders g_m and decoders r_m ----
        self.encoder_l = self._create_encoder(self.d_l, multimodal_config.inter_dim, self.d_l)
        self.encoder_a = self._create_encoder(self.d_l, multimodal_config.inter_dim, self.d_l)
        self.encoder_v = self._create_encoder(self.d_l, multimodal_config.inter_dim, self.d_l)
        self.decoder_l = self._create_decoder(self.d_l * 2, multimodal_config.inter_dim, self.d_l)
        self.decoder_a = self._create_decoder(self.d_l * 2, multimodal_config.inter_dim, self.d_l)
        self.decoder_v = self._create_decoder(self.d_l * 2, multimodal_config.inter_dim, self.d_l)

        # ---- Fusion and predictor h (only invariant reps are used) ----
        self.fusion = nn.Sequential(
            nn.Linear(self.d_l * 3, multimodal_config.inter_dim),
            nn.ReLU(),
            nn.Linear(multimodal_config.inter_dim, self.d_l),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.d_l, multimodal_config.inter_dim),
            nn.ReLU(),
            nn.Linear(multimodal_config.inter_dim, 1),
        )

        # ---- CmIR constraint hyper-parameters ----
        # alpha in Eq. 6 / Table 7 ("Weight alpha in Mutual Information Constraint").
        self.weight_decouple = multimodal_config.weight_decouple
        # K in Eq. 3-4 (number of virtual environments).
        self.num_of_envs = multimodal_config.num_of_envs
        # alpha^(1) in Eq. 3 (base noise coefficient).
        self.noise_scale = multimodal_config.noise_scale

        # ---- OOD test-time corruption (Eq. 14; Table 4 experiments) ----
        # noise_type: 0 = no corruption, 1 = Gaussian, 2 = Laplace / random erasing.
        self.noise_type = multimodal_config.noise_type
        self.percent = multimodal_config.percent      # Laplace / erasing rate NR
        self.percent2 = multimodal_config.percent2    # Gaussian rate NR
        
        self.noise_experiment = multimodal_config.noise_experiment

        self.reconstruction_loss = nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    def _create_encoder(self, input_dim, hidden_dim, out_dim):
        """Encoder g_m: X_m -> (z_inv^m, z_spu^m)."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2),
        )

    def _create_decoder(self, input_dim, hidden_dim, output_dim):
        """Decoder r_m: (z_inv^m, z_spu^m) -> X_hat_m."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def get_network(self, self_type="l", layers=3):
        if self_type in ["l", "a", "v"]:
            embed_dim = self.d_l
        elif self_type == "lav":
            embed_dim = self.d_l * 3
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.unimodal_head,
            layers=layers,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.unimodal_dropout,
            res_dropout=self.unimodal_dropout,
            embed_dropout=self.unimodal_dropout,
            attn_mask=False,
        )

    # ------------------------------------------------------------------ #
    # Constraint losses
    # ------------------------------------------------------------------ #
    def invariance_loss(self, z1, z2):
        """R_inv term for one environment pair (Eq. 4): element-wise L1 distance.

        The regression variant enforces the invariant representations to be
        identical across environments, which is stronger than matching output
        distributions and avoids an extra unimodal predictor (Appendix H).
        """
        return torch.mean(torch.abs(z1 - z2))

    def correlation_loss(self, z_inv, z_spu):
        """R_dec term (Eq. 5-6): orthogonality proxy for I(z_inv; z_spu) ~ 0.

        C = Nor(z_inv)^T Nor(z_spu) / N is the cross-correlation matrix.
        Both diagonal and off-diagonal entries are pushed to zero; alpha weights
        the off-diagonal (same-sample vs cross-sample) term.
        """
        z1_norm = (z_inv - z_inv.mean(0)) / z_inv.std(0)
        z2_norm = (z_spu - z_spu.mean(0)) / z_spu.std(0)

        n = z_inv.size(0)
        c = torch.mm(z1_norm.T, z2_norm) / n

        on_diag = torch.diagonal(c).pow(2).mean()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).mean()
        return on_diag + self.weight_decouple * off_diag

    # ------------------------------------------------------------------ #
    # Test-time corruption (OOD noisy-modality experiments, Eq. 14)
    # ------------------------------------------------------------------ #
    def _apply_test_corruption(self, output_l, output_a, output_v):
        device = output_l.device
        if self.noise_type == 1:
            # Gaussian noise: X' = (1 - NR) X + NR * N, independent per modality.
            output_l = output_l * (1 - self.percent2) + \
                torch.randn_like(output_l) * self.percent2
            output_a = output_a * (1 - self.percent2) + \
                torch.randn_like(output_a) * self.percent2
            output_v = output_v * (1 - self.percent2) + \
                torch.randn_like(output_v) * self.percent2
        elif self.noise_type == 2:
            if np.random.rand() < 0.5:
                # Laplace noise.
                laplace = torch.distributions.Laplace(0, 1)
                output_l = output_l * (1 - self.percent) + \
                    laplace.sample(output_l.shape).to(device) * self.percent
                output_a = output_a * (1 - self.percent) + \
                    laplace.sample(output_a.shape).to(device) * self.percent
                output_v = output_v * (1 - self.percent) + \
                    laplace.sample(output_v.shape).to(device) * self.percent
            else:
                # Random erasing: independently drop a fraction of features.
                def erase(x):
                    mask = torch.ones(x.shape[-1], device=device)
                    n_drop = int(x.shape[-1] * self.percent)
                    idx = np.random.choice(x.shape[-1], n_drop, replace=False)
                    mask[idx] = 0
                    return x * mask

                output_l, output_a, output_v = (
                    erase(output_l), erase(output_a), erase(output_v),
                )
        return output_l, output_a, output_v

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, input_ids, visual, acoustic, label_ids=None,
                feature=False, input_mask=None):
        # ---------------- Unimodal networks (Appendix B) ----------------
        x = self.model(input_ids)[0]              # (bs, seq, text_hidden)
        output_l = self.proj_l(x)                 # (bs, seq, d_l)

        acoustic = self.proj_a(acoustic.transpose(1, 2))   # (bs, d_l, seq)
        visual = self.proj_v(visual.transpose(1, 2))       # (bs, d_l, seq)

        acoustic = self.transa(acoustic.permute(2, 0, 1)).permute(1, 0, 2)
        visual = self.transv(visual.permute(2, 0, 1)).permute(1, 0, 2)

        # Mean pooling over time -> one vector per modality.
        output_l = self.LayerNorm_l(output_l).mean(dim=1)
        output_a = self.LayerNorm_a(acoustic).mean(dim=1)
        output_v = self.LayerNorm_v(visual).mean(dim=1)

        # Optional OOD corruption at test time only.
        if not self.training and self.noise_experiment:
            output_l, output_a, output_v = self._apply_test_corruption(
                output_l, output_a, output_v,
            )

        # ---------------- Disentangling encoders g_m ----------------
        z_l_inv, z_l_sup = self._split(self.encoder_l(output_l))
        z_a_inv, z_a_sup = self._split(self.encoder_a(output_a))
        z_v_inv, z_v_sup = self._split(self.encoder_v(output_v))

        # ---------------- Predictor h (invariant reps only) ----------------
        fusion = self.fusion(torch.cat([z_l_inv, z_a_inv, z_v_inv], dim=-1))
        logits = self.predictor(fusion)

        loss_inv = torch.zeros((), device=logits.device)
        loss_mi = torch.zeros((), device=logits.device)
        loss_recon = torch.zeros((), device=logits.device)

        if self.training:
            # ---------------- Virtual environments (Eq. 3) ----------------
            # Environment e perturbs X with noise coefficient alpha^(e) = e * alpha^(1).
            # The list holds the original (index 0) plus K perturbed variants,
            # giving K + 1 representations per modality.
            total_l = [output_l.unsqueeze(0)]
            total_a = [output_a.unsqueeze(0)]
            total_v = [output_v.unsqueeze(0)]
            scale = self.noise_scale
            for _ in range(self.num_of_envs):
                total_l.append(
                    (output_l + torch.randn_like(output_l) * scale).unsqueeze(0))
                total_a.append(
                    (output_a + torch.randn_like(output_a) * scale).unsqueeze(0))
                total_v.append(
                    (output_v + torch.randn_like(output_v) * scale).unsqueeze(0))
                scale += self.noise_scale

            total_l = torch.cat(total_l, dim=0)   # (K+1, bs, d_l)
            total_a = torch.cat(total_a, dim=0)
            total_v = torch.cat(total_v, dim=0)

            inv_l = self.encoder_l(total_l)[:, :, :self.d_l]
            inv_a = self.encoder_a(total_a)[:, :, :self.d_l]
            inv_v = self.encoder_v(total_v)[:, :, :self.d_l]

            # R_inv: average over ALL K(K+1)/2 environment pairs and 3 modalities.
            num_pairs = 0
            for i in range(self.num_of_envs + 1):
                for j in range(i + 1, self.num_of_envs + 1):
                    loss_inv = loss_inv + self.invariance_loss(inv_l[i], inv_l[j])
                    loss_inv = loss_inv + self.invariance_loss(inv_a[i], inv_a[j])
                    loss_inv = loss_inv + self.invariance_loss(inv_v[i], inv_v[j])
                    num_pairs += 1
            loss_inv = loss_inv / (3 * num_pairs)

            # ---------------- R_dec: orthogonality (Eq. 5-6) ----------------
            loss_mi = loss_mi + self.correlation_loss(z_l_inv, z_l_sup)
            loss_mi = loss_mi + self.correlation_loss(z_a_inv, z_a_sup)
            loss_mi = loss_mi + self.correlation_loss(z_v_inv, z_v_sup)
            loss_mi = loss_mi / 3

            # ---------------- R_rec: reconstruction (Eq. 7-8) ----------------
            rec_l = self.decoder_l(torch.cat([z_l_inv, z_l_sup], dim=-1))
            rec_a = self.decoder_a(torch.cat([z_a_inv, z_a_sup], dim=-1))
            rec_v = self.decoder_v(torch.cat([z_v_inv, z_v_sup], dim=-1))
            # The unimodal target is detached so the reconstruction term shapes
            # the encoder/decoder without destabilising the unimodal networks.
            loss_recon = loss_recon + self.reconstruction_loss(
                output_l.detach(), rec_l).mean()
            loss_recon = loss_recon + self.reconstruction_loss(
                output_a.detach(), rec_a).mean()
            loss_recon = loss_recon + self.reconstruction_loss(
                output_v.detach(), rec_v).mean()
            loss_recon = loss_recon / 3

        return (logits, loss_inv, loss_mi, loss_recon,
                z_l_inv, z_a_inv, z_v_inv, fusion)

    def _split(self, z):
        """Split an encoder output into invariant and spurious halves."""
        return z[:, :self.d_l], z[:, self.d_l:]


class CmIRForRegression(DebertaV2PreTrainedModel):
    """Thin wrapper matching the PretrainedModel interface used by the trainer."""

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.cmir = CmIRModel(config, multimodal_config)
        self.init_weights()

    def forward(self, input_ids, visual, acoustic, label_ids=None,
                feature=False, input_mask=None):
        return self.cmir(
            input_ids, visual, acoustic, label_ids, feature, input_mask,
        )
