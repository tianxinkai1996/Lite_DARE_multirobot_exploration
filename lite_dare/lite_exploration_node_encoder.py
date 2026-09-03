"""Lightweight two/four-layer graph-encoder modules replacing the six-layer DARE encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
torch.cuda.set_per_process_memory_fraction(0.3, device=0)

# Reuse the original attention building blocks without modifying the original file.
from diffusion_policy.model.encoder.exploration_node_encoder import Decoder, Encoder

class LiteExplorationNodeEncoder(nn.Module):
    """Reduced-depth version of DARE's ExplorationNodeEncoder.

    The original encoder uses:
      - graph encoder: 6 self-attention layers, 4 heads;
      - graph decoder: 1 cross-attention layer, 4 heads.

    Lite-DARE changes only the graph encoder depth. The node embedding,
    decoder, output layers, method signatures, and tensor flow remain aligned
    with the original implementation.
    """

    def __init__(self, node_dim, embedding_dim, encoder_n_layer=3, encoder_n_head=4):
        super().__init__()

        if encoder_n_layer <= 0:
            raise ValueError(
                f"encoder_n_layer must be positive, got {encoder_n_layer}."
            )
        if encoder_n_head <= 0:
            raise ValueError(
                f"encoder_n_head must be positive, got {encoder_n_head}."
            )
        if embedding_dim % encoder_n_head != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by "
                f"encoder_n_head={encoder_n_head}."
            )

        self.node_dim = int(node_dim)
        self.embedding_dim = int(embedding_dim)
        self.encoder_n_layer = int(encoder_n_layer)
        self.encoder_n_head = int(encoder_n_head)

        # Identical to the original ExplorationNodeEncoder except for the
        # configurable graph-encoder depth.
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            n_head=encoder_n_head,
            n_layer=encoder_n_layer,
        )

        # Keep the original decoder structure fixed for a controlled ablation.
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            n_head=4,
            n_layer=1,
        )
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)
        self.q_values_layer = nn.Linear(embedding_dim * 2, 1)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(
            src=node_feature,
            key_padding_mask=node_padding_mask,
            attn_mask=edge_mask,
        )
        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(
            enhanced_node_feature,
            1,
            current_index.repeat(1, 1, embedding_dim),
        )
        enhanced_current_node_feature, _ = self.decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask,
        )
        return current_node_feature, enhanced_current_node_feature

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask):
        enhanced_node_feature = self.encode_graph(
            node_inputs,
            node_padding_mask,
            edge_mask,
        )
        _, robot_belief_feature = self.decode_state(
            enhanced_node_feature,
            current_index,
            node_padding_mask,
        )
        return robot_belief_feature
