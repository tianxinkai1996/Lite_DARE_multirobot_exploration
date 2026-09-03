"""Inspect the graph-encoder architecture of a trained exploration policy."""

from __future__ import annotations

import argparse

from diffusion_policy.model.encoder.exploration_node_encoder import (
    ExplorationNodeEncoder,
)
from lite_dare.lite_exploration_node_encoder import (
    LiteExplorationNodeEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect the original and Lite-DARE graph attention depths."
    )
    parser.add_argument("--node-dim", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument(
        "--lite-layers",
        type=int,
        nargs="+",
        default=[4, 2],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    original = ExplorationNodeEncoder(
        node_dim=args.node_dim,
        embedding_dim=args.embedding_dim,
    )

    original_encoder_layers = len(original.encoder.layers)
    original_decoder_layers = len(original.decoder.layers)

    print("=" * 72)
    print("Original ExplorationNodeEncoder")
    print(f"graph encoder layers: {original_encoder_layers}")
    print(f"graph encoder heads:  {original.encoder.layers[0].multiHeadAttention.n_heads}")
    print(f"graph decoder layers: {original_decoder_layers}")
    print(f"graph decoder heads:  {original.decoder.layers[0].multiHeadAttention.n_heads}")

    if original_encoder_layers != 6:
        raise RuntimeError(
            f"Expected six original graph encoder layers, found "
            f"{original_encoder_layers}."
        )
    if original_decoder_layers != 1:
        raise RuntimeError(
            f"Expected one original graph decoder layer, found "
            f"{original_decoder_layers}."
        )

    for layer_count in args.lite_layers:
        lite = LiteExplorationNodeEncoder(
            node_dim=args.node_dim,
            embedding_dim=args.embedding_dim,
            encoder_n_layer=layer_count,
            encoder_n_head=4,
        )
        print("-" * 72)
        print(
            f"Lite variant: "
            f"NodeEncSA_L{len(lite.encoder.layers)}_H4_"
            f"D{args.embedding_dim}_DecL{len(lite.decoder.layers)}_H4"
        )

        if len(lite.encoder.layers) != layer_count:
            raise RuntimeError(
                f"Requested {layer_count} layers but created "
                f"{len(lite.encoder.layers)}."
            )
        if len(lite.decoder.layers) != 1:
            raise RuntimeError("Lite decoder must remain one layer.")

    print("=" * 72)
    print("Encoder structure inspection passed.")


if __name__ == "__main__":
    main()
