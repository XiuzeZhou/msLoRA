import torch
import argparse
import random
import numpy as np
from transformers import AutoTokenizer
import os
from utils import DataLoader, Batchify, now_time
from module import LoRA_Concat, msLoRA

import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt

from module import MyLoraLayer


# ============================================================
# Modality utilities
# ============================================================

def _get_gradient_modalities(model, task):
    """
    Determine which modalities should be analyzed.

    MSR-VTT:
        Text + Image + Audio

    Other datasets:
        Text + Image
    """

    modalities = [
        "text",
        "image",
    ]

    if (
        task.lower() == "msrvtt"
        and getattr(
            model,
            "audio_embeddings",
            None,
        ) is not None
    ):
        modalities.append(
            "audio"
        )

    return modalities


def _get_lora_parameters(
    layer,
    modality,
):
    """
    Return LoRA A and B parameters for a given modality.
    """

    if modality == "text":

        A_name = "lora_A_t"
        B_name = "lora_B_t"

    elif modality == "image":

        A_name = "lora_A_img"
        B_name = "lora_B_img"

    elif modality == "audio":

        A_name = "lora_A_aud"
        B_name = "lora_B_aud"

    else:

        raise ValueError(
            f"Unknown modality: {modality}"
        )

    if (
        not hasattr(layer, A_name)
        or not hasattr(layer, B_name)
    ):
        return None, None

    return (
        getattr(layer, A_name),
        getattr(layer, B_name),
    )


def _get_modality_scaling(
    layer,
    modality,
    image_gain=3.0,
    audio_gain=8.0,
):
    """
    Return the effective scaling used by each modality.

    This supports both implementations:

        self.scaling

    and:

        self.text_scaling
        self.mm_scaling

    image_gain and audio_gain should match the values used
    in MyLoraLayer.forward().
    """

    if modality == "text":

        scale = getattr(
            layer,
            "text_scaling",
            getattr(
                layer,
                "scaling",
                1.0,
            ),
        )

        gain = 1.0

    elif modality == "image":

        scale = getattr(
            layer,
            "mm_scaling",
            getattr(
                layer,
                "scaling",
                1.0,
            ),
        )

        gain = image_gain

    elif modality == "audio":

        scale = getattr(
            layer,
            "mm_scaling",
            getattr(
                layer,
                "scaling",
                1.0,
            ),
        )

        gain = audio_gain

    else:

        raise ValueError(
            modality
        )

    return (
        float(scale)
        * float(gain)
    )


# ============================================================
# Effective DeltaW gradient representation
# ============================================================

def _effective_delta_grad_factors(
    layer,
    modality,
    image_gain=3.0,
    audio_gain=8.0,
    analysis_device="cpu",
):
    """
    Represent the first-order update direction of effective DeltaW.

    Given:

        DeltaW = s * B @ A

    the first-order differential is:

        dDeltaW
        =
        s * (dB @ A + B @ dA)

    It can be factorized as:

        dDeltaW = U @ V

    where:

        U = [s*dB, s*B]

        V = [A
             dA]

    This avoids explicitly materializing the large
    D_out x D_in effective gradient matrix.
    """

    A, B = _get_lora_parameters(
        layer,
        modality,
    )

    if (
        A is None
        or B is None
    ):
        return None

    if (
        A.grad is None
        or B.grad is None
    ):
        return None

    scale = _get_modality_scaling(
        layer,
        modality,
        image_gain=image_gain,
        audio_gain=audio_gain,
    )

    A_value = (
        A.detach()
        .to(
            device=analysis_device,
            dtype=torch.float32,
        )
    )

    B_value = (
        B.detach()
        .to(
            device=analysis_device,
            dtype=torch.float32,
        )
    )

    dA = (
        A.grad.detach()
        .to(
            device=analysis_device,
            dtype=torch.float32,
        )
    )

    dB = (
        B.grad.detach()
        .to(
            device=analysis_device,
            dtype=torch.float32,
        )
    )

    # U shape:
    #
    # [D_out, 2r]
    U = torch.cat(
        [
            scale * dB,
            scale * B_value,
        ],
        dim=1,
    )

    # V shape:
    #
    # [2r, D_in]
    V = torch.cat(
        [
            A_value,
            dA,
        ],
        dim=0,
    )

    return U, V


# ============================================================
# Efficient Frobenius-space operations
# ============================================================

def _low_rank_frobenius_inner(
    U1,
    V1,
    U2,
    V2,
):
    """
    Compute the Frobenius inner product between:

        M1 = U1 @ V1
        M2 = U2 @ V2

    without explicitly constructing M1 and M2.

    The identity used is:

        <M1, M2>_F
        =
        Tr(
            (U1^T U2)
            (V2 V1^T)
        )
    """

    # Effective output dimensions must match.
    if U1.shape[0] != U2.shape[0]:
        return None

    # Effective input dimensions must match.
    if V1.shape[1] != V2.shape[1]:
        return None

    left = (
        U1.T @ U2
    )

    right = (
        V2 @ V1.T
    )

    inner = torch.sum(
        left
        * right.T
    )

    return inner


def _low_rank_frobenius_norm(
    U,
    V,
):
    """
    Compute:

        ||U @ V||_F

    without explicitly constructing U @ V.
    """

    gram_u = (
        U.T @ U
    )

    gram_v = (
        V @ V.T
    )

    norm_squared = torch.sum(
        gram_u
        * gram_v.T
    )

    norm_squared = torch.clamp(
        norm_squared,
        min=0.0,
    )

    return torch.sqrt(
        norm_squared
    )


def _effective_gradient_cosine(
    factors1,
    factors2,
    eps=1e-12,
):
    """
    Compute cosine similarity between two effective DeltaW
    first-order gradient directions.
    """

    if (
        factors1 is None
        or factors2 is None
    ):
        return None

    U1, V1 = factors1
    U2, V2 = factors2

    inner = (
        _low_rank_frobenius_inner(
            U1,
            V1,
            U2,
            V2,
        )
    )

    # This also automatically filters incompatible layers,
    # such as down_proj when modality input spaces differ.
    if inner is None:
        return None

    norm1 = (
        _low_rank_frobenius_norm(
            U1,
            V1,
        )
    )

    norm2 = (
        _low_rank_frobenius_norm(
            U2,
            V2,
        )
    )

    denominator = (
        norm1 * norm2
    )

    if (
        not torch.isfinite(
            denominator
        )
        or denominator.item() <= eps
    ):
        return None

    cosine = (
        inner
        / denominator
    )

    cosine = torch.clamp(
        cosine,
        min=-1.0,
        max=1.0,
    )

    return float(
        cosine.item()
    )


# ============================================================
# Main Gradient Similarity analysis
# ============================================================

def analyze_gradient_similarity(
    model,
    data_loader,
    task,
    num_batches=10,
    output_dir="./analysis_results",
    image_gain=3.0,
    audio_gain=8.0,
    analysis_device="cpu",
    layer_name_contains=None,
):
    """
    Analyze gradient similarity between modality-specific LoRA paths.

    MSR-VTT:
        Text vs Image
        Text vs Audio
        Image vs Audio

    Other datasets:
        Text vs Image

    The comparison is performed in the effective DeltaW space rather
    than directly flattening LoRA parameter gradients. This makes the
    analysis valid when different modalities use different LoRA ranks.

    Args:
        model:
            Trained mmLoRA model.

        data_loader:
            Batchify instance.

        task:
            Dataset name, e.g.:
                "msrvtt"
                "twitter17"
                "twitter15"
                "hateful"
                "scienceqa"
                "flickr"
                "mvsa"

        num_batches:
            Number of batches used for gradient analysis.

        output_dir:
            Directory used to save the heatmap.

        image_gain:
            Image branch gain used in MyLoraLayer.forward().
            Default: 3.0

        audio_gain:
            Audio branch gain used in MyLoraLayer.forward().
            Default: 8.0

        analysis_device:
            Device used for gradient-space calculations.
            "cpu" is recommended to avoid increasing GPU memory usage.

        layer_name_contains:
            Optional layer-name filter.

            Example:
                "q_proj"
                "v_proj"
                "up_proj"

            None means analyzing all compatible MyLoraLayer instances.

    Returns:
        Dictionary containing:
            modalities
            matrix
            pair_means
            pair_stds
            pair_layer_means
            batch_records
            layer_records
            overall_similarity
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Determine active modalities
    # --------------------------------------------------------

    modalities = (
        _get_gradient_modalities(
            model,
            task,
        )
    )

    modality_pairs = list(
        itertools.combinations(
            modalities,
            2,
        )
    )

    # Batch-level averages.
    batch_records = {
        pair: []
        for pair in modality_pairs
    }

    # Every compatible layer value from every batch.
    layer_records = {
        pair: []
        for pair in modality_pairs
    }

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Effective DeltaW Gradient Similarity Analysis"
    )

    print(
        f"Task: {task}"
    )

    print(
        f"Modalities: {modalities}"
    )

    if layer_name_contains is not None:

        print(
            f"Layer filter: "
            f"{layer_name_contains}"
        )

    # --------------------------------------------------------
    # Preserve the original training/evaluation state
    # --------------------------------------------------------

    original_training_state = (
        model.training
    )

    # eval() disables dropout but does not disable gradients.
    model.eval()

    # --------------------------------------------------------
    # Find the correct input device
    # --------------------------------------------------------

    input_device = (
        model.model
        .get_input_embeddings()
        .weight
        .device
    )

    print(
        f"Model input device: "
        f"{input_device}"
    )

    # --------------------------------------------------------
    # Reset the loader if supported
    # --------------------------------------------------------

    if hasattr(
        data_loader,
        "reset_step",
    ):

        data_loader.reset_step()

    successful_batches = 0

    # --------------------------------------------------------
    # Analyze batches
    # --------------------------------------------------------

    total_batches = min(
        num_batches,
        data_loader.total_step,
    )

    for batch_idx in range(
        total_batches
    ):

        # ----------------------------------------------------
        # Retrieve a batch
        #
        # The current training pipeline is expected to return:
        #
        # img_ids,
        # aud_ids,
        # input_ids,
        # attention_mask,
        # target_lens,
        # labels
        # ----------------------------------------------------

        try:

            batch = (
                data_loader.next_batch(
                    mode="train"
                )
            )

        except TypeError:

            batch = (
                data_loader.next_batch()
            )

        if len(batch) < 6:

            raise RuntimeError(
                "Gradient analysis expects the batch format:\n"
                "(img_ids, aud_ids, input_ids, "
                "attention_mask, target_lens, labels)"
            )

        (
            img_ids,
            aud_ids,
            input_ids,
            attention_mask,
            target_lens,
            _,
        ) = batch[:6]

        # ----------------------------------------------------
        # Disable audio for non-MSRT-VTT datasets
        # ----------------------------------------------------

        if "audio" not in modalities:
            aud_ids = None

        input_ids = input_ids.to(
            input_device
        )

        attention_mask = (
            attention_mask.to(
                input_device
            )
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_ids=img_ids,
            aud_ids=aud_ids,
            target_lens=target_lens,
        )

        loss = outputs.loss

        if (
            loss is None
            or not torch.isfinite(
                loss
            )
        ):

            print(
                f"Batch {batch_idx + 1}: "
                f"invalid loss, skipped."
            )

            model.zero_grad(
                set_to_none=True
            )

            continue

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        loss.backward()

        # Store all compatible layer values for this batch.
        current_batch_values = {
            pair: []
            for pair in modality_pairs
        }

        compatible_layers = 0

        # ----------------------------------------------------
        # Analyze each MyLoraLayer
        # ----------------------------------------------------

        for (
            layer_name,
            layer,
        ) in model.named_modules():

            if not isinstance(
                layer,
                MyLoraLayer,
            ):
                continue

            if (
                layer_name_contains
                is not None
                and layer_name_contains
                not in layer_name
            ):
                continue

            # Build the effective gradient representation
            # for each active modality.
            factors = {}

            for modality in modalities:

                factors[
                    modality
                ] = (
                    _effective_delta_grad_factors(
                        layer=layer,
                        modality=modality,
                        image_gain=image_gain,
                        audio_gain=audio_gain,
                        analysis_device=analysis_device,
                    )
                )

            layer_was_used = False

            # ------------------------------------------------
            # Pairwise modality comparison
            # ------------------------------------------------

            for pair in modality_pairs:

                modality_1 = pair[0]
                modality_2 = pair[1]

                cosine = (
                    _effective_gradient_cosine(
                        factors[
                            modality_1
                        ],
                        factors[
                            modality_2
                        ],
                    )
                )

                if cosine is None:
                    continue

                current_batch_values[
                    pair
                ].append(
                    cosine
                )

                layer_records[
                    pair
                ].append(
                    cosine
                )

                layer_was_used = True

            if layer_was_used:
                compatible_layers += 1

        # ----------------------------------------------------
        # Compute the layer-averaged value for this batch
        # ----------------------------------------------------

        valid_batch = False

        print(
            f"\nBatch "
            f"{batch_idx + 1}/{total_batches}"
        )

        print(
            f"Loss: "
            f"{loss.item():.6f}"
        )

        print(
            f"Compatible layers: "
            f"{compatible_layers}"
        )

        for pair in modality_pairs:

            values = (
                current_batch_values[
                    pair
                ]
            )

            if len(values) == 0:
                continue

            batch_mean = float(
                np.mean(values)
            )

            batch_records[
                pair
            ].append(
                batch_mean
            )

            valid_batch = True

            print(
                f"{pair[0]:>5s} vs "
                f"{pair[1]:<5s}: "
                f"{batch_mean:+.6f} "
                f"(layers={len(values)})"
            )

        if valid_batch:
            successful_batches += 1

        # ----------------------------------------------------
        # Release gradients before the next batch
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

        del outputs
        del loss

    # --------------------------------------------------------
    # Restore loader state if supported
    # --------------------------------------------------------

    if hasattr(
        data_loader,
        "reset_step",
    ):

        data_loader.reset_step()

    # --------------------------------------------------------
    # Restore model state
    # --------------------------------------------------------

    if original_training_state:

        model.train()

    else:

        model.eval()

    # --------------------------------------------------------
    # Compute final statistics
    # --------------------------------------------------------

    num_modalities = len(
        modalities
    )

    similarity_matrix = np.full(
        (
            num_modalities,
            num_modalities,
        ),
        np.nan,
        dtype=np.float32,
    )

    np.fill_diagonal(
        similarity_matrix,
        1.0,
    )

    pair_means = {}
    pair_stds = {}
    pair_layer_means = {}

    for pair in modality_pairs:

        batch_values = (
            batch_records[
                pair
            ]
        )

        all_layer_values = (
            layer_records[
                pair
            ]
        )

        key = (
            f"{pair[0]}_"
            f"{pair[1]}"
        )

        if batch_values:

            mean_value = float(
                np.mean(
                    batch_values
                )
            )

            std_value = float(
                np.std(
                    batch_values
                )
            )

        else:

            mean_value = np.nan
            std_value = np.nan

        if all_layer_values:

            layer_mean = float(
                np.mean(
                    all_layer_values
                )
            )

        else:

            layer_mean = np.nan

        pair_means[
            key
        ] = mean_value

        pair_stds[
            key
        ] = std_value

        pair_layer_means[
            key
        ] = layer_mean

        i = modalities.index(
            pair[0]
        )

        j = modalities.index(
            pair[1]
        )

        similarity_matrix[
            i,
            j
        ] = mean_value

        similarity_matrix[
            j,
            i
        ] = mean_value

    # --------------------------------------------------------
    # Compute mean off-diagonal similarity
    # --------------------------------------------------------

    valid_pair_means = [
        value
        for value
        in pair_means.values()
        if not np.isnan(value)
    ]

    if valid_pair_means:

        overall_similarity = float(
            np.mean(
                valid_pair_means
            )
        )

    else:

        overall_similarity = np.nan

    # --------------------------------------------------------
    # Plot heatmap
    # --------------------------------------------------------

    display_names = {
        "text": "Text",
        "image": "Image",
        "audio": "Audio",
    }

    axis_labels = [
        display_names[
            modality
        ]
        for modality in modalities
    ]

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        similarity_matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    ax.set_xticks(
        np.arange(
            num_modalities
        )
    )

    ax.set_yticks(
        np.arange(
            num_modalities
        )
    )

    ax.set_xticklabels(
        axis_labels
    )

    ax.set_yticklabels(
        axis_labels
    )

    # Add numerical values to each cell.
    for i in range(
        num_modalities
    ):

        for j in range(
            num_modalities
        ):

            value = (
                similarity_matrix[
                    i,
                    j
                ]
            )

            if np.isnan(value):

                label = "N/A"

            else:

                label = (
                    f"{value:.4f}"
                )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_title(
        "Modality Gradient Similarity "
        "in Effective $\\Delta W$ Space"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Cosine Similarity"
    )

    plt.tight_layout()

    plot_path = os.path.join(
        output_dir,
        f"gradient_similarity_{task}.png",
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Gradient Similarity Summary"
    )

    print(
        f"Successful batches: "
        f"{successful_batches}/"
        f"{total_batches}"
    )

    for pair in modality_pairs:

        key = (
            f"{pair[0]}_"
            f"{pair[1]}"
        )

        mean_value = (
            pair_means[
                key
            ]
        )

        std_value = (
            pair_stds[
                key
            ]
        )

        layer_mean = (
            pair_layer_means[
                key
            ]
        )

        num_layer_values = len(
            layer_records[
                pair
            ]
        )

        print(
            f"{pair[0]:>5s} vs "
            f"{pair[1]:<5s}: "
            f"batch mean = "
            f"{mean_value:+.6f}, "
            f"batch std = "
            f"{std_value:.6f}, "
            f"all-layer mean = "
            f"{layer_mean:+.6f}, "
            f"N = "
            f"{num_layer_values}"
        )

    print(
        f"\nMean off-diagonal "
        f"similarity: "
        f"{overall_similarity:+.6f}"
    )

    print(
        f"Heatmap saved to:\n"
        f"{plot_path}"
    )

    print(
        "=" * 80
    )

    return {
        "modalities":
            modalities,

        "matrix":
            similarity_matrix,

        "pair_means":
            pair_means,

        "pair_stds":
            pair_stds,

        "pair_layer_means":
            pair_layer_means,

        "batch_records":
            batch_records,

        "layer_records":
            layer_records,

        "overall_similarity":
            overall_similarity,

        "successful_batches":
            successful_batches,

        "plot_path":
            plot_path,
    }


def _get_modalities(model, task):
    """
    Return the modalities used by the current dataset.
    """
    modalities = ["text", "image"]

    if (
        task.lower() == "msrvtt"
        and getattr(model, "audio_embeddings", None) is not None
    ):
        modalities.append("audio")

    return modalities


def _get_lora_parameters(layer, modality):
    """
    Return LoRA A and B matrices for a given modality.
    """

    if modality == "text":
        A_name = "lora_A_t"
        B_name = "lora_B_t"

    elif modality == "image":
        A_name = "lora_A_img"
        B_name = "lora_B_img"

    elif modality == "audio":
        A_name = "lora_A_aud"
        B_name = "lora_B_aud"

    else:
        raise ValueError(
            f"Unknown modality: {modality}"
        )

    if (
        not hasattr(layer, A_name)
        or not hasattr(layer, B_name)
    ):
        return None, None

    return (
        getattr(layer, A_name),
        getattr(layer, B_name),
    )


def _get_modality_scaling(
    layer,
    modality,
    image_gain=3.0,
    audio_gain=8.0,
):
    """
    Return the effective scaling used by each modality.

    This function supports both:
        layer.scaling

    and:
        layer.text_scaling
        layer.mm_scaling
    """

    if modality == "text":

        scale = getattr(
            layer,
            "text_scaling",
            getattr(layer, "scaling", 1.0),
        )

        gain = 1.0

    elif modality == "image":

        scale = getattr(
            layer,
            "mm_scaling",
            getattr(layer, "scaling", 1.0),
        )

        gain = image_gain

    elif modality == "audio":

        scale = getattr(
            layer,
            "mm_scaling",
            getattr(layer, "scaling", 1.0),
        )

        gain = audio_gain

    else:
        raise ValueError(modality)

    return float(scale) * gain


@torch.no_grad()
def compute_full_delta_w(
    A,
    B,
    scaling,
    device="cpu",
):
    """
    Explicitly construct the full effective LoRA update:

        Delta W = scaling * B @ A

    FP32 is intentionally used here because the numerical tail after the
    theoretical rank cutoff is useful for reproducing the original
    long-tail visualization.
    """

    A = A.detach().to(
        device=device,
        dtype=torch.float32,
    )

    B = B.detach().to(
        device=device,
        dtype=torch.float32,
    )

    delta_w = (
        B @ A
    ) * scaling

    return delta_w


def analyze_svd_of_lora_weights(
    model,
    task,
    layer_name_contains="q_proj",
    num_singular_values=None,
    output_dir="./analysis_results",
    image_gain=3.0,
    audio_gain=8.0,
    analysis_device="cpu",
):
    """
    Perform full-matrix SVD analysis for modality-specific LoRA updates.

    MSR-VTT:
        Text + Image + Audio + Fused

    Other datasets:
        Text + Image + Fused

    Full Delta W matrices are explicitly constructed in FP32 so that
    the numerical long-tail behavior after the theoretical rank cutoff
    remains visible in the singular-value spectrum.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    model.eval()

    modalities = _get_modalities(
        model,
        task,
    )

    print("\n" + "=" * 80)
    print("Full SVD Analysis")
    print(f"Task: {task}")
    print(f"Modalities: {modalities}")

    # ---------------------------------------------------------
    # Find a representative multimodal LoRA layer
    # ---------------------------------------------------------

    candidates = []

    for name, module in model.named_modules():

        if not isinstance(
            module,
            MyLoraLayer,
        ):
            continue

        if not hasattr(
            module,
            "lora_A_img",
        ):
            continue

        if (
            layer_name_contains is None
            or layer_name_contains in name
        ):
            candidates.append(
                (name, module)
            )

    if not candidates:
        raise RuntimeError(
            f"No compatible MyLoraLayer containing "
            f"'{layer_name_contains}' was found."
        )

    layer_name, layer = candidates[0]

    print(
        f"Selected layer: {layer_name}"
    )

    print(
        f"Base layer shape: "
        f"{layer.base_layer.in_features} -> "
        f"{layer.base_layer.out_features}"
    )

    # ---------------------------------------------------------
    # Construct modality-specific Delta W matrices
    # ---------------------------------------------------------

    delta_matrices = {}
    ranks = {}

    reference_shape = None
    fused_compatible = True

    for modality in modalities:

        A, B = _get_lora_parameters(
            layer,
            modality,
        )

        if A is None:
            print(
                f"Skipping {modality}: "
                f"LoRA parameters are unavailable."
            )
            continue

        scale = _get_modality_scaling(
            layer,
            modality,
            image_gain=image_gain,
            audio_gain=audio_gain,
        )

        delta_w = compute_full_delta_w(
            A,
            B,
            scaling=scale,
            device=analysis_device,
        )

        delta_matrices[
            modality
        ] = delta_w

        ranks[
            modality
        ] = A.shape[0]

        if reference_shape is None:
            reference_shape = delta_w.shape

        elif delta_w.shape != reference_shape:
            fused_compatible = False

        print(
            f"{modality:>8s}: "
            f"DeltaW={tuple(delta_w.shape)}, "
            f"rank={A.shape[0]}, "
            f"scale={scale:.6f}"
        )

    # ---------------------------------------------------------
    # Construct fused Delta W
    # ---------------------------------------------------------

    if (
        fused_compatible
        and len(delta_matrices) >= 2
    ):

        fused_delta_w = None

        for modality in modalities:

            if modality not in delta_matrices:
                continue

            if fused_delta_w is None:
                fused_delta_w = (
                    delta_matrices[
                        modality
                    ].clone()
                )
            else:
                fused_delta_w = (
                    fused_delta_w
                    + delta_matrices[
                        modality
                    ]
                )

        delta_matrices[
            "fused"
        ] = fused_delta_w

    else:

        print(
            "Fused DeltaW is skipped because the modality-specific "
            "matrices do not share the same shape."
        )

    # ---------------------------------------------------------
    # Perform full SVD
    # ---------------------------------------------------------

    spectra = {}
    results = {}

    for modality, delta_w in delta_matrices.items():

        print(
            f"\nComputing full SVD for {modality}..."
        )

        singular_values = (
            torch.linalg.svdvals(
                delta_w
            )
            .detach()
            .cpu()
            .numpy()
        )

        spectra[
            modality
        ] = singular_values

        total_energy = np.sum(
            singular_values ** 2
        )

        if modality == "fused":
            theoretical_rank = sum(
                ranks.values()
            )
        else:
            theoretical_rank = ranks[
                modality
            ]

        effective_part = (
            singular_values[
                :min(
                    theoretical_rank,
                    len(singular_values),
                )
            ]
        )

        effective_energy = (
            np.sum(
                effective_part ** 2
            )
            / total_energy
            if total_energy > 0
            else 0.0
        )

        results[
            modality
        ] = {
            "theoretical_rank":
                theoretical_rank,

            "num_singular_values":
                len(singular_values),

            "effective_rank_energy":
                effective_energy,

            "largest_singular_value":
                float(
                    singular_values[0]
                ),

            "tail_first_value":
                (
                    float(
                        singular_values[
                            theoretical_rank
                        ]
                    )
                    if theoretical_rank
                    < len(singular_values)
                    else None
                ),
        }

        print(
            f"Theoretical rank: "
            f"{theoretical_rank}"
        )

        print(
            f"Total singular values: "
            f"{len(singular_values)}"
        )

        print(
            f"Energy within theoretical rank: "
            f"{effective_energy * 100:.6f}%"
        )

        if (
            theoretical_rank
            < len(singular_values)
        ):
            print(
                f"First numerical-tail value: "
                f"{singular_values[theoretical_rank]:.3e}"
            )

    # ---------------------------------------------------------
    # Plot full singular-value spectra
    # ---------------------------------------------------------

    plt.figure(
        figsize=(16, 6)
    )

    display_names = {
        "text": r"Text $\Delta W$",
        "image": r"Image $\Delta W$",
        "audio": r"Audio $\Delta W$",
        "fused": r"Fused $\Delta W$",
    }

    for modality, singular_values in spectra.items():

        if num_singular_values is None:
            n = len(
                singular_values
            )
        else:
            n = min(
                num_singular_values,
                len(singular_values),
            )

        values = singular_values[:n]

        # Keep numerical-tail values visible on the logarithmic axis.
        values = np.maximum(
            values,
            1e-12,
        )

        indices = np.arange(
            1,
            n + 1,
        )

        plt.plot(
            indices,
            values,
            marker=".",
            linestyle="-",
            markersize=3,
            linewidth=1.5,
            label=display_names[
                modality
            ],
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        r"Singular Value Index $\log(k)$"
    )

    plt.ylabel(
        r"Singular Value $\log(\sigma_k)$"
    )

    plt.title(
        rf"Singular Value Spectrum of $\Delta W$ Components in {layer_name}"
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.6,
    )

    plt.legend(
        fontsize=11,
        loc="lower left",
    )

    # ---------------------------------------------------------
    # Add cumulative-rank boundaries
    # ---------------------------------------------------------

    cumulative_rank = 0

    active_modalities = [
        m
        for m in modalities
        if m in ranks
    ]

    rank_boundaries = []

    for idx, modality in enumerate(
        active_modalities
    ):

        cumulative_rank += ranks[
            modality
        ]

        rank_boundaries.append(
            cumulative_rank
        )

        plt.axvline(
            x=cumulative_rank,
            linestyle="--",
            alpha=0.65,
            linewidth=1.5,
        )

        if idx == 0:

            rank_label = (
                rf"$r_1={cumulative_rank}$"
            )

        elif idx == len(
            active_modalities
        ) - 1:

            rank_label = (
                rf"$k=\sum r_m="
                rf"{cumulative_rank}$"
            )

        else:

            rank_terms = "+".join(
                [
                    f"r_{j + 1}"
                    for j in range(
                        idx + 1
                    )
                ]
            )

            rank_label = (
                rf"${rank_terms}="
                rf"{cumulative_rank}$"
            )

        y_top = plt.ylim()[1]

        plt.text(
            cumulative_rank * 1.04,
            y_top * 0.35,
            rank_label,
            fontsize=13,
        )

    # ---------------------------------------------------------
    # Annotate information-span extension
    # ---------------------------------------------------------

    if (
        "fused" in spectra
        and rank_boundaries
    ):

        final_rank = (
            rank_boundaries[-1]
        )

        fused_values = spectra[
            "fused"
        ]

        annotation_index = min(
            final_rank - 1,
            len(fused_values) - 1,
        )

        annotation_y = max(
            fused_values[
                annotation_index
            ],
            1e-12,
        )

        plt.annotate(
            "Information Span Extension",
            xy=(
                final_rank,
                annotation_y,
            ),
            xytext=(
                final_rank * 1.5,
                annotation_y * 8,
            ),
            arrowprops=dict(
                arrowstyle="->",
            ),
            fontsize=14,
        )

    plt.tight_layout()

    safe_layer_name = (
        layer_name.replace(
            ".",
            "_",
        )
    )

    plot_path = os.path.join(
        output_dir,
        f"svd_long_tail_{task}_{safe_layer_name}.png",
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSVD figure saved to:\n"
        f"{plot_path}"
    )

    print("=" * 80)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='twitter17', choices=['mvsa', 'hateful', 'scienceqa', 'twitter17', 'twitter15', 'flickr', 'msrvtt'], help='Task name')
    parser.add_argument('-llm_model', type=str, default="./llm/Qwen2.5-3B")
    parser.add_argument('-clip_model', type=str, default="./llm/clip-vit-base-patch32/")
    parser.add_argument('-wav2vec_path', type=str, default="./llm/wav2vec2-base-960h/")
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-r', type=int, default=8)
    parser.add_argument('-lora_modules', type=int, default=7)
    parser.add_argument('-clip_norm', '--clip_norm', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('-noise_std', type=float, default=0.0, help='Standard deviation of Gaussian noise for images')
    parser.add_argument('-gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-lora_name', type=str, default='msLoRA', choices=['msLoRA', 'LoRA'])
    parser.add_argument('-load_in_8bit', action='store_true', help='Load frozen LLM backbone in 8-bit')
    parser.add_argument('-save_path', type=str, default='./checkpoints/', help='Path for trained models.')
    parser.add_argument('-seed', type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.task == 'hateful':
        data_path = './datasets/HatefulMemes/'
    elif args.task == 'scienceqa':
        data_path = './datasets/ScienceQA/'
    elif args.task =='twitter17':
        data_path = './datasets/Twitter17/'
    elif args.task =='twitter15':
        data_path = './datasets/Twitter15/'
    elif args.task == 'flickr':
        data_path = './datasets/flickr_8k/'
    elif args.task =='msrvtt':
        data_path = './datasets/MSR-VTT/'
    else: # mvsa
        data_path = './datasets/MVSA_Single/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(args.seed)

    model_type = "qwen" if "qwen" in args.llm_model.lower() else "llama"
    print(f"{now_time()} Detected Model Type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)

    if model_type == "qwen":
        if tokenizer.pad_token is None:
            if "<|extra_0|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|extra_0|>"
            else:
                tokenizer.pad_token = tokenizer.eos_token
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Fill from the left side when generating
    tokenizer.padding_side = 'left'

    # Get hidden_size: Qwen2.5-3B: 2048，7B: 3584, Llama2-7B: 4096
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.llm_model, trust_remote_code=True)
    llm_hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
    print(f"{now_time()} LLM Hidden Size: {llm_hidden_size}")

    corpus = DataLoader(data_path, tokenizer, args.clip_model, device, wav2vec_path=args.wav2vec_path, task=args.task)
    test_loader = Batchify(corpus.test, tokenizer, args.batch_size, task=args.task)

    if args.lora_name == "msLoRA":
        model = msLoRA(
            args.llm_model, 
            args.r, 
            args.lora_modules, 
            corpus.image_embeddings,
            audio_embeddings=corpus.audio_embeddings,
            load_in_8bit=args.load_in_8bit
            # hidden_size=llm_hidden_size
        )
    else:
        model = LoRA_Concat(
            args.llm_model, 
            args.r, 
            args.lora_modules, 
            corpus.image_embeddings,
            audio_embeddings=corpus.audio_embeddings,
            load_in_8bit=args.load_in_8bit,
            lora_name=args.lora_name
        )

    model_path = os.path.join(args.save_path, 'model.pt')
    print(f"\n{now_time()}Loading best model from {model_path} for final evaluation...")

    if os.path.exists(model_path):
        print(now_time() + f"load the pretrained weights from: {model_path}")
        loaded_state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(loaded_state, strict=False)
        print(now_time() + f'Successfully loaded trainable parameters from {model_path}')
        model = model.to(device)
    else:
        print(now_time() + f"WARNING: Can not find checkpoint: {model_path}")
        
    model.to(device)


    gradient_results = analyze_gradient_similarity(
        model=model,
        data_loader=test_loader,
        task=args.task,
        num_batches=10,
        output_dir=args.save_path,
        image_gain=3.0,
        audio_gain=8.0,
        layer_name_contains="q_proj",
    )

    svd_results = analyze_svd_of_lora_weights(
        model=model,
        task=args.task,
        layer_name_contains="q_proj",
        num_singular_values=100,
        output_dir=args.save_path,
    )
