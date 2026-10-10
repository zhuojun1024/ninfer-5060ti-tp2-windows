"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import cast_direct, fp8_row_maxabs, grouped_absmax, import_encoded

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"
BF16 = "bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


# The masked DFlash2 draft runs entirely on Q4_G64_FP16 projections. Every entry below was
# qualified by the section 8 / section 3.9 campaigns (r62-r67): the draft MLP, attention output and
# dynamic-conv kernels, and the fused QKV parent through its row views. The QKV parent is one
# [6144, 5120] object per layer shared with context_key/context_value; the native Q8-only fused
# consumers are bypassed when it is quantized (see
# src/models/qwen3_5/execution/draft.cpp). DFlash v1 and MTP keep Q8.
#
# The selector codebooks keep their source BF16 representation. The r68 Q4_G64 conversion saved
# 178.09 MiB but measured about 2 pp less long-context acceptance (33k-token prompt, K=7), and those
# bytes land on the TP-2 shard that holds the selector rather than on the masked draft's shard, which
# is the shard that bounds context capacity, so the saving does not buy context. A recipe that wants
# the quantized codebook back can still assign Q4 itself.
DFLASH2_Q4_NAMES = ("dflash2/feature_projection",)
DFLASH2_Q4_ROLES = (
    "/attention/query",
    "/attention/key",
    "/attention/value",
    "/attention/output",
    "/mlp/gate",
    "/mlp/up",
    "/mlp/down",
    "/attention_conv/kernel_projection",
    "/mlp_conv/kernel_projection",
)
DFLASH2_CODEBOOKS = (
    "dflash2/candidate_selector/predecessor_codebook",
    "dflash2/candidate_selector/successor_codebook",
)


def assign_dflash_formats(model, recipe):
    """The qualified draft formats: Q4_G64_FP16 on the projection roles, BF16 on the codebooks.

    The masked DFlash2 draft runs its projections on Q4 and keeps its selector codebooks
    unquantized; DFlash v1 and MTP keep Q8. Every recipe that carries a draft shares this policy, so
    the formats an artifact is built with are the ones the campaign qualified rather than a
    per-recipe choice.
    """

    # The selector codebooks are direct [vocab, rank] parents rather than projections. They are
    # assigned here and skipped below, so their representation does not depend on whether the model
    # happens to declare an input for them.
    for name in DFLASH2_CODEBOOKS:
        if name in model.parameters:
            _assign(recipe, name, BF16)
    for name, parameter in model.parameters.items():
        if name in DFLASH2_CODEBOOKS:
            continue
        if not parameter.projection or not name.startswith(("mtp/", "dflash/", "dflash2/")):
            continue
        if name.endswith(
            ("/moe/router", "/moe/shared_score", "/candidate_selector/hidden_projection")
        ):
            continue
        if name in DFLASH2_Q4_NAMES or (
            name.startswith("dflash2/") and name.endswith(DFLASH2_Q4_ROLES)
        ):
            _assign(recipe, name, Q4)
        else:
            _assign(recipe, name, Q8)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection or not name.startswith("vision/"):
            continue
        if name == "vision/patch_embedding":
            format = Q6
        elif name.startswith("vision/merger/"):
            format = Q8
        elif name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)
    assign_dflash_formats(model, recipe)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
}
