"""Override: store the DFlash2 draft's projections, optionally its codebooks, in Q4_G64_FP16.

The shipped recipes already assign these formats (`tools.convert.official_recipes`), so an ordinary
conversion needs no override. This file is the campaign's lever in explicit form and adds the one
choice the recipes deliberately do not make:

* `configure` assigns every qualified draft projection to `q4_g64_fp16`;
* `configure_with_codebooks` additionally assigns the two selector codebooks to `q4_g64_fp16`.

Measured on the dual RTX 5060 Ti pair (worklog section 8 / section 3.9, r54-r69): the projection set
takes the draft from 2123.6 MiB to 1155.3 MiB, 968.3 MiB (0.946 GiB) below an all-Q8 draft, with no
measurable acceptance cost on pooled sampling; the codebook lever removes a further 178.09 MiB but
measured about 2 pp less acceptance on a 33k-token prompt at K=7, and those bytes land on the shard
that holds the selector rather than on the masked draft's shard, so they buy no context.

Apply with:

    python3 -m tools.convert ... --override tools/convert/overrides/dflash2_draft_q4.py
    python3 -m tools.convert ... --override \
        tools/convert/overrides/dflash2_draft_q4.py:configure_with_codebooks

`tests/convert/test_draft_overrides.py` keeps the projection set in step with
`official_recipes.assign_dflash_formats`.
"""

from tools.convert.methods import grouped_absmax

Q4 = "q4_g64_fp16"

FEATURE_PROJECTION = "dflash2/feature_projection"
CODEBOOKS = (
    "dflash2/candidate_selector/predecessor_codebook",
    "dflash2/candidate_selector/successor_codebook",
)

# The fused QKV parent is one stored object per layer, shared with context_key and context_value, so
# its logical names are assigned together and the stored representation covers every row view.
ROLES = (
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


def _draft_layers(model):
    """Return {layer index: set of role suffixes} for the model's DFlash2 draft."""

    layers = {}
    for name in model.parameters:
        parts = name.split("/")
        if len(parts) < 4 or parts[0] != "dflash2" or parts[1] != "layers":
            continue
        layers.setdefault(parts[2], set()).add("/" + "/".join(parts[3:]))
    return layers


def configure(model, recipe, sources):
    """Assign the qualified DFlash2 draft projections to Q4_G64_FP16."""

    if FEATURE_PROJECTION not in model.parameters:
        raise ValueError(f"dflash2 draft q4 override: {FEATURE_PROJECTION} is not in the model")
    recipe.assign(FEATURE_PROJECTION, format=Q4, method=grouped_absmax)

    layers = _draft_layers(model)
    if not layers:
        raise ValueError("dflash2 draft q4 override: the model has no DFlash2 draft layers")
    order = sorted(layers, key=lambda value: (int(value) if value.isdigit() else 0, value))
    for index in order:
        absent = [role for role in ROLES if role not in layers[index]]
        if absent:
            raise ValueError(
                f"dflash2 draft q4 override: dflash2/layers/{index} has no {', '.join(absent)}"
            )
        for role in ROLES:
            recipe.assign(f"dflash2/layers/{index}{role}", format=Q4, method=grouped_absmax)


def configure_with_codebooks(model, recipe, sources):
    """`configure` plus the two selector codebooks, the trade the recipes do not make."""

    configure(model, recipe, sources)
    for name in CODEBOOKS:
        if name not in model.parameters:
            raise ValueError(f"dflash2 draft q4 override: {name} is not in the model")
        recipe.assign(name, format=Q4, method=grouped_absmax)
