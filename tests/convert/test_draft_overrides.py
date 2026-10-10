from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.convert.__main__ import _function
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import assign_dflash_formats
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

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
CODEBOOKS = (
    "dflash2/candidate_selector/predecessor_codebook",
    "dflash2/candidate_selector/successor_codebook",
)
OVERRIDE = Path(__file__).resolve().parents[2] / "tools/convert/overrides/dflash2_draft_q4.py"


def _parameter(name: str, *, inputs: bool = True) -> Parameter:
    values = torch.ones(4, 128, dtype=torch.bfloat16)
    return Parameter(
        name,
        (4, 128),
        array_source(values, name),
        inputs=(name + "_input",) if inputs else (),
    )


def _model(*, layers: int = 2, roles=ROLES, feature_projection: bool = True) -> Model:
    model = Model({"text": {"config": {}}})
    if feature_projection:
        model.add(_parameter("dflash2/feature_projection"))
    for index in range(layers):
        for role in roles:
            model.add(_parameter(f"dflash2/layers/{index}{role}"))
    model.add(_parameter("dflash2/candidate_selector/hidden_projection"))
    for name in CODEBOOKS:
        model.add(_parameter(name, inputs=False))
    model.add(_parameter("mtp/input_projection"))
    return model


def _apply(model: Model, entry: str | None = None) -> Recipe:
    """Apply the published override through the same entry point the CLI uses."""

    recipe = Recipe(model)
    target = str(OVERRIDE) if entry is None else f"{OVERRIDE}:{entry}"
    _function(target)(model, recipe, {})
    return recipe


def _selected(recipe: Recipe) -> dict:
    return {
        name: tuple(selection.format for selection in selections)
        for name, selections in recipe.selections.items()
    }


def _quantized(selected: dict) -> set:
    return {name for name, formats in selected.items() if formats == ("q4_g64_fp16",)}


def test_projection_set_matches_the_shipped_draft_policy():
    model = _model()
    override = _selected(_apply(model))
    policy = Recipe(model)
    assign_dflash_formats(model, policy)

    assert _quantized(override) == _quantized(_selected(policy))
    assert "dflash2/feature_projection" in _quantized(override)
    assert len(_quantized(override)) == 1 + 2 * len(ROLES)
    # The override is the projection lever only: direct parents and other draft families keep
    # whatever the base recipe assigned them.
    assert override["dflash2/candidate_selector/hidden_projection"] == ("bf16",)
    assert override["mtp/input_projection"] == ("bf16",)


def test_codebooks_stay_direct_by_default_and_quantize_on_request():
    plain = _selected(_apply(_model()))
    for name in CODEBOOKS:
        assert plain[name] == ("bf16",)

    quantized = _selected(_apply(_model(), "configure_with_codebooks"))
    for name in CODEBOOKS:
        assert quantized[name] == ("q4_g64_fp16",)
    assert _quantized(quantized) == _quantized(plain) | set(CODEBOOKS)


def test_override_rejects_a_model_without_the_draft():
    with pytest.raises(ValueError, match="feature_projection is not in the model"):
        _apply(_model(feature_projection=False))


def test_override_rejects_a_model_without_draft_layers():
    with pytest.raises(ValueError, match="no DFlash2 draft layers"):
        _apply(_model(layers=0))


def test_override_rejects_a_draft_layer_missing_a_role():
    with pytest.raises(ValueError, match="has no /mlp_conv/kernel_projection"):
        _apply(_model(layers=1, roles=ROLES[:-1]))
