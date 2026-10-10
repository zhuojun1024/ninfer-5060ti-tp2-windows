"""Convert explicitly selected local weights into a NInfer v3 artifact."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from collections import Counter
from collections.abc import Mapping

from .official_recipes import RECIPES
from .pipeline import convert
from .proposal import DEFAULT_RANKING, add_official_proposal
from .qwen3_5 import build_model
from .recipe import Recipe
from .sources.safetensors import SafetensorsSource
from .gguf_blocks import RECIPES as GGUF_RECIPES
from .sources.gguf import GGUFFile


def _open_named_source(name: str, path: Path):
    """A .gguf source carries its own reader; the block recipe needs no gguf-py."""
    if path.suffix == ".gguf":
        if name == "gguf":
            return GGUFFile(path)
        from .sources.gguf_source import GGUFSource

        return GGUFSource(path)
    return SafetensorsSource(path)


class SourceInputs(Mapping):
    """Named optional sources are opened only when a recipe or component requests one."""

    def __init__(self, base, paths, stack):
        self._sources = {"base": base}
        self._paths = dict(paths)
        self._stack = stack

    def __getitem__(self, name):
        if name not in self._sources:
            if name not in self._paths:
                raise ValueError(
                    f"selected recipe requires source {name!r}; provide --source {name}=PATH"
                )
            self._sources[name] = self._stack.enter_context(
                _open_named_source(name, self._paths[name])
            )
        return self._sources[name]

    def __iter__(self):
        return iter(dict.fromkeys((*self._sources, *self._paths)))

    def __len__(self):
        return len(set(self._sources) | set(self._paths))

    def provenance(self):
        return {
            name: {"path": str(source.path)} for name, source in self._sources.items()
        }


def _pairs(values, label):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{label} requires unique NAME=PATH entries")
        result[name] = Path(path)
    return result


def _function(value: str):
    recipes = {**RECIPES, **GGUF_RECIPES}
    if value in recipes:
        return recipes[value]
    filename, separator, function = value.rpartition(":")
    if not separator or not Path(filename).is_file():
        # An absolute Windows path contains a drive colon, so an existing whole
        # value wins over splitting the entry point off the file.
        filename, function = value, "configure"
    path = Path(filename).resolve()
    spec = importlib.util.spec_from_file_location("ninfer_user_recipe", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load recipe file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    result = getattr(module, function)
    if not callable(result):
        raise TypeError(f"{value}: recipe entry must be callable")
    return result


def _print_plan(recipe):
    """Print the resolved format and method of every planned parameter."""

    counts = Counter(
        selection.format
        for selections in recipe.selections.values()
        for selection in selections
    )
    print(f"plan: {len(recipe.selections)} parameters, {sum(counts.values())} selections")
    for format, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {format}: {count}")
    for name in sorted(recipe.selections):
        for selection in recipe.selections[name]:
            method = getattr(selection.method, "__name__", selection.method)
            print(f"{name} [{selection.begin},{selection.end}) {selection.format} {method}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="primary checkpoint/config and default resources",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="official name or Python file[:function]; defaults to configure in a file",
    )
    parser.add_argument(
        "--override",
        help="Python configuration applied after the selected recipe and proposal",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named source such as quantized, dflash or dflash2",
    )
    parser.add_argument(
        "--components",
        default="text",
        help="comma-separated text,vision,mtp,dflash,dflash2",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="override a final frontend resource",
    )
    parser.add_argument(
        "--proposal",
        action="store_true",
        help="include the existing indexed proposal head",
    )
    parser.add_argument("--proposal-rows", type=int, default=131072)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--name", help="public instance name saved in metadata")
    parser.add_argument("--out", type=Path, help="output artifact path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned selections and write nothing",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--max-file-bytes", type=int, default=32_000_000_000)
    args = parser.parse_args(argv)
    if args.out is None and not args.dry_run:
        parser.error("--out is required unless --dry-run is given")
    components = tuple(args.components.split(","))
    if len(components) != len(set(components)):
        raise ValueError("components must not repeat")
    paths = _pairs(args.source, "source")
    if "base" in paths:
        raise ValueError("select the base source with --model")
    overrides = _pairs(args.resource, "resource")
    with ExitStack() as stack:
        base = stack.enter_context(SafetensorsSource(args.model))
        sources = SourceInputs(base, paths, stack)
        companions = {
            key: sources[key] for key in ("dflash", "dflash2") if key in components
        }
        model = build_model(
            base,
            components=components,
            companions=companions,
            resource_overrides=overrides,
        )
        recipe = Recipe(model)
        _function(args.recipe)(model, recipe, sources)
        if args.proposal:
            add_official_proposal(recipe, ranking=args.ranking, rows=args.proposal_rows)
        if args.override:
            _function(args.override)(model, recipe, sources)
        if args.dry_run:
            _print_plan(recipe)
            print("DRYRUN: plan validated, nothing written", flush=True)
            return

        def progress(index, total, job):
            label = job.parameters[0]
            if len(job.parameters) > 1:
                label += f" (+{len(job.parameters)-1})"
            print(
                f"[{index+1}/{total}] {label}: {job.spec.format} {job.spec.shape}",
                flush=True,
            )

        provenance = {
            "converter": "ninfer-v3",
            "recipe": args.recipe,
            "sources": sources.provenance(),
        }
        if args.override:
            provenance["override"] = args.override
        if args.proposal:
            provenance["ranking"] = str(args.ranking)
        report = convert(
            model,
            recipe,
            args.out,
            name=args.name,
            provenance=provenance,
            device=args.device,
            rows_per_chunk=args.rows_per_chunk,
            max_file_bytes=args.max_file_bytes,
            progress=progress,
        )
        print(
            f"wrote {args.out}: {report['objects']} objects, {len(report['files'])} files, {report['seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
