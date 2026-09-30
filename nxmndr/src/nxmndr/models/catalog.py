# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The PyTorch model catalog: torchvision and torchgeo model classes served by name.

A catalog name is ``<family>:<name>``: ``torchvision:<name>`` for a name that
``torchvision.models.list_models()`` returns and ``torchgeo:<name>`` for one that
``torchgeo.models.list_models()`` returns. A ``ModelSpec`` with format ``PYTORCH``
and a catalog name as ``model_class`` is built by :func:`build` in the PyTorch RPC
worker. Names registered with ``register_pytorch_model`` are resolved before the
catalog and keep their own path.

Served names (:func:`list_model_classes`, the server's Capabilities entry
``pytorch_model_classes``) are the names this module builds and runs on one image
chip. ``tst/integration/test_model_catalog_service.py`` loads and runs every one of
them through the gRPC service and the RPC worker.

- torchvision: the image classification models
  (``list_models(module=torchvision.models)``) and the semantic segmentation models
  (``list_models(module=torchvision.models.segmentation)``). The detection, video,
  optical-flow and quantization builders are not served (:data:`TORCHVISION_EXCLUDED`).
- torchgeo: the names in :data:`TORCHGEO_SERVED`. Every other name of torchgeo 0.10.1
  is in :data:`TORCHGEO_EXCLUDED` with the reason.

A family whose package cannot be imported gives no names. The list is ordered by
family, then name.

Construction (:func:`build`): the family's ``get_model(name, weights=None, ...)``, so
no pretrained weights are downloaded (torchvision segmentation builders also get
``weights_backbone=None``). The constructor arguments are:

1. ``num_classes`` and ``in_channels`` from ``ModelSpec.metadata``, when present;
2. else inferred from the weights file: the class count is the first dimension of
   the class-head weights and the band count is the second dimension of the stem
   weights. The two entries are found by building the model on the ``meta`` device
   with marker values of the two arguments (the entries whose dimension takes the
   marker), so no per-model key list is kept;
3. else the family's defaults.

torchvision segmentation builders get ``aux_loss=True`` when the weights file has
``aux_classifier.*`` entries. The weights load with ``strict=True``. A file that
does not match is a :class:`CatalogError` that names the class, its arguments and
the differing entries.

Input and output: the built model is a :class:`CatalogModel`. It takes the chip as
the server sends it (``[H, W, C]`` float32, no scaling or normalization; CONTRACTS.md
2.4) as one image ``[1, C, H, W]`` and returns one tensor (the ``out`` entry of a
torchvision segmentation model). ``nxmndr.server.dispatch.catalog_output`` turns that
tensor into a result family.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ..logging import get_logger

logger = get_logger("nxmndr.models.catalog")

FAMILY_TORCHVISION = "torchvision"
FAMILY_TORCHGEO = "torchgeo"
FAMILIES = (FAMILY_TORCHVISION, FAMILY_TORCHGEO)
SEPARATOR = ":"

# ModelSpec.metadata keys that set constructor arguments of a catalog model. The model
# cache keys on them (nxmndr.server.model_cache._PYTORCH_CONSTRUCTOR_KEYS).
CONSTRUCTOR_KEYS = ("num_classes", "in_channels")

# The packages whose absence makes a family unavailable, for the error message.
_INSTALL_HINT = {
    FAMILY_TORCHVISION: "torchvision (a dependency of nxmndr)",
    FAMILY_TORCHGEO: "torchgeo (the nxmndr[geo] extra)",
}


class CatalogError(ValueError):
    """A catalog name or its constructor arguments cannot be served as given."""


class CatalogUnavailableError(CatalogError):
    """The family of a catalog name is not installed in this Python environment."""


@dataclass(frozen=True)
class _Entry:
    """How one served name takes its constructor arguments.

    ``class_arg``: the builder keyword for the class count, or None when the model has
    no class head. ``channel_arg``: the builder keyword for the input band count, or
    None when the model takes ``in_channels`` bands only.
    """

    class_arg: Optional[str]
    channel_arg: Optional[str]
    in_channels: int = 3


# torchvision classification and segmentation builders take RGB images and have no
# input-channel argument.
_TORCHVISION_ENTRY = _Entry("num_classes", None, 3)
_TORCHVISION_SERVED_MODULES = ("torchvision.models", "torchvision.models.segmentation")
TORCHVISION_EXCLUDED = {
    "torchvision.models.detection": "returns per-image detection dicts, which no result "
    "family of this server takes",
    "torchvision.models.video": "takes video clips [N, C, T, H, W], not image chips",
    "torchvision.models.optical_flow": "takes two images",
    "torchvision.models.quantization": "int8 inference needs torchvision's quantized "
    "weights (quantize=True); the float architecture is served under its plain name",
}

# Served torchgeo 0.10.1 names. timm builds resnet*, vit_* and ScaleMAE (num_classes,
# in_chans); torchgeo's swin_* are torchvision Swin Transformers (num_classes, RGB);
# segmentation_models_pytorch builds unet (classes, in_channels); TileNet and EarthLoc
# return embeddings.
TORCHGEO_SERVED: Dict[str, _Entry] = {
    **{
        name: _Entry("num_classes", "in_chans")
        for name in (
            "resnet18",
            "resnet50",
            "resnet152",
            "scalemae_large_patch16",
            "vit_base_patch14_dinov2",
            "vit_base_patch16_224",
            "vit_huge_patch14_224",
            "vit_large_patch16_224",
            "vit_small_patch14_dinov2",
            "vit_small_patch16_224",
        )
    },
    **{
        name: _Entry("num_classes", None, 3)
        for name in ("swin_b", "swin_s", "swin_t", "swin_v2_b", "swin_v2_t")
    },
    "unet": _Entry("classes", "in_channels"),
    "tilenet": _Entry(None, "in_channels"),
    "earthloc": _Entry(None, "in_channels"),
}
TORCHGEO_EXCLUDED = {
    "aurora_swin_unet": "needs the aurora package and takes weather batches, not image chips",
    "copernicusfm_base": "its forward needs band metadata (wavelengths, bandwidths) beside "
    "the image",
    "croma_base": "its forward takes separate SAR and optical images",
    "croma_large": "its forward takes separate SAR and optical images",
    "deo_base": "returns channels-last features [N, h, w, D], which no result family takes",
    "dofa_base_patch16_224": "its forward needs the band wavelengths beside the image",
    "dofa_huge_patch14_224": "its forward needs the band wavelengths beside the image",
    "dofa_large_patch16_224": "its forward needs the band wavelengths beside the image",
    "dofa_small_patch16_224": "its forward needs the band wavelengths beside the image",
    "olmoearth_v1": "needs the olmoearth-pretrain-minimal package, which the geo extra "
    "does not install",
    "panopticon_vitb14": "its forward takes a dict of images and channel ids",
    "presto": "takes pixel time series with land cover and coordinates, not image chips",
    "satclip": "takes coordinates, not image chips",
    "tessera": "takes Sentinel-1 and Sentinel-2 pixel time series, not image chips",
}

# Marker values of the class and band arguments for the meta-device probe.
_CLASS_MARK = 7919
_CHANNEL_MARK = 7907
# torchvision builder flags set from the weights file: the flag adds entries with
# these prefixes.
_TORCHVISION_FLAGS = {"aux_loss": ("aux_classifier.",)}


def unknown_class_message(model_class) -> str:
    """The error for a model_class that is neither registered nor a catalog name."""

    return (
        f"PyTorch model_class {model_class!r} is not registered in this server "
        "(nxmndr.models.register_pytorch_model) and is not a catalog name. Catalog names "
        f"are 'torchvision:<name>' and 'torchgeo:<name>' (families: {', '.join(FAMILIES)}), "
        "from torchvision.models.list_models() and torchgeo.models.list_models(); the "
        "server lists the names it serves in its Capabilities entry 'pytorch_model_classes'"
    )


def is_catalog_name(model_class) -> bool:
    """True for a string of the form ``<family>:<name>`` with a known family."""

    if not isinstance(model_class, str):
        return False
    family, sep, name = model_class.partition(SEPARATOR)
    return bool(sep) and family in FAMILIES and bool(name)


@functools.lru_cache(maxsize=None)
def _family_models(family: str):
    """The family's ``models`` module, or None when it cannot be imported."""

    try:
        return importlib.import_module(f"{family}.models")
    except Exception as exc:  # a missing or broken package: the family is absent
        logger.info("model catalog: %s is not available (%s: %s)", family, type(exc).__name__, exc)
        return None


@functools.lru_cache(maxsize=None)
def _family_names(family: str) -> Tuple[str, ...]:
    module = _family_models(family)
    return tuple(module.list_models()) if module is not None else ()


def _exclusion(family: str, name: str) -> str:
    """Why a listed name is not served ("" when it is served)."""

    if family == FAMILY_TORCHVISION:
        module = _family_models(family).get_model_builder(name).__module__.rsplit(".", 1)[0]
        if module in _TORCHVISION_SERVED_MODULES:
            return ""
        return TORCHVISION_EXCLUDED.get(module, f"its builder is in {module}")
    if name in TORCHGEO_SERVED:
        return ""
    return TORCHGEO_EXCLUDED.get(name, "it has not been verified with this server")


def _entry(family: str, name: str) -> _Entry:
    return _TORCHVISION_ENTRY if family == FAMILY_TORCHVISION else TORCHGEO_SERVED[name]


@functools.lru_cache(maxsize=None)
def list_model_classes() -> Tuple[str, ...]:
    """The served catalog names, ordered by family, then name."""

    served = [
        (family, name)
        for family in FAMILIES
        for name in _family_names(family)
        if not _exclusion(family, name)
    ]
    return tuple(f"{family}{SEPARATOR}{name}" for family, name in sorted(served))


def resolve(model_class: str) -> Tuple[str, str]:
    """``(family, name)`` of a served catalog name; ``CatalogError`` otherwise."""

    if not is_catalog_name(model_class):
        raise CatalogError(unknown_class_message(model_class))
    family, _, name = model_class.partition(SEPARATOR)
    if _family_models(family) is None:
        raise CatalogUnavailableError(
            f"{model_class} needs {_INSTALL_HINT[family]}, which this server's Python "
            "environment does not have"
        )
    if name not in _family_names(family):
        raise CatalogError(
            f"{model_class}: {name!r} is not in {family}.models.list_models(); the server "
            "lists the names it serves in its Capabilities entry 'pytorch_model_classes'"
        )
    reason = _exclusion(family, name)
    if reason:
        raise CatalogError(f"{model_class} is not served: {reason}")
    return family, name


def parse_arguments(
    metadata: Optional[Mapping], model_class: Optional[str] = None
) -> Dict[str, int]:
    """The constructor arguments in ``ModelSpec.metadata``: integers >= 1.

    With ``model_class`` (a served name), an argument the model cannot take is refused:
    ``num_classes`` for a model without a class head argument, and an ``in_channels``
    other than the band count of a model without an input-channel argument.
    """

    arguments: Dict[str, int] = {}
    for key in CONSTRUCTOR_KEYS:
        raw = (metadata or {}).get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = int(str(raw).strip())
        except ValueError:
            value = 0
        if value < 1:
            raise CatalogError(f"ModelSpec metadata {key}={raw!r} must be an integer >= 1")
        arguments[key] = value
    if model_class is None:
        return arguments
    entry = _entry(*resolve(model_class))
    if entry.class_arg is None and "num_classes" in arguments:
        raise CatalogError(
            f"{model_class} has no class head; ModelSpec metadata num_classes="
            f"{arguments['num_classes']} cannot be applied"
        )
    in_channels = arguments.get("in_channels", entry.in_channels)
    if entry.channel_arg is None and in_channels != entry.in_channels:
        raise CatalogError(
            f"{model_class} takes {entry.in_channels}-band chips and its builder has no "
            f"input-channel argument; ModelSpec metadata in_channels={in_channels} "
            "cannot be applied"
        )
    return arguments


def _builder_kwargs(family: str, name: str, state: Mapping) -> Dict[str, object]:
    """The arguments every build of ``name`` gets: no pretrained weights, flags from the file."""

    kwargs: Dict[str, object] = {"weights": None}
    if family == FAMILY_TORCHVISION:
        params = inspect.signature(_family_models(family).get_model_builder(name)).parameters
        if "weights_backbone" in params:
            kwargs["weights_backbone"] = None
        for flag, prefixes in _TORCHVISION_FLAGS.items():
            if flag in params:
                kwargs[flag] = any(str(key).startswith(prefixes) for key in state)
    return kwargs


def _construct(family: str, name: str, kwargs: Mapping[str, object]) -> torch.nn.Module:
    return _family_models(family).get_model(name, **kwargs)


def _probe_shapes(family: str, name: str, kwargs: Mapping[str, object]) -> Dict[str, tuple]:
    """State-dict entry shapes of the model built with ``kwargs``, without its weights.

    Built on the ``meta`` device; a builder that needs real tensors while it builds
    (torchvision RegNet computes its widths with ``tolist``) is built on the CPU.
    """

    try:
        with torch.device("meta"):
            model = _construct(family, name, kwargs)
    except (NotImplementedError, RuntimeError):
        model = _construct(family, name, kwargs)
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _dimension(shapes: Mapping, keys, axis: int) -> Optional[int]:
    for key in reversed(keys):
        shape = shapes.get(key)
        if shape is not None and len(shape) > axis:
            return int(shape[axis])
    return None


def _load_state(model_class: str, weights_path) -> Mapping:
    try:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except FileNotFoundError as exc:
        raise CatalogError(f"{model_class}: weights file {weights_path} not found") from exc
    except Exception as exc:
        raise CatalogError(
            f"{model_class}: weights file {weights_path} is not a PyTorch state dict "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        keys = list(state)[:5] if isinstance(state, Mapping) else type(state).__name__
        raise CatalogError(
            f"{model_class}: weights file {weights_path} is not a state dict (a mapping of "
            f"entry names to tensors); it holds {keys}"
        )
    return state


def _few(items) -> str:
    items = list(items)
    shown = ", ".join(items[:5])
    return shown + (f", ... ({len(items)} in all)" if len(items) > 5 else "")


def _mismatch(model_state: Mapping, state: Mapping) -> str:
    """The entries that keep ``state`` from loading strictly, or ""."""

    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    shapes = [
        f"{key} file {list(state[key].shape)} model {list(value.shape)}"
        for key, value in model_state.items()
        if key in state and tuple(state[key].shape) != tuple(value.shape)
    ]
    parts = []
    if missing:
        parts.append(f"missing {_few(missing)}")
    if unexpected:
        parts.append(f"unexpected {_few(unexpected)}")
    if shapes:
        parts.append(f"shape mismatch {_few(shapes)}")
    return "; ".join(parts)


class CatalogModel(torch.nn.Module):
    """A catalog model that takes one ``[H, W, C]`` chip as the server sends it."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        model_class: str,
        family: str,
        num_classes: Optional[int],
        in_channels: int,
    ):
        super().__init__()
        self.model = model
        self.catalog_name = model_class
        self.family = family
        self.num_classes = num_classes  # None: the model has no class head argument
        self.in_channels = int(in_channels)

    def catalog_info(self) -> Dict[str, str]:
        """The load metadata of this model (LoadModelResponse.effective_metadata)."""

        return {
            "model_family": self.family,
            "model_class": self.catalog_name,
            "num_classes": "" if self.num_classes is None else str(self.num_classes),
            "in_channels": str(self.in_channels),
        }

    def forward(self, chip: torch.Tensor) -> torch.Tensor:
        if chip.dim() != 3 or chip.shape[-1] != self.in_channels:
            raise ValueError(
                f"{self.catalog_name} takes one [H, W, {self.in_channels}] chip; got shape "
                f"{list(chip.shape)}"
            )
        out = self.model(chip.permute(2, 0, 1).unsqueeze(0))
        if isinstance(out, Mapping) and "out" in out:  # torchvision segmentation models
            out = out["out"]
        if not torch.is_tensor(out):
            raise TypeError(f"{self.catalog_name} returned {type(out).__name__}, not a tensor")
        return out


def build(model_class: str, weights_path, metadata: Optional[Mapping] = None) -> CatalogModel:
    """Build ``model_class`` with the weights at ``weights_path`` (see the module docstring).

    ``metadata`` holds ``ModelSpec.metadata`` (``num_classes``, ``in_channels``).
    Raises ``CatalogError`` for an unknown or unserved name, bad arguments or weights
    that do not match.
    """

    family, name = resolve(model_class)
    entry = _entry(family, name)
    given = parse_arguments(metadata, model_class)
    state = _load_state(model_class, weights_path)
    base = _builder_kwargs(family, name, state)

    marks = {}
    if entry.class_arg:
        marks[entry.class_arg] = _CLASS_MARK
    if entry.channel_arg:
        marks[entry.channel_arg] = _CHANNEL_MARK
    probe = _probe_shapes(family, name, {**base, **marks}) if marks else {}
    head = [key for key, shape in probe.items() if len(shape) >= 1 and shape[0] == _CLASS_MARK]
    stem = [key for key, shape in probe.items() if len(shape) >= 2 and shape[1] == _CHANNEL_MARK]
    file_shapes = {key: tuple(value.shape) for key, value in state.items()}

    arguments: Dict[str, int] = {}
    if entry.class_arg is not None:
        num_classes = given.get("num_classes") or _dimension(file_shapes, head, 0)
        if num_classes is not None:
            arguments[entry.class_arg] = num_classes
    if entry.channel_arg is not None:
        in_channels = given.get("in_channels") or _dimension(file_shapes, stem, 1)
        if in_channels is not None:
            arguments[entry.channel_arg] = in_channels

    signature = f"{model_class}({', '.join(f'{k}={v}' for k, v in arguments.items())})"
    try:
        model = _construct(family, name, {**base, **arguments})
    except Exception as exc:
        raise CatalogError(f"{signature} could not be built: {type(exc).__name__}: {exc}") from exc
    model_state = model.state_dict()
    mismatch = _mismatch(model_state, state)
    if mismatch:
        raise CatalogError(
            f"weights file {weights_path} does not match {signature} (strict load): {mismatch}"
        )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise CatalogError(
            f"weights file {weights_path} does not load into {signature}: {exc}"
        ) from exc
    model.eval()

    built_shapes = {key: tuple(value.shape) for key, value in model_state.items()}
    effective_classes = None
    if entry.class_arg is not None:
        effective_classes = _dimension(built_shapes, head, 0) or 0
    effective_channels = entry.in_channels
    if entry.channel_arg is not None:
        effective_channels = _dimension(built_shapes, stem, 1) or entry.in_channels
    logger.info("built %s from %s", signature, weights_path)
    return CatalogModel(
        model,
        model_class=model_class,
        family=family,
        num_classes=effective_classes,
        in_channels=effective_channels,
    )


__all__ = [
    "FAMILIES",
    "FAMILY_TORCHGEO",
    "FAMILY_TORCHVISION",
    "CONSTRUCTOR_KEYS",
    "CatalogError",
    "CatalogUnavailableError",
    "CatalogModel",
    "TORCHGEO_EXCLUDED",
    "TORCHGEO_SERVED",
    "TORCHVISION_EXCLUDED",
    "build",
    "is_catalog_name",
    "list_model_classes",
    "parse_arguments",
    "resolve",
    "unknown_class_message",
]
