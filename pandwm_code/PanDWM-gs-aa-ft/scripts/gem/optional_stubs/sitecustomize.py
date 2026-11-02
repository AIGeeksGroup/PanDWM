"""Fail-closed stubs for packages unused by GEM unconditional sampling.

The official sample.py imports its control-metric helper eagerly. That helper
imports Ultralytics and COCO even in unconditional mode, while the official
Dockerfile does not install them. Keep upstream untouched and raise if either
optional evaluator is actually invoked.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import sys
import types


class _UnavailableOptionalDependency:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "An optional control-metric dependency was invoked during GEM "
            "unconditional sampling; refusing to continue with a stub."
        )


if importlib.util.find_spec("ultralytics") is None:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.__spec__ = ModuleSpec("ultralytics", loader=None, is_package=True)
    ultralytics.__path__ = []
    ultralytics.YOLO = _UnavailableOptionalDependency
    sys.modules["ultralytics"] = ultralytics
    print(
        "[GEM adapter] fail-closed Ultralytics stub active for the unused "
        "unconditional path.",
        file=sys.stderr,
    )

if importlib.util.find_spec("pycocotools") is None:
    pycocotools = types.ModuleType("pycocotools")
    pycocotools.__spec__ = ModuleSpec(
        "pycocotools", loader=None, is_package=True
    )
    pycocotools.__path__ = []
    coco = types.ModuleType("pycocotools.coco")
    coco.__spec__ = ModuleSpec("pycocotools.coco", loader=None)
    cocoeval = types.ModuleType("pycocotools.cocoeval")
    cocoeval.__spec__ = ModuleSpec("pycocotools.cocoeval", loader=None)
    coco.COCO = _UnavailableOptionalDependency
    cocoeval.COCOeval = _UnavailableOptionalDependency
    pycocotools.coco = coco
    pycocotools.cocoeval = cocoeval
    sys.modules.update(
        {
            "pycocotools": pycocotools,
            "pycocotools.coco": coco,
            "pycocotools.cocoeval": cocoeval,
        }
    )
    print(
        "[GEM adapter] fail-closed pycocotools stub active for the unused "
        "unconditional path.",
        file=sys.stderr,
    )
