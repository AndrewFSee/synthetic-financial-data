"""Labeled structural-break datasets for testing break / change-point detectors."""

from synfin.structural_breaks.evaluation import auc_report, baseline_scores, summarize, ts_auc
from synfin.structural_breaks.formats import save_adia, to_adia_offline, to_adia_realtime
from synfin.structural_breaks.generator import StructuralBreakDataset, generate
from synfin.structural_breaks.presets import PRESETS, GeneratorConfig, get_preset

__all__ = [
    "generate",
    "get_preset",
    "GeneratorConfig",
    "PRESETS",
    "StructuralBreakDataset",
    "save_adia",
    "to_adia_offline",
    "to_adia_realtime",
    "auc_report",
    "baseline_scores",
    "summarize",
    "ts_auc",
]
