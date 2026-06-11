"""Compute P/R/F1 from classification verdicts."""

from dataclasses import dataclass

from src.evals.classify import ClassifyOutput


@dataclass
class Score:
    supported: int
    contradicted: int
    unsupported: int
    uncovered: int
    precision: float
    recall: float
    f1: float


def compute_score(output: ClassifyOutput) -> Score:
    supported = sum(1 for v in output.verdicts.values() if v.upper() == "SUPPORTED")
    contradicted = sum(1 for v in output.verdicts.values() if v.upper() == "CONTRADICTED")
    unsupported = sum(1 for v in output.verdicts.values() if v.upper() == "UNSUPPORTED")
    uncovered = len(output.uncovered_ref_indices)

    tp = supported
    fp = contradicted + unsupported
    fn = uncovered

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return Score(
        supported=supported,
        contradicted=contradicted,
        unsupported=unsupported,
        uncovered=uncovered,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )
