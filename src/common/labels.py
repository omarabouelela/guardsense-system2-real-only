"""Shared label definitions for GuardSense System 2."""

from __future__ import annotations

from dataclasses import dataclass

LABEL_NORMAL = 0
LABEL_FIGHT = 1

DEFAULT_LABEL_TO_NAME: dict[int, str] = {
    LABEL_NORMAL: "normal",
    LABEL_FIGHT: "fight",
}
DEFAULT_NAME_TO_LABEL: dict[str, int] = {v: k for k, v in DEFAULT_LABEL_TO_NAME.items()}


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """Normalized class metadata."""

    class_id: int
    name: str
    description: str


DEFAULT_LABEL_SPECS: tuple[LabelSpec, ...] = (
    LabelSpec(0, "normal", "Normal / non-fight / hard-negative activity."),
    LabelSpec(1, "fight", "Active physical fighting and violent contact."),
)


def build_label_maps(label_to_name: dict[int, str] | None = None) -> tuple[dict[int, str], dict[str, int]]:
    """Return normalized label maps for the configured class set."""
    labels = dict(sorted((label_to_name or DEFAULT_LABEL_TO_NAME).items(), key=lambda item: item[0]))
    names = {name: class_id for class_id, name in labels.items()}
    return labels, names
