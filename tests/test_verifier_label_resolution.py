from __future__ import annotations

from pathlib import Path

from src.data.video_preprocess import parse_label_from_path, resolve_label_from_rules


def test_resolve_label_from_rules_prefers_specific_nonfight_over_fight() -> None:
    rules = {"fight": 1, "nonfight": 0}
    assert resolve_label_from_rules(Path("dataset/raw/rwf2000/videos/nonfight/clip.mp4"), rules) == 0


def test_resolve_label_from_rules_fight_maps_to_one() -> None:
    rules = {"fight": 1, "nonfight": 0}
    assert resolve_label_from_rules(Path("dataset/raw/rwf2000/videos/fight/clip.mp4"), rules) == 1


def test_parse_label_from_path_nonfight_variants_and_normal_map_to_zero() -> None:
    assert parse_label_from_path(Path("dataset/raw/a/videos/nonfight/clip.mp4")) == 0
    assert parse_label_from_path(Path("dataset/raw/a/videos/non_fight/clip.mp4")) == 0
    assert parse_label_from_path(Path("dataset/raw/a/videos/non-fight/clip.mp4")) == 0
    assert parse_label_from_path(Path("dataset/raw/a/videos/normal/clip.mp4")) == 0
