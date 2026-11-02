"""Configs: every recipe is valid, and bad knobs are all reported before device work."""

from pathlib import Path

from _tiny import ROOT, SCRIPT, TINY_ARCH, knobs, run_checks
from kohakuengine import load_config_file

from latentdragon.training.entry import check
from latentdragon_common.knobs import script_knobs, supplied_knobs

RECIPES = sorted((ROOT / "configs" / "ae" / "recipes").rglob("*.py"))


def every_recipe_is_valid(folder) -> None:
    assert RECIPES
    for path in RECIPES:
        values = supplied_knobs(dict(load_config_file(path).globals_dict))
        assert isinstance(values["ARCH"], dict), path
        assert set(values) <= script_knobs(SCRIPT), path
        assert '"/' not in Path(path).read_text(encoding="utf-8"), path

        merged = knobs(folder, MAX_STEPS=0)
        merged.update(values)
        if "INIT_FROM" in values:
            merged["INIT_FROM"] = str(folder)
        check(merged, str(SCRIPT))


def all_problems_are_reported_together(folder) -> None:
    values = knobs(
        folder,
        GAN_SCHEME="nope",
        MAX_STEPS=0,
        BATCH_SIZE=3,
        GRAD_ACC=2,
        FREEZE=["trunk.nothing"],
        TRAIN_SOURCES=[],
    )
    values["BATCH_SZE"] = 2

    try:
        check(values, str(SCRIPT))
    except SystemExit as err:
        text = str(err)
    else:
        raise AssertionError("bad knobs were accepted")

    expected = (
        "did you mean BATCH_SIZE",
        "nope",
        "MAX_STEPS",
        "not divisible",
        "trunk.nothing",
        "TRAIN_SOURCES",
    )
    for needle in expected:
        assert needle in text, needle


def bad_arch_is_reported(folder) -> None:
    values = knobs(folder, ARCH={**TINY_ARCH, "trunk": "nope"})
    try:
        check(values, str(SCRIPT))
    except SystemExit as err:
        assert "ARCH" in str(err)
    else:
        raise AssertionError("a bad arch was accepted")


if __name__ == "__main__":
    run_checks(
        every_recipe_is_valid,
        all_problems_are_reported_together,
        bad_arch_is_reported,
    )
