"""Tests for the notebook's three modes: smoke, medium and full.

Medium mode sits between the two: one run on a Colab L4 that gives meaningful results with all eight expressions.
The cells that need a GPU cannot run here, so these tests run the configuration cell with stand-ins and read the
source of the other cells.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PARAMETERS = {"DET_BUDGET_MIN": 50.0, "ID_BUDGET_MIN": 55.0, "EXPR_BUDGET_MIN": 12.0}


def builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_cells():
    return [cell[1] for cell in builder().CELLS if cell[0] == "code"]


def markdown_cells():
    return [cell[1] for cell in builder().CELLS if cell[0] == "markdown"]


def one_cell(marker):
    found = [source for source in code_cells() if marker in source]
    assert len(found) == 1, f"expected one code cell with {marker!r}, found {len(found)}"
    return found[0]


def run_config(mode):
    """Runs the configuration cell for a mode and returns its variables."""
    scope = {"MODE": mode, "CPUS": 12, "json": json, "SUMMARY": {}, **PARAMETERS}
    exec(compile(one_cell("CFG = {"), "config-cell", "exec"), scope)
    return scope


def test_medium_mode_reads_the_two_digiface_parts_already_in_drive():
    cfg = run_config("medium")["CFG"]
    assert cfg["digiface_train"] == ["0-1999"] and cfg["digiface_test"] == "8000-9999"


def test_medium_mode_trains_identity_on_about_a_thousand_people_and_tests_on_two_hundred():
    cfg = run_config("medium")["CFG"]
    assert cfg["id_max_ids"] - cfg["id_val_ids"] == 1000
    assert cfg["id_test_ids"] == 200 and cfg["id_depth"] == "iresnet18"


def test_medium_mode_reads_every_emonet_binary_file_so_all_eight_expressions_appear():
    cfg = run_config("medium")["CFG"]
    assert cfg["emonet_files"] == 5 and cfg["emonet_max"] is None
    assert cfg["hq_files"] == 1


def test_medium_mode_uses_three_thousand_wider_training_photos_and_five_hundred_test_photos():
    cfg = run_config("medium")["CFG"]
    assert cfg["wider_train_max"] == 3000 and cfg["wider_val_max"] == 500


def test_medium_mode_trains_to_its_own_time_budgets_and_ignores_the_budget_parameters():
    scope = run_config("medium")
    assert (scope["CFG"]["det_max_steps"], scope["CFG"]["expr_max_steps"], scope["CFG"]["id_max_steps"]) == (0, 0, 0)
    assert (scope["DET_BUDGET_MIN"], scope["ID_BUDGET_MIN"], scope["EXPR_BUDGET_MIN"]) == (6.0, 8.0, 3.0)


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_smoke_and_full_modes_keep_the_budget_parameters(mode):
    scope = run_config(mode)
    assert {name: scope[name] for name in PARAMETERS} == PARAMETERS


def test_portraits_without_a_face_are_centre_cropped_in_every_mode_but_full():
    source = one_cell("build_emonet_array")
    assert "fallback_center_crop=SMOKE" not in source
    assert source.count('fallback_center_crop=MODE != "full"') == 2


def test_the_setup_cell_stops_on_an_unknown_mode_and_claims_the_run_folder_first():
    source = one_cell('os.environ.get("FACEKIT_REPO"')
    assert 'if MODE not in ("smoke", "medium", "full"):' in source
    assert source.index("paths.check_run_name_for_mode(MODE, RUN_ID)") < source.index("RUN = paths.run_dir(EXP, RUN_ID)")
    made = source.index("RUN.mkdir(parents=True, exist_ok=True)")
    claimed = source.index("runlib.claim_run_mode(RUN, MODE)")
    assert made < claimed < source.index('FIG = RUN / "figures"')


def test_capped_wider_subsets_are_spread_over_every_kind_of_event():
    assert 'prepare.evenly_spaced(retina, CFG["wider_train_max"])' in one_cell("split_records(records")
    assert 'max_items=CFG["wider_val_max"], spread=True' in one_cell('rec["keep"][subset]')


def test_the_parameters_cell_names_all_three_modes():
    source = one_cell("# Parameters (papermill overrides these).")
    mode_line = next(line for line in source.splitlines() if line.startswith("MODE ="))
    assert all(f'"{mode}"' in mode_line for mode in ("smoke", "medium", "full"))


def test_the_notebook_explains_medium_mode_with_an_unmeasured_time_estimate():
    text = "\n".join(markdown_cells())
    assert "medium" in text.lower()
    assert "not measured" in text
