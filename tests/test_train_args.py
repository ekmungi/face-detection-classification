"""Tests for how train.py reads the data mode from the experiment specification and the command line."""
import pytest

pytest.importorskip("torch")

import train  # noqa: E402


def write_spec(path, data_table):
    """Writes a minimal experiment specification with the given [data] table text."""
    path.write_text(f"[data]\n{data_table}\n[train]\nepochs = 1\n")
    return str(path)


def test_the_data_mode_defaults_to_fetch():
    assert train.parse_args([]).data_mode == "fetch"


def test_the_data_mode_comes_from_the_specification(tmp_path):
    spec = write_spec(tmp_path / "spec.toml", 'mode = "store"')
    assert train.parse_args(["--spec", spec]).data_mode == "store"


def test_the_command_line_overrides_the_specification(tmp_path):
    spec = write_spec(tmp_path / "spec.toml", 'mode = "store"')
    assert train.parse_args(["--spec", spec, "--data-mode", "fetch"]).data_mode == "fetch"


def test_an_unknown_data_mode_is_rejected():
    with pytest.raises(SystemExit):
        train.parse_args(["--data-mode", "sometimes"])


def write_text_spec(path, text):
    """Writes a specification file from its full text and returns the path as a string."""
    path.write_text(text)
    return str(path)


def test_a_list_of_data_tables_with_one_mode_uses_that_mode(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[[data]]\nname = "A"\nmode = "store"\n[[data]]\nname = "B"\nmode = "store"\n[train]\nepochs = 1\n')
    assert train.parse_args(["--spec", spec]).data_mode == "store"


def test_a_single_data_table_in_a_list_works_and_a_missing_mode_means_fetch(tmp_path):
    only_store = write_text_spec(tmp_path / "a.toml", '[[data]]\nname = "A"\nmode = "store"\n')
    no_mode = write_text_spec(tmp_path / "b.toml", '[[data]]\nname = "A"\n[[data]]\nname = "B"\n')
    assert train.parse_args(["--spec", only_store]).data_mode == "store"
    assert train.parse_args(["--spec", no_mode]).data_mode == "fetch"


def test_data_tables_with_different_modes_stop_with_a_message_that_names_the_choice(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[[data]]\nname = "A"\nmode = "store"\n[[data]]\nname = "B"\nmode = "fetch"\n')
    with pytest.raises(SystemExit) as error:
        train.parse_args(["--spec", spec])
    message = str(error.value)
    assert "--data-mode" in message and "store" in message and "fetch" in message and "A" in message and "B" in message


def test_different_modes_are_fine_when_the_command_line_chooses_one(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[[data]]\nname = "A"\nmode = "store"\n[[data]]\nname = "B"\nmode = "fetch"\n')
    assert train.parse_args(["--spec", spec, "--data-mode", "store"]).data_mode == "store"


def test_a_train_key_that_names_a_fixed_option_becomes_its_default(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[train]\nmodel = "mlp"\naugment = true\nexp = "exp-007"\nmax_steps = 5\nepochs = 2\n')
    args = train.parse_args(["--spec", spec])
    assert (args.model, args.augment, args.exp, args.max_steps, args.epochs) == ("mlp", True, "exp-007", 5, 2)


def test_the_command_line_still_overrides_a_colliding_train_key(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\nmodel = "mlp"\nexp = "exp-007"\n')
    args = train.parse_args(["--spec", spec, "--model", "cnn", "--exp", "exp-002"])
    assert (args.model, args.exp) == ("cnn", "exp-002")


def test_a_train_key_that_names_a_model_that_does_not_exist_is_rejected_clearly(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\nmodel = "transformer-xl"\n')
    with pytest.raises(SystemExit):
        train.parse_args(["--spec", spec])


def test_a_hyphenated_train_key_is_the_same_option_as_the_underscored_one(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\nmax-steps = 5\nbatch-size = 32\nmy-knob = 2\nmodel = "mlp"\n')
    args = train.parse_args(["--spec", spec])
    assert (args.max_steps, args.batch_size, args.my_knob, args.model) == (5, 32, 2, "mlp")
    assert train.parse_args(["--spec", spec, "--max-steps", "9"]).max_steps == 9


def test_a_train_key_written_both_ways_is_an_error_that_names_both(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\nmax-steps = 5\nmax_steps = 6\n')
    with pytest.raises(SystemExit) as error:
        train.parse_args(["--spec", spec])
    assert "max-steps" in str(error.value) and "max_steps" in str(error.value)


@pytest.mark.parametrize("key", ['"my key"', '"1st"', '"a.b"'])
def test_a_train_key_that_cannot_be_an_option_name_is_refused_with_the_key(tmp_path, key):
    spec = write_text_spec(tmp_path / "spec.toml", f"[train]\n{key} = 1\n")
    with pytest.raises(SystemExit) as error:
        train.parse_args(["--spec", spec])
    assert key.strip('"') in str(error.value)


def test_a_train_table_that_is_not_a_table_is_refused(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", 'train = 3\n')
    with pytest.raises(SystemExit, match="train"):
        train.parse_args(["--spec", spec])


def test_a_true_or_false_train_key_is_a_real_flag(tmp_path):
    on = write_text_spec(tmp_path / "on.toml", '[train]\npin_memory = true\n')
    off = write_text_spec(tmp_path / "off.toml", '[train]\npin_memory = false\n')
    assert train.parse_args(["--spec", on]).pin_memory is True
    assert train.parse_args(["--spec", off]).pin_memory is False
    assert train.parse_args(["--spec", on, "--pin-memory", "false"]).pin_memory is False
    assert train.parse_args(["--spec", on, "--pin-memory", "true"]).pin_memory is True
    assert train.parse_args(["--spec", off, "--pin-memory"]).pin_memory is True
    assert train.parse_args(["--spec", off, "--pin-memory", "--epochs", "2"]).pin_memory is True


def test_a_flag_value_that_is_not_true_or_false_is_rejected(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\npin_memory = true\n')
    with pytest.raises(SystemExit):
        train.parse_args(["--spec", spec, "--pin-memory", "maybe"])


def test_the_augment_flag_can_be_turned_off_on_the_command_line_when_the_file_turns_it_on(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\naugment = true\n')
    assert train.parse_args(["--spec", spec]).augment is True
    assert train.parse_args(["--spec", spec, "--augment", "false"]).augment is False
    assert train.parse_args(["--augment"]).augment is True
    assert train.parse_args([]).augment is False


def test_a_list_valued_train_key_takes_one_or_more_values_of_its_own_type(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\nlayers = [32, 64]\nrates = [0.5, 0.25]\ntags = ["a", "b"]\n')
    args = train.parse_args(["--spec", spec])
    assert (args.layers, args.rates, args.tags) == ([32, 64], [0.5, 0.25], ["a", "b"])
    args = train.parse_args(["--spec", spec, "--layers", "16", "--rates", "0.1", "0.2", "0.3"])
    assert (args.layers, args.rates) == ([16], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("value, why", [("[]", "empty list"), ('[1, "a"]', "mixed types"), ("[[1], [2]]", "list of lists"),
                                        ("[true, false]", "list of booleans"), ("{ a = 1 }", "table")])
def test_a_train_value_that_cannot_be_an_option_is_refused_with_the_key(tmp_path, value, why):
    spec = write_text_spec(tmp_path / "spec.toml", f"[train]\nlayers = {value}\n")
    with pytest.raises(SystemExit, match="layers"):
        train.parse_args(["--spec", spec])
