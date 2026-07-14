"""Unit tests for gmmreg._config: shared .ini/.yaml config file loading."""
import textwrap

import pytest

config = pytest.importorskip(
    "gmmreg._config",
    reason="gmmreg package not installed; run `pip install -e ./src` first.",
)


INI_BODY = """
    [FILES]
    model = model.txt
    scene = scene.txt

    [GMMREG_OPT]
    normalize = 1
    level = 2
    sigma = .3 .2
    lambda = .1 0
    max_function_evals = 100 200
    """

YAML_BODY = """
    FILES:
      model: model.txt
      scene: scene.txt

    GMMREG_OPT:
      normalize: 1
      level: 2
      sigma: [0.3, 0.2]
      lambda: [0.1, 0]
      max_function_evals: [100, 200]
    """


def write_config(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return str(path)


@pytest.mark.parametrize("name,body", [("config.ini", INI_BODY), ("config.yaml", YAML_BODY), ("config.yml", YAML_BODY)])
def test_read_sections_parses_files_and_opt(tmp_path, name, body):
    path = write_config(tmp_path, name, body)
    sections = config.read_sections(path)

    assert sections['FILES']['model'] == 'model.txt'
    assert sections['FILES']['scene'] == 'scene.txt'
    assert int(sections['GMMREG_OPT']['normalize']) == 1
    assert int(sections['GMMREG_OPT']['level']) == 2
    assert config.as_list(sections['GMMREG_OPT']['sigma'], float) == [0.3, 0.2]
    assert config.as_list(sections['GMMREG_OPT']['lambda'], float) == [0.1, 0.0]
    assert config.as_list(sections['GMMREG_OPT']['max_function_evals'], int) == [100, 200]


def test_ini_and_yaml_configs_agree(tmp_path):
    ini_sections = config.read_sections(write_config(tmp_path, 'config.ini', INI_BODY))
    yaml_sections = config.read_sections(write_config(tmp_path, 'config.yaml', YAML_BODY))

    for section in ('FILES', 'GMMREG_OPT'):
        for key, ini_value in ini_sections[section].items():
            yaml_value = yaml_sections[section][key]
            if key in ('sigma', 'lambda', 'max_function_evals'):
                assert config.as_list(ini_value, float) == config.as_list(yaml_value, float)
            else:
                assert str(ini_value) == str(yaml_value)


class TestAsList:

    def test_splits_whitespace_separated_string(self):
        assert config.as_list('.3 .2 .1', float) == [0.3, 0.2, 0.1]

    def test_passes_through_native_list(self):
        assert config.as_list([1, 2, 3], int) == [1, 2, 3]

    def test_casts_elements(self):
        assert config.as_list(['1', '2'], int) == [1, 2]
