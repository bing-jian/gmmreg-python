"""Unit tests for gmmreg.demo, including the gmmreg-demo console-script
entry point (main()).

Uses the non-interactive Agg backend so plt.show() (triggered via
demo.test()'s display=True path) doesn't try to open a window; must be set
before matplotlib.pyplot is imported anywhere.
"""
import textwrap
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

demo = pytest.importorskip(
    "gmmreg.demo",
    reason="gmmreg package not installed; run `pip install -e ./src` first.",
)


def write_ini_config(tmp_path, model, scene, transformed_model=None):
    model_file = tmp_path / 'model.txt'
    scene_file = tmp_path / 'scene.txt'
    np.savetxt(model_file, model)
    np.savetxt(scene_file, scene)
    extra = ''
    if transformed_model is not None:
        extra = 'transformed_model = ' + str(transformed_model)

    config = tmp_path / 'config.ini'
    config.write_text(textwrap.dedent(f"""
        [FILES]
        model = {model_file}
        scene = {scene_file}
        {extra}

        [GMMREG_OPT]
        normalize = 1
        level = 1
        sigma = .3
        lambda = .1
        max_function_evals = 20
        """))
    return str(config)


@pytest.fixture
def config(tmp_path):
    rng = np.random.default_rng(0)
    model = rng.uniform(-1, 1, size=(8, 2))
    scene = rng.uniform(-1, 1, size=(8, 2))
    return write_ini_config(tmp_path, model, scene)


class TestTest:

    def test_runs_registration_and_displays(self, config):
        demo.test(config, display=True)

    def test_can_skip_display(self, config):
        demo.test(config, display=False)


class TestMain:

    def test_invokes_test_with_argv(self, config, monkeypatch):
        monkeypatch.setattr('sys.argv', ['gmmreg-demo', config])
        with patch.object(demo, 'test') as mock_test:
            demo.main()
        mock_test.assert_called_once_with(config)

    def test_exits_with_usage_message_for_wrong_argc(self, monkeypatch, capsys):
        monkeypatch.setattr('sys.argv', ['gmmreg-demo'])
        with pytest.raises(SystemExit) as exc_info:
            demo.main()
        assert exc_info.value.code == 1
        assert 'Usage' in capsys.readouterr().err


class TestRunExecutable:

    def test_invokes_subprocess_with_arg_list_and_displays(self, tmp_path):
        model = np.random.default_rng(1).uniform(-1, 1, size=(5, 2))
        scene = np.random.default_rng(2).uniform(-1, 1, size=(6, 2))
        transformed = np.random.default_rng(3).uniform(-1, 1, size=(5, 2))
        transformed_file = tmp_path / 'transformed.txt'
        np.savetxt(transformed_file, transformed)
        config = write_ini_config(tmp_path, model, scene, transformed_model=transformed_file)

        with patch('gmmreg.demo.subprocess.call') as mock_call:
            demo.run_executable('gmmreg_demo_exe', config, 'rigid', display=True)

        mock_call.assert_called_once_with(['gmmreg_demo_exe', config, 'rigid'])
