"""End-to-end tests for gmmreg.run_config(): the full pipeline of loading a
config file (.ini or .yaml), loading the point sets it references, and
running the multi-level TPS registration.

test_config.py already covers read_sections()/as_list() in isolation, and
test_core.py covers run_multi_level() in isolation; this ties the two
together the way a real invocation (e.g. `demo.py some_config.ini`) does, and
locks in that .ini and .yaml configs describing the same problem produce the
same result.
"""
import textwrap

import numpy as np
import pytest

gmmreg = pytest.importorskip(
    "gmmreg",
    reason="gmmreg package not installed; run `pip install -e ./src` first.",
)


def rigidly_transformed(model, theta, translation):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return model @ R.T + translation


INI_TEMPLATE = """
    [FILES]
    model = {model}
    scene = {scene}
    {ctrl_pts_line}

    [GMMREG_OPT]
    normalize = 1
    level = 2
    sigma = .3 .1
    lambda = .1 .01
    max_function_evals = 100 100
    """

YAML_TEMPLATE = """
    FILES:
      model: {model}
      scene: {scene}
      {ctrl_pts_line}

    GMMREG_OPT:
      normalize: 1
      level: 2
      sigma: [0.3, 0.1]
      lambda: [0.1, 0.01]
      max_function_evals: [100, 100]
    """


@pytest.fixture
def problem(tmp_path):
    rng = np.random.default_rng(42)
    model = rng.uniform(-1, 1, size=(15, 2))
    scene = rigidly_transformed(model, theta=0.2, translation=np.array([0.05, -0.03]))

    model_file = tmp_path / 'model.txt'
    scene_file = tmp_path / 'scene.txt'
    np.savetxt(model_file, model)
    np.savetxt(scene_file, scene)
    return dict(tmp_path=tmp_path, model=model, scene=scene,
                model_file=str(model_file), scene_file=str(scene_file))


def write_config(problem, name, template, with_ctrl_pts=True):
    if not with_ctrl_pts:
        ctrl_pts_line = ''
    elif template is INI_TEMPLATE:
        ctrl_pts_line = 'ctrl_pts = ' + problem['model_file']
    else:
        ctrl_pts_line = 'ctrl_pts: ' + problem['model_file']
    text = template.format(model=problem['model_file'], scene=problem['scene_file'],
                           ctrl_pts_line=ctrl_pts_line)
    path = problem['tmp_path'] / name
    path.write_text(textwrap.dedent(text))
    return str(path)


class TestRunConfig:

    def test_ini_and_yaml_configs_agree(self, problem):
        ini_path = write_config(problem, 'config.ini', INI_TEMPLATE)
        yaml_path = write_config(problem, 'config.yaml', YAML_TEMPLATE)

        model_i, scene_i, after_i = gmmreg.run_config(ini_path)
        model_y, scene_y, after_y = gmmreg.run_config(yaml_path)

        np.testing.assert_allclose(model_i, model_y)
        np.testing.assert_allclose(scene_i, scene_y)
        np.testing.assert_allclose(after_i, after_y, atol=1e-8)

    def test_registration_reduces_distance_to_scene(self, problem):
        ini_path = write_config(problem, 'config.ini', INI_TEMPLATE)
        model, scene, after = gmmreg.run_config(ini_path)

        before_err = np.mean(np.sum((model - scene) ** 2, axis=1))
        after_err = np.mean(np.sum((after - scene) ** 2, axis=1))
        assert after_err < before_err

    def test_missing_ctrl_pts_falls_back_to_model(self, problem):
        ini_path = write_config(problem, 'config.ini', INI_TEMPLATE, with_ctrl_pts=False)
        model, scene, after = gmmreg.run_config(ini_path)
        # should run without error and still improve on the initial alignment
        before_err = np.mean(np.sum((model - scene) ** 2, axis=1))
        after_err = np.mean(np.sum((after - scene) ** 2, axis=1))
        assert after_err < before_err

    def test_model_and_scene_loaded_from_referenced_files(self, problem):
        ini_path = write_config(problem, 'config.ini', INI_TEMPLATE)
        model, scene, _ = gmmreg.run_config(ini_path)
        np.testing.assert_allclose(model, problem['model'])
        np.testing.assert_allclose(scene, problem['scene'])
