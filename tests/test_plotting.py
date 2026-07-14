"""Unit tests for gmmreg.plotting.

Uses the non-interactive Agg backend so plt.show() doesn't try to open a
window; must be set before matplotlib.pyplot is imported anywhere, which is
why it happens before the gmmreg.plotting import below.
"""
import textwrap

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

plotting = pytest.importorskip(
    "gmmreg.plotting",
    reason="gmmreg package not installed; run `pip install -e ./src` first.",
)


def random_points(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, d))


class TestDisplayFunctions:

    def test_display2Dpointset_runs_without_error(self):
        plotting.display2Dpointset(random_points(5, 2, seed=0))

    def test_display2Dpointsets_runs_without_error(self):
        plotting.display2Dpointsets(random_points(5, 2, seed=1), random_points(6, 2, seed=2))

    def test_display3Dpointsets_runs_without_error(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plotting.display3Dpointsets(random_points(5, 3, seed=3), random_points(6, 3, seed=4), ax)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_displayABC_runs_without_error(self, dim):
        plotting.displayABC(random_points(5, dim, seed=5),
                            random_points(6, dim, seed=6),
                            random_points(5, dim, seed=7))


class TestDisplayPts:

    def test_reads_files_referenced_by_config_and_displays(self, tmp_path):
        model = random_points(5, 2, seed=8)
        scene = random_points(6, 2, seed=9)
        transformed = random_points(5, 2, seed=10)
        model_file = tmp_path / 'model.txt'
        scene_file = tmp_path / 'scene.txt'
        transformed_file = tmp_path / 'transformed.txt'
        np.savetxt(model_file, model)
        np.savetxt(scene_file, scene)
        np.savetxt(transformed_file, transformed)

        config = tmp_path / 'config.ini'
        config.write_text(textwrap.dedent(f"""
            [FILES]
            model = {model_file}
            scene = {scene_file}
            transformed_model = {transformed_file}
            """))

        plotting.display_pts(str(config))
