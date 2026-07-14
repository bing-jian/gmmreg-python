"""Unit tests for gmmreg._core: the registration math (Gaussian mixture
distances, TPS basis construction, and the multi-level optimizer).

These focus on invariants that are cheap to check independently of any
"golden" numeric reference: analytic gradients vs. finite differences, exact
algebraic identities (e.g. TPS identity initialization), and end-to-end
convergence of the optimizer on a known rigid transform.
"""
import numpy as np
import pytest

core = pytest.importorskip(
    "gmmreg._core",
    reason="gmmreg package not installed; run `pip install -e ./src` first.",
)


def random_points(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, d))


def numerical_gradient(f, x, eps=1e-6):
    """Central-difference gradient of a scalar function f(x) w.r.t. array x."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        x_plus = x.copy()
        x_plus[idx] += eps
        x_minus = x.copy()
        x_minus[idx] -= eps
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


@pytest.fixture(params=[2, 3], ids=['2d', '3d'])
def dim(request):
    return request.param


class TestNormalize:

    def test_result_has_zero_mean_and_unit_scale(self, dim):
        x = random_points(10, dim, seed=1) * 5
        xn, centroid, scale = core.normalize(x)
        np.testing.assert_allclose(xn.mean(0), 0.0, atol=1e-12)
        assert np.linalg.norm(xn, 'fro') / np.sqrt(xn.shape[0]) == pytest.approx(1.0)

    def test_denormalize_is_inverse(self, dim):
        x = random_points(10, dim, seed=2) * 5
        xn, centroid, scale = core.normalize(x)
        np.testing.assert_allclose(core.denormalize(xn, centroid, scale), x, atol=1e-10)


class TestGaussianMixtureDistances:

    @pytest.mark.parametrize("fn", [core.L2_distance, core.correlation])
    def test_gradient_matches_finite_differences(self, dim, fn):
        model = random_points(5, dim, seed=3)
        scene = random_points(6, dim, seed=4)
        scale = 0.7
        _, grad = fn(model, scene, scale)
        numerical = numerical_gradient(lambda m: fn(m, scene, scale)[0], model)
        np.testing.assert_allclose(grad, numerical, atol=1e-5)

    def test_correlation_is_negative(self, dim):
        # correlation() returns -f2^2/f1 with f1, f2 > 0 (sums of Gaussian
        # kernels), so it should always be strictly negative.
        model = random_points(5, dim, seed=5)
        scene = random_points(6, dim, seed=6)
        f, _ = core.correlation(model, scene, 0.7)
        assert f < 0


class TestTPSBasis:

    def test_zero_param_with_ctrl_pts_as_landmarks_is_identity(self, dim):
        # init_param()'s affine block encodes the identity transform and its
        # TPS block is all zeros, so when the basis is evaluated at the
        # control points themselves, transform_points must reproduce them
        # exactly (no bending is applied).
        n = 8
        ctrl_pts = random_points(n, dim, seed=7)
        basis, kernel = core.prepare_TPS_basis(ctrl_pts, ctrl_pts)
        x0 = core.init_param(n, dim)
        after = core.transform_points(x0, basis)
        np.testing.assert_allclose(after, ctrl_pts, atol=1e-8)

    def test_init_param_without_affine_is_tps_block_only(self, dim):
        n = 8
        x0 = core.init_param(n, dim, opt_affine=False)
        assert x0.shape == (dim * n - dim * (dim + 1),)
        np.testing.assert_allclose(x0, 0.0)

    def test_basis_and_kernel_shapes(self, dim):
        n, m = 8, 5
        ctrl_pts = random_points(n, dim, seed=8)
        landmarks = random_points(m, dim, seed=9)
        basis, kernel = core.prepare_TPS_basis(landmarks, ctrl_pts)
        assert basis.shape == (m, n)
        assert kernel.shape == (n - dim - 1, n - dim - 1)

    def test_compute_TPS_K_matrix_is_symmetric(self, dim):
        ctrl_pts = random_points(6, dim, seed=10)
        K, _ = core.compute_TPS_K(ctrl_pts)
        np.testing.assert_allclose(K, K.T)


class TestComputeGRBF:

    def test_diagonal_is_one(self, dim):
        # kernel_func(0, sigma) == exp(0) == 1, so a point's kernel value
        # with itself is always 1 regardless of sigma.
        ctrl_pts = random_points(5, dim, seed=11)
        K, _ = core.compute_GRBF(ctrl_pts, ctrl_pts, 0.5)
        np.testing.assert_allclose(np.diag(K), 1.0)

    def test_matches_reference(self, dim):
        ctrl_pts = random_points(4, dim, seed=12)
        landmarks = random_points(3, dim, seed=13)
        sigma = 0.6
        K, U = core.compute_GRBF(ctrl_pts, landmarks, sigma)

        def ref_kernel(A, B):
            diff = A[:, None, :] - B[None, :, :]
            r = np.linalg.norm(diff, axis=2)
            return np.exp(-(r ** 2) / sigma ** 2)

        np.testing.assert_allclose(K, ref_kernel(ctrl_pts, ctrl_pts))
        np.testing.assert_allclose(U, ref_kernel(landmarks, ctrl_pts))


class TestObjectives:

    def _setup(self, dim, seed):
        n, m = 6, 7
        landmarks = random_points(m, dim, seed=seed)
        ctrl_pts = random_points(n, dim, seed=seed + 1)
        scene = random_points(8, dim, seed=seed + 2)
        basis, kernel = core.prepare_TPS_basis(landmarks, ctrl_pts)
        rng = np.random.default_rng(seed + 3)
        x0 = core.init_param(n, dim) + rng.normal(scale=0.05, size=dim * n)
        return x0, basis, kernel, scene

    def test_obj_L2_TPS_gradient_matches_finite_differences(self, dim):
        x0, basis, kernel, scene = self._setup(dim, seed=14)
        scale, _lambda = 0.5, 0.1
        _, grad = core.obj_L2_TPS(x0, basis, kernel, scene, scale, _lambda)
        numerical = numerical_gradient(
            lambda x: core.obj_L2_TPS(x, basis, kernel, scene, scale, _lambda)[0], x0)
        np.testing.assert_allclose(grad, numerical, atol=1e-5)

    def test_obj_KC_TPS_gradient_matches_finite_differences(self, dim):
        x0, basis, kernel, scene = self._setup(dim, seed=18)
        scale, alpha, beta = 0.5, 1.0, 0.1
        _, grad = core.obj_KC_TPS(x0, basis, kernel, scene, scale, alpha, beta)
        numerical = numerical_gradient(
            lambda x: core.obj_KC_TPS(x, basis, kernel, scene, scale, alpha, beta)[0], x0)
        np.testing.assert_allclose(grad, numerical, atol=1e-5)

    def test_obj_TPS_matches_obj_L2_TPS_for_L2_distance(self, dim):
        x0, basis, kernel, scene = self._setup(dim, seed=22)
        scale, _lambda = 0.5, 0.1
        f1, g1 = core.obj_TPS(core.L2_distance, x0, basis, kernel, scene, scale, _lambda)
        f2, g2 = core.obj_L2_TPS(x0, basis, kernel, scene, scale, _lambda)
        assert f1 == pytest.approx(f2)
        np.testing.assert_allclose(g1, g2)


class TestRunMultiLevel:

    def test_recovers_known_rigid_transform(self, dim):
        # Register a point set against a rigidly transformed copy of itself,
        # using the point set as its own control points. Since the exact
        # correspondence is representable by the TPS model, the optimizer
        # should drive the mean squared error to the scene down to ~0.
        model = random_points(20, dim, seed=23)
        rng = np.random.default_rng(24)
        if dim == 2:
            theta = 0.15
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        else:
            # small rotation about the z-axis
            theta = 0.15
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
        t = rng.uniform(-0.05, 0.05, size=dim)
        scene = model @ R.T + t

        after = core.run_multi_level(
            model, scene, model,
            level=3, scales=[0.5, 0.2, 0.1], lambdas=[0.1, 0.05, 0.01],
            iters=[150, 150, 150])

        before_err = np.mean(np.sum((model - scene) ** 2, axis=1))
        after_err = np.mean(np.sum((after - scene) ** 2, axis=1))
        assert after_err < before_err * 1e-4
