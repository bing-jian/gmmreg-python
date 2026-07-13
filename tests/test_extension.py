"""Unit tests for the gmmreg._extension C extension.

These exercise gauss_transform() and squared_distance_matrix() directly,
independent of the higher-level registration code in gmmreg/_core.py.

The extension must be built first:
    cd src && python setup.py build_ext --inplace
"""
import numpy as np
import pytest

extension = pytest.importorskip(
    "gmmreg._extension",
    reason="C extension not built; run `cd src && python setup.py build_ext --inplace` first.",
)


def ref_gauss_transform(A, B, scale):
    """Reference cross Gauss transform: f = mean_ij exp(-||a_i - b_j||^2 / scale^2)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape[0], B.shape[0]
    diff = A[:, None, :] - B[None, :, :]
    cost = np.exp(-np.sum(diff ** 2, axis=2) / scale ** 2)
    f = cost.sum() / (m * n)
    grad = -2 * np.einsum('ij,ijd->id', cost, diff) / (m * n * scale ** 2)
    return f, grad


def ref_squared_distance_matrix(X, Y, g):
    """Reference metric squared distance: dist[r, c] = (Y_c - X_r)^T g (Y_c - X_r)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    g = np.asarray(g, dtype=float)
    diff = Y[None, :, :] - X[:, None, :]
    return np.einsum('mnd,de,mne->mn', diff, g, diff)


def random_points(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, d))


@pytest.fixture(params=[1, 2, 3], ids=['1d', '2d', '3d'])
def dim(request):
    return request.param


class TestGaussTransform:

    @pytest.mark.parametrize("m,n,scale", [(4, 5, 1.0), (1, 1, 0.7), (3, 7, 2.5)])
    def test_matches_reference(self, dim, m, n, scale):
        A = random_points(m, dim, seed=1)
        B = random_points(n, dim, seed=2)
        f, grad = extension.gauss_transform(A, B, scale)
        f_ref, grad_ref = ref_gauss_transform(A, B, scale)
        assert f == pytest.approx(f_ref, rel=1e-10)
        np.testing.assert_allclose(grad, grad_ref, rtol=1e-10)

    def test_value_is_bounded(self, dim):
        # every pairwise term is a Gaussian kernel in (0, 1], so the mean is too.
        A = random_points(6, dim, seed=3)
        B = random_points(6, dim, seed=4)
        f, _ = extension.gauss_transform(A, B, 1.3)
        assert 0.0 < f <= 1.0

    def test_single_identical_point_gives_max_value(self, dim):
        A = random_points(1, dim, seed=5)
        f, grad = extension.gauss_transform(A, A, 1.0)
        assert f == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_gradient_matches_finite_differences(self, dim):
        A = random_points(5, dim, seed=6)
        B = random_points(6, dim, seed=7)
        scale = 0.9
        _, grad = extension.gauss_transform(A, B, scale)

        eps = 1e-6
        numerical = np.zeros_like(A)
        for i in range(A.shape[0]):
            for d in range(A.shape[1]):
                A_plus, A_minus = A.copy(), A.copy()
                A_plus[i, d] += eps
                A_minus[i, d] -= eps
                f_plus, _ = extension.gauss_transform(A_plus, B, scale)
                f_minus, _ = extension.gauss_transform(A_minus, B, scale)
                numerical[i, d] = (f_plus - f_minus) / (2 * eps)
        np.testing.assert_allclose(grad, numerical, atol=1e-5)

    def test_invalid_ndim_raises_valueerror(self):
        # A has ndim == 3 (shape (2, 2, 2)) -- not a valid (n_points, n_dims)
        # point set -- and should be rejected with ValueError rather than
        # crashing the interpreter.
        A = np.zeros((2, 2, 2))
        B = np.zeros((3, 2))
        with pytest.raises(ValueError):
            extension.gauss_transform(A, B, 1.0)

    def test_accepts_noncontiguous_input(self, dim):
        A = random_points(8, dim, seed=8)[::2]  # strided, non-contiguous view
        B = random_points(5, dim, seed=9)
        assert not A.flags['C_CONTIGUOUS']
        f, grad = extension.gauss_transform(A, B, 1.0)
        f_ref, grad_ref = ref_gauss_transform(A, B, 1.0)
        assert f == pytest.approx(f_ref, rel=1e-10)
        np.testing.assert_allclose(grad, grad_ref, rtol=1e-10)


class TestSquaredDistanceMatrix:

    def test_matches_reference_identity_metric(self, dim):
        X = random_points(4, dim, seed=10)
        Y = random_points(6, dim, seed=11)
        g = np.eye(dim)
        dist = extension.squared_distance_matrix(X, Y, g)
        np.testing.assert_allclose(dist, ref_squared_distance_matrix(X, Y, g), rtol=1e-10)

    def test_matches_reference_general_metric(self, dim):
        rng = np.random.default_rng(12)
        M = rng.uniform(-1, 1, size=(dim, dim))
        g = M @ M.T + dim * np.eye(dim)  # a symmetric positive-definite metric
        X = random_points(5, dim, seed=13)
        Y = random_points(4, dim, seed=14)
        dist = extension.squared_distance_matrix(X, Y, g)
        np.testing.assert_allclose(dist, ref_squared_distance_matrix(X, Y, g), rtol=1e-10)

    def test_zero_distance_for_identical_points(self, dim):
        X = random_points(5, dim, seed=15)
        dist = extension.squared_distance_matrix(X, X, np.eye(dim))
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-12)

    def test_output_shape(self, dim):
        X = random_points(3, dim, seed=16)
        Y = random_points(7, dim, seed=17)
        dist = extension.squared_distance_matrix(X, Y, np.eye(dim))
        assert dist.shape == (3, 7)

    def test_single_point(self, dim):
        X = random_points(1, dim, seed=18)
        Y = random_points(1, dim, seed=19)
        g = np.eye(dim)
        dist = extension.squared_distance_matrix(X, Y, g)
        np.testing.assert_allclose(dist, ref_squared_distance_matrix(X, Y, g), rtol=1e-10)
