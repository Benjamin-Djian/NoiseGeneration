import pytest

from ValueNoise import WorleyNoise1D, WorleyNoise2D, WorleyNoise3D


@pytest.fixture
def worley1d():
    return WorleyNoise1D(size=50, nbr_control_point=10, seed=123)


@pytest.fixture
def worley2d():
    return WorleyNoise2D(size=50, nbr_control_point=10, seed=123)


@pytest.fixture
def worley3d():
    return WorleyNoise3D(size=50, nbr_control_point=10, seed=123)


def test_worley_size_1D(worley1d):
    assert worley1d.noise is None


def test_worley_size_2D(worley2d):
    assert worley2d.noise is None


def test_worley_size_3D(worley3d):
    assert worley3d.noise is None


def test_worley_generate_1d(worley1d):
    worley1d.generate()
    assert worley1d.noise.shape == (50,)


def test_worley_generate_2d(worley2d):
    worley2d.generate()
    assert worley2d.noise.shape == (50, 50)


def test_worley_generate_3d(worley3d):
    worley3d.generate()
    assert worley3d.noise.shape == (50, 50, 50)


def test_worley_deter_1d():
    """Check if the noise is deterministic for the same seed"""
    n_control = 10
    size = 50
    n1 = WorleyNoise1D(size=size, nbr_control_point=n_control, seed=123456)
    n2 = WorleyNoise1D(size=size, nbr_control_point=n_control, seed=123456)
    n1.generate()
    n2.generate()
    assert (n1.noise == n2.noise).all()


def test_worley_deter_2d():
    """Check if the noise is deterministic for the same seed"""
    n_control = 10
    size = 50
    n1 = WorleyNoise2D(size=size, nbr_control_point=n_control, seed=123456)
    n2 = WorleyNoise2D(size=size, nbr_control_point=n_control, seed=123456)
    n1.generate()
    n2.generate()
    assert (n1.noise == n2.noise).all()


def test_worley_deter_3d():
    """Check if the noise is deterministic for the same seed"""
    n_control = 10
    size = 50
    n1 = WorleyNoise3D(size=size, nbr_control_point=n_control, seed=123456)
    n2 = WorleyNoise3D(size=size, nbr_control_point=n_control, seed=123456)
    n1.generate()
    n2.generate()
    assert (n1.noise == n2.noise).all()


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_worley_normalize_1d(worley1d, seed):
    worley1d.generate()
    assert worley1d.noise.min() >= 0 and worley1d.noise.max() <= 1.5 * worley1d.size


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_worley_normalize_2d(worley2d, seed):
    worley2d.generate()
    assert worley2d.noise.min() >= 0 and worley2d.noise.max() <= 1.5 * worley2d.size


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_worley_normalize_3d(worley3d, seed):
    worley3d.generate()
    assert worley3d.noise.min() >= 0 and worley3d.noise.max() <= 1.5 * worley3d.size
