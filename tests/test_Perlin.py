import numpy as np
import pytest

from ValueNoise import PerlinNoise1D, PerlinNoise2D, PerlinNoise3D


@pytest.fixture
def perlin1d():
    return PerlinNoise1D(size=50, grid_size=10, seed=123)


@pytest.fixture
def perlin2d():
    return PerlinNoise2D(size=50, grid_size=10, seed=123)


@pytest.fixture
def perlin3d():
    return PerlinNoise3D(size=50, grid_size=10, seed=123)


def test_perlin_incorrect_gs():
    gs = 9
    with pytest.raises(ValueError, match='grid_size must be a divisor of size'):
        _ = PerlinNoise1D(size=100, grid_size=gs, seed=123456)
    with pytest.raises(ValueError, match='grid_size must be a divisor of size'):
        _ = PerlinNoise2D(size=89, grid_size=gs, seed=123456)
    with pytest.raises(ValueError, match='grid_size must be a divisor of size'):
        _ = PerlinNoise3D(size=91, grid_size=gs, seed=123456)


def test_perlin_size_1D(perlin1d):
    assert perlin1d.noise is None
    assert perlin1d.grid.shape == (10 + 1,)


def test_perlin_size_2D(perlin2d):
    assert perlin2d.noise is None
    assert perlin2d.grid.shape == (10 + 1, 10 + 1, 2)


def test_perlin_size_3D(perlin3d):
    assert perlin3d.noise is None
    assert perlin3d.grid.shape == (10 + 1, 10 + 1, 10 + 1, 3)


def test_perlin_generate_1d(perlin1d):
    perlin1d.generate()
    assert perlin1d.noise.shape == (50,)


def test_perlin_generate_2d(perlin2d):
    perlin2d.generate()
    assert perlin2d.noise.shape == (50, 50)


def test_perlin_generate_3d(perlin3d):
    perlin3d.generate()
    assert perlin3d.noise.shape == (50, 50, 50)


def test_perlin_deter_1d():
    """Check if the noise is deterministic for the same seed"""
    gs = 10
    size = 50
    n1 = PerlinNoise1D(size=size, grid_size=gs, seed=123456)
    n2 = PerlinNoise1D(size=size, grid_size=gs, seed=123456)
    n1.generate()
    n2.generate()
    assert (n1.noise == n2.noise).all()


def test_perlin_deter_2d():
    """Check if the noise is deterministic for the same seed"""
    gs = 10
    size = 50
    n1 = PerlinNoise2D(size=size, grid_size=gs, seed=123456)
    n2 = PerlinNoise2D(size=size, grid_size=gs, seed=123456)
    n1.generate()
    n2.generate()
    assert (n1.noise == n2.noise).all()


def test_perlin_deter_3d():
    """Check if the noise is deterministic for the same seed"""
    gs = 10
    size = 50
    n1 = PerlinNoise3D(size=size, grid_size=gs, seed=123456)
    n2 = PerlinNoise3D(size=size, grid_size=gs, seed=123456)
    n1.generate()
    n2.generate()
    assert (n1.noise == n2.noise).all()


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_perlin_normalize_1d(perlin1d, seed):
    perlin1d.generate()
    assert perlin1d.noise.min() >= -1 and perlin1d.noise.max() <= 1


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_perlin_normalize_2d(perlin2d, seed):
    perlin2d.generate()
    assert perlin2d.noise.min() >= -1 and perlin2d.noise.max() <= 1


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_perlin_normalize_3d(perlin3d, seed):
    perlin3d.generate()
    assert perlin3d.noise.min() >= -1 and perlin3d.noise.max() <= 1


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_perlin_continuity_1d(perlin1d, seed):
    """Check if the noise is continuous"""
    perlin1d.generate()
    diffs = np.abs(np.diff(perlin1d.noise))  # Difference between adjacent values
    assert np.percentile(diffs, 0.99) < 0.1  # Ensure that 99% of the differences are less than 0.1


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_perlin_continuity_2d(perlin2d, seed):
    """Check if the noise is continuous"""
    perlin2d.generate()
    diff_x = np.abs(np.diff(perlin2d.noise, axis=0))
    diff_y = np.abs(np.diff(perlin2d.noise, axis=1))
    assert np.percentile(diff_x, 0.99) < 0.1
    assert np.percentile(diff_y, 0.99) < 0.1


@pytest.mark.parametrize("seed", [0, 1, 42, 123456, 987654321])
def test_perlin_continuity_3d(perlin3d, seed):
    """Check if the noise is continuous"""
    perlin3d.generate()
    diff_x = np.abs(np.diff(perlin3d.noise, axis=0))
    diff_y = np.abs(np.diff(perlin3d.noise, axis=1))
    diff_z = np.abs(np.diff(perlin3d.noise, axis=2))
    assert np.percentile(diff_x, 0.99) < 0.1
    assert np.percentile(diff_y, 0.99) < 0.1
    assert np.percentile(diff_z, 0.99) < 0.1
