import numpy as np
import pytest

from ValueNoise import PerlinNoise1D, PerlinNoise2D, PerlinNoise3D, Noise
from ValueNoise import WorleyNoise1D, WorleyNoise2D, WorleyNoise3D


@pytest.fixture
def empty_noise():
    return Noise(size=50, seed=123)


@pytest.fixture
def perlin1d():
    p = PerlinNoise1D(size=50, grid_size=10, seed=123)
    p.generate()
    return p


@pytest.fixture
def perlin2d():
    p = PerlinNoise2D(size=50, grid_size=10, seed=123)
    p.generate()
    return p


@pytest.fixture
def perlin3d():
    p = PerlinNoise3D(size=50, grid_size=10, seed=123)
    p.generate()
    return p


@pytest.fixture
def worley1d():
    p = WorleyNoise1D(size=50, nbr_control_point=10, seed=123)
    p.generate()
    return p


@pytest.fixture
def worley2d():
    p = WorleyNoise2D(size=50, nbr_control_point=10, seed=123)
    p.generate()
    return p


@pytest.fixture
def worley3d():
    p = WorleyNoise3D(size=50, nbr_control_point=10, seed=123)
    p.generate()
    return p


def test_noise_init(empty_noise):
    assert empty_noise.size == 50
    assert empty_noise.noise is None


def test_empty_noise_str_repr(empty_noise):
    assert str(empty_noise) == "Noise not generated yet"
    assert repr(empty_noise) == "Noise not generated yet"


def test_noise_op_empty(empty_noise, perlin1d):
    with pytest.raises(ValueError, match="Cannot add noise that have not been generated yet"):
        _ = perlin1d + empty_noise
    with pytest.raises(ValueError, match="Cannot subtract noise that have not been generated yet"):
        _ = perlin1d - empty_noise
    with pytest.raises(ValueError, match="Cannot multiply noise that have not been generated yet"):
        _ = perlin1d * empty_noise


def test_noise_op_size(perlin1d, perlin2d):
    with pytest.raises(ValueError, match="Cannot add two noises of different dimensions 1 and 2"):
        _ = perlin1d + perlin2d
    with pytest.raises(ValueError, match="Cannot subtract two noises of different dimensions 1 and 2"):
        _ = perlin1d - perlin2d
    with pytest.raises(ValueError, match="Cannot multiply two noises of different dimensions 1 and 2"):
        _ = perlin1d * perlin2d


def test_noise_op_1d(perlin1d, worley1d):
    add = perlin1d + worley1d
    assert add.noise.shape == perlin1d.noise.shape
    assert (add.noise == perlin1d.noise + worley1d.noise).all()

    sub = perlin1d - worley1d
    assert sub.noise.shape == perlin1d.noise.shape
    assert (sub.noise == perlin1d.noise - worley1d.noise).all()

    mul = perlin1d * worley1d
    assert mul.noise.shape == perlin1d.noise.shape
    assert (mul.noise == np.multiply(perlin1d.noise, worley1d.noise)).all()


def test_noise_op_2d(perlin2d, worley2d):
    add = perlin2d + worley2d
    assert add.noise.shape == perlin2d.noise.shape
    assert (add.noise == perlin2d.noise + worley2d.noise).all()

    sub = perlin2d - worley2d
    assert sub.noise.shape == perlin2d.noise.shape
    assert (sub.noise == perlin2d.noise - worley2d.noise).all()

    mul = perlin2d * worley2d
    assert mul.noise.shape == perlin2d.noise.shape
    assert (mul.noise == np.multiply(perlin2d.noise, worley2d.noise)).all()


def test_noise_op_3d(perlin3d, worley3d):
    add = perlin3d + worley3d
    assert add.noise.shape == perlin3d.noise.shape
    assert (add.noise == perlin3d.noise + worley3d.noise).all()

    sub = perlin3d - worley3d
    assert sub.noise.shape == perlin3d.noise.shape
    assert (sub.noise == perlin3d.noise - worley3d.noise).all()

    mul = perlin3d * worley3d
    assert mul.noise.shape == perlin3d.noise.shape
    assert (mul.noise == np.multiply(perlin3d.noise, worley3d.noise)).all()


def test_noise_normalize(worley3d):
    worley3d.normalize()
    assert worley3d.noise.min() >= 0 and worley3d.noise.max() <= 1
