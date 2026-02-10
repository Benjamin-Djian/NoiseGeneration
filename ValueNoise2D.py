import math
import random

import numpy as np
from matplotlib import pyplot as plt

from Vector import vector_from_dots, vector_from_magdir


def distance(pt1: tuple, pt2: tuple, p: int = 2) -> float:
    """
    :param pt1: Coordinates of the first point
    :param pt2: Coordinates of the first point
    :param p: Parameter of the norm. Defaults to 2 (Euclidean distance)
    :return: The distance between the two points, using the p-norm
    """
    if len(pt1) != len(pt2):
        raise ValueError(f"The two points must have the same dimension, got {len(pt1)} and {len(pt1)} instead")
    return sum([(c1 - c2) ** p for c1, c2 in zip(pt1, pt2)]) ** (1 / p)


class Noise:
    def __init__(self, size: int, seed: int = None):
        """
        :param size: Size of the noise to generate (in pixel)
        :param seed: To make generation reproducible
        """
        self.size = size
        self.seed = seed
        self.noise: np.ndarray | None = None

    def __str__(self):
        if self.noise is None:
            return "Noise not generated yet"
        return "".join([str(row) + '\n' for row in self.noise])

    def __repr__(self):
        if self.noise is None:
            return "Noise not generated yet"
        return str(self.noise)

    def __iter__(self):
        return self.noise.__iter__()

    def __getitem__(self, item):
        return self.noise[item]

    def __add__(self, other, normalize=True):
        """Add two noises together"""
        if normalize:
            self.normalize()
            other.normalize()
        res = Noise(size=self.size)
        res.noise = self.noise + other.noise
        return res

    def __sub__(self, other, normalize=True):
        """Subtract two noises"""
        if normalize:
            self.normalize()
            other.normalize()
        res = Noise(size=self.size)
        res.noise = self.noise - other.noise
        return res

    def __mul__(self, other, normalize=True):
        """Multiply two noises"""
        if normalize:
            self.normalize()
            other.normalize()
        res = Noise(size=self.size)
        res.noise = self.noise * other.noise
        return res

    def normalize(self):
        """Normalize the noise between 0 and 1"""
        self.noise = (self.noise - np.min(self.noise)) / (np.max(self.noise) - np.min(self.noise))

    def generate(self):
        raise NotImplementedError

    def show(self, cmap='gray'):
        """Show the generated noise with matplotlib"""
        if self.noise is None:
            raise ValueError("Noise not generated yet")
        plt.imshow(self.noise, cmap=cmap)
        plt.xticks([]), plt.yticks([])
        plt.show()


class PerlinNoise(Noise):
    def __init__(self, size: int, grid_size: int, seed: int = None):
        """
        :param size: size of the noise to generate (in pixel)
        :param grid_size: the number of rows and columns of the grid. Smaller grid means sharper noise.
        :param seed: To make generation reproducible
        """
        super().__init__(size=size, seed=seed)
        if size % grid_size != 0:
            raise ValueError("grid_size must be a divisor of size")
        self.grid_size = grid_size
        # Number of pixels per grid cell
        self.pix_by_cell = int(size / grid_size)

        # Grid creation: a 2D array of unit vectors of a random direction
        np.random.seed(self.seed)
        random_dirs = np.random.rand(grid_size + 1, grid_size + 1)
        self.grid: np.ndarray = np.empty((grid_size + 1, grid_size + 1), dtype=object)
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                self.grid[i][j] = vector_from_magdir(direction=2 * math.pi * random_dirs[i][j], magnitude=1)

        self.interpolate_fctn = lambda t, a, b: (b - a) * t + a
        # Fade function as defined by Ken Perlin.
        self.fade = lambda t: ((6 * t - 15) * t + 10) * t * t * t

    def compute_dot_product(self, coord: tuple[float, float]) -> list[float]:
        """
        :param coord: Coordinates of the point
        :return: Dot product of offset vector with each corner of the grid
        """
        x, y = coord
        xf, yf = (x - (x % self.pix_by_cell), y - (y % self.pix_by_cell))

        corners: list[tuple[float, float]] = [(xf, yf),
                                              (xf + self.pix_by_cell, yf),
                                              (xf, yf + self.pix_by_cell),
                                              (xf + self.pix_by_cell, yf + self.pix_by_cell)]

        output = []
        for cx, cy in corners:
            # Create an offset vector for each corner
            offset = vector_from_dots((x, y), (cx, cy))
            # Dot product with the random vector of the grid
            output.append(offset * self.grid[cx // self.pix_by_cell, cy // self.pix_by_cell])

        return output

    def interpolate(self, coord: tuple[float, float], dot_products: list[float]) -> float:
        """
        :param coord: coordinates of the point to interpolate
        :param dot_products: Dot products of the points with the 4 corners of its cell
        :return: The interpolation (smoothed) of the 4 dot products of the given point
        """
        x, y = coord
        u = self.fade((x % self.pix_by_cell) / self.pix_by_cell)
        v = self.fade((y % self.pix_by_cell) / self.pix_by_cell)

        interpol_left = self.interpolate_fctn(u, dot_products[0], dot_products[1])
        interpol_right = self.interpolate_fctn(u, dot_products[2], dot_products[3])
        value = self.interpolate_fctn(v, interpol_left, interpol_right)
        return value

    def generate_x_y(self, coord: tuple[float, float]):
        """Generate the value of the noise at a given coordinate"""
        value = self.interpolate(coord, self.compute_dot_product(coord))
        return value

    def generate(self):
        self.noise = np.array([[self.generate_x_y((coord_x, coord_y)) for coord_x in range(self.size)]
                               for coord_y in range(self.size)])


class WorneyNoise(Noise):
    def __init__(self, size: int, nbr_control_point: int, seed: int = None):
        """
        :param size: The size of the noise to generate (in pixel)
        :param nbr_control_point: Number of control points to use to generate the noise.
        :param seed: To make generation reproducible
        """
        super().__init__(size=size)
        self.control_points: list[tuple[int, int]] = []
        self.generate_control_points(nbr_control_point, seed=seed)

    def generate_control_points(self, nbr_control_point, seed: int = None):
        """Generate nbr_control_point random control points"""
        random.seed(seed)
        for _ in range(nbr_control_point):
            random_x = random.randint(0, self.size)
            random_y = random.randint(0, self.size)
            self.control_points.append((random_x, random_y))

    def get_closer_control_point(self, coord: tuple[int, int]) -> float:
        """Return the distance to the closest control point"""
        closest_dist = math.sqrt(self.size ** 2 + self.size ** 2)
        for control_p in self.control_points:
            dist = distance(coord, control_p)
            if dist < closest_dist:
                closest_dist = dist

        return closest_dist

    def generate(self):
        self.noise = np.array(
            [[self.get_closer_control_point((coord_x, coord_y)) for coord_x in range(self.size)]
             for coord_y in range(self.size)])


class SimplexNoise(Noise):
    def __init__(self, size: int, seed: int = None):
        super().__init__(size=size, seed=seed)
        self.F = (math.sqrt(3) - 1) / 2
        self.G = 1 / math.sqrt(2 + math.sqrt(3))

    def coordinate_skewing(self, coord: tuple[int, int]):
        sum_coord = sum(coord)
        return (coord[0] + sum_coord) * self.F, (coord[1] + sum_coord) * self.F

    def simplicial_subdivision(self):
        ...
