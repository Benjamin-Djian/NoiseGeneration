import argparse
import math
import random
from typing import Collection

import numpy as np
from matplotlib import pyplot as plt, animation

from Vector import vector_from_dots, generate_random_unit_vect


def distance(pt1: Collection, pt2: Collection, p: int = 2) -> float:
    """
    Compute the Minkowski distance of order P between two points in any dimension.
    :param pt1: Coordinates of the first point
    :param pt2: Coordinates of the second point
    :param p: Parameter of the norm. Defaults to 2 (Euclidean distance)
    :return: The distance between the two points, using the p-norm
    """
    if isinstance(pt1, Collection) and isinstance(pt2, Collection):
        if len(pt1) != len(pt2):
            raise ValueError(f"The two points must have the same dimension, got {len(pt1)} and {len(pt2)} instead")
        return sum([(c1 - c2) ** p for c1, c2 in zip(pt1, pt2)]) ** (1 / p)
    else:
        raise ValueError(f"The two points must be collections of numbers, got {type(pt1)} and {type(pt2)} instead")


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
        if self.noise.shape != other.noise.shape:
            raise ValueError(f"Cannot add two noises of different dimensions {self.noise.ndim} and {other.noise.ndim}")
        if normalize:
            self.normalize()
            other.normalize()
        res = Noise(size=self.size)
        res.noise = self.noise + other.noise
        return res

    def __sub__(self, other, normalize=True):
        """Subtract two noises"""
        if self.noise.shape != other.noise.shape:
            raise ValueError(
                f"Cannot subtract two noises of different dimensions {self.noise.ndim} and {other.noise.ndim}")
        if normalize:
            self.normalize()
            other.normalize()
        res = Noise(size=self.size)
        res.noise = self.noise - other.noise
        return res

    def __mul__(self, other, normalize=True):
        """Multiply two noises"""
        if self.noise.shape != other.noise.shape:
            raise ValueError(
                f"Cannot multiply two noises of different dimensions {self.noise.ndim} and {other.noise.ndim}")
        if normalize:
            self.normalize()
            other.normalize()
        res = Noise(size=self.size)
        res.noise = np.multiply(self.noise, other.noise)  # Element-wise multiplication
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
        if isinstance(self.noise, np.ndarray):
            fig = plt.figure()
            if self.noise.ndim == 1:
                plt.plot(self.noise)
                plt.show()
            elif self.noise.ndim == 2:
                plt.imshow(self.noise, cmap=cmap)
                plt.xticks([]), plt.yticks([])
                plt.show()
            elif self.noise.ndim == 3:
                images = [[plt.imshow(layer, cmap='gray', animated=True)] for layer in self.noise]
                ani = animation.ArtistAnimation(fig, images, interval=50, blit=True)
                plt.xticks([]), plt.yticks([])
                plt.show()
            else:
                raise ValueError(f"Representation of {self.noise.ndim}-dimensional noise not supported")
        else:
            raise ValueError(f"Noise representation for type {type(self.noise)} not supported")


class PerlinNoise1D(Noise):
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

        # Grid creation: a 1D array of unit vectors of a random direction
        self.grid = np.empty(grid_size + 1, dtype=object)
        for i in range(grid_size + 1):
            self.grid[i] = generate_random_unit_vect(dims=1, seed=self.seed + i if seed is not None else None)

        self.interpolate_fctn = lambda t, a, b: (b - a) * t + a
        # Fade function as defined by Ken Perlin.
        self.fade = lambda t: ((6 * t - 15) * t + 10) * t * t * t

    def compute_dot_product(self, coord: int) -> list[float]:
        """
        :param coord: Coordinates of the point
        :return: Dot product of offset vector with each corner of the grid
        """
        xf = coord - (coord % self.pix_by_cell)

        corners: list[int] = [xf, xf + self.pix_by_cell]

        output = []
        for cx in corners:
            # Create an offset vector for each corner
            offset = vector_from_dots((coord,), (cx,))
            # Dot product with the random vector of the grid
            output.append(offset * self.grid[cx // self.pix_by_cell])

        return output

    def interpolate(self, coord: int, dot_products: list[float]) -> float:
        """
        :param coord: coordinate of the point to interpolate
        :param dot_products: Dot products of the points with the 2 corners of its cell
        :return: The interpolation (smoothed) of the 2 dot products of the given point
        """
        u = self.fade((coord % self.pix_by_cell) / self.pix_by_cell)

        value = self.interpolate_fctn(u, dot_products[0], dot_products[1])
        return value

    def generate_pt(self, coord: int):
        """Generate the value of the noise at a given coordinate"""
        value = self.interpolate(coord, self.compute_dot_product(coord))
        return value

    def generate(self):
        self.noise = np.array([self.generate_pt(coord_x) for coord_x in range(self.size)])


class PerlinNoise2D(Noise):
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
        self.grid = np.empty((grid_size + 1, grid_size + 1), dtype=object)
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                seed = self.seed + i * grid_size + j if seed is not None else None
                self.grid[i][j] = generate_random_unit_vect(dims=2, seed=seed)

        self.interpolate_fctn = lambda t, a, b: (b - a) * t + a
        # Fade function as defined by Ken Perlin.
        self.fade = lambda t: ((6 * t - 15) * t + 10) * t * t * t

    def compute_dot_product(self, coord: tuple[int, int]) -> list[float]:
        """
        :param coord: Coordinates of the point
        :return: Dot product of offset vector with each corner of the grid
        """
        x, y = coord
        xf, yf = (x - (x % self.pix_by_cell), y - (y % self.pix_by_cell))

        corners: list[tuple[int, int]] = [(xf, yf),
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

    def interpolate(self, coord: tuple[int, int], dot_products: list[float]) -> float:
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

    def generate_pt(self, coord: tuple[int, int]):
        """Generate the value of the noise at a given coordinate"""
        value = self.interpolate(coord, self.compute_dot_product(coord))
        return value

    def generate(self):
        self.noise = np.array([[self.generate_pt((coord_x, coord_y)) for coord_x in range(self.size)]
                               for coord_y in range(self.size)])


class PerlinNoise3D(Noise):
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

        self.pix_by_cell = int(size / grid_size)  # Dimension of a cell of the grid

        # Grid creation: a 3D array of unit vectors of a random direction
        self.grid = np.empty((grid_size + 1, grid_size + 1, grid_size + 1), dtype=object)
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                for k in range(grid_size + 1):
                    seed = self.seed + i * grid_size * grid_size + j * grid_size + k if seed is not None else None
                    self.grid[i][j][k] = generate_random_unit_vect(dims=3, seed=seed)

        self.interpolate_fctn = lambda t, a, b: (b - a) * t + a
        # Fade function as defined by Ken Perlin.
        self.fade = lambda t: ((6 * t - 15) * t + 10) * t * t * t

    def compute_dot_product(self, coord: tuple[int, int, int]) -> list[float]:
        """
        :param coord: Coordinates of the point
        :return: Dot product of offset vector with each corner of the grid
        """
        xf, yf, zf = list((c - (c % self.pix_by_cell) for c in coord))

        corners: list[tuple[int, int, int]] = [(xf, yf, zf),
                                               (xf + self.pix_by_cell, yf, zf),
                                               (xf, yf + self.pix_by_cell, zf),
                                               (xf + self.pix_by_cell, yf + self.pix_by_cell, zf),
                                               (xf, yf, zf + self.pix_by_cell),
                                               (xf + self.pix_by_cell, yf, zf + self.pix_by_cell),
                                               (xf, yf + self.pix_by_cell, zf + self.pix_by_cell),
                                               (xf + self.pix_by_cell, yf + self.pix_by_cell, zf + self.pix_by_cell)]

        output = []
        for coord_corners in corners:
            # Create an offset vector for each corner
            offset = vector_from_dots(coord, coord_corners)
            # Dot product with the random vector of the grid
            output.append(offset * self.grid[*(c // self.pix_by_cell for c in coord_corners)])

        return output

    def interpolate(self, coord: tuple[int, int, int], dot_products: list[float]) -> float:
        x, y, z = coord
        u = self.fade((x % self.pix_by_cell) / self.pix_by_cell)
        v = self.fade((y % self.pix_by_cell) / self.pix_by_cell)
        w = self.fade((z % self.pix_by_cell) / self.pix_by_cell)

        interpol_x1 = self.interpolate_fctn(u, dot_products[0], dot_products[1])
        interpol_x2 = self.interpolate_fctn(u, dot_products[2], dot_products[3])
        interpol_x3 = self.interpolate_fctn(u, dot_products[4], dot_products[5])
        interpol_x4 = self.interpolate_fctn(u, dot_products[6], dot_products[7])
        interpol_y1 = self.interpolate_fctn(v, interpol_x1, interpol_x2)
        interpol_y2 = self.interpolate_fctn(v, interpol_x3, interpol_x4)
        value = self.interpolate_fctn(w, interpol_y1, interpol_y2)
        return value

    def generate_pt(self, coord: tuple[int, int, int]):
        """Generate the value of the noise at a given coordinate"""
        value = self.interpolate(coord, self.compute_dot_product(coord))
        return value

    def generate(self):
        self.noise = np.array([[[self.generate_pt((coord_x, coord_y, coord_z))
                                 for coord_x in range(self.size)]
                                for coord_y in range(self.size)]
                               for coord_z in range(self.size)])


class WorleyNoise(Noise):
    def __init__(self, size: int, dims: int, nbr_control_point: int, seed: int = None):
        """
        :param size: The size of the noise to generate (in pixel)
        :param nbr_control_point: Number of control points to use to generate the noise.
        :param dims: Number of dimensions of the noise to generate (must be 1,2 or 3)
        :param seed: To make generation reproducible
        """
        super().__init__(size=size)
        self.control_points: list[tuple[int, int]] = []
        self.dims = dims
        self.generate_control_points(dims, nbr_control_point, seed=seed)

    def generate_control_points(self, dims: int, nbr_control_point: int, seed: int = None):
        """Generate nbr_control_point random control points"""
        random.seed(seed)
        for _ in range(nbr_control_point):
            dim_control_point = []
            for d in range(dims):
                random_d = random.randint(0, self.size)
                dim_control_point.append(random_d)
            self.control_points.append(tuple(dim_control_point))

    def get_closer_control_point(self, coord) -> float:
        """Return the distance to the closest control point"""
        closest_dist = np.inf
        for control_p in self.control_points:
            dist = distance(coord, control_p)
            if dist < closest_dist:
                closest_dist = dist

        return closest_dist

    def generate(self):
        if self.dims == 1:
            self.noise = np.array([self.get_closer_control_point((c,)) for c in range(self.size)])
        elif self.dims == 2:
            self.noise = np.array(
                [[self.get_closer_control_point((coord_x, coord_y))
                  for coord_x in range(self.size)]
                 for coord_y in range(self.size)])
        elif self.dims == 3:
            self.noise = np.array(
                [[[self.get_closer_control_point((coord_x, coord_y, coord_z))
                   for coord_x in range(self.size)]
                  for coord_y in range(self.size)]
                 for coord_z in range(self.size)])
        else:
            raise ValueError(f"Cannot generate noise of dimension {self.dims}")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', choices=['perlin1d', 'perlin2d', 'perlin3d', 'worley1d', 'worley2d', 'worley3d'],
                        default='perlin2d')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ncp', type=int, default=10)  # Number of control points for Worley noise
    parser.add_argument('--gs', type=int, default=10)  # Grid size for Perlin noise
    args = parser.parse_args()
    if args.noise == 'perlin1d':
        noise = PerlinNoise1D(size=args.size, grid_size=args.gs, seed=args.seed)
    elif args.noise == 'perlin2d':
        noise = PerlinNoise2D(size=args.size, grid_size=args.gs, seed=args.seed)
    elif args.noise == 'perlin3d':
        noise = PerlinNoise3D(size=args.size, grid_size=args.gs, seed=args.seed)
    elif args.noise == 'worley1d':
        noise = WorleyNoise(size=args.size, dims=1, nbr_control_point=args.ncp, seed=args.seed)
    elif args.noise == 'worley2d':
        noise = WorleyNoise(size=args.size, dims=2, nbr_control_point=args.ncp, seed=args.seed)
    elif args.noise == 'worley3d':
        noise = WorleyNoise(size=args.size, dims=3, nbr_control_point=args.ncp, seed=args.seed)
    else:
        raise ValueError(f"Noise type {args.noise} not supported")

    noise.generate()
    noise.show()
