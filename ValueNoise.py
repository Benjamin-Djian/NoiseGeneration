import argparse
import math

import numpy as np
from matplotlib import pyplot as plt, animation


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

    def show(self, cmap='gray', savefig: bool = False):
        """Show the generated noise with matplotlib"""
        if self.noise is None:
            raise ValueError("Noise not generated yet")
        if isinstance(self.noise, np.ndarray):
            fig = plt.figure()
            if self.noise.ndim == 1:
                plt.plot(self.noise)
                if savefig:
                    plt.savefig(fname=f"{self.size}px_{self.__class__.__name__}.png", dpi="figure")
                plt.show()
            elif self.noise.ndim == 2:
                plt.imshow(self.noise, cmap=cmap)
                plt.xticks([]), plt.yticks([])
                if savefig:
                    plt.savefig(fname=f"{self.size}px_{self.__class__.__name__}.png", dpi="figure")
                plt.show()
            elif self.noise.ndim == 3:
                images = [[plt.imshow(layer, cmap='gray', animated=True)] for layer in self.noise]
                ani = animation.ArtistAnimation(fig, images, interval=50, blit=True)
                plt.xticks([]), plt.yticks([])
                if savefig:
                    ani.save(filename=f"{self.size}px_{self.__class__.__name__}.gif")
                plt.show()
            else:
                raise ValueError(f"Representation of {self.noise.ndim}-dimensional noise not supported")
        else:
            raise ValueError(f"Noise representation for type {type(self.noise)} not supported")


# region Perlin noise

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

        # The grid is a 2D array of random vectors
        self.grid = np.random.uniform(-1, 1, size=grid_size + 1)

        self.interpolate_fctn = lambda t, a, b: (b - a) * t + a
        # Fade function as defined by Ken Perlin.
        self.fade = lambda t: ((6 * t - 15) * t + 10) * t * t * t

    def generate(self):
        x = np.arange(0, self.size, step=1)
        x_grid = x // self.pix_by_cell  # Indices in the grid
        # Coordinates inside the corresponding cell (offset vectors)
        xf = (x % self.pix_by_cell) / self.pix_by_cell

        # Gradient vectors at 4 corners
        left = self.grid[x_grid]
        right = self.grid[x_grid + 1]

        # Dot products
        dp_left = left * xf
        dp_right = right * (xf - 1)

        # Fade
        u = self.fade(xf)

        # Bilinear interpolation
        self.noise = self.interpolate_fctn(u, dp_left, dp_right)


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
        self.pix_by_cell: int = int(size / grid_size)

        # The grid is a 2D array of random vectors
        self.grid = np.random.randn(grid_size + 1, grid_size + 1, 2)
        # Normalize the vectors of the grid
        self.grid /= np.linalg.norm(self.grid, axis=2, keepdims=True)

        self.interpolate_fctn = lambda t, a, b: (b - a) * t + a
        # Fade function as defined by Ken Perlin.
        self.fade = lambda t: ((6 * t - 15) * t + 10) * t * t * t

    def generate(self):
        x = np.arange(0, self.size, step=1)
        y = np.arange(0, self.size, step=1)
        # real coordinates
        X, Y = np.meshgrid(x, y)
        # Cell coord where the points are located
        x_grid, y_grid = X // self.pix_by_cell, Y // self.pix_by_cell
        # Relative coordinates of the points inside the cell
        xf, yf = (X % self.pix_by_cell) / self.pix_by_cell, (Y % self.pix_by_cell) / self.pix_by_cell

        # Gradient vectors at 4 corners
        top_left = self.grid[x_grid, y_grid]
        bottom_left = self.grid[x_grid + 1, y_grid]
        top_right = self.grid[x_grid, y_grid + 1]
        bottom_right = self.grid[x_grid + 1, y_grid + 1]

        # Dot products
        dp_top_left = top_left[:, :, 0] * xf + top_left[:, :, 1] * yf
        dp_bottom_left = bottom_left[:, :, 0] * (xf - 1) + bottom_left[:, :, 1] * yf
        dp_top_right = top_right[:, :, 0] * xf + top_right[:, :, 1] * (yf - 1)
        dp_bottom_right = bottom_right[:, :, 0] * (xf - 1) + bottom_right[:, :, 1] * (yf - 1)

        # Fade
        u = self.fade(xf)
        v = self.fade(yf)

        # Bilinear interpolation
        interpol_left = self.interpolate_fctn(u, dp_top_left, dp_bottom_left)
        interpol_right = self.interpolate_fctn(u, dp_top_right, dp_bottom_right)
        self.noise = self.interpolate_fctn(v, interpol_left, interpol_right)


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
        # Number of pixels per grid cell
        self.pix_by_cell = int(size / grid_size)

        # The grid is a 2D array of random vectors
        self.grid = np.random.randn(grid_size + 1, grid_size + 1, grid_size + 1, 3)
        # Normalize the vectors of the grid
        self.grid /= np.linalg.norm(self.grid, axis=2, keepdims=True)

        self.interpolate_fctn = lambda t, a, b: (b - a) * t + a
        # Fade function as defined by Ken Perlin.
        self.fade = lambda t: ((6 * t - 15) * t + 10) * t * t * t

    def generate(self):
        x = np.arange(0, self.size, step=1)
        y = np.arange(0, self.size, step=1)
        z = np.arange(0, self.size, step=1)
        X, Y, Z = np.meshgrid(x, y, z)  # real coordinates
        # Indices in the grid
        x_grid, y_grid, z_grid = X // self.pix_by_cell, Y // self.pix_by_cell, Z // self.pix_by_cell
        # Coordinates inside the corresponding cell (offset vectors)
        xf, yf, zf = (X % self.pix_by_cell) / self.pix_by_cell, (Y % self.pix_by_cell) / self.pix_by_cell, (
                Z % self.pix_by_cell) / self.pix_by_cell

        # Gradient vectors at 8 corners
        top_left = self.grid[x_grid, y_grid, z_grid]
        bottom_left = self.grid[x_grid + 1, y_grid, z_grid]
        top_right = self.grid[x_grid, y_grid + 1, z_grid]
        bottom_right = self.grid[x_grid + 1, y_grid + 1, z_grid]
        top_left_up = self.grid[x_grid, y_grid, z_grid + 1]
        bottom_left_up = self.grid[x_grid + 1, y_grid, z_grid + 1]
        top_right_up = self.grid[x_grid, y_grid + 1, z_grid + 1]
        bottom_right_up = self.grid[x_grid + 1, y_grid + 1, z_grid + 1]

        # Dot products
        dp_top_left = top_left[..., 0] * xf + top_left[..., 1] * yf + top_left[..., 2] * zf
        dp_bottom_left = bottom_left[..., 0] * (xf - 1) + bottom_left[..., 1] * yf + bottom_left[..., 2] * zf
        dp_top_right = top_right[..., 0] * xf + top_right[..., 1] * (yf - 1) + top_right[..., 2] * zf
        dp_bottom_right = bottom_right[..., 0] * (xf - 1) + bottom_right[..., 1] * (yf - 1) + bottom_right[..., 2] * zf
        dp_top_left_up = top_left_up[..., 0] * xf + top_left_up[..., 1] * yf + top_left_up[..., 2] * (zf - 1)
        dp_bottom_left_up = bottom_left_up[..., 0] * (xf - 1) + bottom_left_up[..., 1] * yf + bottom_left_up[..., 2] * (
                zf - 1)
        dp_top_right_up = top_right_up[..., 0] * xf + top_right_up[..., 1] * (yf - 1) + top_right_up[..., 2] * (zf - 1)
        dp_bottom_right_up = bottom_right_up[..., 0] * (xf - 1) + bottom_right_up[..., 1] * (yf - 1) + bottom_right_up[
            ..., 2] * (zf - 1)

        # Fade
        u = self.fade(xf)
        v = self.fade(yf)
        w = self.fade(zf)

        # Bilinear interpolation
        interpol_x1 = self.interpolate_fctn(u, dp_top_left, dp_bottom_left)
        interpol_x2 = self.interpolate_fctn(u, dp_top_right, dp_bottom_right)
        interpol_x3 = self.interpolate_fctn(u, dp_top_left_up, dp_bottom_left_up)
        interpol_x4 = self.interpolate_fctn(u, dp_top_right_up, dp_bottom_right_up)
        interpol_y1 = self.interpolate_fctn(v, interpol_x1, interpol_x2)
        interpol_y2 = self.interpolate_fctn(v, interpol_x3, interpol_x4)
        self.noise = self.interpolate_fctn(w, interpol_y1, interpol_y2)


class PerlinOctave(Noise):
    def __init__(self, dims: int, size: int, num_octave: int, seed: int = None):
        super().__init__(size=size, seed=seed)
        if num_octave < 2:
            raise ValueError("The number of octave must be at least 2")
        if self.size % pow(2, num_octave) != 0:
            raise ValueError(f"The size of the noise must be a multiple of {pow(2, num_octave)}")
        self.num_octave = num_octave
        if dims == 1:
            self.noise_generator = PerlinNoise1D
        elif dims == 2:
            self.noise_generator = PerlinNoise2D
        elif dims == 3:
            self.noise_generator = PerlinNoise3D
        else:
            raise ValueError("The dimension of the noise must be 1, 2 or 3")

    def generate(self):
        p_noise = self.noise_generator(size=self.size, grid_size=2, seed=self.seed)
        p_noise.generate()
        p_noise.normalize()
        self.noise = p_noise.noise
        amp = 1.
        freq = 2
        for _ in range(self.num_octave):
            p_noise = self.noise_generator(size=self.size, grid_size=freq, seed=self.seed)
            p_noise.generate()
            p_noise.normalize()
            p_noise.normalize()
            self.noise += amp * p_noise.noise
            amp *= 0.5
            freq *= 2


# endregion
# region Worley noise

class WorleyNoise1D(Noise):
    def __init__(self, size: int, nbr_control_point: int, seed: int = None):
        """
        :param size: The size of the noise to generate (in pixel)
        :param nbr_control_point: Number of control points to use to generate the noise.
        :param seed: To make generation reproducible
        """
        super().__init__(size=size)
        np.random.seed(seed)
        self.control_points = np.random.randint(0, self.size + 1, size=nbr_control_point)

    def generate(self):
        """Generate Worley noise grid"""
        # Create coordinates
        coords = np.arange(self.size)
        # Compute the difference between each control point and each coordinate
        diff = np.subtract.outer(coords, self.control_points)
        # We consider the distance to be the absolute value of the difference
        distances = np.abs(diff)

        self.noise = distances.min(axis=1)


class WorleyNoise2D(Noise):
    def __init__(self, size: int, nbr_control_point: int, seed: int = None):
        """
        :param size: The size of the noise to generate (in pixel)
        :param nbr_control_point: Number of control points to use to generate the noise.
        :param seed: To make generation reproducible
        """
        super().__init__(size=size)
        np.random.seed(seed)
        self.control_points = np.random.randint(0, self.size + 1, size=(nbr_control_point, 2))

    def generate(self):
        """Generate Worley noise grid"""
        # Create coordinates
        x = np.arange(self.size)
        y = np.arange(self.size)
        xx, yy = np.meshgrid(x, y)

        # Compute the difference between each control point and each coordinate
        dx = np.subtract.outer(xx, self.control_points[:, 0])
        dy = np.subtract.outer(yy, self.control_points[:, 1])

        # Considering Euclidean distance
        distances = np.sqrt(dx ** 2 + dy ** 2)

        # Select only the minimum distance
        self.noise = distances.min(axis=2)


class WorleyNoise3D(Noise):
    def __init__(self, size: int, nbr_control_point: int, seed: int = None):
        """
        :param size: The size of the noise to generate (in pixel)
        :param nbr_control_point: Number of control points to use to generate the noise.
        :param seed: To make generation reproducible
        """
        super().__init__(size=size)
        np.random.seed(seed)
        self.control_points = np.random.randint(0, self.size + 1, size=(nbr_control_point, 3))

    def generate(self):
        """Generate Worley noise grid"""
        # Create coordinates
        x = np.arange(self.size)
        y = np.arange(self.size)
        z = np.arange(self.size)
        xx, yy, zz = np.meshgrid(x, y, z)

        # Compute the difference between each control point and each coordinate
        dx = np.subtract.outer(xx, self.control_points[:, 0])
        dy = np.subtract.outer(yy, self.control_points[:, 1])
        dz = np.subtract.outer(zz, self.control_points[:, 2])

        # Considering Euclidean distance
        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Select only the minimum distance
        self.noise = distances.min(axis=3)


# endregion
# region Simplex noise

class SimplexNoise2D(Noise):
    def __init__(self, size: int, seed: int = None):
        super().__init__(size=size, seed=seed)
        self.F = (math.sqrt(3) - 1) / 2
        self.G = 1 / math.sqrt(2 + math.sqrt(3))

    def coordinate_skewing(self, coord: tuple[int, int]):
        sum_coord = sum(coord)
        return (coord[0] + sum_coord) * self.F, (coord[1] + sum_coord) * self.F

    def simplicial_subdivision(self):
        ...


# endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', choices=['perlin1d', 'perlin2d', 'perlin3d', 'worley1d', 'worley2d', 'worley3d'],
                        default='perlin2d')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ncp', type=int, default=10)  # Number of control points for Worley noise
    parser.add_argument('--gs', type=int, default=10)  # Grid size for Perlin noise
    parser.add_argument('--dims', type=int, default=2)  # Number of dimensions for Perlin octave noise
    parser.add_argument('--numoct', type=int, default=2)  # Number of octaves for Perlin octave noise

    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    if args.noise == 'perlin1d':
        noise = PerlinNoise1D(size=args.size, grid_size=args.gs, seed=args.seed)
    elif args.noise == 'perlin2d':
        noise = PerlinNoise2D(size=args.size, grid_size=args.gs, seed=args.seed)
    elif args.noise == 'perlin3d':
        noise = PerlinNoise3D(size=args.size, grid_size=args.gs, seed=args.seed)
    elif args.noise == 'octave':
        noise = PerlinOctave(size=args.size, dims=args.dims, num_octave=args.numoct, seed=args.seed)
    elif args.noise == 'worley1d':
        noise = WorleyNoise1D(size=args.size, nbr_control_point=args.ncp, seed=args.seed)
    elif args.noise == 'worley2d':
        noise = WorleyNoise2D(size=args.size, nbr_control_point=args.ncp, seed=args.seed)
    elif args.noise == 'worley3d':
        noise = WorleyNoise3D(size=args.size, nbr_control_point=args.ncp, seed=args.seed)
    else:
        raise ValueError(f"Noise type {args.noise} not supported")

    noise.generate()
    noise.show(savefig=args.save)
