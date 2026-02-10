import math
import random
from random import uniform
from typing import Collection


class Vector:
    """Class for vector of any dimension"""

    def __init__(self, *args):
        self.coords = args

    def __str__(self):
        return "Instance of vector class : " + str(self.coords)

    def __repr__(self):
        return str(self.coords)

    def __iter__(self):
        return self.coords.__iter__()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, item):
        return self.coords[item]

    def __eq__(self, other):
        return self.coords == other.coords

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("Addition of a vector with type {} is not supported".format(type(other)))
        return Vector(*tuple([x1 + x2 for x1, x2 in zip(self.coords, other.coords)]))

    def __sub__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("Addition of a vector with type {} is not supported".format(type(other)))
        return Vector(*tuple([x1 - x2 for x1, x2 in zip(self.coords, other.coords)]))

    def dot_product(self, other) -> float:
        """Returns the dot product of two vectors"""
        if not isinstance(other, Vector):
            raise TypeError("The dot product requires two vectors")
        if len(self) != len(other):
            raise ValueError(
                "The dot product requires two vectors of the same size. Size given {} and {}"
                .format(len(self), len(other)))
        return sum([x1 * x2 for x1, x2 in zip(self.coords, other.coords)])

    def __mul__(self, other):
        """Returns the dot product of two vectors, or the multiplication with a scalar"""
        if isinstance(other, Vector):
            return self.dot_product(other)
        if isinstance(other, (int, float)):
            return Vector(*tuple([x * other for x in self.coords]))
        else:
            raise ValueError("Multiplication with type {} is not supported".format(type(other)))

    def __rmul__(self, other):
        """ Called if 4 * self for instance """
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(*tuple([x / other for x in self.coords]))
        else:
            raise ValueError("Division with type {} is not supported".format(type(other)))

    def norm(self) -> float:
        """To get the norm of the vector"""
        return math.sqrt(sum([x * x for x in self.coords]))

    def normalize(self):
        """Returns a normalized unit vector"""
        norm = self.norm()
        return Vector(*tuple([x / norm for x in self.coords]))

    def is_collinear(self, other) -> bool:
        """Returns true if self and other are collinear"""
        if not isinstance(other, Vector):
            raise ValueError("This operation is not supported with type {} ".format(type(other)))
        for i in range(len(self.coords)):
            for j in range(0, i):
                if self.coords[i] * other.coords[j] != self.coords[j] * other.coords[i]:
                    return False
        return True


class Vector2D(Vector):
    def __init__(self, *args):
        """Class for 2-dimensional vectors"""
        if len(args) == 2:
            super().__init__(*args)
        else:
            raise ValueError(f"Must provide 2 coordinates, got {len(args)}")

    def direction(self) -> float:
        """Return the direction of a 2-dimensional vector"""
        if len(self) != 2:
            raise ValueError("Unsupported operation for non-2 dimensional vector")
        if not self.coords[0]:
            return math.pi / 2.
        return math.atan(self.coords[1] / self.coords[0])


def vector_from_dots(dot1: Collection, dot2: Collection) -> Vector:
    """Creates a vector from two given points"""
    if len(dot1) != len(dot2):
        raise ValueError("The two given dots must belong to the same dimension. "
                         "One is {}-dimensional, while the other is {}-dimensional".format(len(dot1), len(dot2)))
    return Vector(*tuple(dot1)) - Vector(*tuple(dot2))


def vector_from_magdir(direction: float = 0., magnitude: float = 1.) -> Vector2D:
    """Creates a vector of direction and magnitude"""
    x = magnitude * math.cos(direction)
    y = magnitude * math.sin(direction)
    return Vector2D(x, y)


def generate_random_unit_2d_vect(seed: int = None) -> Vector2D:
    """Generate a 2d unit vector in a random direction."""
    random.seed(seed)
    direction = uniform(0, 2 * math.pi)
    return Vector2D(math.cos(direction), math.sin(direction))
