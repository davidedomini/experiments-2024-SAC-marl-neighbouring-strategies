import random as rnd
import numpy as np
import math

class Vector2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def to_np_array(self):
        return np.array([self.x, self.y], dtype=np.float32)

    def get_random_point(max_x, max_y, min_x=0, min_y=0):
        return Vector2D(rnd.randint(min_x, max_x-1), rnd.randint(min_y, max_y-1))
    
    def get_int_nbrs(v, step=1):
        precision=0
        rounded_v = Vector2D(round(v.x), round(v.y))
        nbr_distances = np.array([[1.0, 0.0],[-1.0, 0.0],[0.0, 1.0],[0.0, -1.0]])*step
        return [Vector2D(nbr[0] + rounded_v.x, nbr[1] + rounded_v.y) for nbr in nbr_distances]

    def round_int(v):
        return Vector2D(round(v.x), round(v.y))

    def copy(v):
        return Vector2D(v.x, v.y)

    def distance_vector(v1, v2):
        return Vector2D(v1.x-v2.x, v1.y-v2.y)
    
    def distance(v1, v2):
        distance_vector = Vector2D.distance_vector(v1, v2)
        return Vector2D.norm(distance_vector)

    def norm(v):
        return math.sqrt(math.pow(v.x, 2) + math.pow(v.y, 2))

    def cast_int(v):
        return Vector2D(int(v.x), int(v.y))

    def unit_vector(v):
        norm = Vector2D.norm(v)
        if norm == 0:
            return Vector2D(0,0)
        return Vector2D(v.x/norm, v.y/norm)

    def from_rad(rad):
        return Vector2D(math.cos(rad), math.sin(rad))

    def similarity(v1, v2):
        x1, y1 = v1.x, v1.y
        x2, y2 = v2.x, v2.y
        v1_value = math.degrees(math.atan2(y1, x1))/180
        v2_value = math.degrees(math.atan2(y2, x2))/180
        diff = abs(v1_value-v2_value)
        similarity = 1 - math.pow(diff, 0.9)
        return similarity
    
    def sum(v1, v2):
        return Vector2D(v1.x+v2.x, v1.y+v2.y)
    
    def mul(v, n):
        return Vector2D(v.x*n, v.y*n)
    
    def __eq__(self, other):
        if isinstance(other, Vector2D):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))