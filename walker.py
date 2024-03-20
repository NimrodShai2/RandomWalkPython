import abc
import numpy as np
import random
from abc import ABC
from typing import List, Optional
from scipy.spatial import KDTree  # type: ignore


def normalize_vector(dimen: int = 2) -> np.ndarray:
    """
    Normalize a random direction vector to the unit size.
    """
    if dimen <= 1:
        raise ValueError("Number of dimensions must be positive and greater than 2.")
    direction = np.random.randn(dimen)
    direction /= np.linalg.norm(direction)
    return direction


def exited_radius(position: List, radius: float) -> bool:
    """
    Check if the walker has exited the radius.
    :param position: Position to check.
    :param radius: Radius to check.
    :return: True if the walker has exited the radius, False otherwise.
    """
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    if radius < np.linalg.norm(np.array(position)):
        return True
    return False


class Walker(ABC):
    """
    Base class for all random walkers.
    """

    def __init__(self, name: str, n_dim: int = 2, magic_gates_placements: Optional[List[List]] = None,
                 magic_gates_dests: Optional[List[List]] = None,
                 obstacles: Optional[List[List]] = None,
                 restart_chance: float = 0,
                 restart_every: int = 0):
        """
        Constructor for the Walker class.
        :param name: Name of the walker.
        :param n_dim: Number of dimensions.
        :param magic_gates_placements: List of magic gates placements -
        if the walker gets there, he will magically appear someplace else.
        :param magic_gates_dests: List of magic gates destinations. Those are the places
        the walker will appear.
        :param obstacles: List of obstacles - positions the walker should skip.
        """
        # Following code is safety checks for the walker constructor
        self.validate_name(name)
        self.validate_dimensions(n_dim)
        self.validate_obstacles(n_dim, obstacles)
        self.validate_gates(magic_gates_dests, magic_gates_placements, n_dim)
        self.validate_restart(restart_chance, restart_every)
        # End of safety checks
        self.__name = name
        self._restart_chance: float = restart_chance
        self._restart_every: int = restart_every
        self._dim: int = n_dim
        # Create KDTree for obstacles and magic gates
        self._magic_gates_placements: Optional[KDTree] = KDTree(
            magic_gates_placements) if magic_gates_placements else None
        self._magic_gates_destinations: Optional[List] = magic_gates_dests
        self._obstacles: Optional[KDTree] = KDTree(obstacles) if obstacles else None
        # Initialize the current position and the path
        self._current_position: List = [0] * n_dim
        self._path: List[List] = [self._current_position]

    @staticmethod
    def validate_restart(restart_chance, restart_every):
        if restart_chance < 0 or restart_chance > 1:
            raise ValueError("Restart chance must be between 0 and 1.")
        if restart_every < 1:
            raise ValueError("Restart-every must be positive.")

    @staticmethod
    def validate_gates(magic_gates_dests, magic_gates_placements, n_dim):
        if magic_gates_placements is not None:
            for gate in magic_gates_placements:
                if len(gate) != n_dim:
                    raise ValueError("Magic gates must be vectors of the same dimension.")
        if magic_gates_dests is not None:
            for gate in magic_gates_dests:
                if len(gate) != n_dim:
                    raise ValueError("Magic gates destinations must be vectors of the same dimension.")
        # Check for None and empty lists in gates and destinations
        if magic_gates_placements == [] and magic_gates_dests != []:
            raise ValueError("Magic gates placements must be provided if magic gates destinations are provided.")
        if magic_gates_placements != [] and magic_gates_dests == []:
            raise ValueError("Magic gates destinations must be provided if magic gates placements are provided.")
        if magic_gates_placements is None and magic_gates_dests is not None:
            raise ValueError("Magic gates placements must be provided if magic gates destinations are provided.")
        if magic_gates_placements is not None and magic_gates_dests is None:
            raise ValueError("Magic gates destinations must be provided if magic gates placements are provided.")
        # Check that gates, vectors, and obstacles are all vectors of numbers
        if magic_gates_placements is not None and magic_gates_dests is not None:
            for gate in magic_gates_placements:
                if not all(isinstance(x, (int, float)) for x in gate):
                    raise ValueError("Magic gates must be vectors of numbers.")
            for gate in magic_gates_dests:
                if not all(isinstance(x, (int, float)) for x in gate):
                    raise ValueError("Magic gates destinations must be vectors of numbers.")

    @staticmethod
    def validate_obstacles(n_dim, obstacles):
        if obstacles is not None:
            for obstacle in obstacles:
                if len(obstacle) != n_dim:
                    raise ValueError("Obstacles must be vectors of the same dimension.")
        if obstacles is not None:
            for obstacle in obstacles:
                if not all(isinstance(x, (int, float)) for x in obstacle):
                    raise ValueError("Obstacles must be vectors of numbers.")

    @staticmethod
    def validate_dimensions(n_dim):
        if n_dim <= 1:
            raise ValueError("Number of dimensions must be positive and greater than 2.")

    @staticmethod
    def validate_name(name):
        if name == "":
            raise ValueError("Name must not be empty.")

    def get_dim(self) -> int:
        """
        Get the number of dimensions.
        :return: Number of dimensions.
        """
        return self._dim

    def dist_from_origin_after(self, n: int) -> float:
        """
        Get the distance from the origin after n steps.
        :param n: Number of steps.
        :return: Distance from the origin.
        """
        if n < 0:
            raise ValueError("Number of steps must be positive.")
        if n >= len(self._path):
            raise ValueError("Number of steps must be less than the length of the path.")
        if type(n) is not int:
            raise ValueError("Number of steps must be an integer.")
        return float(np.linalg.norm(np.array(self._path[n])))  # Calculate the distance from the origin

    def times_crossed_y_axis_after(self, n: int) -> int:
        """
        Get the number of times the walker crossed the y-axis after n steps.
        :param n: Number of steps.
        :return: Number of times the walker crossed the y-axis.
        """
        if n < 0:
            raise ValueError("Number of steps must be positive.")
        if n >= len(self._path):
            raise ValueError("Number of steps must be less than the length of the path.")
        if type(n) is not int:
            raise ValueError("Number of steps must be an integer.")
        count = 0
        for i in range(n - 1):
            if self._path[i][1] > 0 >= self._path[i + 1][1]:
                count += 1
            elif self._path[i][1] < 0 <= self._path[i + 1][1]:
                count += 1
        return count

    def dist_from_axis_after(self, axis: List, n: int) -> float:
        """
        Get the distance from the axis after n steps.
        :param axis: Axis to measure distance from. A base vector for the axis.
        :param n: Number of steps.
        :return: Distance from the axis.
        """
        if type(n) is not int:
            raise ValueError("Number of steps must be an integer.")
        if len(axis) != self._dim:
            raise ValueError("Axis must be a vector of the same dimension as the walker.")
        if np.linalg.norm(axis) != 1:
            raise ValueError("Axis must be a unit vector.")
        if n < 0:
            raise ValueError("Number of steps must be positive.")
        if n >= len(self._path):
            raise ValueError("Number of steps must be less than the length of the path.")
        # Calculate the distance by projection on the axis
        vector = np.array(self._path[n])
        new_axis = np.array(axis)
        projection = np.dot(vector, new_axis) / np.linalg.norm(new_axis)
        projection_vector = projection * new_axis / np.linalg.norm(new_axis)
        return float(np.linalg.norm(vector - projection_vector))

    def exited_radius_at(self, radius: float) -> int:
        """
        Check when the walker has exited the radius.
        :param radius: Radius to check.
        :return: The index of the step at which the walker exited the radius.
        Zero if the walker never exited the radius.
        """
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        if len(self._path) == 0:
            raise ValueError("Path is empty.")
        for i in range(len(self._path)):
            if exited_radius(self._path[i], radius):
                return i
        return 0

    @abc.abstractmethod
    def step(self) -> List[float]:
        pass

    def get_path(self) -> List[List]:
        """
        Get the path of the walker up to the current point.
        :return:
        """
        return self._path[:]

    def get_current_position(self) -> List[float]:
        """
        Get the current position of the walker.
        :return:
        """
        return self._current_position[:]

    def hard_restart(self):
        """
        Restart the walker at the origin.
        """
        self._current_position = [0] * self._dim
        self._path = [self._current_position]
        return self._current_position[:]

    def restart(self):
        """
        Restart the walker at the origin.
        """
        self._current_position = [0] * self._dim
        return self._current_position[:]

    def get_basis_vectors(self) -> List[List]:
        """
        Get the basis vectors of the dimension.
        :return: the basis vectors.
        """
        basis_vectors = []
        for i in range(self._dim):
            basis_vectors.append([0] * self._dim)
            basis_vectors[i][i] = 1
        return basis_vectors

    def walk(self, steps: int):
        """
        Walk n steps.
        Skip obstacles and move through magic gates.
        :param steps: Number of steps to walk.
        """
        if steps <= 0:
            raise ValueError("Number of steps must be positive.")
        for i in range(steps):
            pos_after_step = self.step()
            if self._obstacles:
                while self._obstacles.query_ball_point(pos_after_step, 1):
                    pos_after_step = self.step()
            if self._magic_gates_placements and self._magic_gates_destinations:
                if self._magic_gates_placements.query_ball_point(pos_after_step, 0):
                    pos_after_step = random.choice(self._magic_gates_destinations)
            if i % self._restart_every == 0:
                if random.random() < self._restart_chance:
                    pos_after_step = self.restart()
            self._path.append(pos_after_step)

    def get_name(self):
        """
        Get the name of the walker.
        :return:
        """
        return self.__name[:]

    def get_restart_every(self):
        """
        Get the restart every value.
        :return:
        """
        return self._restart_every

    def set_path(self, path: List):
        """
        Set the path of the walker.
        :param path: Path to set.
        """
        if len(path) == 0:
            raise ValueError("Path must not be empty.")
        self._path = path[:]
        self._current_position = path[-1]


class RandomAngleWalker(Walker):
    """
    Random walker that can go one unit to a random angle each step.
    """

    def __init__(self, name: str, n_dim: int = 2, magic_gates_placements: Optional[List[List]] = None,
                 magic_gates_dests: Optional[List[List]] = None,
                 obstacles: Optional[List[List]] = None,
                 restart_chance: float = 0,
                 restart_every: int = 1):
        super().__init__(name, n_dim, magic_gates_placements, magic_gates_dests, obstacles
                         , restart_chance, restart_every)
        self.__step_size: float = 1

    def step(self) -> List[float]:
        """
        Move one step in a random direction.
        :return:
        """
        self._current_position = self._current_position[:]
        direction = normalize_vector(self._dim)
        self._current_position += direction * self.__step_size
        return list(self._current_position)


class RandomStepWalker(Walker):
    """
    Random walker that can go random units in a random direction each step.
    """

    def __init__(self, name: str, n_dim: int = 2, min_step_size: float = 0.5, max_step_size: float = 1.5,
                 magic_gates_placements: Optional[List[List]] = None,
                 magic_gates_dests: Optional[List[List]] = None,
                 obstacles: Optional[List[List]] = None,
                 restart_chance: float = 0,
                 restart_every: int = 1):
        if min_step_size > max_step_size:
            raise ValueError("Min step size must be less than or equal to max step size.")
        if min_step_size <= 0 or max_step_size <= 0:
            raise ValueError("Step size must be positive.")
        super().__init__(name, n_dim, magic_gates_placements, magic_gates_dests, obstacles
                         , restart_chance, restart_every)
        self.__min_step_size: float = min_step_size
        self.__max_step_size: float = max_step_size

    def step(self) -> List[float]:
        """
        Move one step in a random direction.
        :return:
        """
        step_size = random.uniform(self.__min_step_size, self.__max_step_size)
        self._current_position = self._current_position[:]
        direction = normalize_vector(self._dim)
        self._current_position += direction * step_size
        return list(self._current_position)


class RandomGridWalker(Walker):
    """
    Random walker that can go one unit in a grid in a random direction each step.
    """

    def __init__(self, name: str, n_dim: int = 2, magic_gates_placements: Optional[List[List]] = None,
                 magic_gates_dests: Optional[List[List]] = None,
                 obstacles: Optional[List[List]] = None,
                 restart_chance: float = 0,
                 restart_every: int = 1
                 ):
        super().__init__(name, n_dim, magic_gates_placements, magic_gates_dests, obstacles,
                         restart_chance, restart_every)

    def step(self) -> List[float]:
        """
        Move one step in a random direction on the grid.
        :return:
        """
        self._current_position = self._current_position[:]
        vectors = self.get_basis_vectors()
        sign = random.choice([-1, 1])
        direction = random.randint(0, self._dim - 1)
        # Set new position as plus or minus 1 the direction chosen.
        self._current_position[direction] = self._current_position[direction] + sign * vectors[direction][direction]
        return self._current_position


class BiasedRandomWalker(Walker):
    """
    Random walker that can go random units in a random direction each step,
     but with a bias towards a random direction.
    """

    def __init__(self, name: str, bias_direction: Optional[List[int]] = None, n_dim: int = 2, bias_strength: float = 0,
                 magic_gates_placements: Optional[List[List]] = None,
                 magic_gates_dests: Optional[List[List]] = None,
                 obstacles: Optional[List[List]] = None,
                 restart_chance: float = 0,
                 restart_every: int = 1
                 ):
        """
        Constructor for the biased random walker.
        :param n_dim: Number of dimensions
        :param bias_direction: The direction to bias towards. It is the base vector for the specified
        dimension in the specified direction. If None, the bias is towards the origin.
        :param bias_strength: The strength of the bias. Must be between zero and 1.
        """
        self.validate_bias(bias_direction, bias_strength, n_dim)
        super().__init__(name, n_dim, magic_gates_placements, magic_gates_dests, obstacles,
                         restart_chance, restart_every)
        self.__bias_direction: Optional[List[int]] = bias_direction
        self.__bias_strength: float = bias_strength
        self.__step_size: float = 1

    @staticmethod
    def validate_bias(bias_direction, bias_strength, n_dim):
        if bias_direction:
            if len(bias_direction) != n_dim:
                raise ValueError("Bias direction must be of length n_dim.")
            for i in bias_direction:
                if not isinstance(i, int):
                    raise ValueError("Bias direction must be vectors of numbers.")
            if np.linalg.norm(bias_direction) != 1:
                raise ValueError("Bias direction must be a unit vector.")
        if bias_strength < 0 or bias_strength > 1:
            raise ValueError("Bias strength must be between 0 and 1.")

    def step(self) -> List[float]:
        """
        Move one step in a random direction.
        :return:
        """
        self._current_position = self._current_position[:]
        direction = normalize_vector(self._dim)
        if self.__bias_direction:
            bias_direction = np.array(self.__bias_direction)
        else:
            bias_direction = -np.array(self._current_position)
        # Some linear algebra to create the bias
        if self.__bias_strength < 1:
            combined_direction = ((1 - self.__bias_strength) *
                                  direction + self.__bias_strength * bias_direction)
            combined_direction /= np.linalg.norm(combined_direction)
        else:
            combined_direction = bias_direction
        self._current_position += combined_direction * self.__step_size
        return list(self._current_position)


class RandomSearcher(RandomGridWalker):
    """
    Random grid walker that searches for a set target in a random direction each step.
    """

    def __init__(self, name: str, target: List[int], n_dim: int = 2,
                 magic_gates_placements: Optional[List[List]] = None,
                 magic_gates_dests: Optional[List[List]] = None,
                 obstacles: Optional[List[List]] = None,
                 restart_chance: float = 0,
                 restart_every: int = 1):
        self.validate_target(n_dim, target)
        super().__init__(name, n_dim, magic_gates_placements, magic_gates_dests, obstacles,
                         restart_chance, restart_every)
        self.__target: List[int] = target

    @staticmethod
    def validate_target(n_dim, target):
        if len(target) != n_dim:
            raise ValueError("Target must be of the same dimension.")
        for tar in target:
            if not isinstance(tar, int):
                raise ValueError("Target must be vectors of numbers.")

    def step(self) -> List[float]:
        """
        Move one step in a random direction.
        :return:
        """
        if self._current_position == self.__target:
            return self._current_position
        else:
            return super().step()
