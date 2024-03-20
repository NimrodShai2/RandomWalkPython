from typing import Any, Callable
from walker import *
import numpy as np


class Simulation:
    def __init__(self, times_to_run: int, num_of_steps: int, walker: Walker, axis: List, radius: float):
        if times_to_run <= 0:
            raise ValueError("Times to run must be greater than 0.")
        if num_of_steps <= 0:
            raise ValueError("Number of steps must be greater than 0.")
        if radius <= 0:
            raise ValueError("Radius to check must be greater than 0.")
        if len(axis) == 0:
            raise ValueError("Axis to check must not be empty.")
        if len(axis) != walker.get_dim():
            raise ValueError("Axis to check must be of the same length as the number of dimensions.")
        if walker.get_restart_every() > num_of_steps:
            raise ValueError("Number of steps must be greater than or equal to the restart-every value.")
        self.__axis: List = axis
        self.__radius: float = radius
        self.__times_to_run: int = times_to_run
        self.__walker: Walker = walker
        self.__num_of_steps: int = num_of_steps
        self.__sims: List = []

    def _apply_to_sims(self, func: Callable[[], Any]) -> List:
        """
        Applies a function to each simulation.
        :param func: The function to apply.
        :return: A result list of the function.
        """
        results = []
        for sim in self.__sims:
            self.__walker.set_path(sim)
            results.append(func())
        return results

    def get_walker_name(self) -> str:
        """
        Returns the name of the walker.
        """
        return self.__walker.get_name()

    def run(self):
        """
        Runs the simulation for the number of times specified in the constructor.
        """
        for _ in range(self.__times_to_run):
            self.__walker.hard_restart()
            try:
                self.__walker.walk(self.__num_of_steps)
            except ValueError:
                return
            self.__sims.append(self.__walker.get_path())

    def get_sims(self) -> List[List]:
        """
        Returns the list of paths of each simulation.
        """
        return self.__sims[:]

    def is_abstract(self) -> bool:
        """
        Returns whether the simulation is abstract.
        """
        return self.__walker.get_dim() > 3

    def get_num_of_steps(self):
        """
        Returns the number steps of each simulation.
        """
        return self.__num_of_steps

    def get_times_run(self):
        """
        Returns the number of times the simulation was run.
        """
        return len(self.__sims)

    def get_avg_dist_from_origin_after(self, n: int) -> float:
        """
        Returns the average distance from the origin after n steps.
        :param n: Number of steps.
        :return: Average distance from the origin.
        """
        dists = self._apply_to_sims(lambda: self.__walker.dist_from_origin_after(n))
        return float(np.average(np.array(dists)))

    def get_avg_dist_from_axis_after(self, n: int) -> float:
        """
        Returns the average distance from the axis after n steps.
        :param n: Number of steps.
        :return: Average distance from the axis.
        """
        dists = self._apply_to_sims(lambda: self.__walker.dist_from_axis_after(self.__axis, n))
        return float(np.average(np.array(dists)))

    def get_distances_from_origin_after(self, n: int) -> List[float]:
        """
        Returns the list of distances from the origin.
        :param n: Number of steps.
        :return: List of distances from the origin.
        """
        dists = self._apply_to_sims(lambda: self.__walker.dist_from_origin_after(n))
        return dists[:]

    def get_distances_from_axis_after(self, n: int) -> List[float]:
        """
        Returns the list of distances from the axis after n steps.
        :param n: Number of steps.
        :return: List of distances from the axis.
        """
        dists = self._apply_to_sims(lambda: self.__walker.dist_from_axis_after(self.__axis, n))
        return dists[:]

    def avg_step_exited_radius(self) -> float:
        """
        Returns the average step at which the walker exited the radius.
        :return: Average step at which the walker exited the radius.
        """
        steps = self._apply_to_sims(lambda: self.__walker.exited_radius_at(self.__radius))
        return float(np.average(np.array(steps)))

    def get_step_exited_radius(self) -> List[int]:
        """
        Returns the list of steps at which the walker exited the radius.
        :return: List of steps at which the walker exited the radius.
        """
        steps = self._apply_to_sims(lambda: self.__walker.exited_radius_at(self.__radius))
        steps = [step for step in steps if step != 0]
        return steps[:]

    def get_times_crossed_y_axis_after(self, n: int) -> List[int]:
        """
        Returns the list of times the walker crossed the y-axis.
        :param n: Number of steps.
        :return: List of times the walker crossed the y-axis.
        """
        times = self._apply_to_sims(lambda: self.__walker.times_crossed_y_axis_after(n))
        return times[:]

    def avg_times_crossed_y_axis_after(self, n: int) -> float:
        """
        Returns the average number of times the walker crossed the y-axis.
        :param n: Number of steps.
        :return: Average number of times the walker crossed the y-axis.
        """
        times = self._apply_to_sims(lambda: self.__walker.times_crossed_y_axis_after(n))
        return float(np.average(np.array(times)))

    def get_all_stats_str(self) -> str:
        """
        Returns a string containing all the stats.
        :return: String containing all the stats.
        """
        stats = [
            f"Average distance from origin after {self.__num_of_steps}"
            f" steps: {self.get_avg_dist_from_origin_after(self.__num_of_steps)}",
            f"Average distance from {self.__axis} axis after {self.__num_of_steps}"
            f" steps: {self.get_avg_dist_from_axis_after(self.__num_of_steps)}",
            f"Average step at which the walker exited the {self.__radius}"
            f" radius: {self.avg_step_exited_radius()}",
            f"Average number of times the walker crossed the y-axis:"
            f" {self.avg_times_crossed_y_axis_after(self.__num_of_steps)}\n"]
        return "\n".join(stats)

    def get_avg_path(self) -> List[List[float]]:
        """
        Returns the average path of the simulation.
        :return: Average path of the simulation.
        """
        if len(self.__walker.get_path()) <= 1:
            raise ValueError("No simulations have been run.")
        avg_path = []
        # Initialize the average path with the same number of steps as the simulation.
        for i in range(self.__num_of_steps):
            avg_path.append([0.0] * self.__walker.get_dim())
        for i in range(self.__num_of_steps):
            avg_path.append([0.0] * self.__walker.get_dim())
        # Add all the paths together.
        for sim in self.__sims:
            if sim:
                for i in range(len(sim)):
                    for j in range(self.__walker.get_dim()):
                        avg_path[i][j] += sim[i][j]
        # Divide each element by the number of simulations.
        for i in range(len(avg_path)):
            for j in range(self.__walker.get_dim()):
                avg_path[i][j] /= len(self.__sims)
        return avg_path

    def get_dim(self) -> int:
        """
        Returns the dimension of the simulation.
        :return: Dimension of the simulation.
        """
        return self.__walker.get_dim()

    def get_walker(self):
        """
        Returns the walker of the simulation.
        :return: Walker of the simulation.
        """
        return self.__walker
