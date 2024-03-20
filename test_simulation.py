import time

from simulation import Simulation
from walker import RandomAngleWalker
import pytest


def test_simulations_number():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.get_times_run() == 10
    assert len(sim.get_sims()[0]) == 11
    assert len(sim.get_sims()[9]) == 11


def test_stats_types():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert type(sim.get_avg_dist_from_origin_after(7)) is float
    assert type(sim.get_avg_dist_from_axis_after(5)) is float
    assert type(sim.avg_times_crossed_y_axis_after(7)) is float
    assert type(sim.avg_step_exited_radius()) is float


def test_fields():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert len(sim.get_sims()) == 10
    assert sim.get_times_run() == 10
    assert sim.get_walker_name() == "A"


def test_simulation_init_times_to_run():
    with pytest.raises(ValueError):
        Simulation(0, 10, RandomAngleWalker("A"), [0, 1], 10)


def test_simulation_init_num_of_steps():
    with pytest.raises(ValueError):
        Simulation(10, 0, RandomAngleWalker("A"), [0, 1], 10)


def test_simulation_init_radius():
    with pytest.raises(ValueError):
        Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 0)


def test_simulation_init_axis_empty():
    with pytest.raises(ValueError):
        Simulation(10, 10, RandomAngleWalker("A"), [], 10)


def test_simulation_init_axis_dim():
    with pytest.raises(ValueError):
        Simulation(10, 10, RandomAngleWalker("A"), [0, 1, 2], 10)


def test_get_avg_dist_from_origin_after():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert isinstance(sim.get_avg_dist_from_origin_after(7), float)


def test_get_avg_dist_from_origin_after_zero_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.get_avg_dist_from_origin_after(0) == 0.0


def test_get_avg_dist_from_origin_after_more_than_num_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_avg_dist_from_origin_after(11)


def test_get_avg_dist_from_origin_after_negative_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_avg_dist_from_origin_after(-1)


def test_get_avg_dist_from_origin_after_non_integer_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_avg_dist_from_origin_after(7.5)


def test_get_distances_from_origin_after():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    distances = sim.get_distances_from_origin_after(7)
    assert isinstance(distances, list)
    assert all(isinstance(distance, float) for distance in distances)


def test_get_distances_from_origin_after_zero_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.get_distances_from_origin_after(0) == [0.0] * 10


def test_get_distances_from_origin_after_more_than_num_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_distances_from_origin_after(11)


def test_get_distances_from_origin_after_negative_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_distances_from_origin_after(-1)


def test_get_distances_from_origin_after_non_integer_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_distances_from_origin_after(7.5)


def test_get_avg_dist_from_axis_after():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert isinstance(sim.get_avg_dist_from_axis_after(7), float)


def test_get_avg_dist_from_axis_after_zero_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.get_avg_dist_from_axis_after(0) == 0.0


def test_get_avg_dist_from_axis_after_more_than_num_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_avg_dist_from_axis_after(11)


def test_get_avg_dist_from_axis_after_negative_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_avg_dist_from_axis_after(-1)


def test_get_avg_dist_from_axis_after_non_integer_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_avg_dist_from_axis_after(7.5)


def test_get_avg_dist_from_axis_non_unit_axis():
    axis = [0, 2]
    sim = Simulation(10, 10, RandomAngleWalker("A"), axis, 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_avg_dist_from_axis_after(7)


def test_get_distances_from_axis_after():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    distances = sim.get_distances_from_axis_after(7)
    assert isinstance(distances, list)
    assert all(isinstance(distance, float) for distance in distances)


def test_get_distances_from_axis_after_zero_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.get_distances_from_axis_after(0) == [0.0] * 10


def test_get_distances_from_axis_after_more_than_num_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_distances_from_axis_after(11)


def test_get_distances_from_axis_after_negative_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_distances_from_axis_after(-1)


def test_get_distances_from_axis_after_non_integer_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_distances_from_axis_after(7.5)


def test_get_distances_from_axis_non_unit_axis():
    axis = [0, 2]
    sim = Simulation(10, 10, RandomAngleWalker("A"), axis, 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_distances_from_axis_after(7)


def test_avg_times_crossed_y_axis_after():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert isinstance(sim.avg_times_crossed_y_axis_after(7), float)


def test_avg_times_crossed_y_axis_after_zero_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.avg_times_crossed_y_axis_after(0) == 0.0


def test_avg_times_crossed_y_axis_after_more_than_num_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.avg_times_crossed_y_axis_after(11)


def test_avg_times_crossed_y_axis_after_negative_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.avg_times_crossed_y_axis_after(-1)


def test_avg_times_crossed_y_axis_after_non_integer_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.avg_times_crossed_y_axis_after(7.5)


def test_get_times_crossed_y_axis_after():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    times = sim.get_times_crossed_y_axis_after(7)
    assert isinstance(times, list)
    assert all(isinstance(time, int) for time in times)


def test_get_times_crossed_y_axis_after_zero_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.get_times_crossed_y_axis_after(0) == [0] * 10


def test_get_times_crossed_y_axis_after_more_than_num_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_times_crossed_y_axis_after(11)


def test_get_times_crossed_y_axis_after_negative_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_times_crossed_y_axis_after(-1)


def test_get_times_crossed_y_axis_after_non_integer_steps():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    with pytest.raises(ValueError):
        sim.get_times_crossed_y_axis_after(7.5)


def test_avg_step_exited_radius():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert isinstance(sim.avg_step_exited_radius(), float)


def test_get_step_exited_radius():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    steps = sim.get_step_exited_radius()
    assert isinstance(steps, list)
    assert all(isinstance(step, int) for step in steps)


def test_get_step_exited_radius_non_exited():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert sim.get_step_exited_radius() == []


def test_get_avg_path():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    assert isinstance(sim.get_avg_path(), list)
    assert all(isinstance(step, list) for step in sim.get_avg_path())
    assert all(isinstance(coord, float) for step in sim.get_avg_path() for coord in step)


def test_get_avg_path_empty():
    sim = Simulation(10, 10, RandomAngleWalker("A"), [0, 1], 10)
    with pytest.raises(ValueError):
        sim.get_avg_path()


def test_large_number_of_simulations():
    start_time = time.time()
    sim = Simulation(10000, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    end_time = time.time()
    execution_time = end_time - start_time
    assert execution_time < 1.5


def test_large_number_of_steps():
    sim = Simulation(10, 100000, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    start_time = time.time()
    sim.get_avg_dist_from_origin_after(50000)
    end_time = time.time()
    execution_time = end_time - start_time
    assert execution_time < 1.5


def test_large_avg_path():
    sim = Simulation(100000, 10, RandomAngleWalker("A"), [0, 1], 10)
    sim.run()
    start_time = time.time()
    sim.get_avg_path()
    end_time = time.time()
    execution_time = end_time - start_time
    assert execution_time < 1.5


def test_large_num_steps_runtime():
    sim = Simulation(1, 10000, RandomAngleWalker("A"), [0, 1], 10)
    start_time = time.time()
    sim.run()
    end_time = time.time()
    execution_time = end_time - start_time
    assert execution_time < 1.5
