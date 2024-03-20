from walker import exited_radius, RandomAngleWalker, RandomStepWalker, Walker, RandomGridWalker, BiasedRandomWalker
import pytest


def test_walker_base_class():
    try:
        Walker("")
    except TypeError:
        pass
    else:
        assert False, "Walker should not be instantiable"


def test_walker_class_has_walk_method():
    w = RandomStepWalker("Test")
    assert hasattr(w, 'walk')
    assert callable(w.walk)


def test_random_angle_walker_class():
    w = RandomAngleWalker("Test")
    assert hasattr(w, 'walk')
    assert hasattr(w, 'step')
    assert callable(w.step)
    assert callable(w.walk)
    w2 = RandomAngleWalker("Test")
    w.walk(100)
    w2.walk(100)
    lst1 = w.get_path()
    lst2 = w2.get_path()
    assert len(lst1) == 101
    assert len(lst2) == 101
    assert lst1[0] == [0, 0] == lst2[0]
    assert lst1[-1] != lst2[-1]
    assert w.restart() == [0, 0]


def test_random_step_walker_class():
    w = RandomStepWalker("Test")
    assert hasattr(w, 'walk')
    assert hasattr(w, 'step')
    assert callable(w.step)
    assert callable(w.walk)
    w2 = RandomStepWalker("Test")
    w.walk(100)
    w2.walk(100)
    lst1 = w.get_path()
    lst2 = w2.get_path()
    assert len(lst1) == 101
    assert len(lst2) == 101
    assert lst1[0] == [0, 0] == lst2[0]
    assert lst1[-1] != lst2[-1]
    assert w.restart() == [0, 0]


def test_random_grid_walker():
    w = RandomGridWalker("Test")
    w2 = RandomGridWalker("Test2")
    assert hasattr(w, 'walk')
    assert hasattr(w, 'step')
    assert callable(w.step)
    assert callable(w.walk)
    w.walk(100)
    w2.walk(100)
    lst1 = w.get_path()
    lst2 = w2.get_path()
    assert len(lst1) == 101 == len(lst2)
    assert lst1[0] == [0, 0] == lst2[0]
    assert lst1[-1] != lst2[-1]
    assert type(lst1[-1][-1]) is int
    assert type(lst2[-1][-1]) is int
    assert w.restart() == [0, 0]


def test_biased_walker():
    w = BiasedRandomWalker("Test", bias_direction=[0, 1], bias_strength=0.5)
    w2 = BiasedRandomWalker("Test2", bias_direction=[1, 0], bias_strength=0.5)
    assert hasattr(w, 'walk')
    assert hasattr(w, 'step')
    assert callable(w.step)
    assert callable(w.walk)
    w.walk(100)
    w2.walk(100)
    lst1 = w.get_path()
    lst2 = w2.get_path()
    assert len(lst1) == 101 == len(lst2)
    assert lst1[0] == [0, 0] == lst2[0]
    assert lst1[-1] != lst2[-1]
    assert w.restart() == [0, 0]


def test_walker_exceptions():
    try:
        w = RandomStepWalker("", -1)
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable with negative dimensions"
    try:
        w = RandomAngleWalker("A", 0)
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable with zero dimensions"
    try:
        w = RandomStepWalker("", 1, 0, -1)
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable with negative or zero step size"
    try:
        w = RandomStepWalker("", 1, 3, 1)
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable with max step size greater than min step size"


def test_magic_gates():
    w = RandomGridWalker("Test", magic_gates_dests=[[4, 5]], magic_gates_placements=[[0, 1], [0, -1]])
    w.walk(100)
    lst = w.get_path()
    assert lst[0] == [0, 0]
    assert [4, 5] in lst
    assert [0, 1] not in lst
    assert [0, -1] not in lst
    assert len(lst) == 101


def test_magic_gates_exceptions():
    try:
        w = RandomGridWalker("Test", magic_gates_dests=[[4, 5, 6]], magic_gates_placements=[[0, 1], [0, -1]])
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable with 3-dimensional vector magic gate destinations."
    try:
        w = RandomGridWalker("", magic_gates_dests=[[4, 5]], magic_gates_placements=[[0, 1, 1]])
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable with 3-dimensional vector magic gate placements."
    try:
        w = RandomGridWalker("", magic_gates_placements=[[0, 1], [0, 1]])
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable without magic gate destinations."
    try:
        w = RandomGridWalker("", magic_gates_dests=[[4, 5]])
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable without magic gate placements."


def test_obstacles():
    w = RandomGridWalker("Test", obstacles=[[0, 1], [0, -1]])
    w.walk(100)
    lst = w.get_path()
    assert lst[0] == [0, 0]
    assert [0, 1] not in lst
    assert [0, -1] not in lst
    assert len(lst) == 101


def test_obstacles_exceptions():
    try:
        w = RandomGridWalker("Test", obstacles=[[0, 1, 2]])
    except ValueError:
        pass
    else:
        assert False, "Walker should not be instantiable with 3-dimensional vector obstacles."


def test_exited_radius_inside():
    assert not exited_radius([0, 0], 1), "Should be inside the radius"


def test_exited_radius_edge():
    assert not exited_radius([1, 0], 1), "Should be considered inside at the edge"


def test_exited_radius_outside():
    assert exited_radius([2, 0], 1), "Should be outside the radius"


def test_exited_radius_negative_coordinates():
    assert exited_radius([-2, 0], 1), "Negative coordinates outside the radius"


def test_exited_radius_zero_radius():
    with pytest.raises(ValueError):
        exited_radius([0, 0], 0)


def test_exited_radius_negative_radius():
    with pytest.raises(ValueError):
        exited_radius([0, 0], -1)


def test_dist_from_origin_after():
    w = RandomAngleWalker("Test")
    w.walk(10)
    distance = w.dist_from_origin_after(5)
    assert isinstance(distance, float), "Distance should be a float"
    assert distance >= 0, "Distance should be non-negative"


def test_dist_from_origin_after_negative_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    try:
        w.dist_from_origin_after(-1)
    except ValueError:
        pass
    else:
        assert False, "dist_from_origin_after should raise ValueError for negative steps"


def test_dist_from_origin_after_too_many_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    try:
        w.dist_from_origin_after(11)
    except ValueError:
        pass
    else:
        assert False, "dist_from_origin_after should raise ValueError for steps greater than path length"


def test_times_crossed_y_axis_after():
    w = RandomAngleWalker("Test")
    w.walk(10)
    times_crossed = w.times_crossed_y_axis_after(5)
    assert isinstance(times_crossed, int), "Times crossed should be an integer"
    assert times_crossed >= 0, "Times crossed should be non-negative"


def test_times_crossed_y_axis_after_zero_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    times_crossed = w.times_crossed_y_axis_after(0)
    assert times_crossed == 0, "Times crossed should be zero after zero steps"


def test_times_crossed_y_axis_after_negative_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    try:
        w.times_crossed_y_axis_after(-1)
    except ValueError:
        pass
    else:
        assert False, "times_crossed_y_axis_after should raise ValueError for negative steps"


def test_times_crossed_y_axis_after_too_many_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    try:
        w.times_crossed_y_axis_after(11)
    except ValueError:
        pass
    else:
        assert False, "times_crossed_y_axis_after should raise ValueError for steps greater than path length"


def test_dist_from_axis_after():
    w = RandomAngleWalker("Test")
    w.walk(10)
    distance = w.dist_from_axis_after([1, 0], 5)
    assert isinstance(distance, float), "Distance should be a float"
    assert distance >= 0, "Distance should be non-negative"


def test_dist_from_axis_after_zero_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    distance = w.dist_from_axis_after([1, 0], 0)
    assert distance == 0, "Distance should be zero after zero steps"


def test_dist_from_axis_after_negative_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    try:
        w.dist_from_axis_after([1, 0], -1)
    except ValueError:
        pass
    else:
        assert False, "dist_from_axis_after should raise ValueError for negative steps"


def test_dist_from_axis_after_too_many_steps():
    w = RandomAngleWalker("Test")
    w.walk(10)
    try:
        w.dist_from_axis_after([1, 0], 11)
    except ValueError:
        pass
    else:
        assert False, "dist_from_axis_after should raise ValueError for steps greater than path length"


def test_restart():
    w = RandomAngleWalker("Test")
    w.walk(10)
    w.restart()
    assert w.get_current_position() == [0, 0], "Position should be reset to [0, 0] after restart"
    assert len(w.get_path()) == 11, "Path should not be reset after restart"


def test_hard_restart():
    w = RandomAngleWalker("Test")
    w.walk(10)
    w.hard_restart()
    assert w.get_current_position() == [0, 0], "Position should be reset to [0, 0] after hard restart"
    assert w.get_path() == [[0, 0]], "Path should be reset to [[0, 0]] after hard restart"


def test_random_angle_walker_step():
    w = RandomAngleWalker("Test")
    position_before = w.get_current_position()
    w.step()
    position_after = w.get_current_position()
    assert all(position_before) != all(position_after), "Position should change after step"


def test_biased_random_walker_step():
    w = BiasedRandomWalker("Test", bias_direction=[1, 0], bias_strength=0.5)
    position_before = w.get_current_position()
    w.step()
    position_after = w.get_current_position()
    assert all(position_before) != all(position_after), "Position should change after step"


def test_restart_after_multiple_walks():
    w = RandomAngleWalker("Test")
    w.walk(10)
    w.walk(10)
    w.restart()
    assert w.get_current_position() == [0, 0], "Position should be reset to [0, 0] after restart"
    assert len(w.get_path()) == 21, "Path should not be reset after restart"


def test_hard_restart_after_multiple_walks():
    w = RandomAngleWalker("Test")
    w.walk(10)
    w.walk(10)
    w.hard_restart()
    assert w.get_current_position() == [0, 0], "Position should be reset to [0, 0] after hard restart"
    assert w.get_path() == [[0, 0]], "Path should be reset to [[0, 0]] after hard restart"


def test_dist_from_origin_after_with_different_step_sizes():
    w = RandomStepWalker("Test", min_step_size=2, max_step_size=4)
    w.walk(10)
    distance = w.dist_from_origin_after(5)
    assert isinstance(distance, float), "Distance should be a float"
    assert distance >= 0, "Distance should be non-negative"


def test_walk_with_different_step_sizes():
    w = RandomStepWalker("Test", min_step_size=2, max_step_size=4)
    w.walk(10)
    path = w.get_path()
    for i in range(1, len(path)):
        step_size = ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5
        assert 2 <= step_size <= 4, "Step size should be between 2 and 4"


def test_random_angle_walker_init():
    with pytest.raises(ValueError):
        RandomAngleWalker("")


def test_random_step_walker_init():
    with pytest.raises(ValueError):
        RandomStepWalker("")


def test_random_grid_walker_init():
    with pytest.raises(ValueError):
        RandomGridWalker("")


def test_biased_random_walker_init():
    with pytest.raises(ValueError):
        BiasedRandomWalker("", bias_direction=[0, 1], bias_strength=0.5)


def test_biased_random_walker_init_bias_direction():
    with pytest.raises(ValueError):
        BiasedRandomWalker("Test", bias_direction=[0, 1, 2], bias_strength=0.5)


def test_biased_random_walker_init_bias_strength():
    with pytest.raises(ValueError):
        BiasedRandomWalker("Test", bias_direction=[0, 1], bias_strength=1.5)


def test_step_different_parameters():
    w = RandomStepWalker("Test", min_step_size=0.5, max_step_size=1.5)
    position_before = w.get_current_position()
    w.step()
    position_after = w.get_current_position()
    assert all(position_before) != all(position_after), "Position should change after step"


def test_walk_different_parameters():
    w = RandomStepWalker("Test", min_step_size=0.5, max_step_size=1.5)
    w.walk(10)
    path = w.get_path()
    assert len(path) == 11, "Path should have 11 points after 10 steps"


def test_restart_and_hard_restart():
    w = RandomStepWalker("Test", min_step_size=0.5, max_step_size=1.5)
    w.walk(10)
    w.restart()
    assert w.get_current_position() == [0, 0], "Position should be reset to [0, 0] after restart"
    assert len(w.get_path()) == 11, "Path should not be reset after restart"
    w.hard_restart()
    assert w.get_current_position() == [0, 0], "Position should be reset to [0, 0] after hard restart"
    assert w.get_path() == [[0, 0]], "Path should be reset to [[0, 0]] after hard restart"


def test_get_path_and_get_current_position():
    w = RandomStepWalker("Test", min_step_size=0.5, max_step_size=1.5)
    w.walk(10)
    path = w.get_path()
    assert len(path) == 11, "Path should have 11 points after 10 steps"
    assert all(w.get_current_position()) == all(path[-1]), "Current position should be the last point in the path"
