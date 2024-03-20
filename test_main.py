import argparse
from unittest.mock import patch, mock_open, MagicMock
import pytest
import main
from simulation import Simulation
from walker import RandomAngleWalker, RandomStepWalker


@patch('argparse.ArgumentParser.parse_args')
def test_args(mock_args):
    mock_args.return_value = argparse.Namespace(config_file='config.json', output_file='output.txt',
                                                graphs_output_file='output.pdf')
    args = argparse.ArgumentParser().parse_args()
    assert args.config_file == 'config.json'
    assert args.output_file == 'output.txt'
    assert args.graphs_output_file == 'output.pdf'


@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
@patch('multiprocessing.pool.Pool.map')
def test_create_simulations(mock_pool, mock_file, mock_json):
    mock_json.return_value = {
        "X": {
            "type": "regular",
            "times_to_run": 10,
            "number_of_steps": 100,
            "walker": {
                "n_dim": 2
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    mock_pool.return_value = [Simulation(10, 100, RandomAngleWalker("Test"), [0, 1], 10.0)]
    sims = main.create_simulations(mock_json.return_value)
    assert len(sims) == 1


@patch('seaborn.stripplot')
@patch('matplotlib.pyplot.subplots')
def test_generate_plots(mock_subplots, mock_stripplot):
    mock_subplots.return_value = MagicMock(), MagicMock()
    sims = [Simulation(10, 100, RandomAngleWalker("Test"), [0, 1], 10.0)]
    for sim in sims:
        sim.run()
    main.generate_plots(sims)
    assert mock_stripplot.call_count == 4


@patch('matplotlib.pyplot.subplots')
def test_generate_path_plot(mock_subplots):
    sims = [Simulation(10, 100, RandomAngleWalker("Test"), [0, 1], 10.0)]
    mock_subplots.return_value = MagicMock(), MagicMock()
    for sim in sims:
        sim.run()
    main.generate_path_plot(sims)
    assert mock_subplots.call_count == 1


@patch('argparse.ArgumentParser.parse_args')
def test_create_simulations_different_walkers(mock_json):
    mock_json.return_value = {
        "X": {
            "type": "step",
            "times_to_run": 10,
            "number_of_steps": 100,
            "walker": {
                "n_dim": 2,
                "min_step_size": 0.5,
                "max_step_size": 1.5
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    sims = main.create_simulations(mock_json.return_value)
    assert len(sims) == 1
    assert isinstance(sims[0].get_walker(), RandomStepWalker)


@patch('seaborn.stripplot')
@patch('matplotlib.pyplot.subplots')
def test_generate_plots_different_simulations(mock_subplots, mock_stripplot):
    mock_subplots.return_value = MagicMock(), MagicMock()
    sims = [Simulation(10, 100, RandomStepWalker("Test", min_step_size=0.5, max_step_size=1.5), [0, 1], 10.0)]
    for sim in sims:
        sim.run()
    main.generate_plots(sims)
    assert mock_stripplot.call_count == 4


@patch('matplotlib.pyplot.subplots')
def test_generate_path_plot_different_simulations(mock_subplots):
    sims = [Simulation(10, 100, RandomStepWalker("Test", min_step_size=0.5, max_step_size=1.5), [0, 1], 10.0)]
    mock_subplots.return_value = MagicMock(), MagicMock()
    for sim in sims:
        sim.run()
    main.generate_path_plot(sims)
    assert mock_subplots.call_count == 1


@patch('argparse.ArgumentParser.parse_args')
def test_main_different_arguments(mock_args):
    mock_args.return_value = argparse.Namespace(config_file='config2.json', output_file='output2.txt',
                                                graphs_output_file='output2.pdf')
    args = argparse.ArgumentParser().parse_args()
    assert args.config_file == 'config2.json'
    assert args.output_file == 'output2.txt'
    assert args.graphs_output_file == 'output2.pdf'


def test_worker():
    sim = Simulation(10, 100, RandomAngleWalker("Test"), [0, 1], 10.0)
    assert len(sim.get_sims()) == 0
    main.worker(sim)
    assert len(sim.get_sims()) == 10


@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
def test_create_simulations_invalid(mock_file, mock_json):
    mock_json.return_value = {
        "X": {
            "type": "unknown",
            "times_to_run": 10,
            "number_of_steps": 100,
            "walker": {
                "n_dim": 2
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    with pytest.raises(ValueError):
        main.create_simulations(mock_json.return_value)


@patch('seaborn.stripplot')
@patch('matplotlib.pyplot.subplots')
def test_generate_plots_no_simulations(mock_subplots, mock_stripplot):
    mock_subplots.return_value = MagicMock(), MagicMock()
    sims = []
    with pytest.raises(IndexError):
        main.generate_plots(sims)


@patch('matplotlib.pyplot.subplots')
def test_generate_path_plot_no_simulations(mock_subplots):
    sims = []
    mock_subplots.return_value = MagicMock(), MagicMock()
    with pytest.raises(IndexError):
        main.generate_path_plot(sims)


@patch('builtins.open', new_callable=mock_open)
def test_save_results_invalid_file(mock_file):
    mock_file.side_effect = IOError
    sims = [Simulation(10, 100, RandomAngleWalker("Test"), [0, 1], 10.0)]
    for sim in sims:
        sim.run()
    with pytest.raises(IOError):
        main.save_results(sims, 'invalid_file.txt')


@patch('builtins.open', new_callable=mock_open)
def test_generate_and_save_graphs_invalid_file(mock_file):
    mock_file.side_effect = IOError
    sims = [Simulation(10, 100, RandomAngleWalker("Test"), [0, 1], 10.0)]
    for sim in sims:
        sim.run()
    with pytest.raises(IOError):
        main.generate_and_save_graphs(sims, 'invalid_file.pdf')


@patch('builtins.open', new_callable=mock_open)
def test_save_results(mock_file):
    sims = [Simulation(10, 100, RandomAngleWalker("Test"), [0, 1], 10.0)]
    for sim in sims:
        sim.run()
    main.save_results(sims, 'output.txt')
    assert mock_file.call_count == 1


@patch('argparse.ArgumentParser.parse_args')
def test_parse_arguments(mock_args):
    mock_args.return_value = argparse.Namespace(config_file='config.json', output_file='output.txt',
                                                graphs_output_file='output.pdf')
    args = main.parse_arguments()
    assert args.config_file == 'config.json'
    assert args.output_file == 'output.txt'
    assert args.graphs_output_file == 'output.pdf'


@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
def test_create_simulations_invalid_walker_type(mock_file, mock_json):
    mock_json.return_value = {
        "X": {
            "type": "unknown",
            "times_to_run": 10,
            "number_of_steps": 100,
            "walker": {
                "n_dim": 2
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    with pytest.raises(ValueError):
        main.create_simulations(mock_json.return_value)


@patch('seaborn.stripplot')
@patch('matplotlib.pyplot.subplots')
def test_generate_plots_no_simulations(mock_subplots, mock_stripplot):
    mock_subplots.return_value = MagicMock(), MagicMock()
    sims = []
    with pytest.raises(IndexError):
        main.generate_plots(sims)


@patch('main.generate_plots')
@patch('main.generate_path_plot')
@patch('file_manager.save_graphs')
def test_generate_and_save_graphs_different_plot_counts(mock_save_graphs, mock_generate_path_plot, mock_generate_plots):
    # Create a list of mock simulations
    sims = [MagicMock(spec=Simulation) for _ in range(3)]

    # Mock the return values of generate_plots and generate_path_plot
    mock_generate_plots.return_value = [MagicMock(), MagicMock()]
    mock_generate_path_plot.return_value = [MagicMock()]

    # Call the function with the mock simulations
    main.generate_and_save_graphs(sims, 'output.pdf')

    # Check that save_graphs was called with the correct number of plots
    assert len(mock_save_graphs.call_args[0][1]) == 3


@patch('main.generate_plots')
@patch('main.generate_path_plot')
@patch('file_manager.save_graphs')
def test_generate_and_save_graphs(mock_save_graphs, mock_generate_path_plot, mock_generate_plots):
    # Create a list of mock simulations
    sims = [MagicMock(spec=Simulation) for _ in range(3)]

    # Mock the return values of generate_plots and generate_path_plot
    mock_generate_plots.return_value = [MagicMock(), MagicMock()]
    mock_generate_path_plot.return_value = [MagicMock()]
    # Call the function with the mock simulations
    main.generate_and_save_graphs(sims, 'output.pdf')
    # Check that save_graphs was called with the correct number of plots
    assert mock_save_graphs.call_args[0][1] == mock_generate_plots.return_value[
                                               :-1] + mock_generate_path_plot.return_value
    assert mock_save_graphs.call_args[0][0] == 'output.pdf'


@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
def test_create_simulations_invalid_config(mock_file, mock_json):
    mock_json.return_value = {
        "X": {
            "type": "step",
            "times_to_run": "ten",
            "number_of_steps": 100,
            "walker": {
                "restart_chance": 0,
                "restart_every": 1,
                "n_dim": 3,
                "magic_gates_placements": [],
                "magic_gates_dests": [],
                "obstacles": [],
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    with pytest.raises(TypeError):
        main.create_simulations(mock_json.return_value)


@patch('argparse.ArgumentParser.parse_args')
@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
@patch('multiprocessing.pool.Pool.map')
@patch('file_manager.write_to_file')
def test_integration(mock_write_to_file, mock_pool, mock_file, mock_json, mock_args):
    # Mock the command line arguments
    mock_args.return_value = argparse.Namespace(config_file='config.json', output_file='output.txt',
                                                graphs_output_file='output.pdf')

    # Mock the configuration file
    mock_json.return_value = {
        "X": {
            "type": "regular",
            "times_to_run": 10,
            "number_of_steps": 100,
            "walker": {
                "restart_chance": 0,
                "restart_every": 1,
                "n_dim": 2,
                "magic_gates_placements": [],
                "magic_gates_dests": [],
                "obstacles": [],
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    # Mock the simulations
    mock_pool.return_value = [MagicMock(spec=Simulation) for _ in range(3)]
    # Call the main function
    main.main()
    # Check that the simulations were created and run
    assert mock_pool.call_count == 1
    # Check that the results were saved
    assert mock_write_to_file.call_count == 1


@patch('argparse.ArgumentParser.parse_args')
@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
@patch('multiprocessing.pool.Pool.map')
@patch('file_manager.write_to_file')
def test_integration_different_config(mock_write_to_file, mock_pool, mock_file, mock_json, mock_args):
    # Mock the command line arguments
    mock_args.return_value = argparse.Namespace(config_file='config2.json', output_file='output2.txt',
                                                graphs_output_file='output2.pdf')

    # Mock the configuration file
    mock_json.return_value = {
        "X": {
            "type": "step",
            "times_to_run": 10,
            "number_of_steps": 100,
            "walker": {
                "restart_chance": 0,
                "restart_every": 1,
                "n_dim": 2,
                "magic_gates_placements": [],
                "magic_gates_dests": [],
                "obstacles": [],
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    # Mock the simulations
    mock_pool.return_value = [MagicMock(spec=Simulation) for _ in range(3)]
    # Call the main function
    main.main()
    # Check that the simulations were created and run
    assert mock_pool.call_count == 1
    # Check that the results were saved
    assert mock_write_to_file.call_count == 1


@patch('argparse.ArgumentParser.parse_args')
@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
@patch('multiprocessing.pool.Pool.map')
@patch('file_manager.write_to_file')
def test_integration_invalid_config(mock_write_to_file, mock_pool, mock_file, mock_json, mock_args):
    # Mock the command line arguments
    mock_args.return_value = argparse.Namespace(config_file='config.json', output_file='output.txt',
                                                graphs_output_file='output.pdf')

    # Mock the configuration file
    mock_json.return_value = {
        "X": {
            "type": "step",
            "times_to_run": "10",
            "number_of_steps": 100,
            "walker": {
                "restart_chance": 0,
                "restart_every": 1,
                "n_dim": 2,
                "magic_gates_placements": [],
                "magic_gates_dests": [],
                "obstacles": [],
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    # Mock the simulations
    mock_pool.return_value = [MagicMock(spec=Simulation) for _ in range(3)]
    # Call the main function
    main.main()
    # Check that the simulations were created and run
    assert mock_pool.call_count == 0
    # Check that the results were saved
    assert mock_write_to_file.call_count == 0


@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
def test_create_simulations_empty_config(mock_file, mock_json):
    # Mock the configuration file to be empty
    mock_json.return_value = {}
    # Call the function with the mock configuration file
    with pytest.raises(ValueError):
        main.create_simulations(mock_json.return_value)


def test_simulation_zero_steps_runs():
    # Create a simulation with zero steps and runs
    with pytest.raises(ValueError):
        sim = Simulation(0, 0, RandomAngleWalker("Test"), [0, 1], 10.0)


@patch('seaborn.stripplot')
@patch('matplotlib.pyplot.subplots')
def test_generate_plots_empty_simulations(mock_subplots, mock_stripplot):
    # Call the function with an empty list of simulations
    with pytest.raises(IndexError):
        main.generate_plots([])


@patch('matplotlib.pyplot.subplots')
def test_generate_path_plot_empty_simulations(mock_subplots):
    # Call the function with an empty list of simulations
    with pytest.raises(IndexError):
        main.generate_path_plot([])


@patch('builtins.open', new_callable=mock_open)
def test_save_results_empty_simulations(mock_file):
    # Call the function with an empty list of simulations
    with pytest.raises(IndexError):
        main.save_results([], 'output.txt')


@patch('main.generate_plots')
@patch('main.generate_path_plot')
@patch('file_manager.save_graphs')
def test_generate_and_save_graphs_empty_simulations(mock_save_graphs, mock_generate_path_plot, mock_generate_plots):
    # Call the function with an empty list of simulations
    with pytest.raises(IndexError):
        main.generate_and_save_graphs([], 'output.pdf')


@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
def test_create_simulations_invalid_walker_type(mock_file, mock_json):
    # Mock the configuration file with an invalid walker type
    mock_json.return_value = {
        "X": {
            "type": "unknown",
            "times_to_run": 10,
            "number_of_steps": 100,
            "walker": {
                "n_dim": 2
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    # Call the function with the mock configuration file
    with pytest.raises(ValueError):
        main.create_simulations(mock_json.return_value)


@patch('argparse.ArgumentParser.parse_args')
def test_parsing_invalid_json_file(mock_args):
    mock_args.return_value = argparse.Namespace(config_file='config.jso', output_file='output.txt',
                                                graphs_output_file='output.pdf')
    with pytest.raises(SystemExit):
        main.parse_arguments()


@patch('argparse.ArgumentParser.parse_args')
def test_parsing_invalid_output_file(mock_args):
    mock_args.return_value = argparse.Namespace(config_file='config.json', output_file='output.tx',
                                                graphs_output_file='output.pdf')
    with pytest.raises(SystemExit):
        main.parse_arguments()


@patch('argparse.ArgumentParser.parse_args')
def test_parsing_invalid_graphs_output_file(mock_args):
    mock_args.return_value = argparse.Namespace(config_file='config.json', output_file='output.txt',
                                                graphs_output_file='output.pd')
    with pytest.raises(SystemExit):
        main.parse_arguments()


def test_create_walker():
    mock_data = {
        "name": "Test",
        "n_dim": 2,
        "obstacles": [],
        "magic_gates_placements": [],
        "magic_gates_dests": [],
        "restart_chance": 0,
        "restart_every": 1

    }
    walker = main.create_walker("regular", mock_data)
    assert isinstance(walker, RandomAngleWalker)


def test_create_walker_invalid_type():
    mock_data = {
        "name": "Test",
        "n_dim": 2,
        "obstacles": [],
        "magic_gates_placements": [],
        "magic_gates_dests": [],
        "restart_chance": 0,
        "restart_every": 1

    }
    with pytest.raises(ValueError):
        main.create_walker("unknown", mock_data)


def test_create_walker_invalid_data():
    mock_data = {
        "name": "Test",
        "n_dim": "3",
        "obstacles": [],
        "magic_gates_placements": [],
        "magic_gates_dests": [],
        "restart_chance": 0,
        "restart_every": 1

    }
    with pytest.raises(TypeError):
        main.create_walker("regular", mock_data)


@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
def test_create_simulations_invalid_config(mock_file, mock_json):
    mock_json.return_value = {
        "X": {
            "type": "step",
            "times_to_run": "10",
            "number_of_steps": 100,
            "walker": {
                "restart_chance": 0,
                "restart_every": 1,
                "n_dim": 2,
                "magic_gates_placements": [],
                "magic_gates_dests": [],
                "obstacles": [],
            },
            "axis": [0, 1],
            "radius": 10.0
        }
    }
    with pytest.raises(TypeError):
        main.create_simulations(mock_json.return_value)


@patch('argparse.ArgumentParser.parse_args')
@patch('builtins.open', side_effect=FileNotFoundError)
def test_main_file_not_found(mock_open, mock_args):
    mock_args = argparse.Namespace(config_file='config.json', output_file='output.txt', graphs_output_file='output.pdf')
    with pytest.raises(SystemExit):
        main.main()


@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.Axes')
def test_generate_path_plot_3d(mock_axes, mock_figure):
    # Create a mock simulation with a 3D walker
    sim = Simulation(10, 100, RandomAngleWalker("Test", n_dim=3), [0, 1, 2], 10.0)
    sim.run()
    sims = [sim]

    # Call the function with the mock simulations
    main.generate_path_plot(sims)

    # Check that figure and Axes were called
    count = mock_figure.call_count
    assert count >= 1
