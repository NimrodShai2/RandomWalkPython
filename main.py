import argparse
import file_manager as fm
from walker import RandomAngleWalker, RandomStepWalker, RandomGridWalker, BiasedRandomWalker, RandomSearcher
import simulation
from multiprocessing import pool
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt
from typing import Any, Dict, List

description: str = """
A program that simulates random walks and saves to files of your choosing stats and graphs about them.
To start, you need to specify the configuration file, the output file and the graphs output file.
The configuration file is a json file with the following format:
"X":{ - The name of the walker.
    "walker_type": "regular" | "step" | "grid" | "biased" | "searcher". the type of the walker.
    "times_to_run":  int > 0, the number of times to run the simulation.
    "num_of_steps":  int > 0, the number of steps to take.
    "walker": an object with the following properties:
    {
        "n_dim": 2,3 to produce walking graphs, or any other larger integer in abstract mode (that doesn't produce walking
        graphs).
        "magic_gates_placements": [[0, 0], [1, 1]], etc. It is a list of vectors from the specified dimension
        at which the walker will enter an magically appear in one of the destinations.
        "magic_gates_dests": [[0, 1], [1, 0]], etc. It is a list of vectors from the specified dimension where
        the walker will be teleported to after entering a magic gate.
        "obstacles": [[0.5, 0.5]], etc. It is a list of vectors from the specified dimension where the walker
        will not be able to move.
        Additionally, there can be optional values that determine the possibility of the walker to restart:
        "restart_chance":  float. The chance of the walker to restart after taking a step. Must be between 0 and 1.
        default is 0.
        "restart_every":  int > 0. The number of steps after which the walker will restart. Default is 1.
        If the walker type is "biased", you need to specify the following:
        "bias_direction": [0,1], [1,0]. Unit vector representing the direction the walker will be biased in.
        "bias_strength": float. The strength of the bias. Must be between 0 and 1.
        If the walker type is "step", you need to specify the following:
        "min_step_size": float > 0. The minimum step size of the walker.
        "max_step_size": float > 0. The maximum step size of the walker.
        If the walker type is "searcher", you need to specify the following:
        "target":  [0,1], [2,3], etc. Vector representing the target of the searcher.
    }
    "axis": [0,1], [1,0], etc. Unit vectors of the specified dimension, representing the axis the simulation should
    save stats about.
    "radius":  float. The radius of the circle which the simulation will save stats about.
 """

epilog: str = """Stats that will be saved and presented as graphs in a pdf file for each simulation:
Average distance from origin per number of steps taken.
Average distance from given axis per number of step taken.
Average step at which the walker exited the radius,
 per the average distance at the end of the simulation.
Average times(across the simulations) that the walker crossed the y-axis.
Additionally, the pdf file will include a representation of the
 average path taken by the walker for each non-abstract simulation."""


def worker(sim: simulation.Simulation) -> simulation.Simulation:
    """
    Run the simulation.
    :param sim: The simulation to run.
    :return: The same simulation after running it.
    """
    sim.run()
    return sim


def create_simulations(config: Dict[str, Any]) -> List[simulation.Simulation]:
    """
        Create the simulation objects.
        :param config: The configuration dictionary.
        :return: The list of simulation objects.
        """
    if len(config) == 0:
        raise ValueError("No simulations to create.")
    sims = []
    walker_classes = {
        "regular": RandomAngleWalker,
        "step": RandomStepWalker,
        "grid": RandomGridWalker,
        "biased": BiasedRandomWalker,
        "searcher": RandomSearcher
    }
    for simu in config:
        # Set default restart options, set names.
        data = config[simu]["walker"]
        data.setdefault("restart_every", 1)
        data.setdefault("restart_chance", 0.0)
        data.setdefault("name", simu)
        # Build the walker by type.
        walker_type = config[simu]["type"]
        if walker_type in walker_classes:
            w = walker_classes[walker_type](**data)
        else:
            raise ValueError("Invalid walker type.")
        sim = simulation.Simulation(config[simu]["times_to_run"], config[simu]["number_of_steps"], w
                                    , config[simu]["axis"], config[simu]["radius"])
        sims.append(sim)
    # Run the simulations, using multiprocessing to speed them up.
    with pool.Pool() as p:
        sims = p.map(worker, sims)
    return sims


def generate_plots(sims: List[simulation.Simulation]):
    """
    Generate the plots.
    :param sims: The list of simulation objects.
    :return: The list of the axes of the plots.
    """
    if len(sims) == 0:
        raise IndexError("No simulations to generate plots for.")
    num_of_steps = [sim.get_num_of_steps() for sim in sims]

    def new_stripplot(x: List, y: List, title: str, xlabel: str, ylabel: str) -> plt.Axes:
        """
        Create a new stripplot.
        :param ylabel: The y-axis label.
        :param xlabel: The x-axis label.
        :param title: The title of the plot.
        :param x: The x-axis values.
        :param y: The y-axis values.
        :return: The new stripplot.
        """
        fig, ax = plt.subplots()
        sns.stripplot(x=x, y=y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    plot_data = [
        {
            "x": [sim.get_num_of_steps() for sim in sims],
            "y": [sim.get_avg_dist_from_origin_after(steps) for sim, steps in zip(sims, num_of_steps)],
            "title": "Average distance from origin",
            "xlabel": "Number of steps",
            "ylabel": "Average distance (in units)"
        },
        {
            "x": [sim.get_num_of_steps() for sim in sims],
            "y": [sim.get_avg_dist_from_axis_after(steps) for sim, steps in zip(sims, num_of_steps)],
            "title": "Average distance from axis",
            "xlabel": "Number of steps",
            "ylabel": "Average distance (in units)"
        },
        {
            "x": [round(sim.get_avg_dist_from_origin_after(steps)) for sim, steps in zip(sims, num_of_steps)],
            "y": [sim.avg_step_exited_radius() for sim in sims],
            "title": "Average step in which walker exited the radius",
            "xlabel": "Rounded average distance from the origin after final step",
            "ylabel": "Number of steps taken to cross the radius"
        },
        {
            "x": num_of_steps,
            "y": [sim.avg_times_crossed_y_axis_after(steps) for sim, steps in zip(sims, num_of_steps)],
            "title": "Average number of times the walker crossed the y-axis",
            "xlabel": "Number of steps in total",
            "ylabel": "Number of times crossed"
        }
    ]
    return [new_stripplot(**data) for data in plot_data]  # type: ignore


def generate_path_plot(sims: List[simulation.Simulation]) -> List[plt.Axes]:
    """
    Generate the path plot.
    :param sims: The simulation object.
    :return: The list of the axes of the plots.
    """
    if len(sims) == 0:
        raise IndexError("No simulations to generate path plots for.")

    def new_2D_path_plot(x: List, y: List) -> plt.Axes:
        """
        Create a new 2D path plot.
        :param x: The x-axis values.
        :param y: The y-axis values.
        :return: The new path plot.
        """
        fig, ax = plt.subplots()
        plt.plot(x, y, color='red', linewidth=1.0, linestyle='-')
        plt.title("Walker path")
        plt.xlabel("X-position")
        plt.ylabel("Y-position")
        plt.grid(True)
        return ax

    def new_3D_path_plot(x: List, y: List, z: List) -> plt.Axes:
        """
        Create a new 3D path plot.
        :param x: The x-axis values.
        :param y: The y-axis values.
        :param z: The z-axis values.
        :return: The new path plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, color='red', linewidth=1.0, linestyle='-')
        ax.set_xlabel("X-position")
        ax.set_ylabel("Y-position")
        return ax

    plots = []
    for sim in sims:
        if not sim.is_abstract():
            coordinates = sim.get_avg_path()
            x = [c[0] for c in coordinates]
            y = [c[1] for c in coordinates]
            if sim.get_dim() == 2:
                ax = new_2D_path_plot(x, y)
                ax.set_title(f"Walker path for {sim.get_walker_name()}")
                plots.append(ax)
            elif sim.get_dim() == 3:
                z = [c[2] for c in coordinates]
                ax = new_3D_path_plot(x, y, z)
                ax.set_title(f"Walker path for {sim.get_walker_name()}")
                plots.append(ax)
    return plots


def parse_arguments() -> argparse.Namespace:
    """
    Parse the arguments from the command line.
    :return: Namespace with the parsed arguments.
    """
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config_file", type=str, help="The file with the walks configuration."
                                                      " Must be a JSON file.")
    parser.add_argument("output_file", type=str, help="The file where the output stats will be saved."
                                                      " Must be a .txt file.")
    parser.add_argument("graphs_output_file", type=str, help="The file where the graphs will be saved."
                                                             " Must be a .pdf file.")
    return parser.parse_args()


def save_results(sims: List[simulation.Simulation], output_file: str) -> None:
    """
    Save the results to the output file.
    :param sims: The list of simulation objects.
    :param output_file: The file to save the results to.
    """
    if len(sims) == 0:
        raise IndexError("No simulations to save results for.")
    texts: List[str] = []
    for sim in sims:
        text_to_save = (f"Results for {sim.get_walker_name()}:\n {sim.get_num_of_steps()} steps and "
                        f"{sim.get_times_run()} runs:\n")
        text_to_save += sim.get_all_stats_str()
        texts.append(text_to_save)
    fm.write_to_file(output_file, texts)


def generate_and_save_graphs(sims: List[simulation.Simulation], graphs_output_file: str) -> None:
    """
    Generate and save the graphs.
    :param sims: The list of simulation objects.
    :param graphs_output_file: The file to save the graphs to.
    """
    if len(sims) == 0:
        raise IndexError("No simulations to generate graphs for.")
    plots = generate_plots(sims)
    plots.extend(generate_path_plot(sims))
    fm.save_graphs(graphs_output_file, plots)


def main():
    args = parse_arguments()
    # Check if the files are of correct formats.
    if not args.config_file.endswith(".json"):
        print("The configuration file must be a JSON file.")
        print("The program will now exit.")
        exit(1)
    if not args.output_file.endswith(".txt"):
        print("The output file must be a .txt file.")
        print("The program will now exit.")
        exit(1)
    if not args.graphs_output_file.endswith(".pdf"):
        print("The graphs output file must be a .pdf file.")
        print("The program will now exit.")
        exit(1)
    # Load the config file.
    try:
        d: Dict[str, Any] = fm.load_json(args.config_file)
    except FileNotFoundError as e:
        print(e)
        print("Try again with the correct file name.")
        exit(1)
    # Create the simulations specified in the config file.
    try:
        sims = create_simulations(d)
    except KeyError as e:
        print("Simulations failed."
              "The file consists of a key that does not exist, or lacks a mandatory value."
              " Please fix the file and try again."
              " If in doubt, run --help.")
        print(e)
    except TypeError as e:
        print("Simulations failed."
              "One of the values is of an illegal type."
              " Please choose a legal value type. For more information run --help.")
        print(e)
    except ValueError as e:
        print("Simulations failed.")
        print(e)
    else:
        try:
            save_results(sims, args.output_file)
        except Exception as e:
            print("Failed to write to file.")
            print(e)
        try:
            # If successful, generate the graphs and save them to the graphs output file.
            # Generate the graphs.
            # Save the graphs.
            generate_and_save_graphs(sims, args.graphs_output_file)
            print("Saving graphs...")
            print("Done.")
        except Exception as e:
            print("Failed to save graphs.")
            print(e)


if __name__ == '__main__':
    main()
