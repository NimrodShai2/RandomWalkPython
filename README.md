# Random Walk Simulation

## Introduction
This project simulates random walks in various dimensions and configurations. Each simulation can be executed multiple times, supporting walks in any number of dimensions, though graphical results are generated only for 2D and 3D walks. The simulation includes five types of walkers:
- **Regular**: Moves with a fixed step size in a random direction.
- **Step**: Can change the step size at each step.
- **Grid**: Restricted to moving on a grid.
- **Biased**: Has a directional bias.
- **Searcher**: Searches for a specific point and stops upon reaching it.

## Installation
Clone this repository to your local machine. Ensure you have Python 3.6 or later installed. Some external libraries are required to run the simulation, which can be installed using pip:
```bash
pip install -r requirements.txt
```
## Usage
Run the simulation from the command line by specifying a JSON configuration file, an output file for the results (`.txt`), and a file for the graph results (`.pdf`). 

```bash
python main.py --config_file your_config.json --output_file results.txt --graphs_output_file graph.pdf
```
### Configuration File
The configuration file is a JSON-formatted document that specifies the parameters for the simulation. It defines the type of walker, the number of simulations to run, the number of steps per simulation, and other walker-specific configurations. Below is a detailed breakdown of the configuration file format:

```json
{
  "WalkerName": {
    "walker_type": "Type of the walker. Possible values: 'regular', 'step', 'grid', 'biased', 'searcher'",
    "times_to_run": "Number of simulations to run. Must be a positive integer.",
    "num_of_steps": "Number of steps each simulation will perform. Must be a positive integer.",
    "walker": {
      "n_dim": "Dimension of the walk. Graphs can be produced for 2 or 3 dimensions. Any higher dimension will be simulated without graphical output.",
      "magic_gates_placements": "Optional. A list of coordinates where magic gates are placed. Walkers entering a gate will be teleported to a corresponding destination.",
      "magic_gates_dests": "Optional. A list of destination coordinates for the magic gates. Each destination corresponds to a placement in 'magic_gates_placements'.",
      "obstacles": "Optional. A list of coordinates representing obstacles. Walkers cannot step onto these points.",
      "restart_chance": "Optional. A probability (between 0 and 1) for the walker to restart at the origin after each step. Default is 0.",
      "restart_every": "Optional. Specifies after how many steps the walker should restart at the origin. Default is not to restart."
    }
  }
}
```
Here is an example:
```json
{
  "ExampleWalker": {
    "walker_type": "regular",
    "times_to_run": 100,
    "num_of_steps": 1000,
    "walker": {
      "n_dim": 2,
      "magic_gates_placements": [[0, 0], [1, 1]],
      "magic_gates_dests": [[0, 1], [1, 0]],
      "obstacles": [[0.5, 0.5]],
      "restart_chance": 0.1,
      "restart_every": 50
    }
  }
}
```

