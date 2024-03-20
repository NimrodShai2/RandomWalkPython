import json
from typing import Dict, List
from matplotlib.backends.backend_pdf import PdfPages


def load_json(filename: str) -> Dict:
    """
    Load json file.
    :param filename: File name to load.
    :return: Loaded json file.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def write_to_file(filename: str, data: List[str]) -> None:
    """
    Write data to file.
    :param filename:
    :param data:
    """
    with open(filename, 'w') as f:
        f.writelines(data)


def save_graphs(filename: str, plots) -> None:
    """
    Save graphs to pdf file.
    :param filename: File name to save to.
    :param plots: List of plots to save.
    :return: None.
    """
    with PdfPages(filename) as pdf:
        for plot in plots:
            pdf.savefig(plot.figure)
