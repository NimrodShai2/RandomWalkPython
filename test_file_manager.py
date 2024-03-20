import os

import pytest
from unittest.mock import mock_open, patch

from matplotlib import pyplot as plt

from file_manager import load_json, write_to_file, save_graphs


def test_load_json():
    # Mock the json.load function to return a specific dictionary
    mock_data = {"key": "value"}
    with patch('json.load', return_value=mock_data):
        with patch('builtins.open', new_callable=mock_open):
            result = load_json('dummy_file.json')
    assert result == mock_data


def test_load_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_json('non_existent_file.json')


def test_write_to_file():
    mock_data = ['test data']
    m = mock_open()
    with patch('builtins.open', m):
        write_to_file('dummy_file.txt', mock_data)
    handle = m()
    handle.writelines.assert_called_once_with(mock_data)


def test_save_graphs():
    # Create a list of real Figure instances
    figures = [plt.figure() for _ in range(5)]

    # Call the save_graphs function with a unique filename and the list of Figure instances
    filename = 'test_file.pdf'
    save_graphs(filename, figures)

    # Check if a file with the specified name exists
    assert os.path.exists(filename)

    # Check if the file is not empty
    assert os.path.getsize(filename) > 0

    # Clean up the created file
    os.remove(filename)
