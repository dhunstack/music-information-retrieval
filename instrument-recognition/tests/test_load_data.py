from mirdata.core import Dataset
from utils import load_data

import unittest
from unittest.mock import patch, Mock

@patch('mirdata.initialize')
def test_load_data(mock_initialize):
    # Setup the mock behavior
    mock_dataset = Mock(spec=Dataset)
    mock_initialize.return_value = mock_dataset
    
    # Call the load_data function
    data_home = '/tests/data/mini_medley_solos_db'
    returned_dataset = load_data(data_home)
    
    # Check if the `initialize` method was called with the correct arguments
    mock_initialize.assert_called_once_with('medley_solos_db', data_home=data_home)
    
    # Assert the returned dataset
    assert returned_dataset == mock_dataset, 'Returning the wrong type, should be a mirdata Dataset.'

test_load_data()

