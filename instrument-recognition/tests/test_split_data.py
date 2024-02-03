from unittest.mock import Mock
from utils import split_data

def test_split_data():
    # Create mock tracks
    mock_track_training = Mock()
    mock_track_training.subset = 'training'
    mock_track_validation = Mock()
    mock_track_validation.subset = 'validation'
    mock_track_test = Mock()
    mock_track_test.subset = 'test'
    
    # Create a fake tracks dictionary
    tracks = {
        'train_1': mock_track_training,
        'train_2': mock_track_training,
        'validate_1': mock_track_validation,
        'test_1': mock_track_test,
        'test_2': mock_track_test
    }
    
    # Call the function
    tracks_train, tracks_validate, tracks_test = split_data(tracks)
    
    # Assert the expected numbers
    assert len(tracks_train) == 2, "You're returning the wrong number of training tracks."
    assert len(tracks_validate) == 1, "You're returning the wrong number of validation tracks."
    assert len(tracks_test) == 2, "You're returning the wrong number of test tracks."