import mirdata
from utils import load_data

def test_load_data():
    # Dummy dataset
    data_home = 'tests/data/gtzan_genre'
    dataset_version = 'mini'
    dataset_name = 'gtzan_genre'

    # Call the load_data function
    dataset = load_data(dataset_name, data_home, dataset_version)

    # Assert that the returned object is an instance of mirdata.Dataset
    assert isinstance(dataset, mirdata.core.Dataset), "Returned object is not a mirdata.Dataset instance"
    
    # Assert that dataset contains tracks
    all_tracks = dataset.load_tracks()
    assert len(all_tracks) > 0, "Dataset has no tracks"
    
    # Assert that the dataset is indeed the 'gtzan_genre' version 'mini'
    assert dataset.name == 'gtzan_genre', "Loaded dataset is not 'gtzan_genre'"
    assert dataset.version == 'mini', "Loaded dataset is not 'mini' version"


    # Test that the specific track ID loads its audio correctly
    track_id = 'metal.00002'
    assert track_id in all_tracks, f"Track ID {track_id} not found in dataset"
    
    audio_data = dataset.track(track_id).audio
    assert audio_data, "No audio data returned for track ID"

    # Check that audio data is a tuple with audio signal and sampling rate
    assert isinstance(audio_data, tuple), "Returned audio data is not a tuple"
    assert len(audio_data) == 2, "Audio data tuple should have two elements: audio signal and sampling rate"
    assert isinstance(audio_data[1], int), "Sampling rate should be an integer"
