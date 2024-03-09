from unittest.mock import Mock, patch
from utils import estimate_pitch  
import numpy as np

def test_estimate_pitch():
    # Mocking the output of librosa.load and crepe.predict
    mock_audio = np.array([0.1, 0.2, 0.3])
    mock_fs = 22050
    mock_time = np.array([0.0, 0.01, 0.02])
    mock_frequency = np.array([100.0, 150.0, 200.0])
    mock_confidence = np.array([0.8, 0.6, 0.2])
    mock_activation = np.array([
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.1],
        [0.2, 0.3, 0.5]
    ])
    
    # Patching librosa.load and crepe.predict
    with patch("utils.librosa.load", return_value=(mock_audio, mock_fs)), \
         patch("utils.crepe.predict", return_value=(mock_time, mock_frequency, mock_confidence, mock_activation)) as mock_predict:
         
        time, frequency, confidence, activation = estimate_pitch("dummy_path.wav", voicing_threshold=0.5)
        assert np.array_equal(time, mock_time), "Mismatch in time array."
        assert np.array_equal(frequency, mock_frequency), "Mismatch in frequency array."
        assert np.array_equal(confidence, mock_confidence), "Mismatch in confidence array."
        assert np.array_equal(activation, mock_activation), "Mismatch in activation array."

        # Check if viterbi argument was passed to crepe.predict
        assert mock_predict.call_args[1]['viterbi'] == False, "Viterbi argument not passed correctly to crepe.predict"

        # Test with voicing_threshold=0.7 (expecting only the first frame to be voiced)
        time, frequency, confidence, activation = estimate_pitch("dummy_path.wav", voicing_threshold=0.7, use_viterbi=True)
        assert np.array_equal(frequency, np.array([100.0, 0.0, 0.0])), "Mismatch in frequency array for voicing_threshold=0.7"


