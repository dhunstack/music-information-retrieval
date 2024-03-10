from unittest.mock import patch, Mock
from utils import evaluate_pitch
import numpy as np

def test_evaluate_pitch():
    # Sample data to mimic the track objects and their structure
    mock_data = {
        'track_1': Mock(
            audio_path="path/to/track1.wav",
            pitch=Mock(
                times=np.array([0.0, 0.01, 0.02]),
                frequencies=np.array([100.0, 200.0, 300.0])
            )
        ),
        'track_2': Mock(
            audio_path="path/to/track2.wav",
            pitch=Mock(
                times=np.array([0.0, 0.01, 0.02]),
                frequencies=np.array([150.0, 250.0, 350.0])
            )
        )
    }

    # Mock outputs for estimate_pitch 
    mock_time = np.array([0.0, 0.01, 0.02])
    mock_freq = np.array([100.0, 200.0, 300.0])
    mock_conf = np.array([0.1, 0.6, 0.2])
    mock_activation = np.array([[0.1, 0.7, 0.1], [0.2, 0.8, 0.2], [0.1, 0.9, 0.3]])
    mock_evaluation = {"score": 90.0}

    # Define expected frequency output after applying voicing threshold
    expected_freq = np.where(mock_activation.max(axis=1) < 0.5, 0, mock_freq)

    # Define the side effects of calling the estimate_pitch function for each track
    side_effect = [
        (mock_time, expected_freq, mock_conf, mock_activation),
        (mock_time, expected_freq, mock_conf, mock_activation)
    ]

    with patch("utils.estimate_pitch", side_effect=side_effect) as mock_estimate, \
         patch("utils.mir_eval.melody.evaluate", return_value=mock_evaluation) as mock_evaluate:

        results = evaluate_pitch(mock_data, voicing_threshold=0.5, use_viterbi=True)
        
        # Check if mock functions were called
        assert mock_estimate.call_count == len(mock_data), "estimate_pitch was not called the expected number of times"
        
        # Check if the evaluation function was called
        assert mock_evaluate.called, "mir_eval.melody.evaluate was not called"

        # Check if results are as expected
        assert len(results) == len(mock_data), "Mismatch in the number of evaluation results"
        for key in results:
            assert results[key]['score'] == mock_evaluation['score'], f"Mismatch in evaluation score for {key}"
