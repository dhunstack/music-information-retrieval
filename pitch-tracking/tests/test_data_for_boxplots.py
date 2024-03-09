from collections import OrderedDict
from utils import prepare_boxplot_data


def test_prepare_boxplot_data():
    # Sample pitch_scores input
    pitch_scores = {
        'Track_1': OrderedDict([
            ('Voicing Recall', 0.99),
            ('Voicing False Alarm', 0.45),
            ('Raw Pitch Accuracy', 0.88),
            ('Raw Chroma Accuracy', 0.90),
            ('Overall Accuracy', 0.75)
        ]),
        'Track_2': OrderedDict([
            ('Voicing Recall', 0.98),
            ('Voicing False Alarm', 0.50),
            ('Raw Pitch Accuracy', 0.85),
            ('Raw Chroma Accuracy', 0.93),
            ('Overall Accuracy', 0.80)
        ]),
    }

    # Expected output
    expected_data_dict = {
        'Voicing Recall': [0.99, 0.98],
        'Voicing False Alarm': [0.45, 0.50],
        'Raw Pitch Accuracy': [0.88, 0.85],
        'Raw Chroma Accuracy': [0.90, 0.93],
        'Overall Accuracy': [0.75, 0.80]
    }

    # Call the function
    data_dict = prepare_boxplot_data(pitch_scores)

    # Check if the returned data_dict matches the expected output
    for key in expected_data_dict:
        assert data_dict[key] == expected_data_dict[key], f"Failed on metric: {key}"

