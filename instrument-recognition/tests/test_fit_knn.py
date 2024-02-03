import numpy as np
from unittest.mock import patch, Mock
import matplotlib.pyplot as plt
import utils

@patch('utils.KNeighborsClassifier')
@patch('utils.f1_score')
def test_fit_knn(mocked_f1_score, mocked_knn_classifier):
    # Setup mock behaviors
    mock_knn_instance = Mock()
    mock_knn_instance.predict.return_value = np.array([1, 0, 1, 1, 0])  # dummy prediction for example
    mocked_knn_classifier.return_value = mock_knn_instance

    # Mocked F1 scores for the different k-values
    mocked_f1_score.side_effect = [0.5, 0.6, 0.8, 0.7]

    # Dummy input data
    train_features = np.array([[i, i+1] for i in range(60)])  # 60 samples
    train_labels = np.array([i % 2 for i in range(60)])
    validation_features = np.array([[i, i+1] for i in range(5)])  # Keep this small for the example
    validation_labels = np.array([1, 0, 1, 1, 0])

    best_classifier, best_k = utils.fit_knn(train_features, train_labels, validation_features, validation_labels)

    assert best_k == 10, f"Expected best k to be 10, but got {best_k}"
    assert mocked_knn_classifier.call_count == 4, "KNN Classifier initialization was not called 4 times"
    assert mocked_f1_score.call_count == 4, "f1_score was not called 4 times"

    # Close the plot so it doesn't interfere with the test execution
    plt.close()



