# Exploration of MIR Problems

In this repository, I have compiled some common MIR problems that I've worked on. The list below mentions all the tasks dealt with in this repository.

## Instrument Recognition

The dataset contains sounds from 8 different sources - clarinet, electric guitar, female singer, flute, piano, tenor saxophone, trumpet and violin.

I compute time domain, frequency domain and cepstral domain features for the given audio files. Used KNN for the classification problem to obtain an F1 score of 0.494.

Finally used XGBoost with sample weighting adjusted for unbalanced classes, to obtain an F1 score of 0.668.

## Beat Tracking

We work with the GTZAN-genre mini dataset containing beat annotations for 100 tracks.

We predict the beats using classical signal processing approach, and compare it against the RNN implementation of `madmom` library.

We further calculate `F1-score`, `cemgil` and `continuity` metrics to evaluate beat detections and analyse the common errors.
