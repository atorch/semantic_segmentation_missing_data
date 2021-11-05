# semantic_segmentation_missing_data

Example of semantic segmentation with missing input data

```bash
sudo docker build ~/semantic_segmentation_missing_data --tag=ssmd
sudo docker run --gpus all -it -v ~/semantic_segmentation_missing_data:/home/semantic_segmentation_missing_data ssmd bash
python src/fit.py
```

The input X is distributed on [0, 4], but is sometimes missing (NA or unobserved),
which is encoded as -1.
Can the model learn to ignore missing Xs and base its prediction on nearby non-missing pixels?

Input X (including missing values encoded as -1 in purple):

![X](example_X_9.png)

True class labels:

![Y](example_Y_9.png)

Predictions:

![predictions](example_predictions_9.png)