# semantic_segmentation_missing_data

Example of semantic segmentation with missing input data

```bash
sudo docker build ~/semantic_segmentation_missing_data --tag=ssmd
sudo docker run --gpus all -it -v ~/semantic_segmentation_missing_data:/home/semantic_segmentation_missing_data ssmd bash
python src/fit.py
```
