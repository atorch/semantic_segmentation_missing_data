FROM tensorflow/tensorflow:2.9.3-gpu

WORKDIR /home/semantic_segmentation_missing_data

ADD requirements.txt .

RUN pip install -r requirements.txt
