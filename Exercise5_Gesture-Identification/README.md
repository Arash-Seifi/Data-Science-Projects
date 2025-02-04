# Thumbs Up Detection using TensorFlow

## Overview

This project implements real-time detection of thumbs-up and thumbs-down gestures using the TensorFlow Object Detection API. The system captures video input from a webcam, processes the frames using a trained object detection model, and visualizes the detected gestures in real time.
![Thumsup](https://github.com/user-attachments/assets/b5da6bbc-1ed8-4b64-9665-fc340cf8bdfa)


## Features

- Real-time gesture detection for thumbs-up and thumbs-down.
- Uses TensorFlow 2 and the Object Detection API.
- Processes video frames and overlays detection results.
- Trained using a labeled dataset.
- Can be run locally or on Google Colab.

## Dataset and Labeling

The dataset consists of images labeled with two gesture classes:

- **Thumbs Up** (id: 1)
- **Thumbs Down** (id: 2)

The labeling was defined in a separate file using the following format:

protobuf
item {
    name: 'Thumsup'
    id: 1
}
item {
    name: 'Thumsdown'
    id: 2
}

## Model Training
The training process used the TensorFlow Object Detection API with a pre-configured pipeline. The training command was executed as follows:

python model_main_tf2.py --model_dir=CHECKPOINT_PATH --pipeline_config_path=PIPELINE_CONFIG --num_train_steps=2000

## Inference Pipeline
Loads the trained model checkpoint.
Captures frames from the webcam.
Preprocesses the input image and runs inference.
Extracts detection results and overlays them on the video frame.
Displays the annotated video feed in real time.
