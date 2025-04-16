# Road Detection from Aerial Images

This project implements a deep learning-based system for road detection in aerial/satellite imagery using semantic segmentation techniques.

## Project Overview

The goal is to build a binary semantic segmentation model that can accurately identify road networks in high-resolution aerial imagery. This implementation uses the DeepGlobe Road Extraction Challenge dataset.

## Features

- Data preprocessing and augmentation pipeline
- Training with combined BCE and Dice loss
- Evaluation using IoU and Dice coefficient metrics

## Project Structure

```
road_detection/
├── notebooks/
│   └── road_detection.ipynb     # Main workflow notebook
├── src/
│   ├── __init__.py
│   ├── data.py                  # Dataset, DataLoader, and augmentation
│   ├── losses.py                # Loss functions and metrics
│   ├── training.py              # Training loop and validation
│   ├── visualization.py         # Visualization functions
│   └── utils.py                 # Helper functions and utilities
└── requirements.txt
```

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the DeepGlobe Road Extraction dataset from [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/data)
4. Update the `DATA_DIR` in the notebook to point to your dataset location

## Usage

The main workflow is implemented in the Jupyter notebook `road_detection.ipynb`. The notebook is organized into sections:

1. Environment setup
2. Data loading and preparation
3. Data visualization and augmentation
4. Model initialization
5. Training
6. Evaluation

## Models

### DeepLabV3+

DeepLabV3+ uses atrous spatial pyramid pooling (ASPP) to capture multi-scale context and a decoder module to refine segmentation boundaries.

## Training

The training process includes:
- Combined BCE and Dice loss
- Learning rate scheduling
- Early stopping
- Model checkpointing

## Results

The model achieves a Dice coefficient of approximately 0.7338 and an IoU score of 0.5884 on the validation set.
