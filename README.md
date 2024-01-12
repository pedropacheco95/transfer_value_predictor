# Football Transfer Value Prediction Model (FNNAM)

## Overview
This repository contains the implementation of a Feedforward Neural Network with Attention Mechanism (FNNAM) designed to predict football player transfer values. The project utilizes various data preprocessing techniques, including handling missing data, reducing dimensionality, and managing outliers, before applying a sophisticated neural network model for prediction.

## Repository Structure
- `main.py`: The main script to run the model.
- `treat_df.py`: Contains functions for data cleaning and preprocessing.
- `dimensionality_reduction.py`: Functions for reducing the dimensionality of the dataset, including outlier handling and feature selection.
- `model.py`: The FNNAM model implementation including training, evaluation, and prediction functionalities.
- `images/`: Generated plots and visualizations are saved here.

## Installation
To set up the project, you will need Python installed on your system. Then, install the required libraries using:

```bash
pip install -r requirements.txt
```

## Usage

To use this model, follow these steps:

Place your dataset in the data/ directory.
Run main.py to start the data preprocessing and model training process.
The trained model and various plots will be saved in the respective directories.
Components

### Data Preprocessing
`main.py` initiates the data preprocessing steps:

* Reading and sorting data.
* Cleaning the dataset using `treat_df.py`.
* Dimensionality reduction using `dimensionality_reduction.py`.

### Model Training and Prediction

The `model.py` file defines the FNNAM class responsible for:

* Data preparation and normalization.
* Neural network model creation with attention layers.
* Training the model on preprocessed data.
* Evaluating the model's performance.
* Saving the trained model and making predictions.

### Visualization and Evaluation

The model's training progress and evaluation can be visualized through generated plots, which are saved in the images/ directory.