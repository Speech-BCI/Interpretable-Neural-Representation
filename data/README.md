# README

## Introduction

This data provides a small sample of the original dataset. The sample dataset is provided in the shape of `(3, 127, 64, 512)`, which corresponds to the dimensions of `(trials, channels, frequency, time)`. To simulate the full dataset for testing purposes, this sample can be replicated to form a larger dataset with the desired shape of `(3000, 127, 64, 512)`.

## How to Use

### Data Loading
```python
import h5py

data_dir = 'data_path'

with h5py.File(data_dir, 'r') as file:
    x_data = file['x_data'][:]
    y_data = file['y_data'][:]
```

### Testing Purposes Only

To replicate the sample data and generate a dataset of the same size as the original dataset, you can use the following Python script:
```python
import numpy as np

num_trials = 3000
unique_classes = 88
testing_x_data = np.tile(x_data, (num_trials // x_data.shape[0], 1, 1, 1))
testing_y_data = np.random.randint(0, unique_classes, num_trials)
```

The sample data is provided for testing and debugging purposes, and the full dataset will be made available to all participants through OSF. This replicated dataset is intended for testing purposes only. The sample is provided for debugging, prototyping, and initial testing of code, but it is not a substitute for the full dataset.
