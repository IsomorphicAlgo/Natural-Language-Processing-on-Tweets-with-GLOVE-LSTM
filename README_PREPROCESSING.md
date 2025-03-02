# Preprocessing and Loading Data for Disaster Tweets Analysis

This README explains how to use the preprocessing and loading scripts to save time when working with the disaster tweets dataset.

## Overview

The preprocessing steps for the disaster tweets dataset are time-consuming, especially the spell checking. To avoid repeating these steps every time you run your notebook, you can:

1. Run the preprocessing once and save the results to CSV files
2. Load the preprocessed data in your notebook instead of running the preprocessing again

## Files

- `preprocess_and_save.py`: Script to preprocess the raw data and save it to CSV files
- `load_preprocessed_data.py`: Script to load the preprocessed data from CSV files
- `processed_data/`: Directory where the preprocessed data will be saved

## How to Use

### Step 1: Preprocess the Data (Run Once)

Run the preprocessing script to generate the preprocessed data files:

```bash
python preprocess_and_save.py
```

This will:
- Load the raw data from `train.csv` and `test.csv`
- Apply all preprocessing steps (emoji removal, text cleaning, spell checking, etc.)
- Save the preprocessed data to `processed_data/preprocessed_train.csv` and `processed_data/preprocessed_test.csv`

**Note**: This process may take a while, especially the spell checking step.

### Step 2: Load the Preprocessed Data in Your Notebook

#### Option 1: Use the loading script

```python
from load_preprocessed_data import load_preprocessed_data

# Load the preprocessed data
train_data_frame, test_data_frame = load_preprocessed_data()

# Check if the data was loaded successfully
if train_data_frame is not None and test_data_frame is not None:
    # Continue with your analysis
    print(train_data_frame.head())
else:
    # Handle the case where the data couldn't be loaded
    print("Failed to load preprocessed data")
```

#### Option 2: Load directly in your notebook

```python
import pandas as pd
import os

# Check if preprocessed files exist
train_path = 'processed_data/preprocessed_train.csv'
test_path = 'processed_data/preprocessed_test.csv'

if os.path.exists(train_path) and os.path.exists(test_path):
    # Load preprocessed data
    train_data_frame = pd.read_csv(train_path)
    test_data_frame = pd.read_csv(test_path)
    print("Preprocessed data loaded successfully!")
else:
    # Run your original preprocessing code here
    print("Preprocessed data not found. Running preprocessing steps...")
    # ... your preprocessing code ...
```

## Benefits

- **Time Saving**: Avoid running the time-consuming preprocessing steps every time
- **Consistency**: Ensure the same preprocessed data is used across different runs
- **Flexibility**: Easily modify the preprocessing steps and regenerate the data if needed

## Troubleshooting

If you encounter any issues:

1. Make sure the raw data files (`train.csv` and `test.csv`) are in the correct location
2. Check that the `processed_data` directory exists and is writable
3. Ensure all required packages are installed (`pandas`, `nltk`, `spellchecker`, etc.)
4. If the preprocessing is taking too long, you can modify the `preprocess_and_save.py` script to skip certain steps (e.g., spell checking) 