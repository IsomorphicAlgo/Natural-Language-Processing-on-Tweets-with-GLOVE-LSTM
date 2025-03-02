"""
Example: Loading Preprocessed Disaster Tweets Data

This script demonstrates how to load the preprocessed disaster tweets data
instead of running the time-consuming preprocessing steps every time.
"""

import pandas as pd
import os
import numpy as np

# Option 1: Using the load_preprocessed_data module
print("Option 1: Using the load_preprocessed_data module")
print("-" * 50)

try:
    from load_preprocessed_data import load_preprocessed_data
    
    # Load the preprocessed data
    train_data_frame, test_data_frame = load_preprocessed_data()
    
    # Check if the data was loaded successfully
    if train_data_frame is not None and test_data_frame is not None:
        print("\nFirst 5 rows of training data:")
        print(train_data_frame.head())
        
        print("\nFirst 5 rows of test data:")
        print(test_data_frame.head())
    else:
        print("Failed to load preprocessed data. Please run 'python preprocess_and_save.py' first.")
except ImportError:
    print("Could not import load_preprocessed_data module. Make sure the file exists.")

print("\n\n")

# Option 2: Loading directly in the script
print("Option 2: Loading directly in the script")
print("-" * 50)

# Define paths to preprocessed data files
train_path = 'processed_data/preprocessed_train.csv'
test_path = 'processed_data/preprocessed_test.csv'

# Check if preprocessed files exist
if os.path.exists(train_path) and os.path.exists(test_path):
    # Load preprocessed data
    print(f"Loading preprocessed training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df)} preprocessed training samples.")
    
    print(f"Loading preprocessed test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} preprocessed test samples.")
    
    print("\nPreprocessed data loaded successfully!")
    
    # Display the first few rows
    print("\nFirst 5 rows of training data:")
    print(train_df.head())
    
    print("\nFirst 5 rows of test data:")
    print(test_df.head())
    
    # Continue with analysis
    print("\nContinuing with analysis...")
    train_tweets = train_df['clean_text'].values
    test_tweets = test_df['clean_text'].values
    train_target = train_df['target'].values if 'target' in train_df.columns else None
    
    print(f"Number of training samples: {len(train_tweets)}")
    print(f"Number of test samples: {len(test_tweets)}")
    
    if train_target is not None:
        print(f"Target distribution: {np.bincount(train_target)}")
        print(f"Percentage of disaster tweets: {np.mean(train_target) * 100:.2f}%")
else:
    print("Preprocessed data files not found.")
    print("You would run your original preprocessing code here.")
    print("After preprocessing, you can save the data using:")
    print("train_data_frame.to_csv('processed_data/preprocessed_train.csv', index=False)")
    print("test_data_frame.to_csv('processed_data/preprocessed_test.csv', index=False)") 