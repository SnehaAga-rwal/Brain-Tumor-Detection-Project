import pickle
import os
import numpy as np
from collections import Counter

def inspect_data():
    processed_path = 'data/processed/'
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(processed_path, f'{split}_data.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                X, y, labels = data
                print(f'\n--- {split.capitalize()} Data ---')
                print(f'X shape: {X.shape}')
                print(f'y shape: {y.shape}')
                print(f'Class distribution: {Counter(np.argmax(y, axis=1))}')
                print(f'Labels: {labels}')
        else:
            print(f'File {file_path} does not exist.')

if __name__ == '__main__':
    inspect_data()
