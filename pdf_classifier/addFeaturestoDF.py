import pandas as pd
import dataSplit as ds
import extractFeatures as ef
from typing import Tuple
import numpy as np
from PIL import Image
import pymupdf  # PyMuPDF
import extractFeatures

def addFeaturestoDF():
    train_data, test_data = ds.dataSplit()

    def process_df(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        rgb_features = []

        for index, row in df.iterrows():
            file_path = row['file_path']
            rgb = extractFeatures.extractRGBFeatures(file_path)
            rgb_features.append(rgb)

        features_df = pd.DataFrame(rgb_features, columns=['avg_red', 'avg_green', 'avg_blue'])

        result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

        print(f"{data_type} data with features:")
        print(result_df.head())
        print(result_df.info())

        return result_df

    train_data_with_features = process_df(train_data, "Training")
    test_data_with_features = process_df(test_data, "Test")

    train_data_with_features.to_csv("traindata.csv")
    test_data_with_features.to_csv("testdata.csv")
    return train_data_with_features, test_data_with_features

