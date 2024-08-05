import pandas as pd
import dataSplit as ds
import extractFeatures as ef
from typing import Tuple
import numpy as np
from PIL import Image
import extractFeatures
import os

def addFeaturestoDF():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    documents_path = os.path.join(project_dir, "data", "documents")
    powerpoints_path = os.path.join(project_dir, "data", "powerpoints")
    train_data, test_data = ds.dataSplit(documents_path, powerpoints_path)
    
    def process_df(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        rgb_features = []
        edge_features = []
        for index, row in df.iterrows():
            file_path = row['file_path']
            rgb = extractFeatures.extractRGBFeatures(file_path)
            edges = extractFeatures.extractEdges(file_path)
            rgb_features.append(rgb)
            edge_features.append(edges)
        
        rgb_df = pd.DataFrame(rgb_features, columns=['avg_red', 'avg_green', 'avg_blue'])
        edge_df = pd.DataFrame(edge_features, columns=['edge_ratio', 'edge_intensity'])
        
        features_df = pd.concat([rgb_df, edge_df], axis=1)
        result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        # print(f"{data_type} data with features:")
        # print(result_df.head())
        # print(result_df.info())
        
        return result_df
    
    train_data_with_features = process_df(train_data, "Training")
    test_data_with_features = process_df(test_data, "Test")
    
    train_data_with_features.to_csv("traindata.csv", index=False)
    test_data_with_features.to_csv("testdata.csv", index=False)
    
    return train_data_with_features, test_data_with_features