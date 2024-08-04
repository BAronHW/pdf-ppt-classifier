import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

# this module creates the training and testing data split

def dataSplit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pdf_path = r"C:\Users\Aaron\OneDrive - Lancaster University\Documents\my projects\pdf-ppt-classifier\data\documents"
    ppt_path = r"C:\Users\Aaron\OneDrive - Lancaster University\Documents\my projects\pdf-ppt-classifier\data\powerpoints"

    files: list[str] = []
    labels: list[str] = []

    for pdf in os.listdir(pdf_path):
        files.append(os.path.join(pdf_path, pdf))
        labels.append("pdf")

    for ppt in os.listdir(ppt_path):
        files.append(os.path.join(ppt_path, ppt))
        labels.append("ppt")

    data = pd.DataFrame({
        'file_path': files,
        'labels': labels
    })

    train_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8, stratify=data['labels'], random_state=42)

    print("Training data:", train_data.head())
    print("Test data:", test_data.head())

    return train_data, test_data
