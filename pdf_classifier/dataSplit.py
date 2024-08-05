import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def dataSplit(pdf_path: str, ppt_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

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

    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['labels'], random_state=42)

    return train_data, test_data