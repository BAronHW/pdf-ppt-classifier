import addFeaturestoDF
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
from extractFeatures import extractRGBFeatures
import model
import pandas as pd

def classify_pdf(file_path, model, scaler, label_encoder):
    try:
        rgb_features = extractRGBFeatures(file_path)
        
        features_df: pd.DataFrame  = pd.DataFrame([rgb_features], columns=['avg_red', 'avg_green', 'avg_blue'])
        
        # print(features_df)

        scaled_features = scaler.transform(features_df)
        
        prediction = model.predict(scaled_features)
        
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        return predicted_label
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return "Error in classification"

if __name__ == "__main__":
    best_model, scaler, label_encoder = model.model()
    test_pdf_path = r'C:\Users\Aaron\OneDrive - Lancaster University\Documents\my projects\pdf-ppt-classifier\data\documents\6S9VBU9r7kT2ORCFfGI9HFV36zrD7E.pdf'
    result = classify_pdf(test_pdf_path, best_model, scaler, label_encoder)
    print(f"The document {test_pdf_path} is classified as: {result}")