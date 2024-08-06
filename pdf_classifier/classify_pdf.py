import addFeaturestoDF
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from extractFeatures import extractRGBFeatures, extractEdges
import model

def classify_pdf(file_path, model, scaler, label_encoder) -> str:
    try:
        print("Now running on single document")
        rgb_features = extractRGBFeatures(file_path)
        edge_features = extractEdges(file_path)
        
        all_features = rgb_features + edge_features
        
        feature_names = ['avg_red', 'avg_green', 'avg_blue', 'edge_ratio', 'edge_intensity']
        features_df: pd.DataFrame = pd.DataFrame([all_features], columns=feature_names)
        
        scaled_features = scaler.transform(features_df)
        
        prediction = model.predict(scaled_features)
        
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        return predicted_label
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return "Error in classification"

if __name__ == "__main__":
    best_model, scaler, label_encoder = model.model()
    
    test_pdf_path = r'C:\Users\Aaron\OneDrive - Lancaster University\Documents\my projects\pdf-ppt-classifier\PharmTher 1.1-2024 (ET).pdf'
    result = classify_pdf(test_pdf_path, best_model, scaler, label_encoder)
    print(f"The document {test_pdf_path} is classified as: {result}")