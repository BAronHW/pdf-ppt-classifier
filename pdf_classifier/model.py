import addFeaturestoDF
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

def model():

    train_data_with_features, test_data_with_features = addFeaturestoDF.addFeaturestoDF()

    X_train = train_data_with_features.drop(columns=['labels', 'file_path'])
    X_train.columns = ['avg_red', 'avg_green', 'avg_blue']
    y_train = train_data_with_features['labels']

    X_test = test_data_with_features.drop(columns=['labels', 'file_path'])
    X_test.columns = ['avg_red', 'avg_green', 'avg_blue']
    y_test = test_data_with_features['labels']

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(random_state=42)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)

    return best_model, scaler, le