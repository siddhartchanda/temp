import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

def res(l):
    # Load the dataset
    df = pd.read_csv("Travel_Insurance_Predictor\TRAVEL.csv")

    # Data Preprocessing
    # Handle missing values (if any)
    df.dropna(inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = ['Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Gender', 'Destination']

    for col in categorical_columns:
        # Use handle_unknown='use_encoded_value' to handle unseen labels
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])

    # Split the data into features (X) and target variable (y)
    if 'Claim' in df.columns:
        df['Claim'] = label_encoder.fit_transform(df['Claim'])
        y = df['Claim']
        X = df.drop(columns=['Net Sales', 'Commision (in value)', 'Claim'])
    else:
        raise ValueError("The 'Claim' column is missing in the dataset.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a Sequential model
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Sample input data (you can modify this to match your requirements)
    sample_data = {
        'Agency': [l[0]],
        'Agency Type': [l[1]],
        'Distribution Channel': [l[2]],
        'Product Name': [l[3]],
        'Duration': [int(l[4])],  # Example duration in days
        'Destination': [l[5]],  # Example destination
        'Gender': [l[6]],  # You can specify gender if it's known, or use None
        'Age': [int(l[7])]  # Example age
    }

    # Handle previously unseen labels in other categorical columns in the sample input data
    for col in categorical_columns:
        valid_classes = set(label_encoder.classes_)
        sample_data[col] = [x if x in valid_classes else -1 for x in sample_data[col]]  # Use -1 for unseen labels

    # Define numerical columns
    numerical_columns = X.columns.tolist()

    # Create a DataFrame from the sample data
    sample_df = pd.DataFrame(sample_data)

    # Standardize numerical features in the sample data using the same scaler
    sample_df[numerical_columns] = scaler.transform(sample_df[numerical_columns])
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    # Make a prediction using the trained model
    prediction = model.predict(sample_df)
    #print(prediction[0][0])
    # Convert prediction to a binary claim decision (0 or 1)
    predicted_claim = 'Yes' if prediction[0][0] >= 0.00005 else 'No'
    #print(classification_report(y_test, y_pred,zero_division=1))
    # Print the prediction
    return "Predicted Claim: " + predicted_claim

# Your Flask app setup and routing code here...
