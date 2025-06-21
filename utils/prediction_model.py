import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


class VehicleStateClassifier:
    """
    Vehicle State Classifier
    Predicts vehicle states (idling, on-the-move, stopping) based on GPS data
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['deviceSpeed', 'orientation', 'latitude', 'longitude', 'ignition', 'hour', 'is_weekend']
        self.target_column = 'state'
        self.is_fitted = False
    
    def prepare_features(self, df):
        """
        Prepare feature data
        
        Args:
            df: DataFrame containing GPS data
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        # Ensure ignition is integer type
        if 'ignition' in df.columns:
            df['ignition'] = df['ignition'].astype(int)
        
        # Process time features
        if 'gpsTime' in df.columns:
            df['gpsTime'] = pd.to_datetime(df['gpsTime'])
            df['hour'] = df['gpsTime'].dt.hour
            df['day_of_week'] = df['gpsTime'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def build_model(self, input_shape, num_classes):
        """
        Build neural network model
        
        Args:
            input_shape: Input feature dimension
            num_classes: Number of classes
            
        Returns:
            Compiled model
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, df, epochs=20, batch_size=32, test_size=0.2, random_state=42, verbose=1):
        """
        Train the model
        
        Args:
            df: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size
            test_size: Test set ratio
            random_state: Random seed
            verbose: Training verbosity level
            
        Returns:
            Training history
        """
        # Prepare features
        df = self.prepare_features(df)
        
        # Check if required columns exist
        missing_cols = [col for col in self.feature_columns + [self.target_column] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Encode target variable
        df['state_encoded'] = self.label_encoder.fit_transform(df[self.target_column])
        
        # Prepare features and target
        X = df[self.feature_columns].values
        y = df['state_encoded'].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Build model
        num_classes = len(np.unique(y))
        self.model = self.build_model(X_train.shape[1], num_classes)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(X_test, y_test)
        )
        
        # Save test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        self.is_fitted = True
        
        return history
    
    def predict(self, df):
        """
        Predict vehicle states
        
        Args:
            df: DataFrame with data to predict
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Extract features
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred_probs = self.model.predict(X_scaled, verbose=0)
        y_pred_encoded = np.argmax(y_pred_probs, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_probs,
            'encoded_predictions': y_pred_encoded
        }
    
    def evaluate(self, plot_confusion_matrix=True, return_metrics=False):
        """
        Evaluate model performance
        
        Args:
            plot_confusion_matrix: Whether to plot confusion matrix
            return_metrics: Whether to return evaluation metrics
            
        Returns:
            Evaluation metrics dictionary if return_metrics=True
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predict test set
        y_pred_probs = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Classification report
        labels = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=labels, output_dict=True)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=labels))
        
        # Plot confusion matrix
        if plot_confusion_matrix:
            cm = confusion_matrix(self.y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()
        
        if return_metrics:
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
    
    def save_model(self, model_path="state_classifier.keras", 
                   scaler_path="features_scaler.pkl", 
                   encoder_path="labels_encoder.pkl"):
        """
        Save model and preprocessors
        
        Args:
            model_path: Model save path
            scaler_path: Scaler save path
            encoder_path: Label encoder save path
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Label encoder saved to: {encoder_path}")
    
    def load_model(self, model_path="state_classifier.keras", 
                   scaler_path="features_scaler.pkl", 
                   encoder_path="labels_encoder.pkl"):
        """
        Load saved model and preprocessors
        
        Args:
            model_path: Model file path
            scaler_path: Scaler file path
            encoder_path: Label encoder file path
        """
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_fitted = True
        
        print("Model and preprocessors loaded successfully!")
    
    def predict_single(self, speed, orientation, latitude, longitude, ignition, hour, is_weekend):
        """
        Predict single sample
        
        Args:
            speed: Device speed
            orientation: Orientation
            latitude: Latitude
            longitude: Longitude
            ignition: Ignition status (0 or 1)
            hour: Hour (0-23)
            is_weekend: Weekend flag (0 or 1)
            
        Returns:
            Prediction result dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create DataFrame for single sample
        sample_data = pd.DataFrame({
            'deviceSpeed': [speed],
            'orientation': [orientation],
            'latitude': [latitude],
            'longitude': [longitude],
            'ignition': [ignition],
            'hour': [hour],
            'is_weekend': [is_weekend]
        })
        
        return self.predict(sample_data)


# Example usage functions
def train_classifier_example():
    """
    Example function for training classifier
    """
    # Load data
    df = pd.read_csv("trucks_with_new_states_Mumbai.csv")
    
    # Create classifier instance
    classifier = VehicleStateClassifier()
    
    # Train model
    print("Training model...")
    history = classifier.fit(df, epochs=20, batch_size=32, verbose=1)
    
    # Evaluate model
    print("\nEvaluating model...")
    classifier.evaluate()
    
    # Save model
    classifier.save_model()
    
    return classifier


def load_and_predict_example():
    """
    Example function for loading model and making predictions
    """
    # Create classifier instance
    classifier = VehicleStateClassifier()
    
    # Load model
    classifier.load_model()
    
    # Single sample prediction example
    result = classifier.predict_single(
        speed=25.5,
        orientation=180,
        latitude=19.0760,
        longitude=72.8777,
        ignition=1,
        hour=14,
        is_weekend=0
    )
    
    print("Single prediction result:")
    print(f"Predicted state: {result['predictions'][0]}")
    print(f"Probabilities: {result['probabilities'][0]}")
    
    return classifier


def main():
    """
    Main function - Vehicle State Classifier
    1. Train and evaluate model
    2. Load model and test new data
    """
    
    print("=== Vehicle State Classifier ===")
    print("1. Train and Evaluate Model")
    print("2. Load Model and Test New Data")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nPlease select an option (1-3): ").strip()
            
            if choice == "1":
                # Train and evaluate model
                print("\n--- Training and Evaluation ---")
                data_file = input("Enter training CSV file path (default: trucks_with_new_states_Mumbai.csv): ").strip()
                if not data_file:
                    data_file = "trucks_with_new_states_Mumbai.csv"
                
                if not os.path.exists(data_file):
                    print(f"Error: File {data_file} not found!")
                    continue
                
                try:
                    # Load data
                    df = pd.read_csv(data_file)
                    print(f"Loaded training data with {len(df)} rows")
                    
                    # Display basic data information
                    print(f"Data shape: {df.shape}")
                    if 'state' in df.columns:
                        print("State distribution:")
                        print(df['state'].value_counts())
                    
                    # Create classifier
                    classifier = VehicleStateClassifier()
                    
                    # Get training parameters
                    epochs = int(input("Enter number of epochs (default: 20): ") or "20")
                    batch_size = int(input("Enter batch size (default: 32): ") or "32")
                    
                    print(f"\nStarting training with {epochs} epochs, batch size {batch_size}...")
                    history = classifier.fit(df, epochs=epochs, batch_size=batch_size, verbose=1)
                    
                    print("\n--- Model Evaluation ---")
                    classifier.evaluate(plot_confusion_matrix=True)
                    
                    # Save model
                    save_choice = input("\nSave trained model? (y/n, default: y): ").strip().lower()
                    if save_choice != 'n':
                        classifier.save_model()
                        print("Model saved successfully!")
                    
                    print("Training and evaluation completed!")
                    
                except Exception as e:
                    print(f"Error during training/evaluation: {e}")
            
            elif choice == "2":
                # Load model and test new data
                print("\n--- Load Model and Test New Data ---")
                
                try:
                    # Check if model files exist
                    model_files = ["state_classifier.keras", "features_scaler.pkl", "labels_encoder.pkl"]
                    if not all(os.path.exists(f) for f in model_files):
                        print("Error: Model files not found! Please train a model first.")
                        print("Missing files:", [f for f in model_files if not os.path.exists(f)])
                        continue
                    
                    # Load model
                    classifier = VehicleStateClassifier()
                    classifier.load_model()
                    
                    # Select test method
                    print("\nSelect test method:")
                    print("1. Test with CSV file (batch prediction)")
                    print("2. Test single sample (manual input)")
                    print("3. Test with preset samples")
                    
                    test_choice = input("Enter choice (1-3): ").strip()
                    
                    if test_choice == "1":
                        # CSV file batch prediction
                        test_file = input("Enter test CSV file path: ").strip()
                        if not os.path.exists(test_file):
                            print(f"Error: File {test_file} not found!")
                            continue
                        
                        print(f"Loading test data from {test_file}...")
                        test_df = pd.read_csv(test_file)
                        print(f"Test data shape: {test_df.shape}")
                        
                        # Make predictions
                        print("Making predictions...")
                        results = classifier.predict(test_df)
                        
                        # Add prediction results to DataFrame
                        test_df['predicted_state'] = results['predictions']
                        
                        # If true labels exist, calculate accuracy
                        if 'state' in test_df.columns:
                            accuracy = (test_df['state'] == test_df['predicted_state']).mean()
                            print(f"Test Accuracy: {accuracy:.4f}")
                            
                            print("\nActual vs Predicted distribution:")
                            print("Actual:")
                            print(test_df['state'].value_counts())
                            print("Predicted:")
                            print(test_df['predicted_state'].value_counts())
                        
                        # Save prediction results
                        output_file = test_file.replace('.csv', '_predictions.csv')
                        test_df.to_csv(output_file, index=False)
                        print(f"\nPredictions saved to: {output_file}")
                        
                        # Display sample results
                        print("\nSample predictions:")
                        display_cols = ['predicted_state'] + [col for col in classifier.feature_columns if col in test_df.columns]
                        if 'state' in test_df.columns:
                            display_cols = ['state'] + display_cols
                        print(test_df[display_cols].head(10))
                    
                    elif test_choice == "2":
                        # Manual input for single sample
                        print("\nEnter sample data:")
                        speed = float(input("Device Speed: "))
                        orientation = float(input("Orientation: "))
                        latitude = float(input("Latitude: "))
                        longitude = float(input("Longitude: "))
                        ignition = int(input("Ignition (0 or 1): "))
                        hour = int(input("Hour (0-23): "))
                        is_weekend = int(input("Is Weekend (0 or 1): "))
                        
                        result = classifier.predict_single(speed, orientation, latitude, longitude, 
                                                         ignition, hour, is_weekend)
                        
                        print(f"\n--- Prediction Result ---")
                        print(f"Predicted state: {result['predictions'][0]}")
                        print("\nClass probabilities:")
                        labels = classifier.label_encoder.classes_
                        for i, label in enumerate(labels):
                            print(f"  {label}: {result['probabilities'][0][i]:.4f}")
                    
                    elif test_choice == "3":
                        # Preset sample testing
                        print("\n--- Testing Preset Samples ---")
                        
                        test_samples = [
                            {"name": "Idling vehicle", "params": (0, 180, 19.0760, 72.8777, 1, 14, 0)},
                            {"name": "Moving vehicle", "params": (50, 180, 19.0760, 72.8777, 1, 14, 0)},
                            {"name": "Slow moving/Stopping", "params": (5, 180, 19.0760, 72.8777, 1, 14, 0)},
                            {"name": "Fast moving", "params": (80, 180, 19.0760, 72.8777, 1, 9, 0)},
                            {"name": "Weekend idling", "params": (0, 180, 19.0760, 72.8777, 1, 10, 1)}
                        ]
                        
                        for sample in test_samples:
                            result = classifier.predict_single(*sample["params"])
                            print(f"\n{sample['name']}:")
                            print(f"  Input: Speed={sample['params'][0]}, Ignition={sample['params'][4]}, Hour={sample['params'][5]}, Weekend={sample['params'][6]}")
                            print(f"  Predicted: {result['predictions'][0]} (confidence: {np.max(result['probabilities'][0]):.3f})")
                    
                    else:
                        print("Invalid choice!")
                        continue
                
                except Exception as e:
                    print(f"Error during testing: {e}")
            
            elif choice == "3":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice! Please enter 1, 2, or 3.")
        
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()