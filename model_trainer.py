import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import streamlit as st

class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42, dt_params=None, rf_params=None):
        """
        Initialize the model trainer
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            dt_params (dict): Parameters for Decision Tree
            rf_params (dict): Parameters for Random Forest
        """
        self.test_size = test_size
        self.random_state = random_state
        self.dt_params = dt_params or {'max_depth': 10, 'random_state': random_state}
        self.rf_params = rf_params or {'n_estimators': 100, 'max_depth': 10, 'random_state': random_state}
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, data):
        """
        Prepare data for training by splitting features and target
        
        Args:
            data (pd.DataFrame): Processed housing data
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            # Define features to use for training (excluding price and derived features)
            feature_columns = [
                'area', 'bedrooms', 'bathrooms', 'year_built', 'house_age',
                'location', 'bathroom_category', 'bedroom_category', 'area_category'
            ]
            
            # Filter to only include available columns
            available_features = [col for col in feature_columns if col in data.columns]
            self.feature_columns = available_features
            
            # Prepare features and target
            X = data[self.feature_columns]
            y = data['price']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.X_train = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
            self.X_test = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
            self.y_train = y_train
            self.y_test = y_test
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            raise e
    
    def train_models(self, data, model_types=['linear_regression', 'decision_tree', 'random_forest']):
        """
        Train multiple ML models
        
        Args:
            data (pd.DataFrame): Processed housing data
            model_types (list): List of model types to train
            
        Returns:
            dict: Dictionary containing model results
        """
        try:
            # Prepare data
            self.prepare_data(data)
            
            results = {}
            
            # Initialize models
            models_config = {
                'linear_regression': {
                    'model': LinearRegression(),
                    'name': 'Linear Regression'
                },
                'decision_tree': {
                    'model': DecisionTreeRegressor(**self.dt_params),
                    'name': 'Decision Tree'
                },
                'random_forest': {
                    'model': RandomForestRegressor(**self.rf_params),
                    'name': 'Random Forest'
                }
            }
            
            # Train each selected model
            for model_type in model_types:
                if model_type in models_config:
                    st.write(f"Training {models_config[model_type]['name']}...")
                    
                    model = models_config[model_type]['model']
                    
                    # Train the model
                    model.fit(self.X_train, self.y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(self.X_train)
                    y_pred_test = model.predict(self.X_test)
                    
                    # Calculate metrics
                    train_r2 = r2_score(self.y_train, y_pred_train)
                    test_r2 = r2_score(self.y_test, y_pred_test)
                    test_mse = mean_squared_error(self.y_test, y_pred_test)
                    test_mae = mean_absolute_error(self.y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
                    cv_mean = cv_scores.mean()
                    
                    # Store model and results
                    self.models[model_type] = model
                    
                    results[model_type] = {
                        'name': models_config[model_type]['name'],
                        'model': model,
                        'train_r2': train_r2,
                        'r2_score': test_r2,
                        'mse': test_mse,
                        'mae': test_mae,
                        'cv_score': cv_mean,
                        'cv_scores': cv_scores,
                        'predictions': y_pred_test
                    }
                    
                    st.success(f"✅ {models_config[model_type]['name']} trained successfully!")
            
            return results
            
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            raise e
    
    def get_feature_importance(self, model_type):
        """
        Get feature importance for tree-based models
        
        Args:
            model_type (str): Type of model
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_type not in self.models:
            return None
        
        model = self.models[model_type]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def predict(self, X, model_type):
        """
        Make predictions using a trained model
        
        Args:
            X (pd.DataFrame): Features for prediction
            model_type (str): Type of model to use
            
        Returns:
            np.array: Predictions
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        # Scale features using the same scaler
        X_scaled = self.scaler.transform(X[self.feature_columns])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        return self.models[model_type].predict(X_scaled_df)
    
    def save_models(self, filepath):
        """
        Save trained models to file
        
        Args:
            filepath (str): Path to save the models
        """
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_models(self, filepath):
        """
        Load trained models from file
        
        Args:
            filepath (str): Path to load the models from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
