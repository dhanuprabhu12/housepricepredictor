import pandas as pd
import numpy as np
import streamlit as st
from data_processor import DataProcessor

class Predictor:
    def __init__(self, model_trainer):
        """
        Initialize the predictor with a trained model trainer
        
        Args:
            model_trainer (ModelTrainer): Trained model trainer instance
        """
        self.trainer = model_trainer
        self.processor = DataProcessor()
        
    def predict(self, input_data, model_type):
        """
        Make a price prediction for a single house
        
        Args:
            input_data (dict): Dictionary containing house features
            model_type (str): Type of model to use for prediction
            
        Returns:
            float: Predicted house price
        """
        try:
            # Validate input data
            required_fields = ['area', 'bedrooms', 'bathrooms', 'location', 'year_built']
            missing_fields = [field for field in required_fields if field not in input_data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Prepare features for prediction
            df = self.processor.prepare_features_for_prediction(input_data)
            
            # Make prediction
            prediction = self.trainer.predict(df, model_type)
            
            return float(prediction[0])
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            raise e
    
    def predict_batch(self, input_df, model_type):
        """
        Make predictions for multiple houses
        
        Args:
            input_df (pd.DataFrame): DataFrame containing house features
            model_type (str): Type of model to use for prediction
            
        Returns:
            np.array: Array of predicted house prices
        """
        try:
            predictions = []
            
            for _, row in input_df.iterrows():
                input_data = row.to_dict()
                prediction = self.predict(input_data, model_type)
                predictions.append(prediction)
            
            return np.array(predictions)
            
        except Exception as e:
            st.error(f"Error making batch predictions: {str(e)}")
            raise e
    
    def get_prediction_confidence(self, input_data, model_types=None):
        """
        Get prediction confidence by comparing multiple models
        
        Args:
            input_data (dict): Dictionary containing house features
            model_types (list): List of model types to use (if None, use all available)
            
        Returns:
            dict: Dictionary containing predictions and confidence metrics
        """
        try:
            if model_types is None:
                model_types = list(self.trainer.models.keys())
            
            predictions = {}
            prediction_values = []
            
            for model_type in model_types:
                if model_type in self.trainer.models:
                    pred = self.predict(input_data, model_type)
                    predictions[model_type] = pred
                    prediction_values.append(pred)
            
            if not prediction_values:
                raise ValueError("No valid models available for prediction")
            
            # Calculate confidence metrics
            mean_prediction = np.mean(prediction_values)
            std_prediction = np.std(prediction_values)
            min_prediction = np.min(prediction_values)
            max_prediction = np.max(prediction_values)
            
            # Calculate confidence score (inverse of coefficient of variation)
            cv = std_prediction / mean_prediction if mean_prediction != 0 else float('inf')
            confidence_score = max(0, 1 - cv)  # Higher is better, max 1.0
            
            return {
                'predictions': predictions,
                'mean_prediction': mean_prediction,
                'std_prediction': std_prediction,
                'min_prediction': min_prediction,
                'max_prediction': max_prediction,
                'confidence_score': confidence_score,
                'prediction_range': max_prediction - min_prediction
            }
            
        except Exception as e:
            st.error(f"Error calculating prediction confidence: {str(e)}")
            raise e
    
    def explain_prediction(self, input_data, model_type='random_forest'):
        """
        Provide explanation for the prediction (feature importance)
        
        Args:
            input_data (dict): Dictionary containing house features
            model_type (str): Type of model to use for explanation
            
        Returns:
            dict: Dictionary containing prediction explanation
        """
        try:
            # Get prediction
            prediction = self.predict(input_data, model_type)
            
            # Get feature importance if available
            feature_importance = self.trainer.get_feature_importance(model_type)
            
            # Prepare input features
            df = self.processor.prepare_features_for_prediction(input_data)
            
            explanation = {
                'prediction': prediction,
                'input_features': input_data,
                'processed_features': df.iloc[0].to_dict(),
                'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else None
            }
            
            return explanation
            
        except Exception as e:
            st.error(f"Error explaining prediction: {str(e)}")
            raise e
    
    def validate_input(self, input_data):
        """
        Validate user input data
        
        Args:
            input_data (dict): Dictionary containing house features
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check required fields
            required_fields = ['area', 'bedrooms', 'bathrooms', 'location', 'year_built']
            missing_fields = [field for field in required_fields if field not in input_data]
            
            if missing_fields:
                return False, f"Missing required fields: {missing_fields}"
            
            # Validate data types and ranges
            if not isinstance(input_data['area'], (int, float)) or input_data['area'] <= 0:
                return False, "Area must be a positive number"
            
            if not isinstance(input_data['bedrooms'], (int, float)) or input_data['bedrooms'] < 0:
                return False, "Bedrooms must be a non-negative number"
            
            if not isinstance(input_data['bathrooms'], (int, float)) or input_data['bathrooms'] < 0:
                return False, "Bathrooms must be a non-negative number"
            
            if not isinstance(input_data['year_built'], (int, float)) or input_data['year_built'] < 1800:
                return False, "Year built must be a valid year (>= 1800)"
            
            current_year = pd.Timestamp.now().year
            if input_data['year_built'] > current_year:
                return False, f"Year built cannot be in the future (> {current_year})"
            
            if not isinstance(input_data['location'], str) or len(input_data['location'].strip()) == 0:
                return False, "Location must be a non-empty string"
            
            return True, "Input validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
