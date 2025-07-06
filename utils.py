import pandas as pd
import numpy as np
import streamlit as st
import os

def load_data(filepath='sample_data.csv'):
    """
    Load housing data from CSV file
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded housing data
    """
    try:
        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            return data
        else:
            st.error(f"Data file not found: {filepath}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_sample_data(filename='sample_data.csv', n_samples=1000):
    """
    Create a sample housing dataset for demonstration
    
    Args:
        filename (str): Name of the file to save
        n_samples (int): Number of samples to generate
    """
    try:
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic housing data
        locations = ['Downtown', 'Suburb', 'Urban', 'Rural', 'Waterfront', 'Hills', 'Industrial', 'Airport']
        
        data = []
        for i in range(n_samples):
            # Base features
            bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
            bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], p=[0.15, 0.1, 0.3, 0.15, 0.2, 0.05, 0.05])
            location = np.random.choice(locations)
            year_built = np.random.randint(1950, 2024)
            
            # Area based on bedrooms with some noise
            base_area = 400 + bedrooms * 300 + np.random.normal(0, 200)
            area = max(300, int(base_area))  # Minimum 300 sqft
            
            # Price calculation with location multipliers
            location_multipliers = {
                'Downtown': 1.4,
                'Waterfront': 1.6,
                'Hills': 1.3,
                'Suburb': 1.1,
                'Urban': 1.0,
                'Rural': 0.8,
                'Industrial': 0.7,
                'Airport': 0.9
            }
            
            # Base price calculation
            base_price = (
                area * 150 +  # Base price per sqft
                bedrooms * 10000 +  # Bedroom premium
                bathrooms * 8000 +  # Bathroom premium
                max(0, year_built - 1980) * 500  # Newer home premium
            )
            
            # Apply location multiplier
            location_multiplier = location_multipliers.get(location, 1.0)
            price = base_price * location_multiplier
            
            # Add some realistic noise
            price *= np.random.normal(1.0, 0.15)  # 15% variance
            price = max(50000, int(price))  # Minimum $50k
            
            data.append({
                'price': price,
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': location,
                'year_built': year_built
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some missing values to make it realistic
        missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        missing_columns = np.random.choice(['area', 'year_built'], size=len(missing_indices))
        
        for idx, col in zip(missing_indices, missing_columns):
            df.loc[idx, col] = np.nan
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        st.success(f"Sample dataset created with {n_samples} records and saved as {filename}")
        
        return df
        
    except Exception as e:
        st.error(f"Error creating sample data: {str(e)}")
        return None

def format_currency(amount):
    """
    Format a number as currency
    
    Args:
        amount (float): Amount to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${amount:,.0f}"

def format_number(number):
    """
    Format a number with commas
    
    Args:
        number (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    return f"{number:,.0f}"

def calculate_price_metrics(data):
    """
    Calculate various price metrics for the dataset
    
    Args:
        data (pd.DataFrame): Housing data
        
    Returns:
        dict: Dictionary containing price metrics
    """
    try:
        metrics = {
            'mean_price': data['price'].mean(),
            'median_price': data['price'].median(),
            'min_price': data['price'].min(),
            'max_price': data['price'].max(),
            'std_price': data['price'].std(),
            'price_per_sqft': (data['price'] / data['area']).mean()
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating price metrics: {str(e)}")
        return {}

def validate_dataframe(df, required_columns):
    """
    Validate that a dataframe has required columns and basic data quality
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    error_messages = []
    
    # Check if DataFrame is empty
    if df is None or len(df) == 0:
        error_messages.append("DataFrame is empty")
        return False, error_messages
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_messages.append(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty columns
    for col in required_columns:
        if col in df.columns and df[col].isna().all():
            error_messages.append(f"Column '{col}' is completely empty")
    
    # Check data types
    numeric_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'year_built']
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                error_messages.append(f"Column '{col}' should be numeric")
    
    # Check for reasonable value ranges
    if 'price' in df.columns:
        if (df['price'] <= 0).any():
            error_messages.append("Price values should be positive")
    
    if 'area' in df.columns:
        if (df['area'] <= 0).any():
            error_messages.append("Area values should be positive")
    
    if 'year_built' in df.columns:
        current_year = pd.Timestamp.now().year
        if (df['year_built'] < 1800).any() or (df['year_built'] > current_year).any():
            error_messages.append(f"Year built should be between 1800 and {current_year}")
    
    return len(error_messages) == 0, error_messages

def export_predictions(predictions_df, filename='predictions.csv'):
    """
    Export predictions to CSV file
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions
        filename (str): Name of the output file
    """
    try:
        predictions_df.to_csv(filename, index=False)
        st.success(f"Predictions exported to {filename}")
    except Exception as e:
        st.error(f"Error exporting predictions: {str(e)}")

def get_model_summary(model_results):
    """
    Generate a summary of model performance
    
    Args:
        model_results (dict): Dictionary containing model results
        
    Returns:
        str: Formatted summary string
    """
    try:
        if not model_results:
            return "No model results available"
        
        summary = "Model Performance Summary:\n"
        summary += "=" * 40 + "\n"
        
        for model_name, results in model_results.items():
            summary += f"\n{results['name']}:\n"
            summary += f"  R² Score: {results['r2_score']:.4f}\n"
            summary += f"  MSE: {results['mse']:,.0f}\n"
            summary += f"  MAE: {results['mae']:,.0f}\n"
            summary += f"  CV Score: {results['cv_score']:.4f}\n"
        
        # Best model
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['r2_score'])
        summary += f"\nBest Model: {model_results[best_model]['name']}\n"
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"
