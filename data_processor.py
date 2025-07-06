import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_features = []
        self.categorical_features = []
        
    def process_data(self, data):
        """
        Process the raw housing data for ML training
        
        Args:
            data (pd.DataFrame): Raw housing data
            
        Returns:
            pd.DataFrame: Processed data ready for ML
        """
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Basic data validation
            required_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'location', 'year_built']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove rows with missing target variable
            df = df.dropna(subset=['price'])
            
            # Handle missing values in features
            numeric_columns = ['area', 'bedrooms', 'bathrooms', 'year_built']
            categorical_columns = ['location']
            
            # Impute numeric features with median
            numeric_imputer = SimpleImputer(strategy='median')
            df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
            
            # Impute categorical features with mode
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
            
            # Feature engineering
            df = self._engineer_features(df)
            
            # Remove outliers
            df = self._remove_outliers(df)
            
            # Encode categorical variables
            df = self._encode_categorical_features(df)
            
            # Validate final data
            if len(df) == 0:
                raise ValueError("No data remaining after processing")
            
            return df
            
        except Exception as e:
            st.error(f"Error in data processing: {str(e)}")
            raise e
    
    def _engineer_features(self, df):
        """
        Create new features from existing ones
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        # Calculate house age
        current_year = pd.Timestamp.now().year
        df['house_age'] = current_year - df['year_built']
        
        # Calculate price per square foot (for analysis, not prediction)
        df['price_per_sqft'] = df['price'] / df['area']
        
        # Create bathroom categories
        df['bathroom_category'] = pd.cut(
            df['bathrooms'], 
            bins=[0, 1.5, 2.5, float('inf')], 
            labels=['1_bath', '2_bath', '3plus_bath']
        )
        
        # Create bedroom categories
        df['bedroom_category'] = pd.cut(
            df['bedrooms'], 
            bins=[0, 2, 3, 4, float('inf')], 
            labels=['1-2_bed', '3_bed', '4_bed', '5plus_bed']
        )
        
        # Area categories
        df['area_category'] = pd.cut(
            df['area'], 
            bins=[0, 1000, 1500, 2000, float('inf')], 
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        return df
    
    def _remove_outliers(self, df):
        """
        Remove outliers using IQR method
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe without outliers
        """
        # Remove outliers for price using IQR
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        
        # Remove outliers for area
        Q1_area = df['area'].quantile(0.25)
        Q3_area = df['area'].quantile(0.75)
        IQR_area = Q3_area - Q1_area
        
        lower_bound_area = Q1_area - 1.5 * IQR_area
        upper_bound_area = Q3_area + 1.5 * IQR_area
        
        df = df[(df['area'] >= lower_bound_area) & (df['area'] <= upper_bound_area)]
        
        return df
    
    def _encode_categorical_features(self, df):
        """
        Encode categorical features using label encoding
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        categorical_columns = ['location', 'bathroom_category', 'bedroom_category', 'area_category']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str)
                    
                    # Add unseen labels to the encoder
                    unseen_labels = set(df[col]) - set(le.classes_)
                    if unseen_labels:
                        le.classes_ = np.append(le.classes_, list(unseen_labels))
                    
                    df[col] = le.transform(df[col])
        
        return df
    
    def prepare_features_for_prediction(self, input_data):
        """
        Prepare user input for prediction
        
        Args:
            input_data (dict): User input data
            
        Returns:
            pd.DataFrame: Processed features ready for prediction
        """
        # Create dataframe from input
        df = pd.DataFrame([input_data])
        
        # Engineer features
        current_year = pd.Timestamp.now().year
        df['house_age'] = current_year - df['year_built']
        
        # Create categories
        df['bathroom_category'] = pd.cut(
            df['bathrooms'], 
            bins=[0, 1.5, 2.5, float('inf')], 
            labels=['1_bath', '2_bath', '3plus_bath']
        )
        
        df['bedroom_category'] = pd.cut(
            df['bedrooms'], 
            bins=[0, 2, 3, 4, float('inf')], 
            labels=['1-2_bed', '3_bed', '4_bed', '5plus_bed']
        )
        
        df['area_category'] = pd.cut(
            df['area'], 
            bins=[0, 1000, 1500, 2000, float('inf')], 
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        # Encode categorical features
        categorical_columns = ['location', 'bathroom_category', 'bedroom_category', 'area_category']
        
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str)
                
                # Handle unseen categories by assigning them to the most common class
                mask = df[col].isin(le.classes_)
                if not mask.all():
                    most_common_class = le.classes_[0]  # or use statistics to find most common
                    df.loc[~mask, col] = most_common_class
                
                df[col] = le.transform(df[col])
        
        return df
