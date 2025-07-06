# Home Price Estimator - India

## Overview

This is a Streamlit-based web application that estimates home prices across major Indian cities using machine learning. The application provides an intuitive interface for users to input property details and receive instant price estimates in Indian Rupees. It includes features for market analysis, property comparison, and trend visualization.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface for user interaction
- **Data Processing**: Custom data processor for feature engineering and data cleaning
- **Machine Learning**: Multiple ML models (Linear Regression, Decision Tree, Random Forest) for price prediction
- **Data Sources**: Real estate data scraping and synthetic data generation
- **Visualization**: Plotly for interactive charts and market insights

## Key Components

### 1. Main Application (`app.py`)
- Primary Streamlit interface
- User input handling for property details
- Price prediction display and visualization
- Market trends and insights dashboard
- Session state management for model persistence

### 2. Data Processor (`data_processor.py`)
- Feature engineering and data preprocessing
- Label encoding for categorical variables
- Data validation and imputation
- Standardization and scaling

### 3. Model Trainer (`model_trainer.py`)
- Multiple ML algorithm implementation
- Cross-validation and model evaluation
- Feature importance analysis
- Model persistence and loading

### 4. Predictor (`predictor.py`)
- Single and batch prediction capabilities
- Input validation and preprocessing
- Model selection and inference

### 5. Real Estate Scraper (`real_estate_scraper.py`)
- Web scraping from public real estate sources
- Market data extraction and processing
- Realistic data generation based on market trends

### 6. Utilities (`utils.py`)
- Data loading and saving functions
- Sample data generation
- Helper functions for data manipulation

## Data Flow

1. **Data Collection**: Real estate data is gathered through web scraping and synthetic generation
2. **Data Processing**: Raw data is cleaned, validated, and feature-engineered
3. **Model Training**: Multiple ML models are trained and evaluated
4. **User Input**: Property details are collected through the Streamlit interface
5. **Prediction**: Input data is processed and fed to the selected model
6. **Output**: Price estimate and market insights are displayed to the user

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Plotly**: Interactive data visualization

### Data Sources
- **Trafilatura**: Web content extraction
- **Requests**: HTTP requests for data scraping
- Real estate websites (realtor.com, zillow.com) for market data

### Supported Cities
The application covers 30+ major Indian cities including Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad, Jaipur, and others.

## Deployment Strategy

The application is fully configured for deployment on Render with comprehensive deployment documentation:

- **Platform**: Render (cloud hosting)
- **Runtime**: Python 3
- **Build Command**: `pip install -r render_requirements.txt`
- **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Plan**: Free tier (750 hours/month, sleeps after 15 minutes of inactivity)

### Deployment Files
- `render_requirements.txt`: Python dependencies
- `render.yaml`: Render service configuration
- `Procfile`: Process configuration for deployment
- `DEPLOYMENT_GUIDE.md`: Complete step-by-step deployment instructions
- `render_deployment_checklist.txt`: Quick reference checklist
- GitHub integration for continuous deployment

### Enhanced Features (July 2025)
- **Builders Database**: Real builders and developers for major Indian cities with ratings, experience, and project portfolios
- **Property Overview**: Detailed specifications, amenities, investment insights, and location benefits
- **Virtual Tours**: Interactive room-by-room exploration with 360° view simulation and site visit scheduling
- **Enhanced UI**: Three-tab interface (Builders, Property Overview, Virtual Tour) after price estimation

## Changelog

```
Changelog:
- July 01, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```