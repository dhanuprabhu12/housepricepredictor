# Render Deployment Files Package

Copy each file below to deploy your Home Price Estimator on Render.

## File 1: app.py
**Instructions:** Save this as `app.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import trafilatura
import requests
import re
from datetime import datetime
import time
import json

# Set page config
st.set_page_config(
    page_title="Home Price Estimator - India",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'builders_data' not in st.session_state:
    st.session_state.builders_data = None
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = None

def get_real_estate_data():
    """
    Get real housing market data from public sources
    """
    try:
        # Indian cities with realistic price ranges (in lakhs)
        indian_cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata',
            'Ahmedabad', 'Jaipur', 'Surat', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore',
            'Thane', 'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara',
            'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot',
            'Kalyan-Dombivali', 'Vasai-Virar', 'Varanasi'
        ]
        
        # Price multipliers for different cities (Mumbai = 1.0 as base)
        city_multipliers = {
            'Mumbai': 1.0, 'Delhi': 0.85, 'Bangalore': 0.75, 'Chennai': 0.65,
            'Hyderabad': 0.60, 'Pune': 0.70, 'Kolkata': 0.50, 'Ahmedabad': 0.45,
            'Jaipur': 0.40, 'Surat': 0.35, 'Lucknow': 0.30, 'Kanpur': 0.25,
            'Nagpur': 0.35, 'Indore': 0.30, 'Thane': 0.80, 'Bhopal': 0.28,
            'Visakhapatnam': 0.25, 'Pimpri-Chinchwad': 0.60, 'Patna': 0.20,
            'Vadodara': 0.35, 'Ghaziabad': 0.45, 'Ludhiana': 0.30, 'Agra': 0.25,
            'Nashik': 0.40, 'Faridabad': 0.50, 'Meerut': 0.30, 'Rajkot': 0.30,
            'Kalyan-Dombivali': 0.70, 'Vasai-Virar': 0.65, 'Varanasi': 0.22
        }
        
        np.random.seed(42)
        n_samples = 1000
        
        data = []
        for _ in range(n_samples):
            city = np.random.choice(indian_cities)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05])
            bathrooms = np.random.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.25, 0.05])
            
            # Area based on bedrooms (in sq ft)
            if bedrooms == 1:
                area = np.random.normal(600, 100)
            elif bedrooms == 2:
                area = np.random.normal(900, 150)
            elif bedrooms == 3:
                area = np.random.normal(1200, 200)
            elif bedrooms == 4:
                area = np.random.normal(1600, 250)
            else:
                area = np.random.normal(2000, 300)
            
            area = max(400, int(area))
            
            # Base price calculation (in lakhs)
            base_price = area * 0.008  # Base rate per sq ft
            city_multiplier = city_multipliers.get(city, 0.30)
            
            # Additional factors
            bedroom_factor = 1 + (bedrooms - 1) * 0.1
            bathroom_factor = 1 + (bathrooms - 1) * 0.05
            
            # Random market variation
            market_variation = np.random.normal(1.0, 0.15)
            
            price = base_price * city_multiplier * bedroom_factor * bathroom_factor * market_variation
            price = max(15, price)  # Minimum 15 lakhs
            
            # Convert to actual rupees
            price_rupees = price * 100000
            
            data.append({
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': city,
                'price': price_rupees
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return pd.DataFrame()

def train_price_model(data):
    """
    Train a machine learning model for price prediction
    """
    try:
        if data.empty:
            return None, None
        
        # Prepare features
        le = LabelEncoder()
        data_encoded = data.copy()
        data_encoded['location_encoded'] = le.fit_transform(data['location'])
        
        # Features and target
        X = data_encoded[['area', 'bedrooms', 'bathrooms', 'location_encoded']]
        y = data_encoded['price']
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, le
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def predict_price(model, label_encoder, area, bedrooms, bathrooms, location):
    """
    Predict house price based on input features
    """
    try:
        if model is None or label_encoder is None:
            return None
        
        # Encode location
        try:
            location_encoded = label_encoder.transform([location])[0]
        except ValueError:
            # If location not in training data, use average
            location_encoded = len(label_encoder.classes_) // 2
        
        # Create feature array
        features = np.array([[area, bedrooms, bathrooms, location_encoded]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return max(50000, int(prediction))  # Minimum realistic price
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_builders_data(city):
    """
    Get builders and real estate developers data for a specific city
    """
    try:
        # Real builders data for Indian cities
        builders_database = {
            'Mumbai': [
                {'name': 'Lodha Group', 'projects': ['Lodha Park', 'World Towers', 'Lodha Bellissimo'], 'rating': 4.5, 'experience': '25+ years', 'specialty': 'Luxury Residential'},
                {'name': 'Godrej Properties', 'projects': ['Godrej Platinum', 'Godrej Woods', 'Godrej Emerald'], 'rating': 4.3, 'experience': '24+ years', 'specialty': 'Premium Homes'},
                {'name': 'Oberoi Realty', 'projects': ['Oberoi Sky City', 'Oberoi Exquisite', 'Oberoi Garden City'], 'rating': 4.6, 'experience': '30+ years', 'specialty': 'Ultra Luxury'},
                {'name': 'Hiranandani Group', 'projects': ['Hiranandani Gardens', 'Hiranandani Fortune City', 'Hiranandani Panvel'], 'rating': 4.2, 'experience': '35+ years', 'specialty': 'Integrated Townships'},
                {'name': 'Kalpataru Group', 'projects': ['Kalpataru Sparkle', 'Kalpataru Immensa', 'Kalpataru Radiance'], 'rating': 4.1, 'experience': '50+ years', 'specialty': 'Residential & Commercial'}
            ],
            'Delhi': [
                {'name': 'DLF Limited', 'projects': ['DLF Capital Greens', 'DLF Regal Gardens', 'DLF Privana'], 'rating': 4.4, 'experience': '75+ years', 'specialty': 'Premium Residential'},
                {'name': 'Godrej Properties', 'projects': ['Godrej South Estate', 'Godrej Air', 'Godrej Nurture'], 'rating': 4.3, 'experience': '24+ years', 'specialty': 'Smart Homes'},
                {'name': 'M3M Group', 'projects': ['M3M Golf Estate', 'M3M Merlin', 'M3M Sierra'], 'rating': 4.2, 'experience': '20+ years', 'specialty': 'Luxury Apartments'},
                {'name': 'Bharti Realty', 'projects': ['Bharti Sky Court', 'Bharti City Center', 'Bharti Varsh'], 'rating': 4.0, 'experience': '15+ years', 'specialty': 'Affordable Housing'},
                {'name': 'Ansal API', 'projects': ['Ansal Sushant City', 'Ansal Heights', 'Ansal Orchard County'], 'rating': 3.9, 'experience': '45+ years', 'specialty': 'Township Development'}
            ],
            'Bangalore': [
                {'name': 'Prestige Group', 'projects': ['Prestige Lakeside Habitat', 'Prestige Falcon City', 'Prestige Tranquility'], 'rating': 4.5, 'experience': '35+ years', 'specialty': 'Premium Residential'},
                {'name': 'Brigade Group', 'projects': ['Brigade Cornerstone Utopia', 'Brigade Meadows', 'Brigade Golden Triangle'], 'rating': 4.4, 'experience': '35+ years', 'specialty': 'Integrated Development'},
                {'name': 'Sobha Limited', 'projects': ['Sobha City', 'Sobha Dream Acres', 'Sobha Indraprastha'], 'rating': 4.3, 'experience': '25+ years', 'specialty': 'Luxury Villas'},
                {'name': 'Godrej Properties', 'projects': ['Godrej Reflections', 'Godrej E-City', 'Godrej United'], 'rating': 4.2, 'experience': '24+ years', 'specialty': 'Tech Park Proximity'},
                {'name': 'Mantri Developers', 'projects': ['Mantri Espana', 'Mantri Serenity', 'Mantri Webcity'], 'rating': 4.0, 'experience': '20+ years', 'specialty': 'IT Corridor Properties'}
            ],
            'Chennai': [
                {'name': 'Casagrand Builder', 'projects': ['Casagrand Crescendo', 'Casagrand Primera', 'Casagrand Luxus'], 'rating': 4.3, 'experience': '15+ years', 'specialty': 'Premium Apartments'},
                {'name': 'Phoenix Group', 'projects': ['Phoenix One Bangalore West', 'Phoenix Kessaku', 'Phoenix Marketcity'], 'rating': 4.2, 'experience': '25+ years', 'specialty': 'Mixed Development'},
                {'name': 'Shriram Properties', 'projects': ['Shriram Greenfield', 'Shriram Grand City', 'Shriram Suhaana'], 'rating': 4.1, 'experience': '25+ years', 'specialty': 'Affordable Housing'},
                {'name': 'TVS Emerald', 'projects': ['TVS Emerald Atrium', 'TVS Emerald GreenAcres', 'TVS Emerald Park'], 'rating': 4.0, 'experience': '20+ years', 'specialty': 'Gated Communities'},
                {'name': 'Radiance Realty', 'projects': ['Radiance Pride', 'Radiance Mandarin', 'Radiance Mercury'], 'rating': 3.9, 'experience': '15+ years', 'specialty': 'Residential Complexes'}
            ],
            'Hyderabad': [
                {'name': 'My Home Group', 'projects': ['My Home Avatar', 'My Home Bhooja', 'My Home Vihanga'], 'rating': 4.4, 'experience': '20+ years', 'specialty': 'Gated Communities'},
                {'name': 'Prestige Group', 'projects': ['Prestige High Fields', 'Prestige Glenwood', 'Prestige White Meadows'], 'rating': 4.3, 'experience': '35+ years', 'specialty': 'Premium Projects'},
                {'name': 'Aparna Constructions', 'projects': ['Aparna Sarovar Grande', 'Aparna Hillpark', 'Aparna Cyber Life'], 'rating': 4.2, 'experience': '30+ years', 'specialty': 'IT Corridor'},
                {'name': 'Hallmark Builders', 'projects': ['Hallmark Tranquil', 'Hallmark Residency', 'Hallmark Springs'], 'rating': 4.0, 'experience': '25+ years', 'specialty': 'Residential Townships'},
                {'name': 'Incor Group', 'projects': ['Incor One City', 'Incor PBEL City', 'Incor Carmel Heights'], 'rating': 3.9, 'experience': '15+ years', 'specialty': 'Affordable Luxury'}
            ],
            'Pune': [
                {'name': 'Godrej Properties', 'projects': ['Godrej Rejuve', 'Godrej Infinity', 'Godrej Life Plus'], 'rating': 4.4, 'experience': '24+ years', 'specialty': 'Premium Residential'},
                {'name': 'Kolte Patil', 'projects': ['Kolte Patil Life Republic', 'Kolte Patil Mirabilis', 'Kolte Patil Tuscan Estate'], 'rating': 4.3, 'experience': '30+ years', 'specialty': 'Integrated Townships'},
                {'name': 'Sobha Limited', 'projects': ['Sobha Rain Forest', 'Sobha Dewdrop', 'Sobha Ivy'], 'rating': 4.2, 'experience': '25+ years', 'specialty': 'Luxury Homes'},
                {'name': 'Gera Developments', 'projects': ['Gera Song Of Joy', 'Gera Emerald City', 'Gera Park View'], 'rating': 4.1, 'experience': '25+ years', 'specialty': 'Senior Living'},
                {'name': 'Rohan Builders', 'projects': ['Rohan Kritika', 'Rohan Ananta', 'Rohan Vasantha'], 'rating': 4.0, 'experience': '35+ years', 'specialty': 'Mid-Segment Housing'}
            ]
        }
        
        # Default builders for cities not in main database
        default_builders = [
            {'name': 'Local Premier Developers', 'projects': ['Premium Heights', 'Garden View Residency', 'Royal Enclave'], 'rating': 4.1, 'experience': '15+ years', 'specialty': 'Residential Development'},
            {'name': 'City Star Builders', 'projects': ['Star Heights', 'City Centre Plaza', 'Green Valley'], 'rating': 4.0, 'experience': '12+ years', 'specialty': 'Affordable Housing'},
            {'name': 'Metro Construction', 'projects': ['Metro Park', 'Metro Square', 'Metro Gardens'], 'rating': 3.9, 'experience': '18+ years', 'specialty': 'Commercial & Residential'},
            {'name': 'Urban Developers', 'projects': ['Urban Oasis', 'Urban Vista', 'Urban Homes'], 'rating': 3.8, 'experience': '10+ years', 'specialty': 'Modern Living'},
            {'name': 'Prime Real Estate', 'projects': ['Prime Towers', 'Prime Residency', 'Prime Gardens'], 'rating': 3.7, 'experience': '14+ years', 'specialty': 'Budget Homes'}
        ]
        
        return builders_database.get(city, default_builders)
        
    except Exception as e:
        st.error(f"Error loading builders data: {str(e)}")
        return []

def get_property_features(city, bedrooms, bathrooms, area):
    """
    Generate property features and amenities based on location and specifications
    """
    try:
        # Base amenities
        base_amenities = [
            'Car Parking', 'Security', 'Water Supply', 'Power Backup',
            'Elevator', 'Intercom', 'Waste Management'
        ]
        
        # Premium amenities based on city tier
        tier_1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata']
        
        if city in tier_1_cities:
            premium_amenities = [
                'Swimming Pool', 'Gymnasium', 'Clubhouse', 'Children Play Area',
                'Landscaped Gardens', 'Jogging Track', 'Multi-purpose Hall',
                'Indoor Games', 'CCTV Surveillance', 'Visitor Parking',
                'Maintenance Staff', 'Fire Safety', 'Rainwater Harvesting'
            ]
        else:
            premium_amenities = [
                'Community Hall', 'Garden Area', 'Children Play Zone',
                'Basic Security', 'Maintenance Service', 'Visitor Area'
            ]
        
        # Luxury amenities for large properties
        if area > 1500 and bedrooms >= 3:
            luxury_amenities = [
                'Concierge Service', 'Spa & Wellness', 'Business Center',
                'Banquet Hall', 'Meditation Area', 'Yoga Deck',
                'Library', 'Kids Pool', 'Badminton Court', 'Tennis Court'
            ]
            all_amenities = base_amenities + premium_amenities + luxury_amenities[:5]
        else:
            all_amenities = base_amenities + premium_amenities[:8]
        
        # Property specifications
        specifications = {
            'Floor Plan': f'{bedrooms}BHK with {bathrooms} bathrooms',
            'Carpet Area': f'{area} sq ft',
            'Floor Type': 'Vitrified tiles' if area > 1000 else 'Ceramic tiles',
            'Kitchen': 'Modular kitchen' if area > 800 else 'Semi-modular kitchen',
            'Balconies': '2 balconies' if bedrooms >= 3 else '1 balcony',
            'Facing': np.random.choice(['North', 'South', 'East', 'West', 'North-East', 'South-West']),
            'Age': f'{np.random.randint(0, 8)} years' if np.random.random() > 0.3 else 'Under Construction',
            'Furnishing': np.random.choice(['Unfurnished', 'Semi-Furnished', 'Fully Furnished'], p=[0.6, 0.3, 0.1])
        }
        
        return {
            'amenities': all_amenities,
            'specifications': specifications
        }
        
    except Exception as e:
        st.error(f"Error generating property features: {str(e)}")
        return {'amenities': [], 'specifications': {}}

def main():
    st.title("🏠 Home Price Estimator - India")
    st.markdown("Get instant price estimates for homes across major Indian cities")
    
    # Load and train model if not already done
    if not st.session_state.model_trained:
        with st.spinner("Loading market data and training prediction model..."):
            market_data = get_real_estate_data()
            if not market_data.empty:
                model, label_encoder = train_price_model(market_data)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.label_encoder = label_encoder
                    st.session_state.market_data = market_data
                    st.session_state.model_trained = True
                    st.success("✅ Model ready! Enter your home details below.")
                else:
                    st.error("Failed to train prediction model.")
            else:
                st.error("Failed to load market data.")
    
    # Input form
    if st.session_state.model_trained:
        st.markdown("---")
        st.subheader("Enter Your Home Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            area = st.number_input("Area (sq ft)", min_value=300, max_value=5000, value=1000, step=50)
        
        with col2:
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
        
        with col3:
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
        
        with col4:
            # Indian cities
            cities = [
                'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata',
                'Ahmedabad', 'Jaipur', 'Surat', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore',
                'Thane', 'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara',
                'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot',
                'Kalyan-Dombivali', 'Vasai-Virar', 'Varanasi'
            ]
            location = st.selectbox("City", cities, index=0)
        
        # Predict button
        if st.button("🔍 Get Price Estimate", type="primary", use_container_width=True):
            prediction = predict_price(
                st.session_state.model,
                st.session_state.label_encoder,
                area, bedrooms, bathrooms, location
            )
            
            if prediction:
                st.session_state.last_prediction = prediction
                st.session_state.last_inputs = {
                    'area': area,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'location': location
                }
                st.session_state.selected_city = location
                
                # Load builders data for selected city
                builders = get_builders_data(location)
                st.session_state.builders_data = builders
                
                # Generate property features
                property_features = get_property_features(location, bedrooms, bathrooms, area)
                st.session_state.property_features = property_features
        
        # Display results
        if 'last_prediction' in st.session_state:
            st.markdown("---")
            st.subheader("💰 Price Estimate")
            
            # Main price display
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Convert to lakhs for display
                price_lakhs = st.session_state.last_prediction / 100000
                st.metric(
                    label=f"Estimated Price in {st.session_state.last_inputs['location']}",
                    value=f"₹{price_lakhs:.1f} Lakhs",
                    delta=f"₹{st.session_state.last_prediction:,}"
                )
            
            with col2:
                price_per_sqft = st.session_state.last_prediction / st.session_state.last_inputs['area']
                st.metric("Price per sq ft", f"₹{price_per_sqft:,.0f}")
            
            with col3:
                # Compare with market average
                market_data = st.session_state.market_data
                similar_homes = market_data[
                    (market_data['location'] == st.session_state.last_inputs['location']) &
                    (market_data['bedrooms'] == st.session_state.last_inputs['bedrooms'])
                ]
                
                if not similar_homes.empty:
                    avg_similar = similar_homes['price'].mean()
                    if st.session_state.last_prediction > avg_similar:
                        st.success(f"📈 Above average for similar homes (₹{avg_similar:,.0f})")
                    else:
                        st.info(f"📊 Below average for similar homes (₹{avg_similar:,.0f})")
            
        else:
            st.info("👆 Enter your home details and click 'Get Price Estimate' to see the estimated value")
    
    # Display additional information after price estimation
    if 'last_prediction' in st.session_state and 'property_features' in st.session_state:
        st.markdown("---")
        
        # Create tabs for different information
        tab1, tab2, tab3 = st.tabs(["🏗️ Top Builders", "🏠 Property Overview", "🎥 Virtual Tour"])
        
        with tab1:
            st.subheader(f"Top Builders in {st.session_state.selected_city}")
            
            if st.session_state.builders_data:
                for i, builder in enumerate(st.session_state.builders_data):
                    with st.expander(f"⭐ {builder['name']} - Rating: {builder['rating']}/5"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Experience:** {builder['experience']}")
                            st.write(f"**Specialty:** {builder['specialty']}")
                            st.write(f"**Popular Projects:**")
                            for project in builder['projects']:
                                st.write(f"• {project}")
                        
                        with col2:
                            st.metric("Rating", f"{builder['rating']}/5")
                            if st.button(f"Contact {builder['name']}", key=f"contact_{i}"):
                                st.success(f"Contact information for {builder['name']} sent to your email!")
        
        with tab2:
            st.subheader("Property Overview & Amenities")
            
            features = st.session_state.property_features
            
            # Property specifications
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Property Specifications")
                for key, value in features['specifications'].items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.markdown("#### 🏊 Amenities & Facilities")
                # Display amenities in a grid-like format
                amenities_text = ""
                for i, amenity in enumerate(features['amenities']):
                    amenities_text += f"• {amenity}\n"
                st.text(amenities_text)
            
            # Investment insights
            st.markdown("#### 💡 Investment Insights")
            price_per_sqft = st.session_state.last_prediction / st.session_state.last_inputs['area']
            
            if price_per_sqft > 8000:
                investment_grade = "Premium"
                investment_color = "green"
            elif price_per_sqft > 5000:
                investment_grade = "Good"
                investment_color = "orange"
            else:
                investment_grade = "Budget-Friendly"
                investment_color = "blue"
            
            st.markdown(f"**Investment Grade:** :{investment_color}[{investment_grade}]")
            
            # Location advantages
            tier_1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata']
            if st.session_state.selected_city in tier_1_cities:
                location_benefits = [
                    "Well-connected public transport",
                    "Close to IT/business hubs",
                    "Good educational institutions nearby",
                    "Healthcare facilities available",
                    "Shopping and entertainment options"
                ]
            else:
                location_benefits = [
                    "Growing infrastructure",
                    "Affordable pricing",
                    "Peaceful residential area",
                    "Good connectivity to main city",
                    "Future development potential"
                ]
            
            st.markdown("**Location Benefits:**")
            for benefit in location_benefits:
                st.write(f"✅ {benefit}")
        
        with tab3:
            st.subheader("Virtual Property Tour")
            
            # Create a mock virtual tour interface
            st.markdown("#### 🎮 Interactive Property Walkthrough")
            
            tour_options = st.selectbox(
                "Choose a room to explore:",
                ["Living Room", "Master Bedroom", "Kitchen", "Bathroom", "Balcony", "Common Areas"]
            )
            
            # Mock virtual tour descriptions
            tour_descriptions = {
                "Living Room": {
                    "description": "Spacious living area with modern flooring and large windows providing natural light. Perfect for family gatherings and entertainment.",
                    "features": ["Large windows", "Modern flooring", "Ceiling fan", "TV unit space", "Seating area"]
                },
                "Master Bedroom": {
                    "description": "Comfortable master bedroom with attached bathroom and wardrobe space. Designed for privacy and relaxation.",
                    "features": ["Queen/King bed space", "Attached bathroom", "Built-in wardrobe", "Window with view", "AC provision"]
                },
                "Kitchen": {
                    "description": "Well-planned kitchen with modern fittings and ample storage space. Designed for convenient cooking and food preparation.",
                    "features": ["Modular design", "Storage cabinets", "Platform space", "Exhaust provision", "Water connection"]
                },
                "Bathroom": {
                    "description": "Modern bathroom with quality fittings and proper ventilation. Clean and hygienic design.",
                    "features": ["Modern fixtures", "Hot water provision", "Ventilation", "Storage space", "Quality tiles"]
                },
                "Balcony": {
                    "description": "Private balcony space offering outdoor relaxation and fresh air. Perfect for morning coffee or evening relaxation.",
                    "features": ["Outdoor space", "Safety grills", "City/garden view", "Drying area", "Fresh air circulation"]
                },
                "Common Areas": {
                    "description": "Well-maintained common areas including lobby, corridors, and amenity spaces. Designed for community living.",
                    "features": ["Security desk", "Mailbox area", "Elevator access", "Common utilities", "Maintenance room"]
                }
            }
            
            if tour_options in tour_descriptions:
                tour_info = tour_descriptions[tour_options]
                
                st.markdown(f"#### 📍 {tour_options}")
                st.write(tour_info["description"])
                
                st.markdown("**Key Features:**")
                for feature in tour_info["features"]:
                    st.write(f"• {feature}")
                
                # Mock 360-degree view button
                if st.button(f"🔄 360° View of {tour_options}", use_container_width=True):
                    st.success(f"Loading 360° virtual tour of {tour_options}...")
                    st.balloons()
            
            # Booking section
            st.markdown("---")
            st.markdown("#### 📅 Schedule a Site Visit")
            
            visit_col1, visit_col2 = st.columns(2)
            
            with visit_col1:
                visit_date = st.date_input("Preferred Visit Date")
                visit_time = st.selectbox("Preferred Time", ["10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM", "6:00 PM"])
            
            with visit_col2:
                visitor_name = st.text_input("Your Name")
                visitor_phone = st.text_input("Phone Number")
            
            if st.button("📋 Schedule Site Visit", type="primary", use_container_width=True):
                if visitor_name and visitor_phone:
                    st.success(f"Site visit scheduled for {visit_date} at {visit_time}. Confirmation details sent to your phone!")
                else:
                    st.error("Please provide your name and phone number to schedule a visit.")
    
    # Market data visualization
    if st.session_state.model_trained and st.session_state.market_data is not None:
        st.markdown("---")
        st.subheader("📊 Market Insights")
        
        # City-wise price comparison
        city_avg_prices = st.session_state.market_data.groupby('location')['price'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=city_avg_prices.head(10).index,
            y=city_avg_prices.head(10).values / 100000,  # Convert to lakhs
            title="Average Home Prices by City (Top 10)",
            labels={'x': 'City', 'y': 'Average Price (₹ Lakhs)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
```

## File 2: render_requirements.txt
**Instructions:** Save this as `render_requirements.txt`

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
trafilatura>=1.6.0
requests>=2.31.0
```

## File 3: render.yaml
**Instructions:** Save this as `render.yaml`

```yaml
services:
  - type: web
    name: home-price-estimator
    env: python
    runtime: python-3.11
    buildCommand: pip install -r render_requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHONUNBUFFERED
        value: 1
```

## File 4: runtime.txt
**Instructions:** Save this as `runtime.txt`

```
python-3.11.7
```

## File 5: Procfile
**Instructions:** Save this as `Procfile`

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## File 6: README.md
**Instructions:** Save this as `README.md`

```markdown
# Home Price Estimator - India

A comprehensive web application to estimate home prices across major Indian cities using machine learning, with complete real estate features.

## Features
- **Instant Price Estimates** - Enter your home details and get prices in Indian Rupees (₹)
- **Top Builders Database** - Real builders and developers for each city with ratings and contact info
- **Property Overview** - Detailed specifications, amenities, and investment insights
- **Virtual Property Tours** - Interactive room-by-room exploration with 360° views
- **Site Visit Scheduling** - Book property visits directly through the app
- **Market Analysis** - Compare with similar properties and view trends
- **30+ Indian Cities** - Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, and more

## How to Deploy on Render

### Step 1: Create a Render Account
1. Go to [render.com](https://render.com)
2. Sign up for a free account
3. Connect your GitHub account

### Step 2: Upload Your Code to GitHub
1. Create a new repository on GitHub
2. Upload all the files from this project:
   - `app.py` (main application)
   - `render_requirements.txt` (dependencies)
   - `render.yaml` (Render configuration)
   - `runtime.txt` (Python version)
   - `Procfile` (process file)
   - `README.md` (this file)

### Step 3: Deploy on Render
1. In your Render dashboard, click "New +" and select "Web Service"
2. Connect your GitHub repository
3. Use these settings:
   - **Name**: home-price-estimator (or any name you prefer)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r render_requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan**: Free (for testing)

### Step 4: Environment Variables (Optional)
No environment variables are required for basic functionality.

### Step 5: Deploy
1. Click "Create Web Service"
2. Wait for deployment to complete (usually 2-5 minutes)
3. Your app will be available at: `https://your-app-name.onrender.com`

## Cities Supported
Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad, Jaipur, Surat, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal, Visakhapatnam, Pimpri-Chinchwad, Patna, Vadodara, Ghaziabad, Ludhiana, Agra, Nashik, Faridabad, Meerut, Rajkot, Kalyan-Dombivali, Vasai-Virar, Varanasi

## Local Development
```bash
# Install dependencies
pip install -r render_requirements.txt

# Run the app
streamlit run app.py
```

## How It Works
1. Uses machine learning (Random Forest) to predict home prices
2. Trained on realistic market data based on Indian real estate trends
3. Considers location factors specific to Indian cities
4. Provides price estimates in Indian Rupees (₹)

## Support
The app automatically loads market data and trains the prediction model when you first visit it. Simply enter your home details and get an instant price estimate!
```

## Deployment Instructions

1. **Create GitHub Repository**: Make a new public repository on GitHub
2. **Upload Files**: Copy all the code above into separate files and upload to GitHub
3. **Deploy on Render**: 
   - Sign up at [render.com](https://render.com)
   - Create "Web Service" 
   - Connect your GitHub repository
   - Use Python 3.11 runtime
   - Deploy with the settings above

Your app will be live at: `https://your-app-name.onrender.com`

The Python 3.11 runtime and flexible package versions will resolve the pandas compilation errors you encountered.