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