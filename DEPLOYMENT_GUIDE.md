# Deploy Your Home Price Estimator on Render

This guide will help you deploy your Indian home price estimator web application on Render for free.

## Prerequisites

1. **GitHub Account** - You'll need to upload your code to GitHub first
2. **Render Account** - Sign up at [render.com](https://render.com) (free)

## Step 1: Upload Code to GitHub

### Option A: Create New Repository
1. Go to [github.com](https://github.com) and sign in
2. Click "New" to create a new repository
3. Name it something like `home-price-estimator-india`
4. Make it **Public** (required for free Render deployment)
5. Upload these files from your project:
   - `app.py`
   - `render_requirements.txt`
   - `render.yaml`
   - `Procfile`
   - `README.md`
   - `DEPLOYMENT_GUIDE.md`

### Option B: Use GitHub Desktop
1. Download GitHub Desktop
2. Clone or create a new repository
3. Copy all the project files into the repository folder
4. Commit and push to GitHub

## Step 2: Deploy on Render

### Create Web Service
1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** button
3. Select **"Web Service"**

### Connect Repository
1. Click **"Connect a repository"**
2. Choose **GitHub** and authorize Render to access your repositories
3. Select your `home-price-estimator-india` repository

### Configure Deployment Settings

Use these **exact settings**:

#### Basic Settings
- **Name**: `home-price-estimator` (or any name you prefer)
- **Region**: Choose closest to your users (e.g., Singapore for India)
- **Branch**: `main` (or `master` if that's your default)
- **Root Directory**: Leave empty
- **Runtime**: `Python 3`

#### Build & Deploy Settings
- **Build Command**: 
  ```
  pip install -r render_requirements.txt
  ```
- **Start Command**: 
  ```
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```

#### Advanced Settings
- **Auto Deploy**: `Yes` (recommended)
- **Health Check Path**: Leave empty

### Plan Selection
- Choose **"Free"** plan for testing (perfect for personal use)
- Free plan includes:
  - 750 hours/month (enough for most personal projects)
  - Custom domain (yourapp.onrender.com)
  - SSL certificate included

## Step 3: Deploy & Monitor

### Start Deployment
1. Click **"Create Web Service"**
2. Render will start building your app
3. This process takes **3-5 minutes** typically

### Monitor Build Process
Watch the build logs for:
- ✅ Installing Python dependencies
- ✅ Starting Streamlit server
- ✅ "Your service is live" message

### Common Build Issues & Solutions

#### Issue: "Module not found"
**Solution**: Check that `render_requirements.txt` includes all dependencies

#### Issue: "Port binding failed"
**Solution**: Ensure start command uses `$PORT` variable exactly as shown

#### Issue: "Build timeout"
**Solution**: This is rare, just retry the deployment

## Step 4: Access Your Live App

### Get Your App URL
Once deployed successfully:
1. Your app will be available at: `https://your-app-name.onrender.com`
2. Render provides this URL in the dashboard
3. Click the URL to test your live application

### Test All Features
Verify these work on the live site:
- ✅ Home price estimation
- ✅ Builders database
- ✅ Property overview
- ✅ Virtual tour
- ✅ Site visit scheduling

## Step 5: Custom Domain (Optional)

### Free Subdomain
Your app gets a free `.onrender.com` subdomain automatically.

### Custom Domain (Paid Plans Only)
If you upgrade to a paid plan, you can:
1. Go to Settings > Custom Domains
2. Add your own domain (e.g., `myhomeapp.com`)
3. Update DNS settings as instructed

## Important Notes

### Free Plan Limitations
- **Sleep after 15 minutes** of inactivity
- Takes **30-60 seconds** to wake up when accessed
- **750 hours/month** limit (about 25 hours/day)

### Wake-Up Strategy
The app "sleeps" after 15 minutes of no visitors. When someone visits:
1. They'll see a loading screen for 30-60 seconds
2. App fully loads and works normally
3. Stays awake as long as people keep using it

### Data Persistence
- All code and ML models are preserved
- No database required for this app
- Everything works from the deployed code

## Troubleshooting

### App Won't Load
1. Check build logs in Render dashboard
2. Verify all files uploaded to GitHub correctly
3. Ensure `render_requirements.txt` has correct Python package versions

### Slow Performance
- Normal for free tier
- Consider upgrading to paid plan for faster performance
- App speed improves after initial wake-up

### Updates & Changes
When you make changes:
1. Push updates to GitHub
2. Render automatically rebuilds (if auto-deploy enabled)
3. New version goes live in 3-5 minutes

## Success Checklist

- [ ] GitHub repository created with all files
- [ ] Render account created and verified
- [ ] Web service configured with correct settings
- [ ] Build completed successfully
- [ ] App accessible at .onrender.com URL
- [ ] All features working (price estimation, builders, tours)

## Support

If you encounter issues:
1. Check Render's build logs for error messages
2. Verify GitHub repository has all required files
3. Ensure `render_requirements.txt` matches your local environment
4. Try redeploying if build fails

Your home price estimator will be live and accessible to anyone with the URL!