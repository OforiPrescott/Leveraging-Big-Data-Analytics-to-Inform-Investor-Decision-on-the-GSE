# GSE Sentiment Analysis Dashboard - React & React Native Conversion

This document describes the React web and React Native mobile implementations of the GSE Sentiment Analysis Dashboard, converted from the original Streamlit application.

## 🚀 Project Overview

The GSE Sentiment Analysis system has been converted from Streamlit to:
- **React Web Application**: Full-featured web dashboard with all original functionality
- **React Native Mobile App**: Mobile-optimized dashboard for iOS and Android
- **Flask API Backend**: RESTful API providing data access for both applications

## 📁 Project Structure

```
├── api.py                              # Flask API backend
├── react-dashboard/                   # React web application
│   ├── src/
│   │   ├── App.js                     # Main React app component
│   │   ├── App.css                    # Global styles
│   │   ├── components/                # Reusable React components
│   │   │   ├── Header.js             # Application header
│   │   │   ├── Sidebar.js            # Navigation sidebar
│   │   │   ├── ExecutiveSummary.js   # Executive summary tab
│   │   │   └── ...                   # Other tab components
│   │   └── index.js                  # React app entry point
│   └── package.json
├── GSEMobileDashboard/               # React Native mobile app
│   ├── App.js                        # Main mobile app component
│   ├── screens/                      # Mobile screen components
│   │   ├── HomeScreen.js            # Main dashboard screen
│   │   ├── PredictionsScreen.js     # Real-time predictions
│   │   ├── AnalyticsScreen.js       # Analytics dashboard
│   │   └── SettingsScreen.js        # App settings
│   └── package.json
├── working_dashboard.py              # Original Streamlit dashboard (preserved)
└── gse_sentiment.db                  # SQLite database
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.7+ with Flask and dependencies
- Node.js 16+ (for React development)
- npm or yarn package manager
- Expo CLI (for React Native development)

### 1. Backend API Setup

```bash
# Install Python dependencies
pip install flask flask-cors pandas numpy

# Start the Flask API server
python api.py
```

The API will be available at `http://localhost:5000`

### 2. React Web Application Setup

```bash
# Navigate to React dashboard directory
cd react-dashboard

# Install dependencies
npm install

# Start development server
npm start
```

The React web app will be available at `http://localhost:3000`

### 3. React Native Mobile Application Setup

```bash
# Navigate to mobile app directory
cd GSEMobileDashboard

# Install dependencies
npm install

# Start Expo development server
npm start
```

Use the Expo Go app on your mobile device or an emulator to view the mobile app.

## 📱 Features

### React Web Dashboard

- **Executive Summary**: Key metrics, system status, and research findings
- **Model Performance**: Comparative analysis of ML algorithms
- **Time Series Analysis**: Sentiment trends and market correlations
- **Correlation Studies**: Granger causality and statistical analysis
- **Real-Time Predictions**: Live sentiment-based price predictions
- **Manual Sentiment Input**: User contribution system
- **Data Sources**: Multi-source data collection overview
- **Research Data Export**: Academic data export capabilities

### React Native Mobile App

- **Home Dashboard**: System status and key metrics
- **Real-Time Predictions**: Mobile-optimized prediction interface
- **Analytics**: Performance metrics and insights
- **Settings**: App configuration and preferences

## 🔧 API Endpoints

The Flask API provides the following endpoints:

- `GET /api/sentiment-stats` - System statistics
- `GET /api/sentiment-data` - Raw sentiment data
- `GET /api/company-sentiment/<company>` - Company-specific data
- `GET /api/model-performance` - ML model performance data
- `POST /api/predict` - Generate real-time predictions
- `POST /api/manual-sentiment` - Add manual sentiment entries

## 🎨 Design Features

### Web Dashboard
- Responsive design with dark/light mode support
- Modern UI with gradient backgrounds and animations
- Interactive charts and data visualizations
- Mobile-responsive layout

### Mobile App
- Native mobile UI components
- Bottom tab navigation
- Pull-to-refresh functionality
- Touch-optimized interface
- Platform-specific styling

## 🔄 Data Flow

1. **Original Streamlit App**: Preserved as `working_dashboard.py`
2. **Flask API**: Provides RESTful access to sentiment data
3. **React Web App**: Consumes API data for full dashboard experience
4. **React Native App**: Mobile-optimized interface with API integration

## 🚀 Running the Applications

### Development Mode

1. **Start Flask API**:
   ```bash
   python api.py
   ```

2. **Start React Web App** (in new terminal):
   ```bash
   cd react-dashboard
   npm start
   ```

3. **Start React Native App** (in new terminal):
   ```bash
   cd GSEMobileDashboard
   npm start
   ```

### Production Deployment

For production deployment:

1. **Flask API**: Deploy to services like Heroku, AWS, or DigitalOcean
2. **React Web App**: Build with `npm run build` and deploy to Netlify, Vercel, or Apache
3. **React Native App**: Build APKs for Android or submit to App Store

## 📊 Key Differences from Streamlit Version

### Advantages of React/React Native Conversion

1. **Performance**: Faster rendering and better user experience
2. **Mobile Support**: Native mobile apps for iOS and Android
3. **Modern UI**: Contemporary design with animations and interactions
4. **Scalability**: Better architecture for large-scale applications
5. **Offline Capability**: Potential for offline functionality in mobile app
6. **Progressive Web App**: Web app can be installed as PWA

### Maintained Features

- All original functionality preserved
- Same data sources and analysis methods
- Research-grade accuracy and statistical methods
- Real-time data processing capabilities

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Issues**:
   - Ensure Flask API is running on port 5000
   - Check CORS settings in API
   - Verify network connectivity

2. **React App Issues**:
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility

3. **React Native Issues**:
   - Ensure Expo CLI is installed globally
   - Clear Metro cache: `npx react-native start --reset-cache`
   - Check Expo app version compatibility

### Database Issues

- Ensure `gse_sentiment.db` exists and is accessible
- Check file permissions for database access
- Verify SQLite installation

## 📈 Future Enhancements

- **Real-time WebSocket connections** for live data updates
- **Offline data synchronization** in mobile app
- **Push notifications** for important market events
- **Advanced charting** with interactive Plotly integration
- **User authentication** and personalized dashboards
- **Multi-language support** for international users

## 📚 Documentation

- **Original Research**: See `PROJECT_OVERVIEW.md` and `IMPLEMENTATION_GUIDE.md`
- **API Documentation**: Inline comments in `api.py`
- **Component Documentation**: Inline comments in React components
- **Deployment Guide**: See `DEPLOYMENT.md`

## 🤝 Contributing

The original Streamlit dashboard is preserved for reference. New features should be implemented in both React web and React Native versions to maintain consistency.

## 📄 License

This project maintains the same license as the original GSE Sentiment Analysis system.

---

**Note**: This conversion maintains all research integrity and academic standards of the original implementation while providing modern, accessible interfaces for both web and mobile platforms.