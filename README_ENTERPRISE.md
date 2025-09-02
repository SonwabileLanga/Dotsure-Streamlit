# 🚗 DOTSURE ENTERPRISE - Advanced Telematics Platform

A comprehensive, enterprise-grade telematics and fleet management platform built with Streamlit, featuring advanced AI/ML capabilities, multi-database integration, Azure services, and third-party API connectivity.

## 🌟 Key Features

### 🗄️ Database Integration
- **Multi-Database Support**: SQLite, PostgreSQL, MySQL
- **Real-time Data Processing**: Automated data pipelines
- **Advanced Queries**: Custom SQL execution and optimization
- **Data Persistence**: Reliable data storage and retrieval

### ☁️ Azure Services Integration
- **Blob Storage**: Scalable cloud data storage
- **Cognitive Services**: AI-powered image analysis
- **Key Vault**: Secure credential management
- **Data Pipeline**: Automated cloud data processing

### 🌐 Third-Party API Integration
- **Weather APIs**: Real-time weather data integration
- **Traffic APIs**: Live traffic information and route analysis
- **Geocoding Services**: Address and coordinate conversion
- **Fleet Management APIs**: External system integration

### 🤖 AI/ML Engine
- **Risk Scoring**: Advanced machine learning risk assessment
- **Event Detection**: Anomaly detection and pattern recognition
- **Predictive Analytics**: Future risk and performance prediction
- **Intelligent Alerting**: Smart notification system

### 📊 Advanced Analytics
- **Real-time Dashboards**: Interactive data visualization
- **Geospatial Analysis**: Map-based fleet tracking
- **Performance Metrics**: Comprehensive fleet analytics
- **Trend Analysis**: Historical data insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SonwabileLanga/Dotsure-Streamlit.git
   cd Dotsure-Streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_enterprise.txt
   ```

3. **Run the application**
   ```bash
   streamlit run dotsure_enterprise.py
   ```

4. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - The enterprise dashboard will load with all features available

## 📁 Project Structure

```
telematics/
├── dotsure_enterprise.py          # Main enterprise application
├── database_manager.py            # Database integration module
├── azure_integration.py           # Azure services integration
├── api_integration.py             # Third-party API integration
├── ai_ml_engine.py                # AI/ML and analytics engine
├── requirements_enterprise.txt    # Enterprise dependencies
├── README_ENTERPRISE.md          # This file
├── dotsure_premium.py            # Premium version (simplified)
├── uk_road_safety_app.py         # UK road safety specific app
└── csv_splitter.py               # Large file processing utility
```

## 🔧 Configuration

### Database Setup

#### SQLite (Default)
- No additional configuration required
- Database file created automatically

#### PostgreSQL
```python
# In the database manager UI
Host: localhost
Port: 5432
Database: telematics
Username: your_username
Password: your_password
```

#### MySQL
```python
# In the database manager UI
Host: localhost
Port: 3306
Database: telematics
Username: your_username
Password: your_password
```

### Azure Services Setup

1. **Create Azure Account**: Sign up at [Azure Portal](https://portal.azure.com)
2. **Create Storage Account**: Set up Blob Storage
3. **Configure Cognitive Services**: Enable Vision API
4. **Set up Key Vault**: For secure credential storage

### API Keys Setup

#### Weather API (OpenWeatherMap)
1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your API key
3. Enter in the API Integration tab

#### Google Maps APIs
1. Create project at [Google Cloud Console](https://console.cloud.google.com)
2. Enable Distance Matrix and Geocoding APIs
3. Create API key with appropriate restrictions
4. Enter in the API Integration tab

## 📊 Usage Guide

### 1. Data Import
- **Upload CSV**: Direct file upload (up to 200MB)
- **Database**: Connect to existing database
- **Azure Storage**: Load from cloud storage
- **API Integration**: Real-time data feeds
- **Demo Data**: Sample data for testing

### 2. Database Management
- Connect to multiple database types
- Create and manage tables
- Execute custom queries
- Monitor connection status

### 3. Azure Integration
- Upload/download data to/from Blob Storage
- Analyze images with Cognitive Services
- Manage secrets with Key Vault
- Process data through cloud pipelines

### 4. API Integration
- Configure external API connections
- Enrich data with weather and traffic information
- Geocode addresses and coordinates
- Integrate with fleet management systems

### 5. AI/ML Features
- Train risk scoring models
- Detect anomalies and events
- Analyze driving patterns
- Set up intelligent alerts

## 🎯 Advanced Features

### Risk Scoring Engine
- **Machine Learning Models**: Random Forest, Gradient Boosting
- **Feature Engineering**: Automatic feature extraction
- **Real-time Scoring**: Live risk assessment
- **Model Training**: Continuous learning and improvement

### Event Detection
- **Anomaly Detection**: Isolation Forest algorithm
- **Harsh Driving Events**: Acceleration, braking, speeding
- **Pattern Recognition**: Driving behavior analysis
- **Alert Generation**: Intelligent notification system

### Data Pipeline
- **Automated Processing**: Real-time data transformation
- **Cloud Integration**: Azure-based processing
- **Scalable Architecture**: Handle large datasets
- **Error Handling**: Robust error management

## 🔒 Security Features

- **Credential Management**: Secure API key storage
- **Data Encryption**: Encrypted data transmission
- **Access Control**: Role-based permissions
- **Audit Logging**: Comprehensive activity tracking

## 📈 Performance Optimization

- **Caching**: API response caching
- **Database Pooling**: Connection optimization
- **Lazy Loading**: Efficient data loading
- **Parallel Processing**: Multi-threaded operations

## 🚀 Deployment

### Local Development
```bash
streamlit run dotsure_enterprise.py
```

### Production Deployment
1. **Streamlit Cloud**: Deploy directly from GitHub
2. **Docker**: Containerized deployment
3. **Azure App Service**: Cloud deployment
4. **AWS/GCP**: Multi-cloud deployment

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
WEATHER_API_KEY=your_weather_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and inline comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Email**: Contact the development team

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Multi-database integration
- ✅ Azure services integration
- ✅ Basic AI/ML features
- ✅ API integration

### Phase 2 (Next)
- 🔄 Advanced ML models
- 🔄 Real-time streaming
- 🔄 Mobile app integration
- 🔄 Advanced reporting

### Phase 3 (Future)
- 📋 IoT device integration
- 📋 Blockchain integration
- 📋 Advanced analytics
- 📋 Global deployment

## 🏆 Enterprise Benefits

- **Scalability**: Handle enterprise-scale data
- **Reliability**: Robust error handling and recovery
- **Security**: Enterprise-grade security features
- **Integration**: Seamless third-party integrations
- **Analytics**: Advanced AI/ML capabilities
- **Support**: Comprehensive documentation and support

---

*Built with ❤️ using Streamlit, Python, and Enterprise Technologies*

**DOTSURE ENTERPRISE** - Your comprehensive telematics solution for the modern fleet.
