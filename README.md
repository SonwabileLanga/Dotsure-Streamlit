# 🚗 DOTSURE STREAMLIT - Telematics Analytics Dashboard

A comprehensive offline telematics analytics dashboard built with Streamlit that helps analyze driver behavior, accident risks, and trip performance from historical CSV logs.

## 🎯 Features

### Core Functionality
- **CSV File Upload**: Upload telematics data from local system with validation
- **Interactive Map Visualization**: Plot GPS points with color-coded markers for different events
- **Speed Analysis**: Line charts showing speed vs time with overspeeding detection
- **Trip Analytics**: Calculate total distance, duration, average/max speed
- **Driver Risk Profile**: Generate driver scores based on behavior metrics

### Analytics Capabilities
- **Harsh Event Detection**: Identify harsh braking and acceleration events
- **Speeding Analysis**: Detect and highlight overspeeding events (>120 km/h)
- **Route Visualization**: Interactive map with GPS tracking
- **Risk Scoring**: Comprehensive driver risk assessment
- **Data Export**: Detailed analytics and summaries

## 📋 Data Format

Your CSV file should contain the following columns:

### Required Columns
- `vehicle_id`: Unique identifier for the vehicle
- `timestamp`: Date and time of the GPS reading (YYYY-MM-DD HH:MM:SS)
- `latitude`: GPS latitude coordinate
- `longitude`: GPS longitude coordinate
- `speed`: Vehicle speed in km/h

### Optional Columns
- `accident`: Boolean indicating if an accident occurred (True/False)

### Sample Data Format
```csv
vehicle_id,timestamp,latitude,longitude,speed,accident
V001,2024-01-01 10:00:00,40.7128,-74.0060,45.0,False
V001,2024-01-01 10:01:00,40.7130,-74.0058,52.0,False
```

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Navigate to the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and go to `http://localhost:8501`

### Using the Dashboard
1. **Upload Data**: Use the sidebar to upload your CSV file
2. **View Analytics**: The dashboard will automatically process and display:
   - Trip overview metrics
   - Driver risk profile
   - Interactive route map
   - Speed analysis charts
   - Detailed analytics tabs

## 📊 Analytics Features

### Trip Overview
- Total distance covered (calculated from GPS coordinates)
- Trip duration
- Average and maximum speed
- Number of events (speeding, harsh braking/acceleration)

### Driver Risk Profile
- **Driver Score**: 0-100 scale based on behavior
- **Risk Level**: LOW (80+), MEDIUM (60-79), HIGH (<60)
- **Scoring Factors**:
  - Speeding events (>120 km/h): -2 points each
  - Harsh braking: -3 points each
  - Harsh acceleration: -2 points each
  - Accidents: -25 points each

### Map Visualization
- **Blue markers**: Normal driving
- **Purple markers**: Speeding events
- **Orange markers**: Harsh braking
- **Yellow markers**: Harsh acceleration
- **Red markers**: Accident events

### Speed Analysis
- Speed vs time line chart
- Speed distribution histogram
- Harsh events timeline
- Overspeeding detection with visual indicators

## 🛠️ Technical Details

### Dependencies
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `plotly`: Interactive visualizations
- `folium`: Interactive maps
- `geopy`: Geospatial calculations
- `scikit-learn`: Machine learning utilities

### Key Functions
- `validate_csv_columns()`: Validates required CSV columns
- `calculate_distance()`: Calculates distance between GPS points
- `detect_harsh_events()`: Identifies harsh braking/acceleration
- `calculate_driver_score()`: Computes driver risk score
- `create_map_visualization()`: Generates interactive maps

## 📁 Project Structure
```
telematics/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── sample_telematics_data.csv     # Sample data for testing
└── README.md                      # This file
```

## 🔧 Customization

### Adjusting Risk Scoring
Modify the `calculate_driver_score()` function in `app.py` to adjust:
- Speeding penalties
- Harsh event penalties
- Accident penalties
- Score thresholds

### Adding New Analytics
Extend the dashboard by:
1. Adding new columns to your CSV data
2. Creating new processing functions
3. Adding new visualization components
4. Updating the risk scoring algorithm

## 📝 Sample Data

A sample CSV file (`sample_telematics_data.csv`) is included for testing the application. It contains:
- 2 hours of simulated driving data
- Various speed patterns including overspeeding
- GPS coordinates for a route in New York City area
- No accident events (for testing normal scenarios)

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new analytics features
- Improving the user interface
- Enhancing the risk scoring algorithm
- Adding support for additional data formats

## 📄 License

This project is open source and available under the MIT License.

---

**DOTSURE STREAMLIT** - Making telematics analytics accessible and insightful! 🚗📊
