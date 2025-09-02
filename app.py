import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DOTSURE STREAMLIT - Telematics Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .risk-low {
        color: #00aa44;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def validate_csv_columns(df):
    """Validate that the CSV has required columns"""
    required_columns = ['vehicle_id', 'timestamp', 'latitude', 'longitude', 'speed']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info("Required columns: vehicle_id, timestamp, latitude, longitude, speed")
        return False
    
    return True

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in kilometers"""
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except:
        return 0

def detect_harsh_events(df, speed_threshold=10, time_window=5):
    """Detect harsh braking and acceleration events"""
    df = df.copy()
    df['acceleration'] = df['speed'].diff() / df['speed'].shift(1) * 100
    
    # Harsh braking: rapid deceleration
    df['harsh_braking'] = (df['acceleration'] < -speed_threshold) & (df['speed'] > 20)
    
    # Harsh acceleration: rapid acceleration
    df['harsh_acceleration'] = (df['acceleration'] > speed_threshold) & (df['speed'] > 0)
    
    return df

def calculate_driver_score(df):
    """Calculate driver risk score based on behavior metrics"""
    score = 100  # Start with perfect score
    
    # Speeding penalty (speed > 120 km/h)
    speeding_events = len(df[df['speed'] > 120])
    score -= min(speeding_events * 2, 30)  # Max 30 points penalty
    
    # Harsh braking penalty
    harsh_braking_events = len(df[df.get('harsh_braking', False)])
    score -= min(harsh_braking_events * 3, 25)  # Max 25 points penalty
    
    # Harsh acceleration penalty
    harsh_acceleration_events = len(df[df.get('harsh_acceleration', False)])
    score -= min(harsh_acceleration_events * 2, 20)  # Max 20 points penalty
    
    # Accident penalty (if available)
    if 'accident' in df.columns:
        accident_events = len(df[df['accident'] == True])
        score -= accident_events * 25  # 25 points per accident
    
    return max(score, 0)  # Ensure score doesn't go below 0

def create_map_visualization(df):
    """Create interactive map with GPS points and color-coded markers"""
    if df.empty:
        return None
    
    # Calculate center point
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add GPS points with color coding
    for idx, row in df.iterrows():
        # Determine color based on events
        if 'accident' in df.columns and row.get('accident', False):
            color = 'red'
            icon = 'exclamation-triangle'
        elif row.get('harsh_braking', False):
            color = 'orange'
            icon = 'stop'
        elif row.get('harsh_acceleration', False):
            color = 'yellow'
            icon = 'play'
        elif row['speed'] > 120:
            color = 'purple'
            icon = 'tachometer-alt'
        else:
            color = 'blue'
            icon = 'circle'
        
        # Add marker
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"""
            <b>Time:</b> {row['timestamp']}<br>
            <b>Speed:</b> {row['speed']:.1f} km/h<br>
            <b>Vehicle:</b> {row['vehicle_id']}
            """,
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 DOTSURE STREAMLIT</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Telematics Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload telematics data with columns: vehicle_id, timestamp, latitude, longitude, speed"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            if not validate_csv_columns(df):
                st.stop()
            
            # Data preprocessing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Detect harsh events
            df = detect_harsh_events(df)
            
            # Calculate trip metrics
            total_distance = 0
            if len(df) > 1:
                for i in range(1, len(df)):
                    dist = calculate_distance(
                        df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                        df.iloc[i]['latitude'], df.iloc[i]['longitude']
                    )
                    total_distance += dist
            
            trip_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600  # hours
            avg_speed = df['speed'].mean()
            max_speed = df['speed'].max()
            
            # Calculate driver score
            driver_score = calculate_driver_score(df)
            
            # Display metrics
            st.header("📊 Trip Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Distance", f"{total_distance:.2f} km")
            
            with col2:
                st.metric("Trip Duration", f"{trip_duration:.2f} hours")
            
            with col3:
                st.metric("Average Speed", f"{avg_speed:.1f} km/h")
            
            with col4:
                st.metric("Max Speed", f"{max_speed:.1f} km/h")
            
            # Driver Risk Profile
            st.header("🎯 Driver Risk Profile")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if driver_score >= 80:
                    risk_level = "LOW"
                    risk_class = "risk-low"
                elif driver_score >= 60:
                    risk_level = "MEDIUM"
                    risk_class = "risk-medium"
                else:
                    risk_level = "HIGH"
                    risk_class = "risk-high"
                
                st.markdown(f'<div class="metric-card"><h3>Driver Score: <span class="{risk_class}">{driver_score:.0f}/100</span></h3><p>Risk Level: <span class="{risk_class}">{risk_level}</span></p></div>', unsafe_allow_html=True)
            
            with col2:
                speeding_events = len(df[df['speed'] > 120])
                st.metric("Speeding Events", speeding_events)
            
            with col3:
                harsh_events = len(df[df.get('harsh_braking', False)]) + len(df[df.get('harsh_acceleration', False)])
                st.metric("Harsh Events", harsh_events)
            
            # Map Visualization
            st.header("🗺️ Route Map")
            map_obj = create_map_visualization(df)
            if map_obj:
                st_folium(map_obj, width=700, height=500)
            
            # Speed Analysis
            st.header("📈 Speed Analysis")
            
            # Speed vs Time chart
            fig_speed = px.line(
                df, 
                x='timestamp', 
                y='speed',
                title='Speed vs Time',
                labels={'speed': 'Speed (km/h)', 'timestamp': 'Time'}
            )
            
            # Add speed limit line
            fig_speed.add_hline(y=120, line_dash="dash", line_color="red", 
                              annotation_text="Speed Limit (120 km/h)")
            
            st.plotly_chart(fig_speed, use_container_width=True)
            
            # Speed distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df, 
                    x='speed',
                    title='Speed Distribution',
                    labels={'speed': 'Speed (km/h)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Harsh events over time
                if 'harsh_braking' in df.columns or 'harsh_acceleration' in df.columns:
                    harsh_data = []
                    for idx, row in df.iterrows():
                        if row.get('harsh_braking', False):
                            harsh_data.append({'timestamp': row['timestamp'], 'event': 'Harsh Braking'})
                        if row.get('harsh_acceleration', False):
                            harsh_data.append({'timestamp': row['timestamp'], 'event': 'Harsh Acceleration'})
                    
                    if harsh_data:
                        harsh_df = pd.DataFrame(harsh_data)
                        fig_harsh = px.scatter(
                            harsh_df,
                            x='timestamp',
                            y='event',
                            color='event',
                            title='Harsh Events Timeline'
                        )
                        st.plotly_chart(fig_harsh, use_container_width=True)
                    else:
                        st.info("No harsh events detected in this trip.")
            
            # Detailed Analytics
            st.header("📋 Detailed Analytics")
            
            # Create tabs for different analytics
            tab1, tab2, tab3 = st.tabs(["Speed Events", "Route Analysis", "Data Summary"])
            
            with tab1:
                st.subheader("Speed Events")
                speeding_df = df[df['speed'] > 120].copy()
                if not speeding_df.empty:
                    st.dataframe(speeding_df[['timestamp', 'speed', 'latitude', 'longitude']])
                else:
                    st.success("No speeding events detected!")
            
            with tab2:
                st.subheader("Route Analysis")
                st.write(f"**Total GPS Points:** {len(df)}")
                st.write(f"**Route Efficiency:** {total_distance/trip_duration:.1f} km/h average")
                
                # Show first and last points
                if len(df) > 0:
                    st.write("**Trip Start:**", df.iloc[0]['timestamp'])
                    st.write("**Trip End:**", df.iloc[-1]['timestamp'])
            
            with tab3:
                st.subheader("Data Summary")
                st.dataframe(df.describe())
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample data format
        st.info("👆 Please upload a CSV file to get started!")
        
        st.header("📋 Expected Data Format")
        st.markdown("""
        Your CSV file should contain the following columns:
        - **vehicle_id**: Unique identifier for the vehicle
        - **timestamp**: Date and time of the GPS reading
        - **latitude**: GPS latitude coordinate
        - **longitude**: GPS longitude coordinate
        - **speed**: Vehicle speed in km/h
        
        Optional columns:
        - **accident**: Boolean indicating if an accident occurred
        """)
        
        # Show sample data
        sample_data = pd.DataFrame({
            'vehicle_id': ['V001', 'V001', 'V001'],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', '2024-01-01 10:02:00'],
            'latitude': [40.7128, 40.7130, 40.7132],
            'longitude': [-74.0060, -74.0058, -74.0056],
            'speed': [45.0, 52.0, 38.0]
        })
        
        st.subheader("Sample Data Format:")
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()
