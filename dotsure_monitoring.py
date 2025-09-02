"""
DOTSURE MONITORING DASHBOARD - Grafana-style Telematics Monitoring
Full-featured system monitoring dashboard for vehicle telematics data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DOTSURE Monitoring Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Grafana-style dashboard
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stApp > header {visibility: hidden;}
    .stApp > div > div > div > div {padding-top: 0rem;}
    
    /* Main dashboard styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .dashboard-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .dashboard-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric panels */
    .metric-panel {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #a0aec0;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    
    .metric-subtitle {
        font-size: 0.8rem;
        color: #718096;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-sparkline {
        height: 40px;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-good { color: #48bb78; }
    .status-warning { color: #ed8936; }
    .status-critical { color: #f56565; }
    .status-info { color: #4299e1; }
    
    /* Chart containers */
    .chart-container {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .chart-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #4a5568;
    }
    
    /* Time range selector */
    .time-range {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #1a202c;
    }
    
    /* Data table styling */
    .data-table {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Alert styling */
    .alert-panel {
        background: #2d3748;
        border-left: 4px solid #f56565;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    
    .alert-title {
        font-weight: 600;
        color: #f56565;
        margin: 0 0 0.5rem 0;
    }
    
    .alert-message {
        margin: 0;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

def create_dashboard_header():
    """Create Grafana-style dashboard header"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">
            🚗 DOTSURE MONITORING DASHBOARD
        </div>
        <div class="dashboard-subtitle">
            Real-time Telematics & Fleet Performance Monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_metric_panel(title, value, subtitle="", status="info", sparkline_data=None):
    """Create a metric panel similar to Grafana"""
    status_class = f"status-{status}"
    
    sparkline_html = ""
    if sparkline_data is not None:
        # Create a simple sparkline using CSS
        sparkline_html = f'<div class="metric-sparkline" style="background: linear-gradient(90deg, {status_class} 0%, transparent 100%); height: 2px;"></div>'
    
    return f"""
    <div class="metric-panel">
        <div class="metric-title">{title}</div>
        <div class="metric-value {status_class}">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
        {sparkline_html}
    </div>
    """

def create_quick_overview(df):
    """Create quick overview metrics panel"""
    st.markdown("### 📊 Quick Overview")
    
    # Calculate metrics
    total_vehicles = df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0
    total_records = len(df)
    
    # Calculate uptime (time span of data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_span = df['timestamp'].max() - df['timestamp'].min()
        uptime_days = time_span.days
        uptime_hours = time_span.seconds // 3600
        uptime_str = f"{uptime_days}d {uptime_hours}h"
    else:
        uptime_str = "N/A"
    
    # Calculate average speed
    avg_speed = df['speed'].mean() if 'speed' in df.columns else 0
    
    # Calculate risk level
    if 'risk_score' in df.columns:
        high_risk_count = len(df[df['risk_score'] > 0.7])
        risk_level = "Critical" if high_risk_count > 10 else "Good" if high_risk_count == 0 else "Warning"
    else:
        high_risk_count = 0
        risk_level = "Good"
    
    # Calculate active processes (vehicles with recent data)
    if 'timestamp' in df.columns:
        recent_time = df['timestamp'].max() - timedelta(hours=1)
        active_vehicles = len(df[df['timestamp'] > recent_time]['vehicle_id'].unique())
    else:
        active_vehicles = total_vehicles
    
    # Create metric panels
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_panel("Uptime", uptime_str, "System uptime", "info"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_panel("Active Vehicles", str(active_vehicles), f"of {total_vehicles} total", "good"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_panel("Avg Speed", f"{avg_speed:.1f} km/h", "Fleet average", "info"), unsafe_allow_html=True)
    
    with col4:
        status_color = "critical" if risk_level == "Critical" else "warning" if risk_level == "Warning" else "good"
        st.markdown(create_metric_panel("Risk Level", risk_level, f"{high_risk_count} high risk events", status_color), unsafe_allow_html=True)
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(create_metric_panel("Data Points", f"{total_records:,}", "Total records", "info"), unsafe_allow_html=True)
    
    with col6:
        if 'acceleration' in df.columns:
            harsh_events = len(df[(df['acceleration'] > 2) | (df['acceleration'] < -2)])
            st.markdown(create_metric_panel("Harsh Events", str(harsh_events), "Acceleration/Braking", "warning" if harsh_events > 50 else "good"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_panel("Events", "0", "No acceleration data", "info"), unsafe_allow_html=True)
    
    with col7:
        if 'speed' in df.columns:
            speeding_events = len(df[df['speed'] > 80])
            st.markdown(create_metric_panel("Speeding", str(speeding_events), "Over 80 km/h", "warning" if speeding_events > 100 else "good"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_panel("Speeding", "0", "No speed data", "info"), unsafe_allow_html=True)
    
    with col8:
        if 'fuel_level' in df.columns:
            low_fuel = len(df[df['fuel_level'] < 20])
            st.markdown(create_metric_panel("Low Fuel", str(low_fuel), "Below 20%", "warning" if low_fuel > 0 else "good"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_panel("Alerts", "0", "No fuel data", "info"), unsafe_allow_html=True)

def create_performance_charts(df):
    """Create performance monitoring charts"""
    st.markdown("### 📈 Performance Monitoring")
    
    if 'timestamp' in df.columns and 'speed' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Speed over time
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Speed Over Time</div>', unsafe_allow_html=True)
        
        # Group by hour for better visualization
        df['hour'] = df['timestamp'].dt.floor('h')
        speed_by_hour = df.groupby('hour')['speed'].agg(['mean', 'max', 'min']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=speed_by_hour['hour'], y=speed_by_hour['mean'], 
                               mode='lines', name='Average Speed', line=dict(color='#4299e1')))
        fig.add_trace(go.Scatter(x=speed_by_hour['hour'], y=speed_by_hour['max'], 
                               mode='lines', name='Max Speed', line=dict(color='#f56565', dash='dash')))
        fig.add_trace(go.Scatter(x=speed_by_hour['hour'], y=speed_by_hour['min'], 
                               mode='lines', name='Min Speed', line=dict(color='#48bb78', dash='dash')))
        
        fig.update_layout(
            plot_bgcolor='#2d3748',
            paper_bgcolor='#2d3748',
            font_color='white',
            xaxis=dict(gridcolor='#4a5568'),
            yaxis=dict(gridcolor='#4a5568'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Vehicle performance comparison
    if 'vehicle_id' in df.columns and 'speed' in df.columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Vehicle Performance Comparison</div>', unsafe_allow_html=True)
        
        # Calculate performance metrics per vehicle - dynamic based on available columns
        agg_dict = {}
        if 'speed' in df.columns:
            agg_dict['speed'] = ['mean', 'max', 'std']
        if 'acceleration' in df.columns:
            agg_dict['acceleration'] = ['mean', 'std']
        
        if agg_dict:
            vehicle_perf = df.groupby('vehicle_id').agg(agg_dict).round(2)
        else:
            # Fallback if no numeric columns available
            vehicle_perf = df.groupby('vehicle_id').size().to_frame('record_count')
        
        # Flatten column names
        vehicle_perf.columns = ['_'.join(col).strip() for col in vehicle_perf.columns]
        vehicle_perf = vehicle_perf.reset_index()
        
        # Create performance chart
        fig = px.bar(vehicle_perf, x='vehicle_id', y='speed_mean', 
                    title='Average Speed by Vehicle',
                    color='speed_mean',
                    color_continuous_scale='Viridis')
        
        fig.update_layout(
            plot_bgcolor='#2d3748',
            paper_bgcolor='#2d3748',
            font_color='white',
            xaxis=dict(gridcolor='#4a5568'),
            yaxis=dict(gridcolor='#4a5568'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_risk_monitoring(df):
    """Create risk monitoring section"""
    st.markdown("### ⚠️ Risk Monitoring")
    
    if 'risk_score' in df.columns:
        # Risk distribution
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Risk Score Distribution</div>', unsafe_allow_html=True)
        
        fig = px.histogram(df, x='risk_score', nbins=20, 
                          title='Risk Score Distribution',
                          color_discrete_sequence=['#f56565'])
        
        fig.update_layout(
            plot_bgcolor='#2d3748',
            paper_bgcolor='#2d3748',
            font_color='white',
            xaxis=dict(gridcolor='#4a5568'),
            yaxis=dict(gridcolor='#4a5568'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk alerts
        high_risk = df[df['risk_score'] > 0.7]
        if len(high_risk) > 0:
            st.markdown('<div class="alert-panel">', unsafe_allow_html=True)
            st.markdown('<div class="alert-title">🚨 High Risk Alerts</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alert-message">Found {len(high_risk)} high-risk events requiring immediate attention.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show high risk events
            st.dataframe(high_risk[['vehicle_id', 'timestamp', 'risk_score', 'speed']].head(10))
    else:
        st.info("Risk scores not available. Load data with risk information to see risk monitoring.")

def create_geospatial_monitoring(df):
    """Create geospatial monitoring section"""
    st.markdown("### 🗺️ Geospatial Monitoring")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create map
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, 
                      tiles='OpenStreetMap')
        
        # Add markers with color coding
        for _, row in df.iterrows():
            color = 'red' if 'risk_score' in df.columns and row.get('risk_score', 0) > 0.7 else 'blue'
            popup_text = f"Vehicle: {row.get('vehicle_id', 'Unknown')}<br>Speed: {row.get('speed', 'N/A')} km/h"
            if 'timestamp' in df.columns:
                popup_text += f"<br>Time: {row.get('timestamp', 'N/A')}"
            
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fill=True,
                popup=popup_text
            ).add_to(m)
        
        st_folium(m, width=700, height=500)
        
        # Location statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latitude Range", f"{df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        with col2:
            st.metric("Longitude Range", f"{df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
        with col3:
            # Calculate coverage area
            lat_range = df['latitude'].max() - df['latitude'].min()
            lon_range = df['longitude'].max() - df['longitude'].min()
            st.metric("Coverage Area", f"{lat_range:.4f}° × {lon_range:.4f}°")
    else:
        st.info("Location data not available for geospatial monitoring.")

def create_system_status(df):
    """Create system status section"""
    st.markdown("### 🔧 System Status")
    
    # System health indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Data freshness
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            latest_data = df['timestamp'].max()
            time_since_update = datetime.now() - latest_data
            if time_since_update.total_seconds() < 3600:  # Less than 1 hour
                status = "🟢 Online"
                color = "good"
            elif time_since_update.total_seconds() < 86400:  # Less than 1 day
                status = "🟡 Delayed"
                color = "warning"
            else:
                status = "🔴 Offline"
                color = "critical"
        else:
            status = "❓ Unknown"
            color = "info"
        
        st.markdown(create_metric_panel("Data Status", status, "Last update", color), unsafe_allow_html=True)
    
    with col2:
        # Memory usage (simulated)
        memory_usage = len(df) * 0.001  # Simulate memory usage
        st.markdown(create_metric_panel("Memory Usage", f"{memory_usage:.1f} MB", "Data processing", "info"), unsafe_allow_html=True)
    
    with col3:
        # CPU usage (simulated)
        cpu_usage = min(100, len(df) * 0.01)
        status = "good" if cpu_usage < 50 else "warning" if cpu_usage < 80 else "critical"
        st.markdown(create_metric_panel("CPU Usage", f"{cpu_usage:.1f}%", "Processing load", status), unsafe_allow_html=True)
    
    with col4:
        # Active connections
        active_connections = df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0
        st.markdown(create_metric_panel("Active Connections", str(active_connections), "Vehicle connections", "info"), unsafe_allow_html=True)

def create_sidebar():
    """Create monitoring dashboard sidebar"""
    with st.sidebar:
        st.markdown("### 🚗 DOTSURE MONITORING")
        
        # Data source
        st.markdown("#### 📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source:",
            ["📁 Upload CSV", "📂 Demo Data"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader("Upload Telematics CSV", type="csv")
            if uploaded_file is not None:
                if st.button("🚀 Load Data", type="primary"):
                    with st.spinner("Loading data..."):
                        try:
                            df = pd.read_csv(uploaded_file)
                            st.session_state.data = df
                            st.session_state.data_loaded = True
                            st.success("✅ Data loaded successfully!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        elif data_source == "📂 Demo Data":
            if st.button("📂 Load Demo Data", type="primary"):
                with st.spinner("Loading demo data..."):
                    try:
                        # Create comprehensive demo data
                        np.random.seed(42)
                        n_points = 5000
                        
                        demo_data = pd.DataFrame({
                            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='5min'),
                            'vehicle_id': np.random.choice(['V001', 'V002', 'V003', 'V004', 'V005', 'V006', 'V007', 'V008'], n_points),
                            'latitude': 40.7128 + np.random.normal(0, 0.1, n_points),
                            'longitude': -74.0060 + np.random.normal(0, 0.1, n_points),
                            'speed': np.random.normal(65, 25, n_points).clip(0, 120),
                            'acceleration': np.random.normal(0, 3, n_points),
                            'heading': np.random.uniform(0, 360, n_points),
                            'altitude': np.random.normal(100, 50, n_points),
                            'fuel_level': np.random.uniform(20, 100, n_points),
                            'engine_rpm': np.random.normal(2000, 500, n_points).clip(1000, 4000),
                            'engine_temp': np.random.normal(90, 10, n_points).clip(70, 110),
                            'battery_voltage': np.random.normal(12.6, 0.5, n_points).clip(11.5, 14.0),
                            'weather': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], n_points, p=[0.6, 0.25, 0.1, 0.05]),
                            'road_type': np.random.choice(['Highway', 'City', 'Rural'], n_points, p=[0.3, 0.5, 0.2])
                        })
                        
                        # Add risk scores
                        demo_data['risk_score'] = np.random.beta(2, 5, n_points)
                        
                        st.session_state.data = demo_data
                        st.session_state.data_loaded = True
                        st.success("✅ Demo data loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Time range filter
        if st.session_state.get('data_loaded', False):
            st.markdown("---")
            st.markdown("#### ⏰ Time Range")
            
            df = st.session_state.data
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Quick time range buttons
                if st.button("Last Hour"):
                    cutoff_time = df['timestamp'].max() - timedelta(hours=1)
                    df = df[df['timestamp'] > cutoff_time]
                    st.session_state.data = df
                
                if st.button("Last 24 Hours"):
                    cutoff_time = df['timestamp'].max() - timedelta(hours=24)
                    df = df[df['timestamp'] > cutoff_time]
                    st.session_state.data = df
                
                if st.button("Last 7 Days"):
                    cutoff_time = df['timestamp'].max() - timedelta(days=7)
                    df = df[df['timestamp'] > cutoff_time]
                    st.session_state.data = df
                
                # Custom date range
                date_range = st.date_input(
                    "Custom Range",
                    value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
                    min_value=df['timestamp'].min().date(),
                    max_value=df['timestamp'].max().date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
                    st.session_state.data = df
            
            # Vehicle filter
            if 'vehicle_id' in df.columns:
                st.markdown("#### 🚗 Vehicle Filter")
                selected_vehicles = st.multiselect(
                    "Select Vehicles",
                    options=df['vehicle_id'].unique(),
                    default=df['vehicle_id'].unique()
                )
                if selected_vehicles:
                    df = df[df['vehicle_id'].isin(selected_vehicles)]
                    st.session_state.data = df

def main():
    """Main monitoring dashboard function"""
    create_dashboard_header()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Create sidebar
    create_sidebar()
    
    # Main dashboard content
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="alert-panel">
            <div class="alert-title">📊 DOTSURE MONITORING DASHBOARD</div>
            <div class="alert-message">
                Welcome to the comprehensive telematics monitoring system. Load your CSV data or use demo data to begin monitoring your fleet.
                <br><br>
                <strong>Features:</strong>
                <ul>
                    <li>📈 Real-time performance monitoring</li>
                    <li>⚠️ Risk assessment and alerts</li>
                    <li>🗺️ Geospatial tracking and analysis</li>
                    <li>🔧 System health monitoring</li>
                    <li>📊 Comprehensive metrics and KPIs</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    
    # Create dashboard sections
    create_quick_overview(df)
    create_performance_charts(df)
    create_risk_monitoring(df)
    create_geospatial_monitoring(df)
    create_system_status(df)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #718096; border-top: 1px solid #4a5568; margin-top: 3rem;">
        <p><strong>DOTSURE MONITORING DASHBOARD</strong> - Professional Telematics Monitoring</p>
        <p>Built with ❤️ using Streamlit, Python, and Enterprise Technologies</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
