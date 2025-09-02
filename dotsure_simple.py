"""
DOTSURE ENTERPRISE - Simple Production Version for Render
Minimal dependencies, maximum compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DOTSURE ENTERPRISE",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    .metric-card h3 {
        color: #2a5298;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3c72;
        margin: 0;
    }
    
    .metric-card .subtitle {
        color: #666;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def create_header():
    """Create the enterprise header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚗 DOTSURE ENTERPRISE</h1>
        <p>Advanced Telematics & Fleet Management Platform</p>
        <p>AI-Powered Analytics • Real-time Monitoring • Enterprise Integration</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle=""):
    """Create a metric card"""
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <p class="value">{value}</p>
        <p class="subtitle">{subtitle}</p>
    </div>
    """

def create_dashboard_overview(df):
    """Create the main dashboard overview"""
    st.markdown("### 📊 Fleet Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_vehicles = df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0
        st.markdown(create_metric_card("Active Vehicles", f"{total_vehicles}", "Fleet size"), unsafe_allow_html=True)
    
    with col2:
        total_records = len(df)
        st.markdown(create_metric_card("Data Points", f"{total_records:,}", "Records processed"), unsafe_allow_html=True)
    
    with col3:
        if 'speed' in df.columns:
            avg_speed = df['speed'].mean()
            st.markdown(create_metric_card("Avg Speed", f"{avg_speed:.1f} km/h", "Fleet average"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Status", "✅ Active", "System operational"), unsafe_allow_html=True)
    
    with col4:
        if 'risk_score' in df.columns:
            high_risk = len(df[df['risk_score'] > 0.7])
            st.markdown(create_metric_card("High Risk", f"{high_risk}", "Requires attention"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Alerts", "0", "No active alerts"), unsafe_allow_html=True)

def create_analytics_tab(df):
    """Create analytics tab"""
    st.markdown("### 📈 Advanced Analytics")
    
    # Performance Analytics
    st.markdown("#### Performance Analytics")
    
    if 'speed' in df.columns and 'timestamp' in df.columns:
        # Speed over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        speed_over_time = df.groupby(df['timestamp'].dt.hour)['speed'].mean()
        
        fig = px.line(x=speed_over_time.index, y=speed_over_time.values,
                     title="Average Speed by Hour", labels={'x': 'Hour', 'y': 'Speed (km/h)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Vehicle performance comparison
    if 'vehicle_id' in df.columns and 'speed' in df.columns:
        # Build aggregation dictionary dynamically
        agg_dict = {'speed': ['mean', 'max', 'std']}
        
        if 'acceleration' in df.columns:
            agg_dict['acceleration'] = ['mean', 'std']
        
        vehicle_performance = df.groupby('vehicle_id').agg(agg_dict).round(2)
        st.dataframe(vehicle_performance)
    
    # Risk Analysis
    st.markdown("#### Risk Analysis")
    
    if 'risk_score' in df.columns:
        # Risk distribution
        fig = px.histogram(df, x='risk_score', title='Risk Score Distribution',
                         color_discrete_sequence=['#ff6b6b'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk by vehicle
        if 'vehicle_id' in df.columns:
            risk_by_vehicle = df.groupby('vehicle_id')['risk_score'].mean().sort_values(ascending=False)
            fig = px.bar(x=risk_by_vehicle.index, y=risk_by_vehicle.values,
                       title='Average Risk Score by Vehicle')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Risk scores not available. Load data with risk information to see risk analysis.")
    
    # Geospatial Analysis
    st.markdown("#### Geospatial Analysis")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create scatter plot for geospatial data
        fig = px.scatter(df, x='longitude', y='latitude', 
                        color='speed' if 'speed' in df.columns else None,
                        title='Vehicle Locations',
                        labels={'latitude': 'Latitude', 'longitude': 'Longitude'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Location statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude Range", f"{df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        with col2:
            st.metric("Longitude Range", f"{df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    else:
        st.info("Location data not available for geospatial analysis.")

def create_sidebar():
    """Create the sidebar"""
    with st.sidebar:
        st.markdown("### 🚗 DOTSURE ENTERPRISE")
        
        # Data source selection
        st.markdown("#### 📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source:",
            ["📁 Upload CSV", "📂 Demo Data"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV File", type="csv")
            if uploaded_file is not None:
                if st.button("🚀 Load Data", type="primary"):
                    with st.spinner("Processing..."):
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
                        n_points = 2000
                        
                        demo_data = pd.DataFrame({
                            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='30min'),
                            'vehicle_id': np.random.choice(['V001', 'V002', 'V003', 'V004', 'V005'], n_points),
                            'latitude': 40.7128 + np.random.normal(0, 0.05, n_points),
                            'longitude': -74.0060 + np.random.normal(0, 0.05, n_points),
                            'speed': np.random.normal(65, 25, n_points).clip(0, 120),
                            'acceleration': np.random.normal(0, 3, n_points),
                            'heading': np.random.uniform(0, 360, n_points),
                            'altitude': np.random.normal(100, 50, n_points),
                            'fuel_level': np.random.uniform(20, 100, n_points),
                            'engine_rpm': np.random.normal(2000, 500, n_points).clip(1000, 4000),
                            'weather': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], n_points, p=[0.6, 0.25, 0.1, 0.05]),
                            'road_type': np.random.choice(['Highway', 'City', 'Rural'], n_points, p=[0.3, 0.5, 0.2])
                        })
                        
                        st.session_state.data = demo_data
                        st.session_state.data_loaded = True
                        st.success("✅ Demo data loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Quick filters
        if st.session_state.get('data_loaded', False):
            st.markdown("---")
            st.markdown("#### 🔍 Quick Filters")
            
            df = st.session_state.data
            
            # Date filter
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                date_range = st.date_input(
                    "Date Range",
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
                selected_vehicles = st.multiselect(
                    "Vehicles",
                    options=df['vehicle_id'].unique(),
                    default=df['vehicle_id'].unique()
                )
                if selected_vehicles:
                    df = df[df['vehicle_id'].isin(selected_vehicles)]
                    st.session_state.data = df

def main():
    """Main application function"""
    create_header()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Create sidebar
    create_sidebar()
    
    # Main content
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="alert-info">
            <h3>🚗 Welcome to DOTSURE ENTERPRISE</h3>
            <p>Your comprehensive telematics and fleet management platform.</p>
            <p><strong>Core Features:</strong></p>
            <ul>
                <li>📊 Advanced analytics and visualizations</li>
                <li>🗺️ Geospatial analysis and mapping</li>
                <li>📈 Performance metrics and KPIs</li>
                <li>🔍 Data filtering and exploration</li>
                <li>📤 Data export and reporting</li>
                <li>🚀 Real-time dashboards</li>
            </ul>
            <p>Select a data source from the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("### 🚀 Enterprise Capabilities")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Analytics</h3>
                <p class="value">Advanced</p>
                <p class="subtitle">Performance & Risk Analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🗺️ Geospatial</h3>
                <p class="value">Interactive</p>
                <p class="subtitle">Location & Mapping</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Real-time</h3>
                <p class="value">Live Data</p>
                <p class="subtitle">Fleet Monitoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🔧 Enterprise</h3>
                <p class="value">Production</p>
                <p class="subtitle">Deployed on Render</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    df = st.session_state.data
    
    # Main dashboard
    create_dashboard_overview(df)
    
    # Analytics tab
    create_analytics_tab(df)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>DOTSURE ENTERPRISE</strong> - Advanced Telematics Platform</p>
        <p>Built with ❤️ using Streamlit, Python, and Enterprise Technologies</p>
        <p>Deployed on Render • Production Ready</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
