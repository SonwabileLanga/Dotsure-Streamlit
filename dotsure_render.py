"""
DOTSURE ENTERPRISE - Render Production Version
Optimized for Render deployment with proper environment handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import logging
import os
from datetime import datetime, timedelta
import json

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules with error handling
try:
    from database_manager import DatabaseManager, create_database_ui
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database module not available: {e}")
    DATABASE_AVAILABLE = False

try:
    from azure_integration import AzureServicesManager, TelematicsDataPipeline, create_azure_ui
    AZURE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Azure module not available: {e}")
    AZURE_AVAILABLE = False

try:
    from api_integration import APIManager, create_api_ui
    API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"API module not available: {e}")
    API_AVAILABLE = False

try:
    from ai_ml_engine import RiskScoringEngine, EventDetectionEngine, AlertingSystem, create_ai_ml_ui
    AI_ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI/ML module not available: {e}")
    AI_ML_AVAILABLE = False

try:
    from tecdoc_integration import TecDocAPI, VehicleDataManager, create_tecdoc_ui
    TECDOC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TecDoc module not available: {e}")
    TECDOC_AVAILABLE = False

# Page configuration for production
st.set_page_config(
    page_title="DOTSURE ENTERPRISE",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise styling
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
    
    .status-excellent { border-left-color: #28a745; }
    .status-good { border-left-color: #17a2b8; }
    .status-poor { border-left-color: #dc3545; }
    
    .alert-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
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
    
    .hide-footer {
        visibility: hidden;
    }
    
    .hide-header {
        visibility: hidden;
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

def create_metric_card(title, value, subtitle="", status="info"):
    """Create an enterprise metric card"""
    status_class = f"status-{status}" if status in ["excellent", "good", "poor"] else ""
    
    return f"""
    <div class="metric-card {status_class}">
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

def create_advanced_analytics(df):
    """Create advanced analytics section"""
    st.markdown("### 📈 Advanced Analytics")
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Analysis", "Geospatial", "Predictions"])
    
    with tab1:
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
            # Build aggregation dictionary dynamically based on available columns
            agg_dict = {'speed': ['mean', 'max', 'std']}
            
            if 'acceleration' in df.columns:
                agg_dict['acceleration'] = ['mean', 'std']
            
            vehicle_performance = df.groupby('vehicle_id').agg(agg_dict).round(2)
            
            st.dataframe(vehicle_performance)
    
    with tab2:
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
            st.info("Risk scores not available. Please run the AI/ML engine first.")
    
    with tab3:
        st.markdown("#### Geospatial Analysis")
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Create interactive map
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add markers
            for _, row in df.iterrows():
                color = 'red' if 'risk_score' in df.columns and row.get('risk_score', 0) > 0.7 else 'blue'
                popup_text = f"Vehicle: {row.get('vehicle_id', 'Unknown')}"
                if 'speed' in df.columns:
                    popup_text += f"<br>Speed: {row.get('speed', 'N/A')} km/h"
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
        else:
            st.info("Location data not available for geospatial analysis.")
    
    with tab4:
        st.markdown("#### Predictive Analytics")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if 'risk_score' in df.columns:
                # Risk trend prediction
                daily_risk = df.groupby(df['timestamp'].dt.date)['risk_score'].mean()
                
                # Simple trend line
                x = np.arange(len(daily_risk))
                y = daily_risk.values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_risk.index, y=daily_risk.values, 
                                       mode='lines+markers', name='Actual Risk'))
                fig.add_trace(go.Scatter(x=daily_risk.index, y=p(x), 
                                       mode='lines', name='Trend', line=dict(dash='dash')))
                
                fig.update_layout(title="Risk Score Trend & Prediction")
                st.plotly_chart(fig, use_container_width=True)
            elif 'speed' in df.columns:
                # Speed trend prediction
                daily_speed = df.groupby(df['timestamp'].dt.date)['speed'].mean()
                
                # Simple trend line
                x = np.arange(len(daily_speed))
                y = daily_speed.values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_speed.index, y=daily_speed.values, 
                                       mode='lines+markers', name='Actual Speed'))
                fig.add_trace(go.Scatter(x=daily_speed.index, y=p(x), 
                                       mode='lines', name='Trend', line=dict(dash='dash')))
                
                fig.update_layout(title="Speed Trend & Prediction")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for predictions. Please ensure risk scores or speed data with timestamps are available.")
        else:
            st.info("Timestamp data required for predictions.")

def create_enterprise_sidebar():
    """Create the enterprise sidebar"""
    with st.sidebar:
        st.markdown("### 🚗 DOTSURE ENTERPRISE")
        
        # Data source selection
        st.markdown("#### 📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source:",
            ["📁 Upload CSV", "🗄️ Database", "☁️ Azure Storage", "🌐 API Integration", "📂 Demo Data"]
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

def create_module_ui(module_name, create_ui_func, available):
    """Create UI for a module with availability check"""
    if available:
        create_ui_func()
    else:
        st.markdown(f"### {module_name}")
        st.warning(f"⚠️ {module_name} module is not available in this deployment. Some dependencies may be missing.")

def main():
    """Main application function"""
    create_header()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Create sidebar
    create_enterprise_sidebar()
    
    # Main content
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="alert-info">
            <h3>🚗 Welcome to DOTSURE ENTERPRISE</h3>
            <p>Your comprehensive telematics and fleet management platform.</p>
            <p><strong>Enterprise Features:</strong></p>
            <ul>
                <li>🗄️ Multi-database integration (SQLite, PostgreSQL, MySQL)</li>
                <li>☁️ Azure services integration (Blob Storage, Cognitive Services)</li>
                <li>🌐 Third-party API integration (Weather, Traffic, Geocoding)</li>
                <li>🔧 TecDoc Catalog integration (VIN decoding, Parts search)</li>
                <li>🤖 AI/ML-powered risk scoring and event detection</li>
                <li>📊 Advanced analytics and predictive insights</li>
                <li>⚠️ Intelligent alerting and monitoring</li>
                <li>📈 Real-time dashboards and reporting</li>
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
                <h3>🗄️ Database Integration</h3>
                <p class="value">Multi-DB</p>
                <p class="subtitle">SQLite, PostgreSQL, MySQL</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>☁️ Azure Services</h3>
                <p class="value">Cloud-Ready</p>
                <p class="subtitle">Blob Storage, Cognitive Services</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 AI/ML Engine</h3>
                <p class="value">Smart Analytics</p>
                <p class="subtitle">Risk scoring, Event detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🔧 TecDoc Catalog</h3>
                <p class="value">Auto Parts</p>
                <p class="subtitle">VIN decoding, Parts search</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    df = st.session_state.data
    
    # Main dashboard
    create_dashboard_overview(df)
    
    # Main tabs - only show available modules
    available_tabs = ["📊 Analytics"]
    tab_functions = [create_advanced_analytics]
    
    if DATABASE_AVAILABLE:
        available_tabs.append("🗄️ Database")
        tab_functions.append(lambda: create_module_ui("Database Management", create_database_ui, DATABASE_AVAILABLE))
    
    if AZURE_AVAILABLE:
        available_tabs.append("☁️ Azure")
        tab_functions.append(lambda: create_module_ui("Azure Services", create_azure_ui, AZURE_AVAILABLE))
    
    if API_AVAILABLE:
        available_tabs.append("🌐 APIs")
        tab_functions.append(lambda: create_module_ui("API Integration", create_api_ui, API_AVAILABLE))
    
    if AI_ML_AVAILABLE:
        available_tabs.append("🤖 AI/ML")
        tab_functions.append(lambda: create_module_ui("AI/ML Engine", create_ai_ml_ui, AI_ML_AVAILABLE))
    
    if TECDOC_AVAILABLE:
        available_tabs.append("🔧 TecDoc")
        tab_functions.append(lambda: create_module_ui("TecDoc Catalog", create_tecdoc_ui, TECDOC_AVAILABLE))
    
    # Create tabs dynamically
    tabs = st.tabs(available_tabs)
    
    for i, (tab, func) in enumerate(zip(tabs, tab_functions)):
        with tab:
            func()
    
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
