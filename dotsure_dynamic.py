"""
DOTSURE DYNAMIC MONITORING DASHBOARD
Automatically adapts to ANY CSV data structure
Creates filters and displays based on actual CSV columns
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
    page_title="DOTSURE Dynamic Dashboard",
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
    
    /* Data info panel */
    .data-info {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    
    .data-info h4 {
        color: #4299e1;
        margin: 0 0 1rem 0;
    }
    
    .column-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .column-item {
        background: #1a202c;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 3px solid #4299e1;
    }
    
    .column-name {
        font-weight: 600;
        color: #ffffff;
    }
    
    .column-type {
        font-size: 0.8rem;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

def create_dashboard_header():
    """Create Grafana-style dashboard header"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">
            🚗 DOTSURE DYNAMIC DASHBOARD
        </div>
        <div class="dashboard-subtitle">
            Automatically Adapts to ANY CSV Data Structure
        </div>
    </div>
    """, unsafe_allow_html=True)

def analyze_csv_structure(df):
    """Analyze CSV structure and return column information"""
    column_info = {}
    
    for col in df.columns:
        col_data = df[col]
        col_info = {
            'name': col,
            'type': str(col_data.dtype),
            'non_null_count': col_data.count(),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique(),
            'is_numeric': pd.api.types.is_numeric_dtype(col_data),
            'is_datetime': pd.api.types.is_datetime64_any_dtype(col_data),
            'is_categorical': col_data.dtype == 'object' and col_data.nunique() < 20,
            'sample_values': col_data.dropna().head(3).tolist()
        }
        
        # Try to detect if it's a timestamp column
        if col.lower() in ['timestamp', 'time', 'date', 'datetime', 'created_at', 'updated_at']:
            col_info['is_timestamp'] = True
        else:
            col_info['is_timestamp'] = False
            
        # Try to detect if it's a location column
        if col.lower() in ['latitude', 'lat', 'longitude', 'lon', 'lng', 'location']:
            col_info['is_location'] = True
        else:
            col_info['is_location'] = False
            
        # Try to detect if it's a vehicle ID column
        if col.lower() in ['vehicle_id', 'vehicleid', 'vehicle', 'id', 'car_id', 'fleet_id']:
            col_info['is_vehicle_id'] = True
        else:
            col_info['is_vehicle_id'] = False
            
        column_info[col] = col_info
    
    return column_info

def create_data_info_panel(df, column_info):
    """Create data information panel"""
    st.markdown('<div class="data-info">', unsafe_allow_html=True)
    st.markdown('<h4>📊 Data Structure Analysis</h4>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numeric_cols = sum(1 for info in column_info.values() if info['is_numeric'])
        st.metric("Numeric Columns", numeric_cols)
    with col4:
        categorical_cols = sum(1 for info in column_info.values() if info['is_categorical'])
        st.metric("Categorical Columns", categorical_cols)
    
    # Column details
    st.markdown('<div class="column-info">', unsafe_allow_html=True)
    for col, info in column_info.items():
        st.markdown(f'''
        <div class="column-item">
            <div class="column-name">{col}</div>
            <div class="column-type">
                Type: {info['type']} | 
                Non-null: {info['non_null_count']:,} | 
                Unique: {info['unique_count']:,}
            </div>
            <div class="column-type">
                Sample: {info['sample_values'][:2]}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_dynamic_metrics(df, column_info):
    """Create dynamic metrics based on available columns"""
    st.markdown("### 📊 Dynamic Metrics")
    
    # Find key columns
    timestamp_col = None
    vehicle_id_col = None
    location_cols = []
    numeric_cols = []
    
    for col, info in column_info.items():
        if info['is_timestamp']:
            timestamp_col = col
        if info['is_vehicle_id']:
            vehicle_id_col = col
        if info['is_location']:
            location_cols.append(col)
        if info['is_numeric']:
            numeric_cols.append(col)
    
    # Create metrics based on available data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if vehicle_id_col:
            total_vehicles = df[vehicle_id_col].nunique()
            st.metric("Total Vehicles", total_vehicles)
        else:
            st.metric("Total Records", len(df))
    
    with col2:
        if timestamp_col:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                time_span = df[timestamp_col].max() - df[timestamp_col].min()
                uptime_str = f"{time_span.days}d {time_span.seconds//3600}h"
                st.metric("Data Time Span", uptime_str)
            except:
                st.metric("Data Time Span", "N/A")
        else:
            st.metric("Data Points", len(df))
    
    with col3:
        if numeric_cols:
            # Use the first numeric column for average
            first_numeric = numeric_cols[0]
            avg_value = df[first_numeric].mean()
            st.metric(f"Avg {first_numeric.title()}", f"{avg_value:.2f}")
        else:
            st.metric("Categorical Data", "Yes")
    
    with col4:
        if location_cols:
            st.metric("Location Data", "Available")
        else:
            st.metric("Location Data", "Not Available")

def create_dynamic_filters(df, column_info):
    """Create dynamic filters based on available columns"""
    st.markdown("### 🔍 Dynamic Filters")
    
    filtered_df = df.copy()
    
    # Create filters for each column type
    for col, info in column_info.items():
        if info['is_categorical'] and info['unique_count'] > 1:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 20:  # Only show filter if reasonable number of options
                selected_values = st.multiselect(
                    f"Filter by {col}",
                    options=unique_values,
                    default=unique_values
                )
                if selected_values:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
        
        elif info['is_numeric']:
            min_val = df[col].min()
            max_val = df[col].max()
            if not pd.isna(min_val) and not pd.isna(max_val):
                range_values = st.slider(
                    f"Range for {col}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=(float(min_val), float(max_val))
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= range_values[0]) & 
                    (filtered_df[col] <= range_values[1])
                ]
    
    return filtered_df

def create_dynamic_charts(df, column_info):
    """Create dynamic charts based on available columns"""
    st.markdown("### 📈 Dynamic Visualizations")
    
    # Find key columns
    timestamp_col = None
    vehicle_id_col = None
    numeric_cols = []
    
    for col, info in column_info.items():
        if info['is_timestamp']:
            timestamp_col = col
        if info['is_vehicle_id']:
            vehicle_id_col = col
        if info['is_numeric']:
            numeric_cols.append(col)
    
    # Time series chart if timestamp available
    if timestamp_col and numeric_cols:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Time Series Analysis</div>', unsafe_allow_html=True)
        
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Group by hour for better visualization
            df['hour'] = df[timestamp_col].dt.floor('h')
            
            # Create subplot for multiple numeric columns
            if len(numeric_cols) > 1:
                fig = make_subplots(
                    rows=len(numeric_cols), cols=1,
                    subplot_titles=numeric_cols,
                    vertical_spacing=0.1
                )
                
                for i, col in enumerate(numeric_cols[:3]):  # Limit to 3 columns
                    hourly_data = df.groupby('hour')[col].mean().reset_index()
                    fig.add_trace(
                        go.Scatter(x=hourly_data['hour'], y=hourly_data[col], 
                                 mode='lines', name=col),
                        row=i+1, col=1
                    )
            else:
                col = numeric_cols[0]
                hourly_data = df.groupby('hour')[col].agg(['mean', 'max', 'min']).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['mean'], 
                                       mode='lines', name=f'Avg {col}'))
                fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['max'], 
                                       mode='lines', name=f'Max {col}'))
                fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['min'], 
                                       mode='lines', name=f'Min {col}'))
            
            fig.update_layout(
                plot_bgcolor='#2d3748',
                paper_bgcolor='#2d3748',
                font_color='white',
                height=400 * min(len(numeric_cols), 3)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating time series chart: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Vehicle comparison chart if vehicle ID available
    if vehicle_id_col and numeric_cols:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Vehicle Comparison</div>', unsafe_allow_html=True)
        
        try:
            # Use the first numeric column for comparison
            col = numeric_cols[0]
            vehicle_perf = df.groupby(vehicle_id_col)[col].agg(['mean', 'max', 'std']).round(2)
            vehicle_perf = vehicle_perf.reset_index()
            
            fig = px.bar(vehicle_perf, x=vehicle_id_col, y='mean', 
                        title=f'Average {col} by Vehicle',
                        color='mean',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(
                plot_bgcolor='#2d3748',
                paper_bgcolor='#2d3748',
                font_color='white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating vehicle comparison chart: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution charts for numeric columns
    if numeric_cols:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Data Distributions</div>', unsafe_allow_html=True)
        
        # Create histogram for each numeric column
        for col in numeric_cols[:3]:  # Limit to 3 columns
            try:
                fig = px.histogram(df, x=col, nbins=20, title=f'{col} Distribution')
                fig.update_layout(
                    plot_bgcolor='#2d3748',
                    paper_bgcolor='#2d3748',
                    font_color='white',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating histogram for {col}: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_dynamic_map(df, column_info):
    """Create dynamic map if location data is available"""
    location_cols = []
    for col, info in column_info.items():
        if info['is_location']:
            location_cols.append(col)
    
    if len(location_cols) >= 2:
        st.markdown("### 🗺️ Dynamic Map Visualization")
        
        try:
            # Assume first two location columns are lat/lon
            lat_col = location_cols[0]
            lon_col = location_cols[1]
            
            # Filter out invalid coordinates
            valid_coords = df.dropna(subset=[lat_col, lon_col])
            valid_coords = valid_coords[
                (valid_coords[lat_col] >= -90) & (valid_coords[lat_col] <= 90) &
                (valid_coords[lon_col] >= -180) & (valid_coords[lon_col] <= 180)
            ]
            
            if len(valid_coords) > 0:
                center_lat = valid_coords[lat_col].mean()
                center_lon = valid_coords[lon_col].mean()
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                
                # Add markers
                for _, row in valid_coords.head(1000).iterrows():  # Limit to 1000 points
                    popup_text = f"Lat: {row[lat_col]:.4f}<br>Lon: {row[lon_col]:.4f}"
                    
                    # Add other column data to popup
                    for col in df.columns:
                        if col not in [lat_col, lon_col]:
                            popup_text += f"<br>{col}: {row[col]}"
                    
                    folium.CircleMarker(
                        [row[lat_col], row[lon_col]],
                        radius=5,
                        color='blue',
                        fill=True,
                        popup=popup_text
                    ).add_to(m)
                
                st_folium(m, width=700, height=500)
                
                # Location statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latitude Range", f"{valid_coords[lat_col].min():.4f} to {valid_coords[lat_col].max():.4f}")
                with col2:
                    st.metric("Longitude Range", f"{valid_coords[lon_col].min():.4f} to {valid_coords[lon_col].max():.4f}")
                with col3:
                    st.metric("Valid Points", len(valid_coords))
            else:
                st.info("No valid coordinate data found for mapping.")
                
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
    else:
        st.info("Location data not available for mapping.")

def create_sidebar():
    """Create dynamic sidebar"""
    with st.sidebar:
        st.markdown("### 🚗 DOTSURE DYNAMIC")
        
        # Data source
        st.markdown("#### 📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source:",
            ["📁 Upload CSV", "📂 Demo Data"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader("Upload ANY CSV File", type="csv")
            if uploaded_file is not None:
                if st.button("🚀 Load & Analyze Data", type="primary"):
                    with st.spinner("Loading and analyzing data..."):
                        try:
                            df = pd.read_csv(uploaded_file)
                            column_info = analyze_csv_structure(df)
                            st.session_state.data = df
                            st.session_state.column_info = column_info
                            st.session_state.data_loaded = True
                            st.success("✅ Data loaded and analyzed successfully!")
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
                            'vehicle_id': np.random.choice(['V001', 'V002', 'V003', 'V004', 'V005'], n_points),
                            'latitude': 40.7128 + np.random.normal(0, 0.1, n_points),
                            'longitude': -74.0060 + np.random.normal(0, 0.1, n_points),
                            'speed': np.random.normal(65, 25, n_points).clip(0, 120),
                            'acceleration': np.random.normal(0, 3, n_points),
                            'fuel_level': np.random.uniform(20, 100, n_points),
                            'engine_temp': np.random.normal(90, 10, n_points).clip(70, 110),
                            'weather': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], n_points, p=[0.6, 0.25, 0.1, 0.05]),
                            'road_type': np.random.choice(['Highway', 'City', 'Rural'], n_points, p=[0.3, 0.5, 0.2])
                        })
                        
                        column_info = analyze_csv_structure(demo_data)
                        st.session_state.data = demo_data
                        st.session_state.column_info = column_info
                        st.session_state.data_loaded = True
                        st.success("✅ Demo data loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Show data info if loaded
        if st.session_state.get('data_loaded', False):
            st.markdown("---")
            st.markdown("#### 📋 Data Summary")
            df = st.session_state.data
            column_info = st.session_state.column_info
            
            st.write(f"**Records:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            
            # Show detected column types
            timestamp_cols = [col for col, info in column_info.items() if info['is_timestamp']]
            vehicle_cols = [col for col, info in column_info.items() if info['is_vehicle_id']]
            location_cols = [col for col, info in column_info.items() if info['is_location']]
            numeric_cols = [col for col, info in column_info.items() if info['is_numeric']]
            
            if timestamp_cols:
                st.write(f"**Time:** {timestamp_cols[0]}")
            if vehicle_cols:
                st.write(f"**Vehicle ID:** {vehicle_cols[0]}")
            if location_cols:
                st.write(f"**Location:** {', '.join(location_cols)}")
            if numeric_cols:
                st.write(f"**Numeric:** {len(numeric_cols)} columns")

def main():
    """Main dynamic dashboard function"""
    create_dashboard_header()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Create sidebar
    create_sidebar()
    
    # Main dashboard content
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="data-info">
            <h4>🚀 DOTSURE DYNAMIC DASHBOARD</h4>
            <p>Upload ANY CSV file and the dashboard will automatically:</p>
            <ul>
                <li>🔍 Analyze the data structure</li>
                <li>📊 Create appropriate visualizations</li>
                <li>🔧 Generate dynamic filters</li>
                <li>🗺️ Show maps if location data exists</li>
                <li>📈 Adapt charts to your data types</li>
            </ul>
            <p><strong>No configuration needed - just upload and go!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    column_info = st.session_state.column_info
    
    # Create dashboard sections
    create_data_info_panel(df, column_info)
    create_dynamic_metrics(df, column_info)
    
    # Apply filters
    filtered_df = create_dynamic_filters(df, column_info)
    
    # Update session state with filtered data
    st.session_state.data = filtered_df
    
    # Create visualizations
    create_dynamic_charts(filtered_df, column_info)
    create_dynamic_map(filtered_df, column_info)
    
    # Show filtered data
    st.markdown("### 📋 Filtered Data Preview")
    st.dataframe(filtered_df.head(100))
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #718096; border-top: 1px solid #4a5568; margin-top: 3rem;">
        <p><strong>DOTSURE DYNAMIC DASHBOARD</strong> - Automatically Adapts to Your Data</p>
        <p>Built with ❤️ using Streamlit and Python</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
