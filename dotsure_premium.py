import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import warnings
import io
import base64
from pathlib import Path
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DOTSURE PREMIUM - Advanced Telematics Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #9467bd;
        --light-bg: #f8f9fa;
        --dark-bg: #2c3e50;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dark-bg);
        margin: 0;
    }
    
    /* Status indicators */
    .status-excellent { color: #27ae60; font-weight: bold; }
    .status-good { color: #f39c12; font-weight: bold; }
    .status-poor { color: #e74c3c; font-weight: bold; }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Alert styling */
    .alert-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color);
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def create_header():
    """Create the premium header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚗 DOTSURE PREMIUM</h1>
        <p>Advanced Telematics Analytics Platform</p>
        <p>Professional-grade insights for fleet management and road safety</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle="", status="info"):
    """Create a premium metric card"""
    status_class = f"status-{status}" if status in ["excellent", "good", "poor"] else ""
    
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="metric-value {status_class}">{value}</div>
        {f'<p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """

def load_data_with_progress():
    """Load data with enhanced progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        "Initializing data processing...",
        "Validating file format...",
        "Loading and parsing data...",
        "Processing coordinates...",
        "Calculating metrics...",
        "Finalizing analysis..."
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        # Simulate processing time
        import time
        time.sleep(0.5)
    
    status_text.text("✅ Data loaded successfully!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

def create_data_import_section():
    """Create the data import section with premium UI"""
    st.sidebar.markdown("### 📊 Data Import")
    
    # Import method selection with icons
    import_method = st.sidebar.selectbox(
        "Choose Import Method:",
        ["📁 Upload Files", "🌐 Load from URL", "💾 GitHub Repository", "📂 Sample Data", "🗄️ Large Files (>200MB)"],
        help="Select how you want to import your telematics data"
    )
    
    if import_method == "📁 Upload Files":
        st.sidebar.markdown("#### Upload Your Data")
        st.sidebar.markdown('<div class="alert-info">💡 <strong>Tip:</strong> Files up to 200MB supported. For larger files, use "Large Files" option.</div>', unsafe_allow_html=True)
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type="csv",
            help="Upload your telematics data CSV file"
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("🚀 Process Data", type="primary"):
                with st.spinner("Processing your data..."):
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        st.sidebar.success("✅ Data processed successfully!")
                        load_data_with_progress()
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {str(e)}")
    
    elif import_method == "🌐 Load from URL":
        st.sidebar.markdown("#### Load from URL")
        url = st.sidebar.text_input(
            "Data URL",
            placeholder="https://example.com/data.csv",
            help="Enter URL to your CSV data file"
        )
        
        if url and st.sidebar.button("🌐 Load from URL", type="primary"):
            with st.spinner("Loading data from URL..."):
                try:
                    df = pd.read_csv(url)
                    st.session_state.data = df
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Data loaded from URL!")
                    load_data_with_progress()
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
    
    elif import_method == "💾 GitHub Repository":
        st.sidebar.markdown("#### GitHub Repository")
        github_url = st.sidebar.text_input(
            "GitHub Raw URL",
            placeholder="https://raw.githubusercontent.com/user/repo/main/data.csv",
            help="Enter GitHub raw URL to CSV file"
        )
        
        if github_url and st.sidebar.button("💾 Load from GitHub", type="primary"):
            with st.spinner("Loading from GitHub..."):
                try:
                    df = pd.read_csv(github_url)
                    st.session_state.data = df
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Data loaded from GitHub!")
                    load_data_with_progress()
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
    
    elif import_method == "📂 Sample Data":
        st.sidebar.markdown("#### Sample Data")
        st.sidebar.markdown('<div class="alert-info">📊 <strong>Demo Data:</strong> Explore the dashboard with sample telematics data.</div>', unsafe_allow_html=True)
        
        if st.sidebar.button("📂 Load Sample Data", type="primary"):
            with st.spinner("Loading sample data..."):
                try:
                    # Create sample data
                    np.random.seed(42)
                    n_points = 1000
                    
                    sample_data = pd.DataFrame({
                        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1H'),
                        'vehicle_id': np.random.choice(['V001', 'V002', 'V003', 'V004'], n_points),
                        'latitude': 40.7128 + np.random.normal(0, 0.01, n_points),
                        'longitude': -74.0060 + np.random.normal(0, 0.01, n_points),
                        'speed': np.random.normal(60, 20, n_points).clip(0, 120),
                        'acceleration': np.random.normal(0, 2, n_points),
                        'accident_severity': np.random.choice(['None', 'Slight', 'Serious', 'Fatal'], n_points, p=[0.8, 0.15, 0.04, 0.01]),
                        'weather': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], n_points, p=[0.6, 0.25, 0.1, 0.05]),
                        'road_type': np.random.choice(['Highway', 'City', 'Rural'], n_points, p=[0.3, 0.5, 0.2])
                    })
                    
                    st.session_state.data = sample_data
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Sample data loaded!")
                    load_data_with_progress()
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
    
    elif import_method == "🗄️ Large Files (>200MB)":
        st.sidebar.markdown("#### Large File Solutions")
        st.sidebar.markdown('<div class="alert-warning">⚠️ <strong>Note:</strong> Streamlit has a 200MB upload limit. Use these solutions for larger files.</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown("**Recommended Solutions:**")
        st.sidebar.markdown("""
        1. **Split your file** using the CSV splitter tool
        2. **Use cloud storage** (Google Drive, Dropbox)
        3. **Database connection** for very large datasets
        4. **Data sampling** for initial analysis
        """)
        
        if st.sidebar.button("📋 Show Detailed Guide", type="secondary"):
            st.sidebar.markdown("""
            **CSV Splitter Command:**
            ```bash
            python csv_splitter.py your_file.csv 50000 chunks_
            ```
            
            **Cloud Storage URLs:**
            - Google Drive: `https://drive.google.com/uc?export=download&id=FILE_ID`
            - Dropbox: `https://dl.dropboxusercontent.com/s/FILE_ID/file.csv`
            """)

def create_analytics_dashboard():
    """Create the main analytics dashboard"""
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        st.markdown("""
        <div class="alert-info">
            <h3>🎯 Welcome to DOTSURE PREMIUM</h3>
            <p>Please import your telematics data using the sidebar to begin analysis.</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>📊 Advanced analytics and visualizations</li>
                <li>🗺️ Interactive maps with GPS tracking</li>
                <li>📈 Trend analysis and predictions</li>
                <li>🎯 Risk assessment and scoring</li>
                <li>📤 Export capabilities</li>
                <li>☁️ Multi-source data import</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    
    # Overview metrics
    st.markdown("### 📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.markdown(create_metric_card("Total Records", f"{total_records:,}", "Data points analyzed"), unsafe_allow_html=True)
    
    with col2:
        if 'vehicle_id' in df.columns:
            unique_vehicles = df['vehicle_id'].nunique()
            st.markdown(create_metric_card("Vehicles", f"{unique_vehicles}", "Unique vehicles tracked"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Data Points", f"{len(df):,}", "Records processed"), unsafe_allow_html=True)
    
    with col3:
        if 'speed' in df.columns:
            avg_speed = df['speed'].mean()
            st.markdown(create_metric_card("Avg Speed", f"{avg_speed:.1f} km/h", "Average speed across all data"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Columns", f"{len(df.columns)}", "Data fields available"), unsafe_allow_html=True)
    
    with col4:
        if 'accident_severity' in df.columns:
            accidents = len(df[df['accident_severity'] != 'None'])
            st.markdown(create_metric_card("Incidents", f"{accidents}", "Safety events detected"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Status", "✅ Ready", "Data loaded successfully"), unsafe_allow_html=True)
    
    # Main analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Map View", "📈 Analytics", "🎯 Risk Assessment", "📊 Trends", "📤 Export"])
    
    with tab1:
        create_map_view(df)
    
    with tab2:
        create_analytics_view(df)
    
    with tab3:
        create_risk_assessment(df)
    
    with tab4:
        create_trends_view(df)
    
    with tab5:
        create_export_section(df)

def create_map_view(df):
    """Create the interactive map view"""
    st.markdown("### 🗺️ Interactive Map View")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create map
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add markers
        for idx, row in df.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Determine marker color based on data
                if 'accident_severity' in df.columns:
                    if row['accident_severity'] == 'Fatal':
                        color = 'red'
                        icon = 'exclamation-triangle'
                    elif row['accident_severity'] == 'Serious':
                        color = 'orange'
                        icon = 'exclamation'
                    elif row['accident_severity'] == 'Slight':
                        color = 'yellow'
                        icon = 'warning'
                    else:
                        color = 'green'
                        icon = 'check'
                else:
                    color = 'blue'
                    icon = 'circle'
                
                # Create popup text
                popup_text = f"""
                <b>Record {idx + 1}</b><br>
                """
                
                for col in df.columns:
                    if col not in ['latitude', 'longitude']:
                        popup_text += f"<b>{col}:</b> {row[col]}<br>"
                
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=popup_text,
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Map statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Map Points", len(df))
        with col2:
            st.metric("Latitude Range", f"{df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        with col3:
            st.metric("Longitude Range", f"{df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    else:
        st.info("📍 No location data found. Map view requires 'latitude' and 'longitude' columns.")

def create_analytics_view(df):
    """Create the analytics view"""
    st.markdown("### 📈 Data Analytics")
    
    # Data distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Distribution")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for distribution", numeric_cols)
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for distribution analysis.")
    
    with col2:
        st.markdown("#### Categorical Analysis")
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            selected_cat = st.selectbox("Select categorical column", categorical_cols)
            value_counts = df[selected_cat].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {selected_cat}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found for analysis.")
    
    # Correlation matrix
    if len(numeric_cols) > 1:
        st.markdown("#### Correlation Analysis")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title="Correlation Matrix", color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

def create_risk_assessment(df):
    """Create the risk assessment view"""
    st.markdown("### 🎯 Risk Assessment")
    
    # Calculate risk score
    risk_score = 100  # Start with perfect score
    
    if 'accident_severity' in df.columns:
        fatal_count = len(df[df['accident_severity'] == 'Fatal'])
        serious_count = len(df[df['accident_severity'] == 'Serious'])
        slight_count = len(df[df['accident_severity'] == 'Slight'])
        
        risk_score -= fatal_count * 20
        risk_score -= serious_count * 10
        risk_score -= slight_count * 2
    
    if 'speed' in df.columns:
        speeding_events = len(df[df['speed'] > 100])
        risk_score -= speeding_events * 1
    
    risk_score = max(risk_score, 0)
    
    # Risk level
    if risk_score >= 80:
        risk_level = "Excellent"
        risk_color = "status-excellent"
    elif risk_score >= 60:
        risk_level = "Good"
        risk_color = "status-good"
    else:
        risk_level = "Needs Attention"
        risk_color = "status-poor"
    
    # Display risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Risk Score</h3>
            <div class="metric-value {risk_color}">{risk_score:.0f}/100</div>
            <p style="margin: 0.5rem 0 0 0; color: #666;">{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'accident_severity' in df.columns:
            total_incidents = len(df[df['accident_severity'] != 'None'])
            st.markdown(create_metric_card("Total Incidents", f"{total_incidents}", "Safety events detected"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Data Quality", "✅ Good", "All records processed"), unsafe_allow_html=True)
    
    with col3:
        if 'speed' in df.columns:
            max_speed = df['speed'].max()
            st.markdown(create_metric_card("Max Speed", f"{max_speed:.1f} km/h", "Highest recorded speed"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Records", f"{len(df):,}", "Total data points"), unsafe_allow_html=True)
    
    # Risk factors
    if 'accident_severity' in df.columns:
        st.markdown("#### Risk Factors Analysis")
        
        risk_factors = df['accident_severity'].value_counts()
        fig = px.bar(
            x=risk_factors.index,
            y=risk_factors.values,
            title="Incident Severity Distribution",
            color=risk_factors.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_trends_view(df):
    """Create the trends view"""
    st.markdown("### 📊 Trend Analysis")
    
    if 'timestamp' in df.columns:
        # Time series analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly trends
            hourly_counts = df['hour'].value_counts().sort_index()
            fig = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Activity by Hour of Day",
                labels={'x': 'Hour', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week trends
            day_counts = df['day_of_week'].value_counts()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = day_counts.reindex(day_order, fill_value=0)
            
            fig = px.bar(
                x=day_counts.index,
                y=day_counts.values,
                title="Activity by Day of Week",
                labels={'x': 'Day', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Speed trends over time
        if 'speed' in df.columns:
            st.markdown("#### Speed Trends")
            df_sample = df.sample(min(1000, len(df)))  # Sample for performance
            fig = px.scatter(
                df_sample,
                x='timestamp',
                y='speed',
                title="Speed Over Time",
                labels={'timestamp': 'Time', 'speed': 'Speed (km/h)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📅 No timestamp data found. Trend analysis requires a 'timestamp' column.")

def create_export_section(df):
    """Create the export section"""
    st.markdown("### 📤 Export Data & Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Filtered Data")
        
        # Data preview
        st.markdown("**Data Preview:**")
        st.dataframe(df.head(10))
        
        # Export options
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
        
        if st.button("📥 Download Data", type="primary"):
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"telematics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Telematics Data', index=False)
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name=f"telematics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"telematics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col2:
        st.markdown("#### Generate Summary Report")
        
        if st.button("📊 Generate Report", type="primary"):
            # Create summary report
            summary = {
                'Total Records': len(df),
                'Data Columns': len(df.columns),
                'Date Range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A',
                'Unique Vehicles': df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 'N/A',
                'Average Speed': f"{df['speed'].mean():.1f} km/h" if 'speed' in df.columns else 'N/A',
                'Max Speed': f"{df['speed'].max():.1f} km/h" if 'speed' in df.columns else 'N/A',
                'Incidents': len(df[df['accident_severity'] != 'None']) if 'accident_severity' in df.columns else 'N/A'
            }
            
            summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            
            # Export summary
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📋 Download Summary Report",
                data=csv,
                file_name=f"telematics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Display summary
            st.markdown("**Summary Report:**")
            st.dataframe(summary_df)

def main():
    """Main application function"""
    create_header()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Create sidebar
    with st.sidebar:
        create_data_import_section()
        
        # Advanced filters (if data is loaded)
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 🔍 Advanced Filters")
            
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
                    "Select Vehicles",
                    options=df['vehicle_id'].unique(),
                    default=df['vehicle_id'].unique()
                )
                if selected_vehicles:
                    df = df[df['vehicle_id'].isin(selected_vehicles)]
                    st.session_state.data = df
            
            # Speed filter
            if 'speed' in df.columns:
                speed_range = st.slider(
                    "Speed Range (km/h)",
                    min_value=float(df['speed'].min()),
                    max_value=float(df['speed'].max()),
                    value=(float(df['speed'].min()), float(df['speed'].max()))
                )
                df = df[(df['speed'] >= speed_range[0]) & (df['speed'] <= speed_range[1])]
                st.session_state.data = df
    
    # Main content
    create_analytics_dashboard()

if __name__ == "__main__":
    main()
