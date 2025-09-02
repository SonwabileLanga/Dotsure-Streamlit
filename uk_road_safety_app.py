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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DOTSURE STREAMLIT - UK Road Safety Analytics",
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
    .severity-fatal {
        color: #ff0000;
        font-weight: bold;
    }
    .severity-serious {
        color: #ff8800;
        font-weight: bold;
    }
    .severity-slight {
        color: #00aa44;
        font-weight: bold;
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

def load_uk_data():
    """Load UK road safety data with multiple encoding attempts"""
    try:
        # Load accident data with multiple encoding attempts
        st.info("Loading accident data... This may take a moment for large files.")
        accidents_df = None
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                accidents_df = pd.read_csv('Accident_Information.csv', encoding=encoding)
                st.success(f"Accident data loaded successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if accidents_df is None:
            st.error("Could not load accident data with any supported encoding")
            return None, None
        
        # Load vehicle data with multiple encoding attempts
        st.info("Loading vehicle data... This may take a moment for large files.")
        vehicles_df = None
        
        for encoding in encodings:
            try:
                vehicles_df = pd.read_csv('Vehicle_Information.csv', encoding=encoding)
                st.success(f"Vehicle data loaded successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if vehicles_df is None:
            st.error("Could not load vehicle data with any supported encoding")
            return None, None
        
        return accidents_df, vehicles_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def preprocess_uk_data(accidents_df, vehicles_df):
    """Preprocess UK road safety data"""
    try:
        # Clean and process accident data
        accidents_df = accidents_df.copy()
        
        # Convert date and time with better error handling
        accidents_df['Date'] = pd.to_datetime(accidents_df['Date'], errors='coerce')
        accidents_df['Time'] = pd.to_datetime(accidents_df['Time'], format='%H:%M', errors='coerce').dt.time
        
        # Create datetime column more safely
        accidents_df['DateTime'] = pd.to_datetime(
            accidents_df['Date'].astype(str) + ' ' + accidents_df['Time'].astype(str), 
            errors='coerce'
        )
        
        # Clean coordinates with better error handling
        accidents_df['Latitude'] = pd.to_numeric(accidents_df['Latitude'], errors='coerce')
        accidents_df['Longitude'] = pd.to_numeric(accidents_df['Longitude'], errors='coerce')
        
        # Remove rows with invalid coordinates
        initial_count = len(accidents_df)
        accidents_df = accidents_df.dropna(subset=['Latitude', 'Longitude'])
        final_count = len(accidents_df)
        
        if initial_count != final_count:
            st.warning(f"Removed {initial_count - final_count} rows with invalid coordinates")
        
        # Clean vehicle data
        vehicles_df = vehicles_df.copy()
        
        # Merge datasets
        merged_df = pd.merge(accidents_df, vehicles_df, on='Accident_Index', how='left')
        
        st.success(f"Data preprocessing completed. {len(merged_df)} records ready for analysis.")
        
        return merged_df
        
    except Exception as e:
        st.error(f"Error during data preprocessing: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe on error

def create_accident_severity_map(df):
    """Create map showing accident severity"""
    if df.empty:
        return None
    
    # Calculate center point
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Color mapping for severity
    severity_colors = {
        'Fatal': 'red',
        'Serious': 'orange', 
        'Slight': 'yellow'
    }
    
    # Add markers for each accident
    for idx, row in df.iterrows():
        severity = row['Accident_Severity']
        color = severity_colors.get(severity, 'blue')
        
        # Create popup text
        popup_text = f"""
        <b>Accident Index:</b> {row['Accident_Index']}<br>
        <b>Severity:</b> {severity}<br>
        <b>Date:</b> {row['Date']}<br>
        <b>Time:</b> {row['Time']}<br>
        <b>Vehicles:</b> {row['Number_of_Vehicles']}<br>
        <b>Casualties:</b> {row['Number_of_Casualties']}<br>
        <b>Speed Limit:</b> {row['Speed_limit']} mph<br>
        <b>Weather:</b> {row['Weather_Conditions']}<br>
        <b>Road Type:</b> {row['Road_Type']}
        """
        
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=popup_text,
            icon=folium.Icon(color=color, icon='car-crash', prefix='fa')
        ).add_to(m)
    
    return m

def analyze_accident_patterns(df):
    """Analyze accident patterns and risk factors"""
    analysis = {}
    
    # Severity distribution
    analysis['severity_dist'] = df['Accident_Severity'].value_counts()
    
    # Time patterns
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
    analysis['hourly_dist'] = df['Hour'].value_counts().sort_index()
    
    # Day of week patterns
    analysis['day_dist'] = df['Day_of_Week'].value_counts()
    
    # Weather conditions
    analysis['weather_dist'] = df['Weather_Conditions'].value_counts()
    
    # Road conditions
    analysis['road_surface_dist'] = df['Road_Surface_Conditions'].value_counts()
    
    # Speed limit analysis
    analysis['speed_limit_dist'] = df['Speed_limit'].value_counts().sort_index()
    
    # Vehicle types (if available)
    if 'Vehicle_Type' in df.columns:
        analysis['vehicle_type_dist'] = df['Vehicle_Type'].value_counts()
    
    return analysis

def calculate_risk_score(df):
    """Calculate overall risk score for the dataset"""
    score = 100  # Start with perfect score
    
    # Fatal accidents penalty
    fatal_count = len(df[df['Accident_Severity'] == 'Fatal'])
    score -= fatal_count * 20  # 20 points per fatal accident
    
    # Serious accidents penalty
    serious_count = len(df[df['Accident_Severity'] == 'Serious'])
    score -= serious_count * 10  # 10 points per serious accident
    
    # Slight accidents penalty
    slight_count = len(df[df['Accident_Severity'] == 'Slight'])
    score -= slight_count * 2  # 2 points per slight accident
    
    # High casualty accidents
    high_casualty = len(df[df['Number_of_Casualties'] > 2])
    score -= high_casualty * 5  # 5 points per high casualty accident
    
    return max(score, 0)  # Ensure score doesn't go below 0

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 DOTSURE STREAMLIT</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">UK Road Safety Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Data Analysis")
    
    # Advanced Filters Section
    st.sidebar.header("🔍 Advanced Filters")
    
    # Initialize session state for filters
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    
    # Load data button
    if st.sidebar.button("🔄 Load UK Road Safety Data"):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Starting data load...")
            progress_bar.progress(10)
            
            accidents_df, vehicles_df = load_uk_data()
            progress_bar.progress(50)
            
            if accidents_df is not None and vehicles_df is not None:
                status_text.text("Data loaded successfully! Processing...")
                progress_bar.progress(75)
                
                st.session_state.accidents_df = accidents_df
                st.session_state.vehicles_df = vehicles_df
                st.session_state.data_loaded = True
                
                progress_bar.progress(100)
                status_text.text("Ready for analysis!")
                st.success("✅ Data loaded and processed successfully!")
                
                # Clear progress indicators after a moment
                import time
                time.sleep(2)
                progress_bar.empty()
                status_text.empty()
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Failed to load data. Please check your CSV files.")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during data loading: {str(e)}")
    
    # Check if data is loaded
    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
        # Preprocess data
        with st.spinner("Processing data..."):
            merged_df = preprocess_uk_data(st.session_state.accidents_df, st.session_state.vehicles_df)
        
        # Advanced Filters
        if not merged_df.empty:
            st.sidebar.subheader("📅 Date Range Filter")
            date_min = merged_df['Date'].min()
            date_max = merged_df['Date'].max()
            
            if pd.notna(date_min) and pd.notna(date_max):
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=(date_min.date(), date_max.date()),
                    min_value=date_min.date(),
                    max_value=date_max.date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    merged_df = merged_df[
                        (merged_df['Date'].dt.date >= start_date) & 
                        (merged_df['Date'].dt.date <= end_date)
                    ]
            
            st.sidebar.subheader("🚨 Severity Filter")
            severity_options = ['All'] + list(merged_df['Accident_Severity'].unique())
            selected_severity = st.sidebar.selectbox("Select Severity", severity_options)
            
            if selected_severity != 'All':
                merged_df = merged_df[merged_df['Accident_Severity'] == selected_severity]
            
            st.sidebar.subheader("🌤️ Weather Filter")
            weather_options = ['All'] + list(merged_df['Weather_Conditions'].unique())
            selected_weather = st.sidebar.selectbox("Select Weather", weather_options)
            
            if selected_weather != 'All':
                merged_df = merged_df[merged_df['Weather_Conditions'] == selected_weather]
            
            st.sidebar.subheader("🛣️ Road Type Filter")
            road_options = ['All'] + list(merged_df['Road_Type'].unique())
            selected_road = st.sidebar.selectbox("Select Road Type", road_options)
            
            if selected_road != 'All':
                merged_df = merged_df[merged_df['Road_Type'] == selected_road]
            
            st.sidebar.subheader("🚦 Speed Limit Filter")
            speed_options = ['All'] + sorted([x for x in merged_df['Speed_limit'].unique() if pd.notna(x)])
            selected_speed = st.sidebar.selectbox("Select Speed Limit", speed_options)
            
            if selected_speed != 'All':
                merged_df = merged_df[merged_df['Speed_limit'] == selected_speed]
            
            st.sidebar.subheader("📍 Location Filter")
            if st.sidebar.checkbox("Enable Location Filter"):
                lat_min_val = float(merged_df['Latitude'].min())
                lat_max_val = float(merged_df['Latitude'].max())
                lat_min, lat_max = st.sidebar.slider(
                    "Latitude Range",
                    lat_min_val,
                    lat_max_val,
                    (lat_min_val, lat_max_val))
                
                lon_min_val = float(merged_df['Longitude'].min())
                lon_max_val = float(merged_df['Longitude'].max())
                lon_min, lon_max = st.sidebar.slider(
                    "Longitude Range",
                    lon_min_val,
                    lon_max_val,
                    (lon_min_val, lon_max_val))
                
                merged_df = merged_df[
                    (merged_df['Latitude'] >= lat_min) & (merged_df['Latitude'] <= lat_max) &
                    (merged_df['Longitude'] >= lon_min) & (merged_df['Longitude'] <= lon_max)
                ]
            
            # Show filter summary
            st.sidebar.subheader("📊 Filter Summary")
            st.sidebar.write(f"**Filtered Records:** {len(merged_df)}")
            st.sidebar.write(f"**Original Records:** {len(st.session_state.accidents_df)}")
            
            if len(merged_df) < len(st.session_state.accidents_df):
                st.sidebar.success(f"Filters applied: {len(st.session_state.accidents_df) - len(merged_df)} records filtered out")
            
            # Reset filters button
            if st.sidebar.button("🔄 Reset All Filters"):
                st.session_state.filters_applied = False
                st.rerun()
        
        if merged_df.empty:
            st.error("No valid data found after preprocessing.")
            return
        
        # Display basic statistics
        st.header("📊 Dataset Overview")
        
        # Add export and preview options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.checkbox("🔍 Show Data Preview"):
                st.subheader("Data Preview")
                st.dataframe(merged_df.head(10))
                st.write(f"**Total columns:** {len(merged_df.columns)}")
                st.write(f"**Total rows:** {len(merged_df)}")
        
        with col2:
            # Export filtered data
            if st.button("📥 Export Filtered Data"):
                csv = merged_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"filtered_accident_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Export summary report
            if st.button("📊 Export Summary Report"):
                # Create summary statistics
                summary = {
                    'Total_Accidents': len(merged_df),
                    'Date_Range': f"{merged_df['Date'].min()} to {merged_df['Date'].max()}",
                    'Fatal_Accidents': len(merged_df[merged_df['Accident_Severity'] == 'Fatal']),
                    'Serious_Accidents': len(merged_df[merged_df['Accident_Severity'] == 'Serious']),
                    'Slight_Accidents': len(merged_df[merged_df['Accident_Severity'] == 'Slight']),
                    'Total_Casualties': merged_df['Number_of_Casualties'].sum(),
                    'Total_Vehicles': merged_df['Number_of_Vehicles'].sum(),
                    'Most_Common_Weather': merged_df['Weather_Conditions'].mode().iloc[0] if not merged_df['Weather_Conditions'].mode().empty else 'N/A',
                    'Most_Common_Road_Type': merged_df['Road_Type'].mode().iloc[0] if not merged_df['Road_Type'].mode().empty else 'N/A'
                }
                
                summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=csv,
                    file_name=f"accident_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Accidents", len(merged_df))
        
        with col2:
            date_min = merged_df['Date'].min()
            date_max = merged_df['Date'].max()
            if pd.notna(date_min) and pd.notna(date_max):
                st.metric("Date Range", f"{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
            else:
                st.metric("Date Range", "Invalid dates")
        
        with col3:
            casualties_sum = merged_df['Number_of_Casualties'].sum()
            st.metric("Total Casualties", casualties_sum if pd.notna(casualties_sum) else 0)
        
        with col4:
            vehicles_sum = merged_df['Number_of_Vehicles'].sum()
            st.metric("Total Vehicles", vehicles_sum if pd.notna(vehicles_sum) else 0)
        
        # Risk Assessment
        st.header("🎯 Risk Assessment")
        risk_score = calculate_risk_score(merged_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if risk_score >= 80:
                risk_level = "LOW"
                risk_class = "risk-low"
            elif risk_score >= 60:
                risk_level = "MEDIUM"
                risk_class = "risk-medium"
            else:
                risk_level = "HIGH"
                risk_class = "risk-high"
            
            st.markdown(f'<div class="metric-card"><h3>Risk Score: <span class="{risk_class}">{risk_score:.0f}/100</span></h3><p>Risk Level: <span class="{risk_class}">{risk_level}</span></p></div>', unsafe_allow_html=True)
        
        with col2:
            fatal_count = len(merged_df[merged_df['Accident_Severity'] == 'Fatal'])
            st.metric("Fatal Accidents", fatal_count)
        
        with col3:
            serious_count = len(merged_df[merged_df['Accident_Severity'] == 'Serious'])
            st.metric("Serious Accidents", serious_count)
        
        # Accident Severity Distribution
        st.header("📈 Accident Severity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            severity_counts = merged_df['Accident_Severity'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Accident Severity Distribution",
                color_discrete_map={
                    'Fatal': '#ff0000',
                    'Serious': '#ff8800',
                    'Slight': '#00aa44'
                }
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Casualties vs Vehicles
            fig_casualties = px.scatter(
                merged_df,
                x='Number_of_Vehicles',
                y='Number_of_Casualties',
                color='Accident_Severity',
                title='Casualties vs Number of Vehicles',
                color_discrete_map={
                    'Fatal': '#ff0000',
                    'Serious': '#ff8800',
                    'Slight': '#00aa44'
                }
            )
            st.plotly_chart(fig_casualties, use_container_width=True)
        
        # Map Visualization
        st.header("🗺️ Accident Location Map")
        
        # Sample data for map (to avoid performance issues with large datasets)
        sample_size = min(1000, len(merged_df))
        sample_df = merged_df.sample(n=sample_size, random_state=42)
        
        map_obj = create_accident_severity_map(sample_df)
        if map_obj:
            st_folium(map_obj, width=700, height=500)
            st.info(f"Showing {sample_size} random accidents for performance. Total accidents: {len(merged_df)}")
        
        # Time Analysis
        st.header("⏰ Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            merged_df['Hour'] = pd.to_datetime(merged_df['Time'], format='%H:%M', errors='coerce').dt.hour
            hourly_counts = merged_df['Hour'].value_counts().sort_index()
            
            # Create DataFrame for Plotly
            hourly_df = pd.DataFrame({
                'Hour': hourly_counts.index,
                'Accidents': hourly_counts.values
            })
            
            fig_hourly = px.bar(
                hourly_df,
                x='Hour',
                y='Accidents',
                title="Accidents by Hour of Day",
                labels={'Hour': 'Hour', 'Accidents': 'Number of Accidents'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Day of week distribution
            day_counts = merged_df['Day_of_Week'].value_counts()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = day_counts.reindex(day_order, fill_value=0)
            
            # Create DataFrame for Plotly
            day_df = pd.DataFrame({
                'Day': day_counts.index,
                'Accidents': day_counts.values
            })
            
            fig_days = px.bar(
                day_df,
                x='Day',
                y='Accidents',
                title="Accidents by Day of Week",
                labels={'Day': 'Day', 'Accidents': 'Number of Accidents'}
            )
            st.plotly_chart(fig_days, use_container_width=True)
        
        # Environmental Factors
        st.header("🌤️ Environmental Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weather conditions
            weather_counts = merged_df['Weather_Conditions'].value_counts()
            
            # Create DataFrame for Plotly
            weather_df = pd.DataFrame({
                'Weather': weather_counts.index,
                'Accidents': weather_counts.values
            })
            
            fig_weather = px.bar(
                weather_df,
                x='Accidents',
                y='Weather',
                orientation='h',
                title="Accidents by Weather Conditions",
                labels={'Accidents': 'Number of Accidents', 'Weather': 'Weather'}
            )
            st.plotly_chart(fig_weather, use_container_width=True)
        
        with col2:
            # Road surface conditions
            road_counts = merged_df['Road_Surface_Conditions'].value_counts()
            
            # Create DataFrame for Plotly
            road_df = pd.DataFrame({
                'Road_Surface': road_counts.index,
                'Accidents': road_counts.values
            })
            
            fig_road = px.bar(
                road_df,
                x='Accidents',
                y='Road_Surface',
                orientation='h',
                title="Accidents by Road Surface Conditions",
                labels={'Accidents': 'Number of Accidents', 'Road_Surface': 'Road Surface'}
            )
            st.plotly_chart(fig_road, use_container_width=True)
        
        # Speed Limit Analysis
        st.header("🚦 Speed Limit Analysis")
        
        speed_analysis = merged_df.groupby('Speed_limit')['Accident_Severity'].value_counts().unstack(fill_value=0)
        
        fig_speed = px.bar(
            speed_analysis,
            title="Accidents by Speed Limit and Severity",
            labels={'value': 'Number of Accidents', 'index': 'Speed Limit (mph)'}
        )
        st.plotly_chart(fig_speed, use_container_width=True)
        
        # Advanced Analytics
        st.header("🧠 Advanced Analytics")
        
        # Create tabs for different analytics
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fatal Accidents", "High Risk Areas", "Trend Analysis", "Correlation Analysis", "Data Summary"])
        
        with tab1:
            st.subheader("Fatal Accidents Analysis")
            fatal_accidents = merged_df[merged_df['Accident_Severity'] == 'Fatal']
            
            if not fatal_accidents.empty:
                st.write(f"**Total Fatal Accidents:** {len(fatal_accidents)}")
                
                # Show fatal accidents by location
                fatal_locations = fatal_accidents[['Accident_Index', 'Date', 'Time', 'Latitude', 'Longitude', 
                                                 'Number_of_Casualties', 'Number_of_Vehicles', 'Weather_Conditions']]
                st.dataframe(fatal_locations.head(20))
            else:
                st.success("No fatal accidents in this dataset!")
        
        with tab2:
            st.subheader("High Risk Areas")
            
            # Group by location (rounded coordinates for clustering)
            merged_df['Lat_rounded'] = merged_df['Latitude'].round(3)
            merged_df['Lon_rounded'] = merged_df['Longitude'].round(3)
            
            risk_areas = merged_df.groupby(['Lat_rounded', 'Lon_rounded']).agg({
                'Accident_Index': 'count',
                'Accident_Severity': lambda x: (x == 'Fatal').sum(),
                'Number_of_Casualties': 'sum'
            }).rename(columns={'Accident_Index': 'Accident_Count'})
            
            risk_areas = risk_areas.sort_values('Accident_Count', ascending=False)
            st.dataframe(risk_areas.head(20))
        
        with tab3:
            st.subheader("Trend Analysis")
            
            # Monthly trend analysis
            if 'Date' in merged_df.columns:
                merged_df['Year'] = merged_df['Date'].dt.year
                merged_df['Month'] = merged_df['Date'].dt.month
                
                # Monthly accidents trend
                monthly_trend = merged_df.groupby(['Year', 'Month']).size().reset_index(name='Accidents')
                monthly_trend['Date'] = pd.to_datetime(monthly_trend[['Year', 'Month']].assign(day=1))
                
                fig_trend = px.line(
                    monthly_trend,
                    x='Date',
                    y='Accidents',
                    title='Monthly Accident Trends',
                    labels={'Date': 'Month', 'Accidents': 'Number of Accidents'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Year-over-year comparison
                yearly_comparison = merged_df.groupby('Year').agg({
                    'Accident_Index': 'count',
                    'Accident_Severity': lambda x: (x == 'Fatal').sum()
                }).rename(columns={'Accident_Index': 'Total_Accidents', 'Accident_Severity': 'Fatal_Accidents'})
                
                fig_yearly = px.bar(
                    yearly_comparison,
                    title='Year-over-Year Accident Comparison',
                    labels={'value': 'Number of Accidents', 'index': 'Year'}
                )
                st.plotly_chart(fig_yearly, use_container_width=True)
        
        with tab4:
            st.subheader("Correlation Analysis")
            
            # Create correlation matrix for numeric columns
            numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = merged_df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    correlation_matrix,
                    title='Correlation Matrix of Numeric Variables',
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Show top correlations
                st.write("**Top Correlations:**")
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_pairs.append({
                            'Variable 1': correlation_matrix.columns[i],
                            'Variable 2': correlation_matrix.columns[j],
                            'Correlation': correlation_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
                st.dataframe(corr_df.head(10))
            else:
                st.info("Not enough numeric columns for correlation analysis.")
        
        with tab5:
            st.subheader("Data Summary")
            st.write("**Dataset Statistics:**")
            st.dataframe(merged_df.describe(include='all'))
            
            # Data quality report
            st.write("**Data Quality Report:**")
            quality_report = {
                'Total Records': len(merged_df),
                'Missing Values': merged_df.isnull().sum().sum(),
                'Duplicate Records': merged_df.duplicated().sum(),
                'Unique Accidents': merged_df['Accident_Index'].nunique() if 'Accident_Index' in merged_df.columns else 'N/A'
            }
            
            for key, value in quality_report.items():
                st.write(f"**{key}:** {value}")
    
    else:
        # Show instructions
        st.info("👆 Click 'Load UK Road Safety Data' in the sidebar to begin analysis!")
        
        st.header("📋 UK Road Safety Data Analysis")
        st.markdown("""
        This dashboard analyzes UK road safety data including:
        
        ### Accident Information
        - Accident severity (Fatal, Serious, Slight)
        - Location data (GPS coordinates)
        - Date and time information
        - Weather and road conditions
        - Speed limits and road types
        
        ### Vehicle Information
        - Vehicle types and makes
        - Driver demographics
        - Vehicle maneuvers
        - Impact points
        
        ### Analytics Features
        - **Risk Assessment**: Overall risk scoring based on accident severity
        - **Geographic Analysis**: Interactive maps showing accident locations
        - **Temporal Analysis**: Time patterns and trends
        - **Environmental Factors**: Weather and road condition impacts
        - **Speed Limit Analysis**: Relationship between speed limits and accidents
        """)
        
        st.header("🎯 Key Insights")
        st.markdown("""
        The dashboard will help you identify:
        - **High-risk locations** with frequent accidents
        - **Time patterns** when accidents are most likely
        - **Environmental factors** that contribute to accidents
        - **Speed limit relationships** with accident severity
        - **Vehicle and driver factors** in accident causation
        """)

if __name__ == "__main__":
    main()
