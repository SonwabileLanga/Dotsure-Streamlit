"""
Third-party API Integration for DOTSURE Telematics Platform
Handles external APIs for weather, traffic, mapping, and other services
"""

import pandas as pd
import streamlit as st
import requests
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import hashlib
import hmac
import base64
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherAPI:
    """Weather API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get current weather for coordinates"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {}
    
    def get_weather_forecast(self, lat: float, lon: float, days: int = 5) -> List[Dict[str, Any]]:
        """Get weather forecast for coordinates"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecasts = []
            
            for item in data['list'][:days * 8]:  # 8 forecasts per day
                forecasts.append({
                    'datetime': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'weather': item['weather'][0]['main'],
                    'description': item['weather'][0]['description'],
                    'wind_speed': item['wind']['speed'],
                    'precipitation': item.get('rain', {}).get('3h', 0)
                })
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Weather forecast API error: {e}")
            return []

class TrafficAPI:
    """Traffic API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    def get_traffic_info(self, origin: Tuple[float, float], 
                        destination: Tuple[float, float]) -> Dict[str, Any]:
        """Get traffic information between two points"""
        try:
            params = {
                'origins': f"{origin[0]},{origin[1]}",
                'destinations': f"{destination[0]},{destination[1]}",
                'key': self.api_key,
                'departure_time': 'now',
                'traffic_model': 'best_guess'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK' and data['rows'][0]['elements'][0]['status'] == 'OK':
                element = data['rows'][0]['elements'][0]
                
                return {
                    'distance': element['distance']['value'],
                    'duration': element['duration']['value'],
                    'duration_in_traffic': element.get('duration_in_traffic', {}).get('value', 0),
                    'traffic_delay': element.get('duration_in_traffic', {}).get('value', 0) - element['duration']['value'],
                    'status': 'OK'
                }
            else:
                return {'status': 'ERROR', 'message': data.get('error_message', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"Traffic API error: {e}")
            return {'status': 'ERROR', 'message': str(e)}

class GeocodingAPI:
    """Geocoding API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def geocode_address(self, address: str) -> Dict[str, Any]:
        """Convert address to coordinates"""
        try:
            params = {
                'address': address,
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                result = data['results'][0]
                location = result['geometry']['location']
                
                return {
                    'latitude': location['lat'],
                    'longitude': location['lng'],
                    'formatted_address': result['formatted_address'],
                    'place_id': result['place_id'],
                    'types': result['types']
                }
            else:
                return {'status': 'ERROR', 'message': data.get('error_message', 'No results found')}
                
        except Exception as e:
            logger.error(f"Geocoding API error: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        """Convert coordinates to address"""
        try:
            params = {
                'latlng': f"{lat},{lon}",
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                result = data['results'][0]
                
                return {
                    'formatted_address': result['formatted_address'],
                    'place_id': result['place_id'],
                    'types': result['types'],
                    'address_components': result['address_components']
                }
            else:
                return {'status': 'ERROR', 'message': data.get('error_message', 'No results found')}
                
        except Exception as e:
            logger.error(f"Reverse geocoding API error: {e}")
            return {'status': 'ERROR', 'message': str(e)}

class FleetManagementAPI:
    """Fleet management API integration"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_vehicle_status(self, vehicle_id: str) -> Dict[str, Any]:
        """Get vehicle status from fleet management system"""
        try:
            url = f"{self.base_url}/vehicles/{vehicle_id}/status"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Fleet API error: {e}")
            return {}
    
    def send_alert(self, vehicle_id: str, alert_type: str, 
                  message: str, severity: str = "medium") -> bool:
        """Send alert to fleet management system"""
        try:
            url = f"{self.base_url}/alerts"
            
            payload = {
                'vehicle_id': vehicle_id,
                'alert_type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(url, headers=self.headers, 
                                   json=payload, timeout=10)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Alert sending failed: {e}")
            return False

class APIManager:
    """Manages all API integrations"""
    
    def __init__(self):
        self.weather_api = None
        self.traffic_api = None
        self.geocoding_api = None
        self.fleet_api = None
        self.api_cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def setup_weather_api(self, api_key: str):
        """Setup weather API"""
        self.weather_api = WeatherAPI(api_key)
        logger.info("Weather API configured")
    
    def setup_traffic_api(self, api_key: str):
        """Setup traffic API"""
        self.traffic_api = TrafficAPI(api_key)
        logger.info("Traffic API configured")
    
    def setup_geocoding_api(self, api_key: str):
        """Setup geocoding API"""
        self.geocoding_api = GeocodingAPI(api_key)
        logger.info("Geocoding API configured")
    
    def setup_fleet_api(self, base_url: str, api_key: str):
        """Setup fleet management API"""
        self.fleet_api = FleetManagementAPI(base_url, api_key)
        logger.info("Fleet API configured")
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data"""
        if key in self.api_cache:
            data, timestamp = self.api_cache[key]
            if time.time() - timestamp < self.cache_duration:
                return data
            else:
                del self.api_cache[key]
        return None
    
    def set_cached_data(self, key: str, data: Any):
        """Set cached data"""
        self.api_cache[key] = (data, time.time())
    
    def enrich_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich vehicle data with external API data"""
        try:
            enriched_df = df.copy()
            
            # Add weather data
            if self.weather_api and 'latitude' in df.columns and 'longitude' in df.columns:
                weather_data = []
                for _, row in df.iterrows():
                    cache_key = f"weather_{row['latitude']:.4f}_{row['longitude']:.4f}"
                    weather = self.get_cached_data(cache_key)
                    
                    if weather is None:
                        weather = self.weather_api.get_current_weather(
                            row['latitude'], row['longitude']
                        )
                        if weather:
                            self.set_cached_data(cache_key, weather)
                    
                    weather_data.append(weather)
                
                # Add weather columns
                if weather_data and weather_data[0]:
                    weather_df = pd.DataFrame(weather_data)
                    enriched_df = pd.concat([enriched_df, weather_df], axis=1)
            
            # Add geocoding data
            if self.geocoding_api and 'latitude' in df.columns and 'longitude' in df.columns:
                geocoding_data = []
                for _, row in df.iterrows():
                    cache_key = f"geocode_{row['latitude']:.4f}_{row['longitude']:.4f}"
                    geocode = self.get_cached_data(cache_key)
                    
                    if geocode is None:
                        geocode = self.geocoding_api.reverse_geocode(
                            row['latitude'], row['longitude']
                        )
                        if geocode and geocode.get('status') != 'ERROR':
                            self.set_cached_data(cache_key, geocode)
                    
                    geocoding_data.append(geocode)
                
                # Add geocoding columns
                if geocoding_data and geocoding_data[0]:
                    geocode_df = pd.DataFrame(geocoding_data)
                    enriched_df = pd.concat([enriched_df, geocode_df], axis=1)
            
            logger.info(f"Enriched {len(enriched_df)} records with external data")
            return enriched_df
            
        except Exception as e:
            logger.error(f"Data enrichment failed: {e}")
            return df
    
    def get_route_analysis(self, origin: Tuple[float, float], 
                          destination: Tuple[float, float]) -> Dict[str, Any]:
        """Get comprehensive route analysis"""
        try:
            analysis = {
                'origin': origin,
                'destination': destination,
                'timestamp': datetime.now().isoformat()
            }
            
            # Traffic information
            if self.traffic_api:
                traffic_info = self.traffic_api.get_traffic_info(origin, destination)
                analysis['traffic'] = traffic_info
            
            # Weather at origin and destination
            if self.weather_api:
                origin_weather = self.weather_api.get_current_weather(origin[0], origin[1])
                dest_weather = self.weather_api.get_current_weather(destination[0], destination[1])
                analysis['weather'] = {
                    'origin': origin_weather,
                    'destination': dest_weather
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Route analysis failed: {e}")
            return {}

# Streamlit UI for API integration
def create_api_ui():
    """Create Streamlit UI for API integration"""
    st.markdown("### 🌐 API Integration")
    
    # Initialize API manager
    if 'api_manager' not in st.session_state:
        st.session_state.api_manager = APIManager()
    
    api_manager = st.session_state.api_manager
    
    # API configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Configuration", "Weather", "Traffic", "Data Enrichment"])
    
    with tab1:
        st.markdown("#### API Configuration")
        
        # Weather API
        st.markdown("##### Weather API (OpenWeatherMap)")
        weather_key = st.text_input("Weather API Key", type="password")
        if st.button("Setup Weather API"):
            if weather_key:
                api_manager.setup_weather_api(weather_key)
                st.success("✅ Weather API configured")
            else:
                st.error("❌ Please enter API key")
        
        # Traffic API
        st.markdown("##### Traffic API (Google Maps)")
        traffic_key = st.text_input("Traffic API Key", type="password")
        if st.button("Setup Traffic API"):
            if traffic_key:
                api_manager.setup_traffic_api(traffic_key)
                st.success("✅ Traffic API configured")
            else:
                st.error("❌ Please enter API key")
        
        # Geocoding API
        st.markdown("##### Geocoding API (Google Maps)")
        geocoding_key = st.text_input("Geocoding API Key", type="password")
        if st.button("Setup Geocoding API"):
            if geocoding_key:
                api_manager.setup_geocoding_api(geocoding_key)
                st.success("✅ Geocoding API configured")
            else:
                st.error("❌ Please enter API key")
        
        # Fleet API
        st.markdown("##### Fleet Management API")
        col1, col2 = st.columns(2)
        with col1:
            fleet_url = st.text_input("Fleet API Base URL")
        with col2:
            fleet_key = st.text_input("Fleet API Key", type="password")
        
        if st.button("Setup Fleet API"):
            if fleet_url and fleet_key:
                api_manager.setup_fleet_api(fleet_url, fleet_key)
                st.success("✅ Fleet API configured")
            else:
                st.error("❌ Please enter URL and API key")
    
    with tab2:
        st.markdown("#### Weather Data")
        
        if api_manager.weather_api:
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude", value=40.7128, format="%.4f")
            with col2:
                lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
            
            if st.button("Get Current Weather"):
                weather = api_manager.weather_api.get_current_weather(lat, lon)
                if weather:
                    st.success("✅ Weather data retrieved")
                    st.json(weather)
                else:
                    st.error("❌ Failed to get weather data")
            
            if st.button("Get Weather Forecast"):
                forecast = api_manager.weather_api.get_weather_forecast(lat, lon, 3)
                if forecast:
                    st.success("✅ Weather forecast retrieved")
                    forecast_df = pd.DataFrame(forecast)
                    st.dataframe(forecast_df)
                else:
                    st.error("❌ Failed to get weather forecast")
        else:
            st.info("Please configure Weather API first")
    
    with tab3:
        st.markdown("#### Traffic Data")
        
        if api_manager.traffic_api:
            st.markdown("##### Route Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Origin**")
                origin_lat = st.number_input("Origin Latitude", value=40.7128, format="%.4f")
                origin_lon = st.number_input("Origin Longitude", value=-74.0060, format="%.4f")
            with col2:
                st.markdown("**Destination**")
                dest_lat = st.number_input("Destination Latitude", value=40.7589, format="%.4f")
                dest_lon = st.number_input("Destination Longitude", value=-73.9851, format="%.4f")
            
            if st.button("Get Traffic Info"):
                traffic_info = api_manager.traffic_api.get_traffic_info(
                    (origin_lat, origin_lon), (dest_lat, dest_lon)
                )
                if traffic_info.get('status') == 'OK':
                    st.success("✅ Traffic data retrieved")
                    st.json(traffic_info)
                else:
                    st.error(f"❌ {traffic_info.get('message', 'Failed to get traffic data')}")
        else:
            st.info("Please configure Traffic API first")
    
    with tab4:
        st.markdown("#### Data Enrichment")
        
        if 'data' in st.session_state:
            st.markdown("##### Enrich Vehicle Data")
            st.info("This will add weather and location data to your vehicle data")
            
            if st.button("Enrich Data"):
                with st.spinner("Enriching data with external APIs..."):
                    enriched_df = api_manager.enrich_vehicle_data(st.session_state.data)
                    
                    if len(enriched_df.columns) > len(st.session_state.data.columns):
                        st.session_state.data = enriched_df
                        st.success("✅ Data enriched successfully")
                        st.dataframe(enriched_df.head())
                    else:
                        st.warning("⚠️ No additional data was added")
        else:
            st.info("Please load vehicle data first")

if __name__ == "__main__":
    # Test API integration
    api_manager = APIManager()
    print("API integration module loaded successfully")
