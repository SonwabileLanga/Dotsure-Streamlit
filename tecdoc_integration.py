"""
TecDoc Catalog API Integration for DOTSURE Telematics Platform
Provides automotive parts, vehicle data, and VIN decoding capabilities
"""

import http.client
import json
import pandas as pd
import streamlit as st
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TecDocAPI:
    """TecDoc Catalog API integration"""
    
    def __init__(self, api_key: str = "9e3c362234msh82f82b2975093c9p19a713jsn8c7c37be92e0"):
        self.api_key = api_key
        self.base_url = "tecdoc-catalog.p.rapidapi.com"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.base_url
        }
    
    def make_request(self, endpoint: str, method: str = "GET") -> Dict[str, Any]:
        """Make HTTP request to TecDoc API"""
        try:
            conn = http.client.HTTPSConnection(self.base_url)
            conn.request(method, endpoint, headers=self.headers)
            res = conn.getresponse()
            data = res.read()
            conn.close()
            
            if res.status == 200:
                return json.loads(data.decode("utf-8"))
            else:
                logger.error(f"API request failed with status {res.status}")
                return {"error": f"API request failed with status {res.status}"}
                
        except Exception as e:
            logger.error(f"TecDoc API request failed: {e}")
            return {"error": str(e)}
    
    def get_all_languages(self) -> List[Dict[str, Any]]:
        """Get all available languages"""
        try:
            result = self.make_request("/languages/list")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get languages: {e}")
            return []
    
    def get_all_countries(self) -> List[Dict[str, Any]]:
        """Get all available countries"""
        try:
            result = self.make_request("/countries/list")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get countries: {e}")
            return []
    
    def get_countries_by_language(self, lang_id: int) -> List[Dict[str, Any]]:
        """Get countries by language ID"""
        try:
            result = self.make_request(f"/countries/list-countries-by-lang-id/{lang_id}")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get countries by language: {e}")
            return []
    
    def get_all_suppliers(self) -> List[Dict[str, Any]]:
        """Get all suppliers"""
        try:
            result = self.make_request("/suppliers/list")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get suppliers: {e}")
            return []
    
    def get_vehicle_types(self) -> List[Dict[str, Any]]:
        """Get all vehicle types"""
        try:
            result = self.make_request("/types/list-vehicles-type")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get vehicle types: {e}")
            return []
    
    def decode_vin_v1(self, vin: str) -> Dict[str, Any]:
        """Decode VIN using version 1"""
        try:
            result = self.make_request(f"/vin/decoder-v1/{vin}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"VIN decoding v1 failed: {e}")
            return {}
    
    def decode_vin_v2(self, vin: str) -> Dict[str, Any]:
        """Decode VIN using version 2"""
        try:
            result = self.make_request(f"/vin/decoder-v2/{vin}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"VIN decoding v2 failed: {e}")
            return {}
    
    def decode_vin_v3(self, vin: str) -> Dict[str, Any]:
        """Decode VIN using version 3 (BETA)"""
        try:
            result = self.make_request(f"/vin/decoder-v3/{vin}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"VIN decoding v3 failed: {e}")
            return {}
    
    def search_article_by_number(self, lang_id: int, article_number: str) -> Dict[str, Any]:
        """Search article by number"""
        try:
            result = self.make_request(f"/articles/search/lang-id/{lang_id}/article-search/{article_number}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"Article search failed: {e}")
            return {}
    
    def search_article_by_supplier(self, lang_id: int, supplier_id: int, article_number: str) -> Dict[str, Any]:
        """Search article by supplier ID and number"""
        try:
            result = self.make_request(f"/articles/search/lang-id/{lang_id}/supplier-id/{supplier_id}/article-search/{article_number}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"Article search by supplier failed: {e}")
            return {}
    
    def get_article_details(self, article_id: int, lang_id: int, country_filter_id: int) -> Dict[str, Any]:
        """Get complete article details by ID"""
        try:
            result = self.make_request(f"/articles/article-id-details/{article_id}/lang-id/{lang_id}/country-filter-id/{country_filter_id}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"Failed to get article details: {e}")
            return {}
    
    def get_article_media(self, article_id: int, lang_id: int) -> Dict[str, Any]:
        """Get article media information"""
        try:
            result = self.make_request(f"/articles/article-all-media-info/{article_id}/lang-id/{lang_id}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"Failed to get article media: {e}")
            return {}
    
    def get_manufacturer_details(self, manufacturer_id: int) -> Dict[str, Any]:
        """Get manufacturer details by ID"""
        try:
            result = self.make_request(f"/manufacturers/find-by-id/{manufacturer_id}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"Failed to get manufacturer details: {e}")
            return {}
    
    def get_models_by_manufacturer(self, manufacturer_id: int, lang_id: int, country_filter_id: int, type_id: int) -> List[Dict[str, Any]]:
        """Get models list by manufacturer ID"""
        try:
            result = self.make_request(f"/models/list/manufacturer-id/{manufacturer_id}/lang-id/{lang_id}/country-filter-id/{country_filter_id}/type-id/{type_id}")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def get_vehicle_type_details(self, vehicle_id: int, manufacturer_id: int, lang_id: int, country_filter_id: int, type_id: int) -> Dict[str, Any]:
        """Get vehicle type detailed information"""
        try:
            result = self.make_request(f"/types/vehicle-type-details/{vehicle_id}/manufacturer-id/{manufacturer_id}/lang-id/{lang_id}/country-filter-id/{country_filter_id}/type-id/{type_id}")
            return result if "error" not in result else {}
        except Exception as e:
            logger.error(f"Failed to get vehicle type details: {e}")
            return {}
    
    def get_engine_types(self, model_series_id: int, manufacturer_id: int, lang_id: int, country_filter_id: int, type_id: int) -> List[Dict[str, Any]]:
        """Get all vehicle engine types"""
        try:
            result = self.make_request(f"/types/list-vehicles-types/{model_series_id}/manufacturer-id/{manufacturer_id}/lang-id/{lang_id}/country-filter-id/{country_filter_id}/type-id/{type_id}")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get engine types: {e}")
            return []
    
    def get_category_products_v1(self, vehicle_id: int, manufacturer_id: int, lang_id: int, country_filter_id: int, type_id: int) -> List[Dict[str, Any]]:
        """Get category products groups variant 1"""
        try:
            result = self.make_request(f"/category/category-products-groups-variant-1/{vehicle_id}/manufacturer-id/{manufacturer_id}/lang-id/{lang_id}/country-filter-id/{country_filter_id}/type-id/{type_id}")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get category products v1: {e}")
            return []
    
    def get_category_products_v2(self, vehicle_id: int, manufacturer_id: int, lang_id: int, country_filter_id: int, type_id: int) -> List[Dict[str, Any]]:
        """Get category products groups variant 2"""
        try:
            result = self.make_request(f"/category/category-products-groups-variant-2/{vehicle_id}/manufacturer-id/{manufacturer_id}/lang-id/{lang_id}/country-filter-id/{country_filter_id}/type-id/{type_id}")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get category products v2: {e}")
            return []
    
    def get_category_products_v3(self, vehicle_id: int, manufacturer_id: int, lang_id: int, country_filter_id: int, type_id: int) -> List[Dict[str, Any]]:
        """Get category products groups variant 3"""
        try:
            result = self.make_request(f"/category/category-products-groups-variant-3/{vehicle_id}/manufacturer-id/{manufacturer_id}/lang-id/{lang_id}/country-filter-id/{country_filter_id}/type-id/{type_id}")
            return result.get("data", []) if "error" not in result else []
        except Exception as e:
            logger.error(f"Failed to get category products v3: {e}")
            return []

class VehicleDataManager:
    """Manages vehicle data from TecDoc API"""
    
    def __init__(self, tecdoc_api: TecDocAPI):
        self.tecdoc_api = tecdoc_api
        self.vehicle_cache = {}
        self.parts_cache = {}
    
    def enrich_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich vehicle data with TecDoc information"""
        try:
            enriched_df = df.copy()
            
            # Add VIN decoding if VIN column exists
            if 'vin' in df.columns:
                vin_data = []
                for vin in df['vin'].dropna():
                    if vin not in self.vehicle_cache:
                        # Try VIN decoding
                        vin_info = self.tecdoc_api.decode_vin_v2(vin)
                        self.vehicle_cache[vin] = vin_info
                    else:
                        vin_info = self.vehicle_cache[vin]
                    
                    vin_data.append(vin_info)
                
                # Add VIN data to DataFrame
                if vin_data:
                    vin_df = pd.DataFrame(vin_data)
                    enriched_df = pd.concat([enriched_df, vin_df], axis=1)
            
            # Add manufacturer information if manufacturer_id exists
            if 'manufacturer_id' in df.columns:
                manufacturer_data = []
                for manufacturer_id in df['manufacturer_id'].dropna():
                    if manufacturer_id not in self.vehicle_cache:
                        manufacturer_info = self.tecdoc_api.get_manufacturer_details(manufacturer_id)
                        self.vehicle_cache[manufacturer_id] = manufacturer_info
                    else:
                        manufacturer_info = self.vehicle_cache[manufacturer_id]
                    
                    manufacturer_data.append(manufacturer_info)
                
                if manufacturer_data:
                    manufacturer_df = pd.DataFrame(manufacturer_data)
                    enriched_df = pd.concat([enriched_df, manufacturer_df], axis=1)
            
            logger.info(f"Enriched {len(enriched_df)} records with TecDoc data")
            return enriched_df
            
        except Exception as e:
            logger.error(f"Vehicle data enrichment failed: {e}")
            return df
    
    def search_parts_for_vehicle(self, vehicle_id: int, manufacturer_id: int, 
                                lang_id: int = 4, country_filter_id: int = 63, 
                                type_id: int = 1) -> List[Dict[str, Any]]:
        """Search parts for a specific vehicle"""
        try:
            cache_key = f"{vehicle_id}_{manufacturer_id}_{lang_id}_{country_filter_id}_{type_id}"
            
            if cache_key not in self.parts_cache:
                # Get category products
                parts = self.tecdoc_api.get_category_products_v1(vehicle_id, manufacturer_id, lang_id, country_filter_id, type_id)
                self.parts_cache[cache_key] = parts
            else:
                parts = self.parts_cache[cache_key]
            
            return parts
            
        except Exception as e:
            logger.error(f"Parts search failed: {e}")
            return []
    
    def get_vehicle_specifications(self, vehicle_id: int, manufacturer_id: int,
                                 lang_id: int = 4, country_filter_id: int = 63,
                                 type_id: int = 1) -> Dict[str, Any]:
        """Get detailed vehicle specifications"""
        try:
            # Get vehicle type details
            vehicle_details = self.tecdoc_api.get_vehicle_type_details(
                vehicle_id, manufacturer_id, lang_id, country_filter_id, type_id
            )
            
            # Get engine types
            engine_types = self.tecdoc_api.get_engine_types(
                vehicle_id, manufacturer_id, lang_id, country_filter_id, type_id
            )
            
            return {
                'vehicle_details': vehicle_details,
                'engine_types': engine_types,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get vehicle specifications: {e}")
            return {}

# Streamlit UI for TecDoc integration
def create_tecdoc_ui():
    """Create Streamlit UI for TecDoc integration"""
    st.markdown("### 🔧 TecDoc Catalog Integration")
    
    # Initialize TecDoc API
    if 'tecdoc_api' not in st.session_state:
        st.session_state.tecdoc_api = TecDocAPI()
    
    if 'vehicle_data_manager' not in st.session_state:
        st.session_state.vehicle_data_manager = VehicleDataManager(st.session_state.tecdoc_api)
    
    tecdoc_api = st.session_state.tecdoc_api
    vehicle_manager = st.session_state.vehicle_data_manager
    
    # TecDoc tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["VIN Decoder", "Vehicle Data", "Parts Search", "Manufacturers", "API Status"])
    
    with tab1:
        st.markdown("#### VIN Decoder")
        
        vin_input = st.text_input("Enter VIN Number", placeholder="1HGBH41JXMN109186")
        decoder_version = st.selectbox("Decoder Version", ["V2 (Recommended)", "V1", "V3 (Beta)"])
        
        if st.button("Decode VIN"):
            if vin_input:
                with st.spinner("Decoding VIN..."):
                    if decoder_version == "V1":
                        result = tecdoc_api.decode_vin_v1(vin_input)
                    elif decoder_version == "V2 (Recommended)":
                        result = tecdoc_api.decode_vin_v2(vin_input)
                    else:  # V3
                        result = tecdoc_api.decode_vin_v3(vin_input)
                    
                    if result and "error" not in result:
                        st.success("✅ VIN decoded successfully!")
                        st.json(result)
                        
                        # Display key information
                        if isinstance(result, dict):
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'make' in result:
                                    st.metric("Make", result['make'])
                                if 'model' in result:
                                    st.metric("Model", result['model'])
                            with col2:
                                if 'year' in result:
                                    st.metric("Year", result['year'])
                                if 'engine' in result:
                                    st.metric("Engine", result['engine'])
                    else:
                        st.error("❌ VIN decoding failed")
            else:
                st.warning("Please enter a VIN number")
    
    with tab2:
        st.markdown("#### Vehicle Data")
        
        col1, col2 = st.columns(2)
        with col1:
            vehicle_id = st.number_input("Vehicle ID", value=1, min_value=1)
            manufacturer_id = st.number_input("Manufacturer ID", value=1, min_value=1)
        with col2:
            lang_id = st.number_input("Language ID", value=4, min_value=1)
            country_filter_id = st.number_input("Country Filter ID", value=63, min_value=1)
            type_id = st.number_input("Type ID", value=1, min_value=1)
        
        if st.button("Get Vehicle Details"):
            with st.spinner("Fetching vehicle data..."):
                vehicle_specs = vehicle_manager.get_vehicle_specifications(
                    vehicle_id, manufacturer_id, lang_id, country_filter_id, type_id
                )
                
                if vehicle_specs:
                    st.success("✅ Vehicle data retrieved!")
                    st.json(vehicle_specs)
                else:
                    st.error("❌ Failed to retrieve vehicle data")
        
        # Enrich existing data
        if 'data' in st.session_state and st.session_state.data_loaded:
            st.markdown("#### Enrich Vehicle Data")
            if st.button("Enrich with TecDoc Data"):
                with st.spinner("Enriching vehicle data..."):
                    enriched_df = vehicle_manager.enrich_vehicle_data(st.session_state.data)
                    if len(enriched_df.columns) > len(st.session_state.data.columns):
                        st.session_state.data = enriched_df
                        st.success("✅ Data enriched with TecDoc information!")
                        st.dataframe(enriched_df.head())
                    else:
                        st.info("No additional TecDoc data was added")
    
    with tab3:
        st.markdown("#### Parts Search")
        
        col1, col2 = st.columns(2)
        with col1:
            search_vehicle_id = st.number_input("Vehicle ID for Parts", value=1, min_value=1)
            search_manufacturer_id = st.number_input("Manufacturer ID for Parts", value=1, min_value=1)
        with col2:
            search_lang_id = st.number_input("Language ID for Parts", value=4, min_value=1)
            search_country_id = st.number_input("Country ID for Parts", value=63, min_value=1)
            search_type_id = st.number_input("Type ID for Parts", value=1, min_value=1)
        
        if st.button("Search Parts"):
            with st.spinner("Searching parts..."):
                parts = vehicle_manager.search_parts_for_vehicle(
                    search_vehicle_id, search_manufacturer_id, search_lang_id, search_country_id, search_type_id
                )
                
                if parts:
                    st.success(f"✅ Found {len(parts)} parts!")
                    parts_df = pd.DataFrame(parts)
                    st.dataframe(parts_df)
                else:
                    st.info("No parts found for this vehicle")
        
        # Article search
        st.markdown("#### Article Search")
        article_number = st.text_input("Article Number", placeholder="Enter article number")
        article_lang_id = st.number_input("Language ID for Article", value=4, min_value=1)
        
        if st.button("Search Article"):
            if article_number:
                with st.spinner("Searching article..."):
                    article = tecdoc_api.search_article_by_number(article_lang_id, article_number)
                    if article and "error" not in article:
                        st.success("✅ Article found!")
                        st.json(article)
                    else:
                        st.error("❌ Article not found")
            else:
                st.warning("Please enter an article number")
    
    with tab4:
        st.markdown("#### Manufacturers & Suppliers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Manufacturers")
            if st.button("Get All Manufacturers"):
                with st.spinner("Fetching manufacturers..."):
                    # Note: This would need a specific endpoint for all manufacturers
                    st.info("Use the Vehicle Data tab to search for specific manufacturers by ID")
        
        with col2:
            st.markdown("##### Suppliers")
            if st.button("Get All Suppliers"):
                with st.spinner("Fetching suppliers..."):
                    suppliers = tecdoc_api.get_all_suppliers()
                    if suppliers:
                        st.success(f"✅ Found {len(suppliers)} suppliers!")
                        suppliers_df = pd.DataFrame(suppliers)
                        st.dataframe(suppliers_df)
                    else:
                        st.error("❌ Failed to fetch suppliers")
    
    with tab5:
        st.markdown("#### API Status & Configuration")
        
        # API Key configuration
        st.markdown("##### API Configuration")
        new_api_key = st.text_input("TecDoc API Key", value=tecdoc_api.api_key, type="password")
        if st.button("Update API Key"):
            tecdoc_api.api_key = new_api_key
            tecdoc_api.headers['x-rapidapi-key'] = new_api_key
            st.success("✅ API key updated!")
        
        # Test API connection
        st.markdown("##### API Test")
        if st.button("Test API Connection"):
            with st.spinner("Testing API connection..."):
                # Test with a simple request
                languages = tecdoc_api.get_all_languages()
                if languages:
                    st.success("✅ API connection successful!")
                    st.info(f"Found {len(languages)} languages")
                else:
                    st.error("❌ API connection failed")
        
        # API Information
        st.markdown("##### API Information")
        st.info("""
        **TecDoc Catalog API Features:**
        - VIN Decoding (3 versions)
        - Vehicle specifications and details
        - Parts catalog and search
        - Manufacturer and supplier data
        - Article search and details
        - Category and product groups
        
        **Rate Limits:** Please check your RapidAPI subscription for rate limits.
        **Documentation:** https://rapidapi.com/tecdoc-catalog/api/tecdoc-catalog
        """)

if __name__ == "__main__":
    # Test TecDoc integration
    tecdoc_api = TecDocAPI()
    vehicle_manager = VehicleDataManager(tecdoc_api)
    
    print("TecDoc integration module loaded successfully")
    
    # Test VIN decoding
    test_vin = "1HGBH41JXMN109186"
    print(f"Testing VIN decoding for: {test_vin}")
    result = tecdoc_api.decode_vin_v2(test_vin)
    print(f"VIN decode result: {result}")
