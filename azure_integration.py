"""
Azure Services Integration for DOTSURE Telematics Platform
Handles Azure Blob Storage, SQL Database, and Cognitive Services
"""

import pandas as pd
import streamlit as st
import io
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureServicesManager:
    """Manages Azure services integration"""
    
    def __init__(self):
        self.blob_service_client = None
        self.cognitive_vision_client = None
        self.keyvault_client = None
        self.credentials = None
        
    def authenticate(self, tenant_id: str = None, client_id: str = None, 
                   client_secret: str = None, use_default: bool = True):
        """Authenticate with Azure services"""
        try:
            if use_default:
                self.credentials = DefaultAzureCredential()
            else:
                self.credentials = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            logger.info("Azure authentication successful")
            return True
        except Exception as e:
            logger.error(f"Azure authentication failed: {e}")
            return False
    
    def connect_blob_storage(self, account_name: str, container_name: str = "telematics-data"):
        """Connect to Azure Blob Storage"""
        try:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=self.credentials
            )
            
            # Create container if it doesn't exist
            try:
                self.blob_service_client.create_container(container_name)
                logger.info(f"Created container: {container_name}")
            except:
                logger.info(f"Container {container_name} already exists")
            
            self.container_name = container_name
            logger.info(f"Connected to Azure Blob Storage: {account_name}")
            return True
            
        except Exception as e:
            logger.error(f"Blob Storage connection failed: {e}")
            return False
    
    def upload_dataframe_to_blob(self, df: pd.DataFrame, blob_name: str, 
                                format: str = "csv") -> bool:
        """Upload DataFrame to Azure Blob Storage"""
        if not self.blob_service_client:
            logger.error("Blob Storage not connected")
            return False
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            if format.lower() == "csv":
                data = df.to_csv(index=False)
            elif format.lower() == "json":
                data = df.to_json(orient='records', indent=2)
            elif format.lower() == "parquet":
                data = df.to_parquet(index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Uploaded {blob_name} to Blob Storage")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def download_dataframe_from_blob(self, blob_name: str, format: str = "csv") -> pd.DataFrame:
        """Download DataFrame from Azure Blob Storage"""
        if not self.blob_service_client:
            logger.error("Blob Storage not connected")
            return pd.DataFrame()
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            data = blob_client.download_blob().readall()
            
            if format.lower() == "csv":
                df = pd.read_csv(io.StringIO(data.decode('utf-8')))
            elif format.lower() == "json":
                df = pd.read_json(io.StringIO(data.decode('utf-8')))
            elif format.lower() == "parquet":
                df = pd.read_parquet(io.BytesIO(data))
            else:
                logger.error(f"Unsupported format: {format}")
                return pd.DataFrame()
            
            logger.info(f"Downloaded {blob_name} from Blob Storage")
            return df
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return pd.DataFrame()
    
    def list_blobs(self, prefix: str = "") -> List[str]:
        """List blobs in container"""
        if not self.blob_service_client:
            logger.error("Blob Storage not connected")
            return []
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"List blobs failed: {e}")
            return []
    
    def delete_blob(self, blob_name: str) -> bool:
        """Delete blob from storage"""
        if not self.blob_service_client:
            logger.error("Blob Storage not connected")
            return False
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Delete blob failed: {e}")
            return False
    
    def connect_cognitive_vision(self, endpoint: str, subscription_key: str):
        """Connect to Azure Cognitive Services Vision"""
        try:
            self.cognitive_vision_client = ComputerVisionClient(
                endpoint=endpoint,
                credentials=CognitiveServicesCredentials(subscription_key)
            )
            logger.info("Connected to Azure Cognitive Services Vision")
            return True
        except Exception as e:
            logger.error(f"Cognitive Vision connection failed: {e}")
            return False
    
    def analyze_image(self, image_url: str) -> Dict[str, Any]:
        """Analyze image using Azure Cognitive Services"""
        if not self.cognitive_vision_client:
            logger.error("Cognitive Vision not connected")
            return {}
        
        try:
            # Analyze image
            analysis = self.cognitive_vision_client.analyze_image(
                image_url,
                visual_features=['Categories', 'Description', 'Tags', 'Objects', 'Faces']
            )
            
            result = {
                'categories': [cat.name for cat in analysis.categories],
                'description': analysis.description.captions[0].text if analysis.description.captions else "",
                'tags': [tag.name for tag in analysis.tags],
                'objects': [obj.object_property for obj in analysis.objects],
                'faces': len(analysis.faces) if analysis.faces else 0
            }
            
            logger.info("Image analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {}
    
    def connect_keyvault(self, vault_url: str):
        """Connect to Azure Key Vault"""
        try:
            self.keyvault_client = SecretClient(
                vault_url=vault_url,
                credential=self.credentials
            )
            logger.info("Connected to Azure Key Vault")
            return True
        except Exception as e:
            logger.error(f"Key Vault connection failed: {e}")
            return False
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault"""
        if not self.keyvault_client:
            logger.error("Key Vault not connected")
            return ""
        
        try:
            secret = self.keyvault_client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error(f"Get secret failed: {e}")
            return ""
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set secret in Azure Key Vault"""
        if not self.keyvault_client:
            logger.error("Key Vault not connected")
            return False
        
        try:
            self.keyvault_client.set_secret(secret_name, secret_value)
            logger.info(f"Secret {secret_name} set successfully")
            return True
        except Exception as e:
            logger.error(f"Set secret failed: {e}")
            return False

class TelematicsDataPipeline:
    """Data pipeline for telematics data processing"""
    
    def __init__(self, azure_manager: AzureServicesManager):
        self.azure_manager = azure_manager
        self.processed_data = {}
    
    def process_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process vehicle data with Azure services"""
        try:
            # Add processing timestamp
            df['processed_at'] = datetime.now()
            
            # Calculate additional metrics
            if 'speed' in df.columns:
                df['speed_category'] = pd.cut(
                    df['speed'], 
                    bins=[0, 30, 60, 90, 120, float('inf')],
                    labels=['Very Slow', 'Slow', 'Normal', 'Fast', 'Very Fast']
                )
            
            if 'acceleration' in df.columns:
                df['acceleration_category'] = pd.cut(
                    df['acceleration'],
                    bins=[float('-inf'), -2, -0.5, 0.5, 2, float('inf')],
                    labels=['Harsh Braking', 'Gentle Braking', 'Normal', 'Gentle Acceleration', 'Harsh Acceleration']
                )
            
            # Calculate risk indicators
            df['risk_score'] = self.calculate_risk_score(df)
            
            logger.info(f"Processed {len(df)} vehicle data records")
            return df
            
        except Exception as e:
            logger.error(f"Vehicle data processing failed: {e}")
            return df
    
    def calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk score for each record"""
        risk_score = pd.Series(0.0, index=df.index)
        
        # Speed risk
        if 'speed' in df.columns:
            speed_risk = (df['speed'] - 60) / 60 * 0.3  # Normalize around 60 km/h
            risk_score += speed_risk.clip(0, 0.3)
        
        # Acceleration risk
        if 'acceleration' in df.columns:
            accel_risk = abs(df['acceleration']) / 5 * 0.2  # Normalize around 5 m/s²
            risk_score += accel_risk.clip(0, 0.2)
        
        # Time risk (night driving)
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            night_risk = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int) * 0.1
            risk_score += night_risk
        
        return risk_score.clip(0, 1)  # Normalize to 0-1
    
    def upload_processed_data(self, df: pd.DataFrame, vehicle_id: str) -> bool:
        """Upload processed data to Azure Blob Storage"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"processed_data/{vehicle_id}_{timestamp}.csv"
            
            return self.azure_manager.upload_dataframe_to_blob(df, blob_name)
            
        except Exception as e:
            logger.error(f"Upload processed data failed: {e}")
            return False
    
    def generate_analytics_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate analytics report"""
        try:
            report = {
                'total_records': len(df),
                'date_range': {
                    'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                },
                'vehicles': df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0,
                'average_speed': df['speed'].mean() if 'speed' in df.columns else 0,
                'max_speed': df['speed'].max() if 'speed' in df.columns else 0,
                'risk_distribution': df['risk_score'].value_counts().to_dict() if 'risk_score' in df.columns else {},
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {}

# Streamlit UI for Azure services
def create_azure_ui():
    """Create Streamlit UI for Azure services"""
    st.markdown("### ☁️ Azure Services Integration")
    
    # Initialize Azure manager
    if 'azure_manager' not in st.session_state:
        st.session_state.azure_manager = AzureServicesManager()
    
    azure_manager = st.session_state.azure_manager
    
    # Azure services tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Authentication", "Blob Storage", "Cognitive Services", "Data Pipeline"])
    
    with tab1:
        st.markdown("#### Azure Authentication")
        
        auth_method = st.radio(
            "Authentication Method",
            ["Default Credentials", "Service Principal"]
        )
        
        if auth_method == "Service Principal":
            col1, col2 = st.columns(2)
            with col1:
                tenant_id = st.text_input("Tenant ID")
                client_id = st.text_input("Client ID")
            with col2:
                client_secret = st.text_input("Client Secret", type="password")
            
            if st.button("Authenticate"):
                if azure_manager.authenticate(tenant_id, client_id, client_secret, use_default=False):
                    st.success("✅ Authentication successful")
                else:
                    st.error("❌ Authentication failed")
        else:
            if st.button("Use Default Credentials"):
                if azure_manager.authenticate(use_default=True):
                    st.success("✅ Default authentication successful")
                else:
                    st.error("❌ Default authentication failed")
    
    with tab2:
        st.markdown("#### Azure Blob Storage")
        
        col1, col2 = st.columns(2)
        with col1:
            account_name = st.text_input("Storage Account Name")
            container_name = st.text_input("Container Name", value="telematics-data")
        
        with col2:
            if st.button("Connect Blob Storage"):
                if azure_manager.connect_blob_storage(account_name, container_name):
                    st.success("✅ Connected to Blob Storage")
                else:
                    st.error("❌ Connection failed")
        
        # Blob operations
        if azure_manager.blob_service_client:
            st.markdown("#### Blob Operations")
            
            operation = st.selectbox(
                "Select Operation",
                ["Upload Data", "Download Data", "List Blobs", "Delete Blob"]
            )
            
            if operation == "Upload Data" and 'data' in st.session_state:
                blob_name = st.text_input("Blob Name", value="vehicle_data.csv")
                format_type = st.selectbox("Format", ["csv", "json", "parquet"])
                
                if st.button("Upload"):
                    if azure_manager.upload_dataframe_to_blob(
                        st.session_state.data, blob_name, format_type
                    ):
                        st.success("✅ Data uploaded successfully")
                    else:
                        st.error("❌ Upload failed")
            
            elif operation == "Download Data":
                blob_name = st.text_input("Blob Name to Download")
                format_type = st.selectbox("Format", ["csv", "json", "parquet"])
                
                if st.button("Download"):
                    df = azure_manager.download_dataframe_from_blob(blob_name, format_type)
                    if not df.empty:
                        st.success("✅ Data downloaded successfully")
                        st.dataframe(df.head())
                    else:
                        st.error("❌ Download failed")
            
            elif operation == "List Blobs":
                prefix = st.text_input("Prefix (optional)")
                if st.button("List Blobs"):
                    blobs = azure_manager.list_blobs(prefix)
                    if blobs:
                        st.success(f"Found {len(blobs)} blobs")
                        for blob in blobs:
                            st.write(f"📄 {blob}")
                    else:
                        st.info("No blobs found")
            
            elif operation == "Delete Blob":
                blob_name = st.text_input("Blob Name to Delete")
                if st.button("Delete"):
                    if azure_manager.delete_blob(blob_name):
                        st.success("✅ Blob deleted successfully")
                    else:
                        st.error("❌ Delete failed")
    
    with tab3:
        st.markdown("#### Azure Cognitive Services")
        
        col1, col2 = st.columns(2)
        with col1:
            vision_endpoint = st.text_input("Vision Endpoint")
            vision_key = st.text_input("Vision Subscription Key", type="password")
        
        with col2:
            if st.button("Connect Vision Service"):
                if azure_manager.connect_cognitive_vision(vision_endpoint, vision_key):
                    st.success("✅ Connected to Vision Service")
                else:
                    st.error("❌ Connection failed")
        
        # Image analysis
        if azure_manager.cognitive_vision_client:
            st.markdown("#### Image Analysis")
            image_url = st.text_input("Image URL")
            
            if st.button("Analyze Image"):
                result = azure_manager.analyze_image(image_url)
                if result:
                    st.success("✅ Image analysis completed")
                    st.json(result)
                else:
                    st.error("❌ Analysis failed")
    
    with tab4:
        st.markdown("#### Data Pipeline")
        
        if 'data' in st.session_state and azure_manager.blob_service_client:
            if st.button("Process and Upload Data"):
                pipeline = TelematicsDataPipeline(azure_manager)
                
                # Process data
                processed_df = pipeline.process_vehicle_data(st.session_state.data)
                
                # Generate report
                report = pipeline.generate_analytics_report(processed_df)
                
                # Upload processed data
                if 'vehicle_id' in processed_df.columns:
                    vehicle_id = processed_df['vehicle_id'].iloc[0]
                    if pipeline.upload_processed_data(processed_df, vehicle_id):
                        st.success("✅ Data processed and uploaded")
                        st.json(report)
                    else:
                        st.error("❌ Upload failed")
                else:
                    st.error("❌ No vehicle_id column found")
        else:
            st.info("Load data and connect to Azure services to use the pipeline")

if __name__ == "__main__":
    # Test Azure services
    azure_manager = AzureServicesManager()
    
    # Test authentication
    if azure_manager.authenticate():
        print("Azure authentication successful")
    
    print("Azure services module loaded successfully")
