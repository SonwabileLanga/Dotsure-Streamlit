"""
AI/ML Engine for DOTSURE Telematics Platform
Handles risk scoring, alerting, and event detection using machine learning
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from scipy import stats
from scipy.signal import find_peaks
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskScoringEngine:
    """Advanced risk scoring engine using machine learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for risk scoring"""
        try:
            features_df = df.copy()
            
            # Time-based features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                features_df['hour'] = df['timestamp'].dt.hour
                features_df['day_of_week'] = df['timestamp'].dt.dayofweek
                features_df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
                features_df['is_night'] = ((df['timestamp'].dt.hour >= 22) | 
                                         (df['timestamp'].dt.hour <= 6)).astype(int)
            
            # Speed-based features
            if 'speed' in df.columns:
                features_df['speed_above_limit'] = (df['speed'] - 60).clip(0, None)
                features_df['speed_variance'] = df.groupby('vehicle_id')['speed'].transform('std').fillna(0)
                features_df['speed_percentile'] = df.groupby('vehicle_id')['speed'].transform(
                    lambda x: x.rank(pct=True)
                )
            
            # Acceleration-based features
            if 'acceleration' in df.columns:
                features_df['harsh_acceleration'] = (df['acceleration'] > 2).astype(int)
                features_df['harsh_braking'] = (df['acceleration'] < -2).astype(int)
                features_df['acceleration_variance'] = df.groupby('vehicle_id')['acceleration'].transform('std').fillna(0)
            
            # Location-based features
            if 'latitude' in df.columns and 'longitude' in df.columns:
                # Calculate distance between consecutive points
                features_df['distance'] = self.calculate_distance(df)
                features_df['speed_calculated'] = features_df['distance'] / 3600  # km/h assuming 1-hour intervals
                
                # Urban vs rural classification (simplified)
                features_df['is_urban'] = self.classify_urban_rural(df)
            
            # Weather features (if available)
            if 'weather' in df.columns:
                weather_encoder = LabelEncoder()
                features_df['weather_encoded'] = weather_encoder.fit_transform(df['weather'].fillna('Unknown'))
            
            # Rolling window features
            if 'speed' in df.columns:
                features_df['speed_rolling_mean'] = df.groupby('vehicle_id')['speed'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
                features_df['speed_rolling_std'] = df.groupby('vehicle_id')['speed'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std()
                )
            
            logger.info(f"Extracted {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return df
    
    def calculate_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance between consecutive GPS points"""
        try:
            distances = []
            for i in range(len(df)):
                if i == 0:
                    distances.append(0)
                else:
                    lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
                    lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
                    
                    # Haversine formula
                    R = 6371  # Earth's radius in kilometers
                    dlat = math.radians(lat2 - lat1)
                    dlon = math.radians(lon2 - lon1)
                    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                         math.sin(dlon/2) * math.sin(dlon/2))
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    distance = R * c
                    distances.append(distance)
            
            return pd.Series(distances, index=df.index)
            
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return pd.Series(0, index=df.index)
    
    def classify_urban_rural(self, df: pd.DataFrame) -> pd.Series:
        """Classify locations as urban or rural (simplified)"""
        try:
            # Simple classification based on coordinate density
            # In a real implementation, this would use proper geographic data
            urban_threshold = 0.01  # degrees
            
            classifications = []
            for _, row in df.iterrows():
                lat, lon = row['latitude'], row['longitude']
                
                # Count nearby points (simplified urban density)
                nearby_points = df[
                    (abs(df['latitude'] - lat) < urban_threshold) & 
                    (abs(df['longitude'] - lon) < urban_threshold)
                ]
                
                # If more than 10 points nearby, consider it urban
                is_urban = len(nearby_points) > 10
                classifications.append(int(is_urban))
            
            return pd.Series(classifications, index=df.index)
            
        except Exception as e:
            logger.error(f"Urban/rural classification failed: {e}")
            return pd.Series(0, index=df.index)
    
    def train_risk_model(self, df: pd.DataFrame, target_column: str = 'accident_severity'):
        """Train risk scoring model"""
        try:
            # Extract features
            features_df = self.extract_features(df)
            
            # Prepare features and target
            feature_columns = [col for col in features_df.columns 
                             if col not in ['vehicle_id', 'timestamp', 'accident_severity', 'latitude', 'longitude']]
            
            X = features_df[feature_columns].fillna(0)
            
            # Create target variable (1 for accidents, 0 for no accidents)
            if target_column in df.columns:
                y = (df[target_column] != 'None').astype(int)
            else:
                # Create synthetic target based on risk indicators
                y = self.create_synthetic_target(features_df)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['risk_scaler'] = scaler
            
            # Train Random Forest model
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            rf_model.fit(X_scaled, y)
            self.models['risk_classifier'] = rf_model
            
            # Calculate feature importance
            self.feature_importance['risk'] = dict(zip(feature_columns, rf_model.feature_importances_))
            
            # Train regression model for continuous risk scores
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            gb_model.fit(X_scaled, y)
            self.models['risk_regressor'] = gb_model
            
            self.is_trained = True
            logger.info("Risk scoring model trained successfully")
            
            return {
                'feature_columns': feature_columns,
                'model_score': rf_model.score(X_scaled, y),
                'feature_importance': self.feature_importance['risk']
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {}
    
    def create_synthetic_target(self, df: pd.DataFrame) -> pd.Series:
        """Create synthetic target variable for training"""
        try:
            # Create risk score based on multiple factors
            risk_score = pd.Series(0.0, index=df.index)
            
            # Speed risk
            if 'speed' in df.columns:
                speed_risk = (df['speed'] - 60) / 60 * 0.3
                risk_score += speed_risk.clip(0, 0.3)
            
            # Acceleration risk
            if 'acceleration' in df.columns:
                accel_risk = abs(df['acceleration']) / 5 * 0.2
                risk_score += accel_risk.clip(0, 0.2)
            
            # Time risk
            if 'is_night' in df.columns:
                risk_score += df['is_night'] * 0.1
            
            # Weekend risk
            if 'is_weekend' in df.columns:
                risk_score += df['is_weekend'] * 0.05
            
            # Convert to binary (threshold at 0.3)
            return (risk_score > 0.3).astype(int)
            
        except Exception as e:
            logger.error(f"Synthetic target creation failed: {e}")
            return pd.Series(0, index=df.index)
    
    def predict_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict risk scores for new data"""
        try:
            if not self.is_trained:
                logger.error("Model not trained yet")
                return df
            
            # Extract features
            features_df = self.extract_features(df)
            
            # Get feature columns (same as training)
            feature_columns = list(self.feature_importance['risk'].keys())
            X = features_df[feature_columns].fillna(0)
            
            # Scale features
            X_scaled = self.scalers['risk_scaler'].transform(X)
            
            # Predict risk scores
            risk_probabilities = self.models['risk_classifier'].predict_proba(X_scaled)[:, 1]
            risk_scores = self.models['risk_regressor'].predict(X_scaled)
            
            # Combine predictions
            df['risk_probability'] = risk_probabilities
            df['risk_score'] = risk_scores
            df['risk_level'] = pd.cut(risk_scores, 
                                    bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                    labels=['Low', 'Medium', 'High', 'Critical'])
            
            logger.info(f"Predicted risk scores for {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return df

class EventDetectionEngine:
    """Advanced event detection engine"""
    
    def __init__(self):
        self.anomaly_model = None
        self.is_trained = False
        
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in vehicle data"""
        try:
            # Prepare features for anomaly detection
            feature_columns = ['speed', 'acceleration']
            if 'latitude' in df.columns and 'longitude' in df.columns:
                feature_columns.extend(['latitude', 'longitude'])
            
            X = df[feature_columns].fillna(0)
            
            # Train Isolation Forest for anomaly detection
            if not self.is_trained:
                self.anomaly_model = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self.anomaly_model.fit(X)
                self.is_trained = True
            
            # Predict anomalies
            anomaly_scores = self.anomaly_model.decision_function(X)
            anomaly_predictions = self.anomaly_model.predict(X)
            
            df['anomaly_score'] = anomaly_scores
            df['is_anomaly'] = (anomaly_predictions == -1).astype(int)
            
            logger.info(f"Detected {df['is_anomaly'].sum()} anomalies")
            return df
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return df
    
    def detect_harsh_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect harsh driving events"""
        try:
            events = []
            
            if 'acceleration' in df.columns:
                # Harsh acceleration detection
                harsh_accel = df[df['acceleration'] > 2]
                for _, row in harsh_accel.iterrows():
                    events.append({
                        'timestamp': row.get('timestamp', datetime.now()),
                        'vehicle_id': row.get('vehicle_id', 'Unknown'),
                        'event_type': 'Harsh Acceleration',
                        'severity': 'High' if row['acceleration'] > 3 else 'Medium',
                        'value': row['acceleration'],
                        'latitude': row.get('latitude', 0),
                        'longitude': row.get('longitude', 0)
                    })
                
                # Harsh braking detection
                harsh_brake = df[df['acceleration'] < -2]
                for _, row in harsh_brake.iterrows():
                    events.append({
                        'timestamp': row.get('timestamp', datetime.now()),
                        'vehicle_id': row.get('vehicle_id', 'Unknown'),
                        'event_type': 'Harsh Braking',
                        'severity': 'High' if row['acceleration'] < -3 else 'Medium',
                        'value': row['acceleration'],
                        'latitude': row.get('latitude', 0),
                        'longitude': row.get('longitude', 0)
                    })
            
            if 'speed' in df.columns:
                # Speeding detection
                speeding = df[df['speed'] > 80]  # Assuming 80 km/h is speeding
                for _, row in speeding.iterrows():
                    events.append({
                        'timestamp': row.get('timestamp', datetime.now()),
                        'vehicle_id': row.get('vehicle_id', 'Unknown'),
                        'event_type': 'Speeding',
                        'severity': 'High' if row['speed'] > 100 else 'Medium',
                        'value': row['speed'],
                        'latitude': row.get('latitude', 0),
                        'longitude': row.get('longitude', 0)
                    })
            
            events_df = pd.DataFrame(events)
            logger.info(f"Detected {len(events)} harsh driving events")
            return events_df
            
        except Exception as e:
            logger.error(f"Harsh event detection failed: {e}")
            return pd.DataFrame()
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect driving patterns and behaviors"""
        try:
            patterns = {}
            
            if 'vehicle_id' in df.columns:
                for vehicle_id in df['vehicle_id'].unique():
                    vehicle_data = df[df['vehicle_id'] == vehicle_id]
                    
                    vehicle_patterns = {
                        'total_distance': 0,
                        'average_speed': 0,
                        'max_speed': 0,
                        'harsh_events': 0,
                        'night_driving': 0,
                        'weekend_driving': 0
                    }
                    
                    # Calculate patterns
                    if 'speed' in vehicle_data.columns:
                        vehicle_patterns['average_speed'] = vehicle_data['speed'].mean()
                        vehicle_patterns['max_speed'] = vehicle_data['speed'].max()
                    
                    if 'acceleration' in vehicle_data.columns:
                        vehicle_patterns['harsh_events'] = len(
                            vehicle_data[(vehicle_data['acceleration'] > 2) | 
                                       (vehicle_data['acceleration'] < -2)]
                        )
                    
                    if 'timestamp' in vehicle_data.columns:
                        vehicle_data['timestamp'] = pd.to_datetime(vehicle_data['timestamp'])
                        vehicle_patterns['night_driving'] = len(
                            vehicle_data[(vehicle_data['timestamp'].dt.hour >= 22) | 
                                       (vehicle_data['timestamp'].dt.hour <= 6)]
                        )
                        vehicle_patterns['weekend_driving'] = len(
                            vehicle_data[vehicle_data['timestamp'].dt.dayofweek >= 5]
                        )
                    
                    patterns[vehicle_id] = vehicle_patterns
            
            logger.info(f"Detected patterns for {len(patterns)} vehicles")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {}

class AlertingSystem:
    """Intelligent alerting system"""
    
    def __init__(self):
        self.alert_rules = {}
        self.alert_history = []
        
    def add_alert_rule(self, rule_name: str, condition: str, severity: str, 
                      message_template: str):
        """Add custom alert rule"""
        self.alert_rules[rule_name] = {
            'condition': condition,
            'severity': severity,
            'message_template': message_template,
            'created_at': datetime.now()
        }
    
    def evaluate_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluate all alert rules against data"""
        try:
            alerts = []
            
            # Default alert rules
            if 'risk_score' in df.columns:
                high_risk = df[df['risk_score'] > 0.8]
                for _, row in high_risk.iterrows():
                    alerts.append({
                        'timestamp': datetime.now(),
                        'vehicle_id': row.get('vehicle_id', 'Unknown'),
                        'alert_type': 'High Risk Score',
                        'severity': 'Critical',
                        'message': f"Vehicle {row.get('vehicle_id', 'Unknown')} has high risk score: {row['risk_score']:.2f}",
                        'data': row.to_dict()
                    })
            
            if 'is_anomaly' in df.columns:
                anomalies = df[df['is_anomaly'] == 1]
                for _, row in anomalies.iterrows():
                    alerts.append({
                        'timestamp': datetime.now(),
                        'vehicle_id': row.get('vehicle_id', 'Unknown'),
                        'alert_type': 'Anomaly Detected',
                        'severity': 'High',
                        'message': f"Anomalous behavior detected for vehicle {row.get('vehicle_id', 'Unknown')}",
                        'data': row.to_dict()
                    })
            
            # Custom alert rules
            for rule_name, rule in self.alert_rules.items():
                # Simple condition evaluation (in production, use proper expression evaluator)
                if self.evaluate_condition(df, rule['condition']):
                    alerts.append({
                        'timestamp': datetime.now(),
                        'vehicle_id': 'Multiple',
                        'alert_type': rule_name,
                        'severity': rule['severity'],
                        'message': rule['message_template'],
                        'data': {}
                    })
            
            self.alert_history.extend(alerts)
            logger.info(f"Generated {len(alerts)} alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Alert evaluation failed: {e}")
            return []
    
    def evaluate_condition(self, df: pd.DataFrame, condition: str) -> bool:
        """Evaluate alert condition (simplified)"""
        try:
            # Simple condition evaluation
            if 'speed > 100' in condition:
                return (df['speed'] > 100).any()
            elif 'acceleration > 3' in condition:
                return (df['acceleration'] > 3).any()
            elif 'risk_score > 0.9' in condition:
                return (df['risk_score'] > 0.9).any()
            else:
                return False
        except:
            return False

# Streamlit UI for AI/ML features
def create_ai_ml_ui():
    """Create Streamlit UI for AI/ML features"""
    st.markdown("### 🤖 AI/ML Engine")
    
    # Initialize engines
    if 'risk_engine' not in st.session_state:
        st.session_state.risk_engine = RiskScoringEngine()
    
    if 'event_engine' not in st.session_state:
        st.session_state.event_engine = EventDetectionEngine()
    
    if 'alerting_system' not in st.session_state:
        st.session_state.alerting_system = AlertingSystem()
    
    risk_engine = st.session_state.risk_engine
    event_engine = st.session_state.event_engine
    alerting_system = st.session_state.alerting_system
    
    # AI/ML tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Scoring", "Event Detection", "Pattern Analysis", "Alerting"])
    
    with tab1:
        st.markdown("#### Risk Scoring Engine")
        
        if 'data' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Train Risk Model"):
                    with st.spinner("Training risk scoring model..."):
                        result = risk_engine.train_risk_model(st.session_state.data)
                        if result:
                            st.success("✅ Risk model trained successfully")
                            st.json(result)
                        else:
                            st.error("❌ Model training failed")
            
            with col2:
                if st.button("Predict Risk Scores"):
                    if risk_engine.is_trained:
                        with st.spinner("Predicting risk scores..."):
                            df_with_scores = risk_engine.predict_risk_scores(st.session_state.data)
                            st.session_state.data = df_with_scores
                            st.success("✅ Risk scores predicted")
                            
                            # Display risk distribution
                            if 'risk_score' in df_with_scores.columns:
                                fig = px.histogram(df_with_scores, x='risk_score', 
                                                 title='Risk Score Distribution')
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Please train the model first")
            
            # Feature importance
            if risk_engine.feature_importance:
                st.markdown("#### Feature Importance")
                importance_df = pd.DataFrame(
                    list(risk_engine.feature_importance['risk'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                           title='Top 10 Most Important Features')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please load data first")
    
    with tab2:
        st.markdown("#### Event Detection Engine")
        
        if 'data' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        df_with_anomalies = event_engine.detect_anomalies(st.session_state.data)
                        st.session_state.data = df_with_anomalies
                        st.success("✅ Anomaly detection completed")
                        
                        if 'is_anomaly' in df_with_anomalies.columns:
                            anomaly_count = df_with_anomalies['is_anomaly'].sum()
                            st.metric("Anomalies Detected", anomaly_count)
            
            with col2:
                if st.button("Detect Harsh Events"):
                    with st.spinner("Detecting harsh driving events..."):
                        events_df = event_engine.detect_harsh_events(st.session_state.data)
                        if not events_df.empty:
                            st.success(f"✅ Detected {len(events_df)} harsh events")
                            st.dataframe(events_df)
                        else:
                            st.info("No harsh events detected")
            
            # Anomaly visualization
            if 'anomaly_score' in st.session_state.data.columns:
                st.markdown("#### Anomaly Analysis")
                
                fig = px.scatter(st.session_state.data, x='speed', y='acceleration',
                               color='anomaly_score', title='Anomaly Detection Results')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please load data first")
    
    with tab3:
        st.markdown("#### Pattern Analysis")
        
        if 'data' in st.session_state:
            if st.button("Analyze Driving Patterns"):
                with st.spinner("Analyzing driving patterns..."):
                    patterns = event_engine.detect_patterns(st.session_state.data)
                    
                    if patterns:
                        st.success("✅ Pattern analysis completed")
                        
                        # Convert patterns to DataFrame for visualization
                        patterns_df = pd.DataFrame(patterns).T
                        
                        # Display patterns
                        st.dataframe(patterns_df)
                        
                        # Visualize patterns
                        if 'average_speed' in patterns_df.columns:
                            fig = px.bar(patterns_df, x=patterns_df.index, y='average_speed',
                                       title='Average Speed by Vehicle')
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No patterns detected")
        else:
            st.info("Please load data first")
    
    with tab4:
        st.markdown("#### Alerting System")
        
        # Add custom alert rules
        st.markdown("##### Custom Alert Rules")
        
        with st.form("alert_rule_form"):
            rule_name = st.text_input("Rule Name")
            condition = st.selectbox("Condition", 
                                   ["speed > 100", "acceleration > 3", "risk_score > 0.9"])
            severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
            message = st.text_area("Alert Message")
            
            if st.form_submit_button("Add Alert Rule"):
                alerting_system.add_alert_rule(rule_name, condition, severity, message)
                st.success("✅ Alert rule added")
        
        # Evaluate alerts
        if 'data' in st.session_state:
            if st.button("Evaluate Alerts"):
                with st.spinner("Evaluating alerts..."):
                    alerts = alerting_system.evaluate_alerts(st.session_state.data)
                    
                    if alerts:
                        st.success(f"✅ Generated {len(alerts)} alerts")
                        
                        # Display alerts
                        for alert in alerts:
                            st.warning(f"🚨 {alert['alert_type']}: {alert['message']}")
                    else:
                        st.info("No alerts generated")
        
        # Alert history
        if alerting_system.alert_history:
            st.markdown("##### Alert History")
            history_df = pd.DataFrame(alerting_system.alert_history)
            st.dataframe(history_df)

if __name__ == "__main__":
    # Test AI/ML engine
    risk_engine = RiskScoringEngine()
    event_engine = EventDetectionEngine()
    alerting_system = AlertingSystem()
    
    print("AI/ML engine module loaded successfully")
