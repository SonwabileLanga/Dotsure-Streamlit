"""
Database Manager for DOTSURE Telematics Platform
Handles multiple database connections and operations
"""

import pandas as pd
import sqlite3
import psycopg2
import pymysql
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import logging
from typing import Dict, List, Optional, Any
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Unified database manager for multiple database types"""
    
    def __init__(self):
        self.connections = {}
        self.engines = {}
        
    def connect_sqlite(self, db_path: str = "telematics.db") -> bool:
        """Connect to SQLite database"""
        try:
            engine = create_engine(f"sqlite:///{db_path}", echo=False)
            self.engines['sqlite'] = engine
            self.connections['sqlite'] = engine.connect()
            logger.info(f"Connected to SQLite database: {db_path}")
            return True
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            return False
    
    def connect_postgresql(self, host: str, port: int, database: str, 
                          username: str, password: str) -> bool:
        """Connect to PostgreSQL database"""
        try:
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            self.engines['postgresql'] = engine
            self.connections['postgresql'] = engine.connect()
            logger.info(f"Connected to PostgreSQL: {host}:{port}/{database}")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def connect_mysql(self, host: str, port: int, database: str, 
                     username: str, password: str) -> bool:
        """Connect to MySQL database"""
        try:
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            self.engines['mysql'] = engine
            self.connections['mysql'] = engine.connect()
            logger.info(f"Connected to MySQL: {host}:{port}/{database}")
            return True
        except Exception as e:
            logger.error(f"MySQL connection failed: {e}")
            return False
    
    def create_tables(self, db_type: str = 'sqlite'):
        """Create telematics tables if they don't exist"""
        if db_type not in self.engines:
            logger.error(f"Database {db_type} not connected")
            return False
        
        try:
            engine = self.engines[db_type]
            
            # Vehicle data table
            vehicle_table_sql = """
            CREATE TABLE IF NOT EXISTS vehicle_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id VARCHAR(50) NOT NULL,
                timestamp DATETIME NOT NULL,
                latitude DECIMAL(10, 8),
                longitude DECIMAL(11, 8),
                speed DECIMAL(5, 2),
                acceleration DECIMAL(5, 2),
                heading DECIMAL(5, 2),
                altitude DECIMAL(8, 2),
                fuel_level DECIMAL(5, 2),
                engine_rpm INTEGER,
                odometer DECIMAL(10, 2),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Events table
            events_table_sql = """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id VARCHAR(50) NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                event_severity VARCHAR(20),
                timestamp DATETIME NOT NULL,
                latitude DECIMAL(10, 8),
                longitude DECIMAL(11, 8),
                speed DECIMAL(5, 2),
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Risk scores table
            risk_scores_table_sql = """
            CREATE TABLE IF NOT EXISTS risk_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id VARCHAR(50) NOT NULL,
                driver_id VARCHAR(50),
                score_date DATE NOT NULL,
                overall_score DECIMAL(5, 2),
                speeding_score DECIMAL(5, 2),
                braking_score DECIMAL(5, 2),
                acceleration_score DECIMAL(5, 2),
                cornering_score DECIMAL(5, 2),
                time_score DECIMAL(5, 2),
                distance_score DECIMAL(5, 2),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Execute table creation
            with engine.connect() as conn:
                conn.execute(text(vehicle_table_sql))
                conn.execute(text(events_table_sql))
                conn.execute(text(risk_scores_table_sql))
                conn.commit()
            
            logger.info(f"Tables created successfully in {db_type}")
            return True
            
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            return False
    
    def insert_vehicle_data(self, df: pd.DataFrame, db_type: str = 'sqlite') -> bool:
        """Insert vehicle data into database"""
        if db_type not in self.engines:
            logger.error(f"Database {db_type} not connected")
            return False
        
        try:
            engine = self.engines[db_type]
            
            # Prepare data for insertion
            df_clean = df.copy()
            df_clean['created_at'] = datetime.now()
            
            # Insert data
            df_clean.to_sql('vehicle_data', engine, if_exists='append', index=False)
            logger.info(f"Inserted {len(df_clean)} records into vehicle_data table")
            return True
            
        except Exception as e:
            logger.error(f"Data insertion failed: {e}")
            return False
    
    def insert_events(self, events_df: pd.DataFrame, db_type: str = 'sqlite') -> bool:
        """Insert events data into database"""
        if db_type not in self.engines:
            logger.error(f"Database {db_type} not connected")
            return False
        
        try:
            engine = self.engines[db_type]
            events_df['created_at'] = datetime.now()
            events_df.to_sql('events', engine, if_exists='append', index=False)
            logger.info(f"Inserted {len(events_df)} events into events table")
            return True
            
        except Exception as e:
            logger.error(f"Events insertion failed: {e}")
            return False
    
    def get_vehicle_data(self, vehicle_id: str = None, 
                        start_date: datetime = None, 
                        end_date: datetime = None,
                        db_type: str = 'sqlite') -> pd.DataFrame:
        """Retrieve vehicle data from database"""
        if db_type not in self.engines:
            logger.error(f"Database {db_type} not connected")
            return pd.DataFrame()
        
        try:
            engine = self.engines[db_type]
            
            query = "SELECT * FROM vehicle_data WHERE 1=1"
            params = {}
            
            if vehicle_id:
                query += " AND vehicle_id = :vehicle_id"
                params['vehicle_id'] = vehicle_id
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql(query, engine, params=params)
            logger.info(f"Retrieved {len(df)} records from vehicle_data")
            return df
            
        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            return pd.DataFrame()
    
    def get_events(self, vehicle_id: str = None, 
                  event_type: str = None,
                  start_date: datetime = None,
                  end_date: datetime = None,
                  db_type: str = 'sqlite') -> pd.DataFrame:
        """Retrieve events from database"""
        if db_type not in self.engines:
            logger.error(f"Database {db_type} not connected")
            return pd.DataFrame()
        
        try:
            engine = self.engines[db_type]
            
            query = "SELECT * FROM events WHERE 1=1"
            params = {}
            
            if vehicle_id:
                query += " AND vehicle_id = :vehicle_id"
                params['vehicle_id'] = vehicle_id
            
            if event_type:
                query += " AND event_type = :event_type"
                params['event_type'] = event_type
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql(query, engine, params=params)
            logger.info(f"Retrieved {len(df)} events")
            return df
            
        except Exception as e:
            logger.error(f"Events retrieval failed: {e}")
            return pd.DataFrame()
    
    def get_risk_scores(self, vehicle_id: str = None, 
                       start_date: datetime = None,
                       end_date: datetime = None,
                       db_type: str = 'sqlite') -> pd.DataFrame:
        """Retrieve risk scores from database"""
        if db_type not in self.engines:
            logger.error(f"Database {db_type} not connected")
            return pd.DataFrame()
        
        try:
            engine = self.engines[db_type]
            
            query = "SELECT * FROM risk_scores WHERE 1=1"
            params = {}
            
            if vehicle_id:
                query += " AND vehicle_id = :vehicle_id"
                params['vehicle_id'] = vehicle_id
            
            if start_date:
                query += " AND score_date >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND score_date <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY score_date DESC"
            
            df = pd.read_sql(query, engine, params=params)
            logger.info(f"Retrieved {len(df)} risk scores")
            return df
            
        except Exception as e:
            logger.error(f"Risk scores retrieval failed: {e}")
            return pd.DataFrame()
    
    def execute_custom_query(self, query: str, params: Dict = None, 
                           db_type: str = 'sqlite') -> pd.DataFrame:
        """Execute custom SQL query"""
        if db_type not in self.engines:
            logger.error(f"Database {db_type} not connected")
            return pd.DataFrame()
        
        try:
            engine = self.engines[db_type]
            df = pd.read_sql(query, engine, params=params or {})
            logger.info(f"Custom query executed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Custom query failed: {e}")
            return pd.DataFrame()
    
    def close_connections(self):
        """Close all database connections"""
        for name, connection in self.connections.items():
            try:
                connection.close()
                logger.info(f"Closed {name} connection")
            except Exception as e:
                logger.error(f"Error closing {name} connection: {e}")
        
        for name, engine in self.engines.items():
            try:
                engine.dispose()
                logger.info(f"Disposed {name} engine")
            except Exception as e:
                logger.error(f"Error disposing {name} engine: {e}")
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get status of all database connections"""
        status = {}
        for name, engine in self.engines.items():
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                status[name] = True
            except:
                status[name] = False
        return status

# Streamlit UI for database management
def create_database_ui():
    """Create Streamlit UI for database management"""
    st.markdown("### 🗄️ Database Management")
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Database connection tabs
    tab1, tab2, tab3, tab4 = st.tabs(["SQLite", "PostgreSQL", "MySQL", "Status"])
    
    with tab1:
        st.markdown("#### SQLite Connection")
        db_path = st.text_input("Database Path", value="telematics.db")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect SQLite"):
                if db_manager.connect_sqlite(db_path):
                    st.success("✅ Connected to SQLite")
                else:
                    st.error("❌ Connection failed")
        
        with col2:
            if st.button("Create Tables"):
                if db_manager.create_tables('sqlite'):
                    st.success("✅ Tables created")
                else:
                    st.error("❌ Table creation failed")
    
    with tab2:
        st.markdown("#### PostgreSQL Connection")
        col1, col2 = st.columns(2)
        
        with col1:
            pg_host = st.text_input("Host", value="localhost")
            pg_port = st.number_input("Port", value=5432)
            pg_database = st.text_input("Database", value="telematics")
        
        with col2:
            pg_username = st.text_input("Username")
            pg_password = st.text_input("Password", type="password")
        
        if st.button("Connect PostgreSQL"):
            if db_manager.connect_postgresql(pg_host, pg_port, pg_database, pg_username, pg_password):
                st.success("✅ Connected to PostgreSQL")
            else:
                st.error("❌ Connection failed")
    
    with tab3:
        st.markdown("#### MySQL Connection")
        col1, col2 = st.columns(2)
        
        with col1:
            mysql_host = st.text_input("MySQL Host", value="localhost")
            mysql_port = st.number_input("MySQL Port", value=3306)
            mysql_database = st.text_input("MySQL Database", value="telematics")
        
        with col2:
            mysql_username = st.text_input("MySQL Username")
            mysql_password = st.text_input("MySQL Password", type="password")
        
        if st.button("Connect MySQL"):
            if db_manager.connect_mysql(mysql_host, mysql_port, mysql_database, mysql_username, mysql_password):
                st.success("✅ Connected to MySQL")
            else:
                st.error("❌ Connection failed")
    
    with tab4:
        st.markdown("#### Connection Status")
        status = db_manager.get_connection_status()
        
        for db_type, is_connected in status.items():
            if is_connected:
                st.success(f"✅ {db_type.upper()} - Connected")
            else:
                st.error(f"❌ {db_type.upper()} - Disconnected")
        
        if st.button("Close All Connections"):
            db_manager.close_connections()
            st.success("All connections closed")

if __name__ == "__main__":
    # Test the database manager
    db_manager = DatabaseManager()
    
    # Test SQLite connection
    if db_manager.connect_sqlite():
        print("SQLite connected successfully")
        
        # Create tables
        if db_manager.create_tables():
            print("Tables created successfully")
        
        # Test data insertion
        test_data = pd.DataFrame({
            'vehicle_id': ['V001', 'V002'],
            'timestamp': [datetime.now(), datetime.now()],
            'latitude': [40.7128, 40.7589],
            'longitude': [-74.0060, -73.9851],
            'speed': [65.5, 72.3],
            'acceleration': [0.2, -0.5]
        })
        
        if db_manager.insert_vehicle_data(test_data):
            print("Test data inserted successfully")
        
        # Test data retrieval
        retrieved_data = db_manager.get_vehicle_data()
        print(f"Retrieved {len(retrieved_data)} records")
    
    db_manager.close_connections()
