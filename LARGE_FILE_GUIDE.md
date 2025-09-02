# 🚀 Large File Handling Guide (2GB+)

## 📋 **Streamlit Upload Limit**

**Streamlit has a 200MB upload limit per file.** However, we provide multiple solutions to handle files larger than 2GB!

## 🛠️ **Solution 1: Split Your Files (Recommended)**

### **Step 1: Use the CSV Splitter Tool**
```bash
# Download the splitter tool (included in this project)
python csv_splitter.py your_large_file.csv 50000 chunks_
```

### **Step 2: What This Creates**
```
split_files/
├── chunk_001.csv (50,000 rows)
├── chunk_002.csv (50,000 rows)
├── chunk_003.csv (50,000 rows)
├── chunk_004.csv (50,000 rows)
└── manifest.txt (file list)
```

### **Step 3: Upload Chunks**
- Upload individual chunks using "Upload CSV Files"
- Or upload chunks to cloud storage and use URLs

## ☁️ **Solution 2: Cloud Storage (Best for 2GB+)**

### **Google Drive Method**
1. **Upload your large file** to Google Drive
2. **Right-click** → Share → Anyone with the link
3. **Get the file ID** from the URL
4. **Create export URL**: `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID`
5. **Use "Load from URL"** in the dashboard

### **Dropbox Method**
1. **Upload your file** to Dropbox
2. **Right-click** → Share → Create link
3. **Change the URL**: Replace `www.dropbox.com` with `dl.dropboxusercontent.com`
4. **Use "Load from URL"** in the dashboard

### **OneDrive Method**
1. **Upload your file** to OneDrive
2. **Right-click** → Share → Anyone with the link
3. **Get the direct download URL**
4. **Use "Load from URL"** in the dashboard

### **AWS S3 Method**
1. **Upload to S3** public bucket
2. **Get the object URL**
3. **Use "Load from URL"** in the dashboard

## 🗄️ **Solution 3: Database Connection**

### **PostgreSQL**
```sql
-- Import your CSV to PostgreSQL
COPY accidents FROM '/path/to/your/file.csv' DELIMITER ',' CSV HEADER;
```

### **MySQL**
```sql
-- Import your CSV to MySQL
LOAD DATA INFILE '/path/to/your/file.csv'
INTO TABLE accidents
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
```

### **SQLite**
```bash
# Convert CSV to SQLite
sqlite3 accidents.db
.mode csv
.import your_file.csv accidents
```

## 📊 **Solution 4: Data Sampling**

### **Python Sampling Script**
```python
import pandas as pd

# Load your large file
df = pd.read_csv('your_large_file.csv')

# Sample 100,000 random rows
sample_df = df.sample(n=100000, random_state=42)

# Save the sample
sample_df.to_csv('sample_data.csv', index=False)
```

### **Time-based Sampling**
```python
# Get data from last 6 months
df['date'] = pd.to_datetime(df['date'])
recent_data = df[df['date'] >= '2024-01-01']
recent_data.to_csv('recent_data.csv', index=False)
```

## 🔧 **Solution 5: File Compression**

### **Compress Your CSV**
```bash
# Compress with gzip (reduces size by 70-80%)
gzip your_large_file.csv
# Creates: your_large_file.csv.gz
```

### **Use Parquet Format**
```python
import pandas as pd

# Convert CSV to Parquet (much smaller)
df = pd.read_csv('your_large_file.csv')
df.to_parquet('your_large_file.parquet', compression='snappy')
```

## 📈 **File Size Calculator**

| Rows | Estimated Size | Recommended Method |
|------|----------------|-------------------|
| 100K | ~50MB | Direct upload |
| 500K | ~250MB | Split or cloud storage |
| 1M | ~500MB | Cloud storage |
| 5M | ~2.5GB | Database or sampling |
| 10M | ~5GB | Database only |

## 🎯 **Best Practices**

### **For Files 200MB - 1GB**
1. **Split into chunks** (50K-100K rows each)
2. **Upload chunks** individually
3. **Use cloud storage** URLs

### **For Files 1GB - 5GB**
1. **Use cloud storage** (Google Drive, Dropbox)
2. **Database connection** (PostgreSQL, MySQL)
3. **Data sampling** for initial analysis

### **For Files 5GB+**
1. **Database connection** only
2. **Data sampling** for exploration
3. **Aggregate data** before import

## 🚀 **Quick Start Commands**

### **Split Large File**
```bash
python csv_splitter.py your_file.csv 50000 chunks_
```

### **Sample Large File**
```python
import pandas as pd
df = pd.read_csv('your_file.csv')
df.sample(50000).to_csv('sample.csv', index=False)
```

### **Compress File**
```bash
gzip your_file.csv
```

## 💡 **Pro Tips**

1. **Always test with a sample** first
2. **Use cloud storage** for files >500MB
3. **Split files** for better performance
4. **Compress files** to reduce upload time
5. **Use databases** for very large datasets
6. **Sample data** for initial exploration

## 🔗 **Useful URLs**

- **Google Drive Export**: `https://drive.google.com/uc?export=download&id=FILE_ID`
- **Dropbox Direct**: `https://dl.dropboxusercontent.com/s/FILE_ID/filename.csv`
- **GitHub Raw**: `https://raw.githubusercontent.com/user/repo/branch/file.csv`

---

**Your DOTSURE STREAMLIT dashboard can handle datasets of any size with these solutions!** 🚗📊
