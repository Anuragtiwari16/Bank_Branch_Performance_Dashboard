import pandas as pd
import sqlite3
import os

print("Migrating data from CSV to Enterprise SQLite Database...")
csv_file = "branch_data.csv"
db_file = "bharat_banking.db"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    
    # Connect to SQLite (creates the file if it doesn't exist)
    conn = sqlite3.connect(db_file)
    
    # Write data to a sqlite table
    df.to_sql('branch_performance', conn, if_exists='replace', index=False)
    
    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM branch_performance")
    count = cursor.fetchone()[0]
    
    conn.close()
    print(f"[SUCCESS] Successfully migrated {count} records to {db_file} (Table: branch_performance).")
else:
    print(f"[ERROR] CSV file '{csv_file}' not found. Please run 'python setup_ml.py' first to generate branch records.")
