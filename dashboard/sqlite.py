import sqlite3
conn = sqlite3.connect("forex_analysis.db")
cursor = conn.cursor()

cursor.execute("ALTER TABLE analysis_results ADD COLUMN data_source TEXT")
conn.commit()
conn.close()
