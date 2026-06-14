import sqlite3
import pandas as pd

# Connect to your local database
conn = sqlite3.connect("data/sentiment_pipeline.db")

# Load your news headlines into a Pandas DataFrame
print("\n--- RECENTLY COLLECTED NEWS HEADLINES ---")
df_news = pd.read_sql_query("SELECT ticker, market_date, headline, source FROM sentiment_scores ORDER BY id DESC LIMIT 10;", conn)
print(df_news)

# Load your job logs to see system performance metrics
print("\n--- BACKGROUND JOB LOGS ---")
df_jobs = pd.read_sql_query("SELECT job_name, ran_at, inserted, skipped FROM job_log ORDER BY id DESC LIMIT 5;", conn)
print(df_jobs)

conn.close()