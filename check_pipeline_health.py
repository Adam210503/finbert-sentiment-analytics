from pathlib import Path
import pandas as pd
from src.storage.db_manager import DatabaseManager

# Initialize database manager instance
db_path = Path("data/sentiment_pipeline.db")
db = DatabaseManager(db_path)

# Extract statistics via built-in health engine
stats = db.get_health()

print("\n" + "="*60)
print("                SENTIMENT PIPELINE STATUS CARD               ")
print("="*60)
print(f" Total Headlines Stored : {stats['total_headlines']:,}")
print(f"   ├─ Scored (FinBERT)  : {stats['scored_headlines']:,}")
print(f"   └─ Unscored (Pending): {stats['unscored_headlines']:,}")
print(f" Total Price Records    : {stats['total_price_records']:,}")
print("-"*60)
print(f" Newest Headline Added  : {stats['last_news_timestamp']}")
print(f" Newest Price Date Added: {stats['last_price_date']}")
print("="*60)

# Extract and format the job execution ledger
print("\n--- RECENT BACKGROUND JOB TIMESTAMPS (UTC) ---")
if stats['recent_jobs']:
    df_jobs = pd.DataFrame(stats['recent_jobs'])
    # Reordering columns for scannability
    print(df_jobs[['ran_at', 'job_name', 'inserted', 'skipped']])
else:
    print("No background logs found yet.")
print("\n" + "="*60 + "\n")