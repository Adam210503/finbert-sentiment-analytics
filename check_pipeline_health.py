from pathlib import Path
from src.storage.db_manager import DatabaseManager

# Initialize the manager pointing to your local DB
db_path = Path("data/sentiment_pipeline.db")
db = DatabaseManager(db_path)

# Retrieve the summary statistics dictionary
stats = db.get_health()

# Print a formatted observability dashboard
print("\n" + "="*50)
print("          SENTIMENT PIPELINE HEALTH REPORT          ")
print("="*50)
print(f"Total Headlines Collected : {stats['total_headlines']:,}")
print(f"  └─ Scored by FinBERT    : {stats['scored_headlines']:,}")
print(f"  └─ Pending Inference    : {stats['unscored_headlines']:,}")
print(f"Total Price Records       : {stats['total_price_records']:,}")
print("-"*50)
print(f"Last News Timestamp       : {stats['last_news_timestamp']}")
print(f"Last Price Data Date      : {stats['last_price_date']}")
print("="*50 + "\n")