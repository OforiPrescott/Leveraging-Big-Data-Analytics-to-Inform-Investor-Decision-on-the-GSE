import sqlite3
from datetime import datetime, timedelta
import random

# Connect to database
conn = sqlite3.connect('gse_sentiment.db')
cursor = conn.cursor()

# Companies to add sample data for
companies = ['MTN', 'GCB', 'EGH', 'TOTAL', 'FML', 'SCB', 'CAL', 'ACCESS', 'AGA', 'GOIL']

print('Adding sample data for all GSE companies...')

total_entries = 0

# Sample sentiment data for each company
for company in companies:
    # Add 5-10 entries per company
    num_entries = random.randint(5, 10)
    print(f'Adding {num_entries} entries for {company}')

    for i in range(num_entries):
        sentiment_score = random.uniform(-0.8, 0.8)
        sentiment_label = 'positive' if sentiment_score > 0.05 else 'negative' if sentiment_score < -0.05 else 'neutral'
        confidence = random.uniform(0.6, 0.95)
        timestamp = datetime.now() - timedelta(days=random.randint(1, 30))

        cursor.execute('''
            INSERT INTO sentiment_data
            (timestamp, source, content, sentiment_score, sentiment_label, company, url, confidence, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(),
            f'Sample News Source {random.randint(1,5)}',
            f'Sample content about {company} performance and market sentiment analysis {i+1}',
            sentiment_score,
            sentiment_label,
            company,
            f'https://sample-news.com/{company.lower()}/article{i+1}',
            confidence,
            f'{company}_sample_{i}_{random.randint(1000,9999)}'
        ))

        total_entries += 1

conn.commit()

# Check final count
cursor.execute('SELECT company, COUNT(*) as count FROM sentiment_data GROUP BY company ORDER BY company')
rows = cursor.fetchall()

print('\nFinal database contents:')
print('Company | Count')
print('-' * 15)
for row in rows:
    print(f'{row[0]:8} | {row[1]}')

print(f'\nTotal entries: {total_entries}')
print(f'Companies with data: {len(rows)}')

conn.close()
print('✅ Sample data insertion completed successfully!')