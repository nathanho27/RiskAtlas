import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv(
    "DATABASE_URL"
)

print(
    f"DATABASE_URL set: {bool(database_url)}"
)

if not database_url:
    raise ValueError(
        "DATABASE_URL is not set."
    )


connection = psycopg2.connect(
    database_url
)

cursor = connection.cursor()

cursor.execute(
    """
    SELECT COUNT(*)
    FROM current_risk_predictions_v3;
    """
)

row_count = cursor.fetchone()[0]

cursor.execute(
    """
    SELECT
        ticker,
        risk_score,
        risk_level
    FROM current_risk_predictions
    ORDER BY risk_score DESC
    LIMIT 5;
    """
)

rows = cursor.fetchall()

print(
    f"Rows found: {row_count}"
)

for row in rows:
    print(row)


cursor.close()
connection.close()