
from database import get_connection

try:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            result = cur.fetchone()
            print("PostgreSQL connection successful!")
            print(result[0])

except Exception as error:
    print("Connection failed:", error)
