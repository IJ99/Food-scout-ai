from datetime import datetime
import sqlite3

DB_FILE = "food_scout.db"

def get_db_connection():
    return sqlite3.connect(DB_FILE)

def get_user_by_email(email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2]}
    return None

def create_user(name, email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id

def save_search(user_id: int, food: str, city: str):
    conn = get_db_connection()
    cur = conn.cursor()

    timestamp = datetime.now().isoformat()

    cur.execute("""
        INSERT INTO search_history (user_id, food, city, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_id, food, city, timestamp))

    conn.commit()
    conn.close()

def get_last_search(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT food, city FROM search_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {"food": row[0], "city": row[1]}
    return None

def get_all_searches(email):
    conn = sqlite3.connect("food_scout.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT food, city, timestamp 
        FROM search_history 
        WHERE user_id = (
            SELECT id FROM users WHERE email = ?
        )
        ORDER BY timestamp DESC
    """, (email,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {"food": row[0], "city": row[1], "timestamp": row[2]}
        for row in rows
    ]
