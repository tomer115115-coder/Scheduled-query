'''
מחברת זו נועדה לטעון את כל השאילתות מה-WhoIsActive
ולשמור אותם ב-SQLite database במקום קובץ אקסל/CSV.

יתרונות SQLite על פני Excel/CSV:
  - ללא מגבלת שורות (מיליוני שורות ללא בעיה)
  - תמיכה מלאה בשאילתות SQL לניתוח
  - טעינה יעילה בזיכרון (לא טוען הכל בכל ריצה)
  - index אוטומטי לביצועי query מהירים

יש לתזמן את המחברת כך שתרוץ מספר פעמים ביום לפי הדרישה
'''

from mssqlutils import db_utils
import pandas as pd
import sqlite3
import time
import os
import warnings
import regex as re

warnings.filterwarnings("ignore")

user_name     = 'mac-ad\\BI_QLIK_ANALYST'
user_password = 'nIy}A^FD6h36'

DB_PATH    = r'//mnt/analyst_project/sql_similarity_check/sql_queries.db'
TABLE_NAME = 'sql_queries'
COLS       = ['dd hh:mm:ss.mss', 'session_id', 'sql_text', 'login_name', 'CPU',
              'wait_info', 'used_memory', 'sql_code', 'start_time', 'reads',
              'writes', 'database_name']
DEDUP_KEYS = ['sql_code', 'login_name', 'start_time']


# ── פונקציות עיבוד טקסט ──────────────────────────────────────────────────────

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def normalize_new_line(text):
    return re.sub(r'\n+', '\n', text).strip()

def preprocess_query(query):
    query = str(query).replace('/r', '\r').replace('/n', '\n').replace('/t', '\t').replace('_x000D_', '')
    query = query.lower()
    query = query.replace('[', '').replace(']', '').replace('inner', '')
    query = normalize_whitespace(query)
    query = normalize_new_line(query)
    return query


# ── פונקציות SQLite ───────────────────────────────────────────────────────────

def _ensure_index(conn):
    """יוצר index על עמודות הדה-דופליקציה אם לא קיים."""
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_dedup "
        f"ON {TABLE_NAME} (sql_code, login_name, start_time)"
    )

def migrate_legacy_data():
    """
    מיגרציה חד-פעמית: מעביר נתונים קיימים מ-Excel או CSV ל-SQLite.
    הפונקציה רצה רק כשה-DB עדיין לא קיים.
    """
    excel_path = r'//mnt/analyst_project/sql_similarity_check/sql_queries.xlsx'
    csv_path   = r'//mnt/analyst_project/sql_similarity_check/sql_queries.csv'

    if os.path.exists(excel_path):
        print("Migrating existing data from Excel to SQLite...")
        legacy_df = pd.read_excel(excel_path)
    elif os.path.exists(csv_path):
        print("Migrating existing data from CSV to SQLite...")
        legacy_df = pd.read_csv(csv_path)
    else:
        print("No legacy data found — starting fresh DB.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        legacy_df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        _ensure_index(conn)
    print(f"Migration complete: {len(legacy_df):,} rows imported to {DB_PATH}")

def get_row_count():
    """מחזיר את סך השורות הנוכחי ב-DB."""
    if not os.path.exists(DB_PATH):
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

def get_existing_keys():
    """
    טוען רק את עמודות המפתח לצורך דה-דופליקציה.
    חוסך זיכרון בהשוואה לטעינת כל הנתונים.
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=DEDUP_KEYS)
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            f"SELECT {', '.join(DEDUP_KEYS)} FROM {TABLE_NAME}",
            conn
        )

def append_new_rows(df):
    """מוסיף שורות חדשות ל-DB (append בלבד, לא כותב מחדש)."""
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        _ensure_index(conn)


# ── מיגרציה חד-פעמית ─────────────────────────────────────────────────────────

if not os.path.exists(DB_PATH):
    migrate_legacy_data()

previous_count = get_row_count()
print(f"Existing rows in DB: {previous_count:,}")


# ── איסוף נתונים מ-WhoIsActive ───────────────────────────────────────────────

n_range = 350

df = pd.DataFrame()

for i in range(n_range):
    print('tables left:', n_range - i, end='  ')

    connection = db_utils.connect(user=user_name, password=user_password, database='dw_dim')
    try:
        new_df = pd.read_sql('exec sp_WhoIsActive', connection)
    finally:
        connection.close()

    df = pd.concat([df, new_df]).reset_index(drop=True)
    df['sql_code'] = df['sql_text'].apply(preprocess_query)
    df = df.sort_values(by='dd hh:mm:ss.mss', ascending=False)
    df = df.drop_duplicates(subset=DEDUP_KEYS, keep='first')
    df = df[COLS]

    if i < n_range - 1:
        time.sleep(30)


# ── דה-דופליקציה מול הנתונים הקיימים ב-DB ────────────────────────────────────

df = df.dropna(subset=['sql_text'])

# המרה לסטרינג לצורך השוואה עקבית (SQL Server מחזיר datetime, SQLite שומר טקסט)
df['start_time'] = df['start_time'].astype(str)

existing_keys = get_existing_keys()

if not existing_keys.empty:
    existing_keys['start_time'] = existing_keys['start_time'].astype(str)
    merged   = df.merge(existing_keys, on=DEDUP_KEYS, how='left', indicator=True)
    new_rows = df[merged['_merge'].values == 'left_only'].copy()
else:
    new_rows = df.copy()


# ── שמירה ל-DB ────────────────────────────────────────────────────────────────

if len(new_rows) > 0:
    append_new_rows(new_rows)
    print(f"\nAdded {len(new_rows):,} new rows.")
else:
    print("\nNo new rows to add (all duplicates).")

new_count = get_row_count()
print(f"Previous: {previous_count:,}  |  Added: {len(new_rows):,}  |  Total: {new_count:,}")
print(f"Database: {DB_PATH}")

# ── דוגמאות לניתוח (להרצה נפרדת) ────────────────────────────────────────────
#
# import sqlite3, pandas as pd
# conn = sqlite3.connect(r'//mnt/analyst_project/sql_similarity_check/sql_queries.db')
#
# # 10 השאילתות הכבדות ביותר לפי CPU
# pd.read_sql("""
#     SELECT login_name, sql_code, MAX(CAST(CPU AS INTEGER)) AS max_cpu, COUNT(*) AS appearances
#     FROM sql_queries
#     GROUP BY sql_code, login_name
#     ORDER BY max_cpu DESC
#     LIMIT 10
# """, conn)
#
# # פעילות לפי תאריך
# pd.read_sql("""
#     SELECT DATE(start_time) AS day, COUNT(*) AS queries
#     FROM sql_queries
#     GROUP BY day
#     ORDER BY day DESC
# """, conn)
