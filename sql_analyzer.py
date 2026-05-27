# %% [markdown]
# # SQL Analyzer
# קורא נתונים מ-SQLite, מריץ ניתוח דמיון שאילתות ומייצר דוחות

# %%
from mssqlutils import db_utils
import sqlite3
import pandas as pd
import warnings
import regex as re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── הגדרות ────────────────────────────────────────────────────────────────────

DB_PATH    = r'//mnt/analyst_project/sql_similarity_check/sql_queries.db'
TABLE_NAME = 'sql_queries'

# ⭐ הופחת מ-20000 ל-2000 כדי למנוע MemoryError בחישוב cosine similarity
# (מטריצה של n×n float64: 20000²×8=3.2GB נגד 2000²×8=32MB)
CHUNK_SIZE = 2000

user_name     = 'mac-ad\\BI_QLIK_ANALYST'
user_password = 'nIy}A^FD6h36'


# ── פונקציות עזר ─────────────────────────────────────────────────────────────

def display_scrollable_dataframe(df, max_height=400, max_width='100%'):
    try:
        from IPython.display import display, HTML
        display(HTML(f"""
            <div style='max-height: {max_height}px; max-width: {max_width};
                        overflow-y: auto; overflow-x: auto; border: 1px solid black'>
                {df.to_html(index=False)}
            </div>
        """))
    except ImportError:
        print(df.to_string())


def add_labels(ax, data, x_col, y_col):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9), textcoords='offset points')


# %% [markdown]
# ## טעינת נתונים מ-SQLite

# %%
print("Loading data from SQLite...")
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)

df = df.dropna(subset=['sql_text'])
print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

# %% [markdown]
# ## עיבוד תאריכים וסינון

# %%
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce', dayfirst=True)
df['start_date'] = df['start_time'].dt.date
print(f"Date range: {df['start_date'].min()} → {df['start_date'].max()}")

# סינון ידני — ניתן לשנות את התאריך
target_date = pd.to_datetime('2025/06/01').date()
df = df[df['start_date'] > target_date]
print(f"After date filter: {df.shape[0]:,} rows")

# %% [markdown]
# ## פיצול לחלקים ואלגוריתם דמיון שאילתות

# %%
def split_dataframe(df, chunk_size):
    df = df.sort_values(by='start_date').reset_index(drop=True)
    n = (len(df) // chunk_size) + 1
    return [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n)], n


def add_similar_queries(df):
    def _weight_query(query):
        try:
            # ⭐ תוקן: \\s+ → \s+ (הbאג מנע מציאת שמות טבלאות)
            tables = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', query, re.IGNORECASE)
            tables = [t for sublist in tables for t in sublist if t]
            return query + ' ' + ' '.join(tables * 8)
        except:
            return query

    df = df.copy()
    df['weighted_sql_code'] = df['sql_code'].apply(_weight_query)

    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['weighted_sql_code'])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

    similarity_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)
    threshold     = 0.8
    group_keys    = [-1] * len(df)
    current_key   = 0

    for i in range(len(similarity_df)):
        if group_keys[i] == -1:
            group_keys[i] = current_key
            for j in range(i + 1, len(similarity_df)):
                if similarity_df.iloc[i, j] > threshold:
                    group_keys[j] = current_key
            current_key += 1

    df['group_key']       = group_keys
    df['group_key_count'] = df.groupby('group_key').transform('count')['sql_code']
    df['from_table']      = df['sql_code'].apply(lambda x: set(re.findall(
        r'\n*\s+FROM\s+(?!openquery)\s*(\w+\.*\w*\.*\w*)', x, re.IGNORECASE)))
    df['join_tables']     = df['sql_code'].apply(lambda x: set(re.findall(
        r'\n*\s+join\s+(?!openquery)\s*(\w+\.*\w*\.*\w*)', x, re.IGNORECASE)))
    return df


split_dfs, n = split_dataframe(df, CHUNK_SIZE)
print(f"Split into {n} chunks of ~{CHUNK_SIZE} rows each")

result_dfs = []
for i, chunk in enumerate(split_dfs):
    print(f"  Chunk {i + 1}/{n} ({len(chunk)} rows)...")
    processed = add_similar_queries(chunk)
    processed['group_key'] = processed['group_key'].apply(lambda x: str(x) + str(i))
    result_dfs.append(processed)

df = pd.concat(result_dfs).reset_index(drop=True)
print(f"Similarity analysis done. Total rows: {len(df):,}")

# %% [markdown]
# ## חילוץ שמות טבלאות

# %%
def extract_table_name(table_set):
    if len(table_set) == 0:
        return table_set
    name = list(table_set)[0]
    try:
        return name.split('.')[-1]
    except:
        return name


def join_from_tables(row):
    lst = [row['from_table_name']]
    for j in row['join_table_name']:
        lst.append(j)
    try:
        lst = sorted(set(lst))
    except:
        pass
    return lst


df['from_table_name'] = df['from_table'].apply(extract_table_name)
df['join_table_name'] = df['join_tables'].apply(
    lambda x: [extract_table_name({y}) for y in x])
df['all_tables'] = df.apply(join_from_tables, axis=1)

# %% [markdown]
# ## סיווג משתמשים / מערכות

# %%
non_analyst_users = [
    'BI_ML', 'MAC-AD\\Informatica-MSG', 'MAC-AD\\SPOTLIGHT',
    'MAC-AD\\SentryOne_MON', 'mac-ad\\qliksrvprd', 'MAC-AD\\precise_app',
    'MAC-AD\\sql-dw9-pprod-s$', 'mac-ad\\uc4sql', 'MAC-AD\\digital_health',
    'MAC-AD\\SQLSentryMonSvc', 'MAC-AD\\SQLSrv2k5Service', 'Informatica_Users',
    'DWH_MD_Extract', 'MAC-AD\\uc4sql_test', 'HDF_NiFi_User',
    'MAC-AD\\Py_BI_Click_XML', 'nifi_bi_user', 'ESBProd',
    'MAC-AD\\sql-dw9-pprod-a$', 'Logim_1', 'MAC-AD\\SpdBasePPSvc$'
]

all_users      = list(df['login_name'].value_counts().keys())
analyst_users  = [u for u in all_users if u not in non_analyst_users]
df['analyst_users'] = df['login_name'].apply(lambda x: 1 if x in analyst_users else 0)

df['login_name'] = df['login_name'].astype(str)
df['emp_name']   = df['login_name'].apply(lambda x: x[x.find('\\') + 1:].lower())

# %% [markdown]
# ## הבאת מידע על עובדים מ-SQL Server

# %%
analyst_login_names = df[df['analyst_users'] == 1]['login_name'].unique()
flat_names = []
for x in analyst_login_names:
    found = re.findall(r'(?<=\\)\w+', x)
    flat_names.append(found[0] if found else x)

names_sql_list = ', '.join(f"'{n}'" for n in flat_names)

users_sql_info = f"""
SELECT EMP_USER_NAME, EMP_NUMBER, emp_name, BRANCH_DESC, DISTRICT_DESC,
       OCCUPATION_FULL_DESC, BRNAHC_CITY_DESC, OCC_GROUP_FULL_DESC,
       EMP_GROUP_DESC, HR_EMP_KEY_PK
FROM dw_hr.dbo.dw_dim_hr_emp_keys_flat_data WITH (NOLOCK)
WHERE EMP_USER_NAME IN ({names_sql_list})
  AND ind_current = 1
"""

connection = db_utils.connect(user=user_name, password=user_password, database='dw_dim')
users_info = pd.read_sql(users_sql_info, connection)
connection.close()

users_info['EMP_USER_NAME'] = users_info['EMP_USER_NAME'].str.lower()
users_info = users_info.drop_duplicates(subset='EMP_NUMBER')
df = df.merge(users_info, left_on='emp_name', right_on='EMP_USER_NAME', how='left')

# %% [markdown]
# ## חישוב עמודות זמן

# %%
df['start_hour']      = df['start_time'].dt.hour
df['day_of_the_week'] = df['start_time'].dt.weekday
df['month']           = df['start_time'].dt.month

df['start_time']      = df['start_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Jerusalem')
df['day_of_the_week'] = df['start_time'].dt.day_name()


def run_time_minutes(t):
    try:
        days    = int(t[:2]) * 24 * 60
        hours   = int(t[3:5]) * 60
        minutes = int(t[6:8])
        seconds = int(t[9:11]) / 60
        return days + hours + minutes + seconds
    except:
        return 0


df['total_run_time'] = df['dd hh:mm:ss.mss'].apply(run_time_minutes)
df['start_date']     = df['start_time'].dt.date

# שמירה לאקסל (timezone מוסר כי Excel לא תומך בו)
df['start_time'] = df['start_time'].dt.tz_localize(None)
df.to_excel('sql_similiratiy_check.xlsx')
print("Saved: sql_similiratiy_check.xlsx")

# %% [markdown]
# ## יצירת דגלים (outliers, flags)

# %%
deleted_question = '<?query -- exec sp_whoisactive --?>'
df = df[~(df['sql_code'] == deleted_question)]

df['count_questions'] = df.groupby('login_name').transform('count')['sql_text']

q1  = df[df['analyst_users'] == 1]['count_questions'].quantile(0.25)
q3  = df[df['analyst_users'] == 1]['count_questions'].quantile(0.75)
iqr = q3 - q1


def question_count_flag_check(row):
    if row['analyst_users'] == 0:
        return 0
    return 1 if row['count_questions'] < q1 or row['count_questions'] > q3 + 1.5 * iqr else 0


df['question_count_flag'] = df.apply(question_count_flag_check, axis=1)

df['CPU']          = df['CPU'].fillna(0).astype(str).str.replace(',', '').astype(int)
cpu_q1, cpu_q3     = df['CPU'].quantile(0.25), df['CPU'].quantile(0.75)
df['cpu_flag']     = df['CPU'].apply(
    lambda x: 1 if x < cpu_q1 - 5.5 * (cpu_q3 - cpu_q1) or x > cpu_q3 + 5.5 * (cpu_q3 - cpu_q1) else 0)

df['used_memory']  = df['used_memory'].fillna(0).astype(str).str.replace(',', '').astype(int)
mem_q1, mem_q3     = df['used_memory'].quantile(0.25), df['used_memory'].quantile(0.75)
df['used_memory_flag'] = df['used_memory'].apply(
    lambda x: 1 if x < mem_q1 - 5.5 * (mem_q3 - mem_q1) or x > mem_q3 + 5.5 * (mem_q3 - mem_q1) else 0)

gk_q1 = df[df['analyst_users'] == 1]['group_key_count'].quantile(0.25)
df['group_key_count_flag'] = df['group_key_count'].apply(lambda x: 1 if x < gk_q1 else 0)

df['query_count_per_month'] = df.groupby(['login_name', 'month']).transform('count')['sql_text']
df['low_usage_user_flag']   = df['query_count_per_month'].apply(lambda x: 1 if x < 5 else 0)
df['working_hour_flag']     = df['start_hour'].apply(lambda x: 1 if 6 < x < 20 else 0)
df['run_time_flag']         = df['total_run_time'].apply(lambda x: 0 if x < 10 else 1)
df['nolock_flag']           = df['sql_text'].apply(lambda x: 1 if 'nolock' in x.lower() else 0)

# %% [markdown]
# ## סינון לשבועיים האחרונים + שאילתות כבדות

# %%
two_weeks_ago        = pd.to_datetime((datetime.now() - timedelta(14)).strftime('%Y-%m-%d')).date()
filterd_df_for_dates = df[df['start_date'] >= two_weeks_ago]
filterd_df_for_dates.to_excel('check_08012026.xlsx')
print("Saved: check_08012026.xlsx")

heavy_queries = filterd_df_for_dates[filterd_df_for_dates['run_time_flag'] == 1]
print(f"Heavy queries: {heavy_queries['start_date'].min()} → {heavy_queries['start_date'].max()}")

# %% [markdown]
# # ניתוח מערכות

# %%
system_df = heavy_queries[
    (heavy_queries['analyst_users'] == 0) & (heavy_queries['working_hour_flag'] == 1)
]

cpu_mean_sql_count = system_df.groupby('login_name').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).sort_values('CPU', ascending=False).head(10).reset_index()

memory_mean = system_df.groupby('login_name').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).sort_values('used_memory', ascending=False).head(10).reset_index()

sql_text_mean = system_df.groupby('login_name').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).sort_values('sql_text', ascending=False).head(10).reset_index()

system_run_mean = system_df.groupby('login_name').agg(
    CPU_mean=('CPU', 'mean'),
    sql_text_count=('sql_text', 'count'),
    used_memory_mean=('used_memory', 'mean'),
    quantile_95_run_time=('total_run_time', lambda x: x.quantile(0.95)),
    run_time_median=('total_run_time', 'median')
).reset_index().sort_values('run_time_median', ascending=False).head(10)

run_mean_melted = system_run_mean.melt(
    id_vars=['login_name'],
    value_vars=['quantile_95_run_time', 'run_time_median'],
    var_name='Metric', value_name='Value'
)

fig, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns.barplot(x='login_name', y='CPU',       data=cpu_mean_sql_count, ax=ax1[1][0], color='blue')
sns.barplot(x='login_name', y='used_memory', data=memory_mean,      ax=ax1[1][1], color='red')
sns.barplot(x='login_name', y='sql_text',  data=sql_text_mean,      ax=ax1[0][0], color='red')
sns.barplot(x='login_name', y='Value',     data=run_mean_melted,    ax=ax1[0][1], hue='Metric')
add_labels(ax1[0][1], system_run_mean, 'login_name', 'run_time_median')

titles = [('CPU Mean', 'CPU Mean', 'Mean of CPU by login_name'),
          ('Used Memory Mean', 'Used Memory Mean', 'Mean Used Memory by login_name'),
          ('Question Count', 'Question Count', 'Question Count by login_name'),
          ('Run Time', 'Run Time', 'Median and 95th percentile Run Time by login_name')]
for ax, (xlabel, ylabel, title) in zip(ax1.flat, titles):
    ax.set_xlabel('login_name'); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

plt.subplots_adjust(hspace=0.5)
plt.show()

# %%
needed_columns = ['start_time', 'start_hour', 'login_name', 'CPU', 'used_memory',
                  'sql_code', 'sql_text', 'start_date', 'total_run_time', 'group_key', 'group_key_count']
heavy_system_users = list(system_run_mean['login_name'])

for login in heavy_system_users:
    print(f'\nSystem: {login}')
    check_system = system_df[system_df['login_name'] == login][needed_columns].reset_index(drop=True)
    unique_que   = check_system['group_key'].nunique()
    print(f'  {len(check_system)} שאילתות כבדות, מתוכן {unique_que} ייחודיות')

    check_system_grp = check_system.groupby('group_key').agg(
        CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
        used_memory=('used_memory', 'mean'), total_run_time=('total_run_time', 'median'),
        start_hour=('start_hour', 'mean'),
        start_date_min=('start_date', 'min'), start_date_max=('start_date', 'max')
    ).sort_values('total_run_time', ascending=False).head(2).reset_index()

    for _, row in check_system_grp.iterrows():
        print(f'  שאילתה {row["group_key"]}: {row["sql_text"]:.0f} הרצות, '
              f'זמן ריצה ממוצע {row["total_run_time"]:.0f} דקות, '
              f'שעה ~{2 + round(row["start_hour"]):.0f}, '
              f'{row["start_date_min"]} → {row["start_date_max"]}')

# %% [markdown]
# # ניתוח משתמשים

# %%
users_df = heavy_queries[
    (heavy_queries['analyst_users'] == 1) &
    (heavy_queries['working_hour_flag'] == 1) &
    (heavy_queries['low_usage_user_flag'] == 0)
]
users_df['OCCUPATION_FULL_DESC_REORDER'] = users_df['OCCUPATION_FULL_DESC'].apply(lambda x: str(x)[::-1])

cpu_mean_sql_count = users_df.groupby('OCCUPATION_FULL_DESC_REORDER').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).reset_index().sort_values('CPU', ascending=False).head(10)

memory_mean = users_df.groupby('OCCUPATION_FULL_DESC_REORDER').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).reset_index().sort_values('used_memory', ascending=False).head(10)

sql_text_mean = users_df.groupby('OCCUPATION_FULL_DESC_REORDER').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
    used_memory=('used_memory', 'mean'), EMP_USER_NAME=('EMP_USER_NAME', 'nunique')
).reset_index()
sql_text_mean['queries_per_user'] = sql_text_mean['sql_text'] / sql_text_mean['EMP_USER_NAME']
sql_text_mean = sql_text_mean.sort_values('queries_per_user', ascending=False).head(10)

occupation_run_mean = users_df.groupby('OCCUPATION_FULL_DESC_REORDER').agg(
    CPU_mean=('CPU', 'mean'), sql_text_count=('sql_text', 'count'),
    used_memory_mean=('used_memory', 'mean'),
    OCCUPATION_FULL_DESC=('OCCUPATION_FULL_DESC', 'max'),
    quantile_95_run_time=('total_run_time', lambda x: x.quantile(0.95)),
    run_time_median=('total_run_time', 'median')
).reset_index().sort_values('run_time_median', ascending=False).head(10)

run_mean_melted = occupation_run_mean.melt(
    id_vars=['OCCUPATION_FULL_DESC_REORDER'],
    value_vars=['quantile_95_run_time', 'run_time_median'],
    var_name='Metric', value_name='Value'
)

fig, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns.barplot(x='OCCUPATION_FULL_DESC_REORDER', y='CPU',             data=cpu_mean_sql_count, ax=ax1[1][0], color='blue')
sns.barplot(x='OCCUPATION_FULL_DESC_REORDER', y='used_memory',     data=memory_mean,        ax=ax1[1][1], color='red')
sns.barplot(x='OCCUPATION_FULL_DESC_REORDER', y='queries_per_user', data=sql_text_mean,     ax=ax1[0][0], color='red')
sns.barplot(x='OCCUPATION_FULL_DESC_REORDER', y='Value',           data=run_mean_melted,    ax=ax1[0][1], hue='Metric')
add_labels(ax1[0][1], occupation_run_mean, 'OCCUPATION_FULL_DESC_REORDER', 'run_time_median')

for ax in ax1.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
plt.subplots_adjust(hspace=0.5)
plt.show()

# %%
emp_id_lst          = []
heavy_occupations   = list(occupation_run_mean['OCCUPATION_FULL_DESC'])

for occ in heavy_occupations:
    print(f'\noccupation: {occ}')
    print('~' * 50)
    check_system = users_df[users_df['OCCUPATION_FULL_DESC'] == occ].reset_index(drop=True)
    print(f'  {len(check_system)} שאילתות כבדות, {check_system["group_key"].nunique()} ייחודיות')

    check_user_grp = check_system.groupby('emp_name_y').agg(
        CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
        used_memory=('used_memory', 'mean'), total_run_time=('total_run_time', 'median'),
        start_hour=('start_hour', 'mean'), group_key=('group_key', 'nunique'),
        start_date_min=('start_date', 'min'), start_date_max=('start_date', 'max'),
        BRANCH_DESC=('BRANCH_DESC', 'max'), EMP_NUMBER=('EMP_NUMBER', 'max')
    ).sort_values('total_run_time', ascending=False).head(5).reset_index()

    for _, row in check_user_grp.iterrows():
        emp_id_lst.append(row['EMP_NUMBER'])
        print(f'  {row["emp_name_y"]} מ-{row["BRANCH_DESC"]}: '
              f'{row["sql_text"]:.0f} שאילתות ({row["group_key"]:.0f} ייחודיות), '
              f'זמן ריצה חציוני {row["total_run_time"]:.0f} דקות, '
              f'שעה ~{2 + round(row["start_hour"]):.0f}')

# %% [markdown]
# # ניתוח משתמשים — חלוקה לפי מחלקות

# %%
users_df['BRANCH_DESC_REORDER'] = users_df['BRANCH_DESC'].apply(lambda x: str(x)[::-1])

branch_run_mean = users_df.groupby('BRANCH_DESC_REORDER').agg(
    CPU_mean=('CPU', 'mean'), sql_text_count=('sql_text', 'count'),
    used_memory_mean=('used_memory', 'mean'),
    BRANCH_DESC=('BRANCH_DESC', 'max'),
    quantile_95_run_time=('total_run_time', lambda x: x.quantile(0.95)),
    run_time_median=('total_run_time', 'median')
).reset_index().sort_values('run_time_median', ascending=False).head(10)

run_mean_melted = branch_run_mean.melt(
    id_vars=['BRANCH_DESC_REORDER'],
    value_vars=['run_time_median', 'quantile_95_run_time'],
    var_name='Metric', value_name='Value'
)

cpu_mean_sql_count = users_df.groupby('BRANCH_DESC_REORDER').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).reset_index().sort_values('CPU', ascending=False).head(10)

memory_mean = users_df.groupby('BRANCH_DESC_REORDER').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).reset_index().sort_values('used_memory', ascending=False).head(10)

sql_text_mean = users_df.groupby('BRANCH_DESC_REORDER').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
    used_memory=('used_memory', 'mean'), EMP_USER_NAME=('EMP_USER_NAME', 'nunique')
).reset_index()
sql_text_mean['queries_per_user'] = sql_text_mean['sql_text'] / sql_text_mean['EMP_USER_NAME']
sql_text_mean = sql_text_mean.sort_values('queries_per_user', ascending=False).head(10)

fig, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
sns.barplot(x='BRANCH_DESC_REORDER', y='CPU',             data=cpu_mean_sql_count, ax=ax1[1][0], color='blue')
sns.barplot(x='BRANCH_DESC_REORDER', y='used_memory',     data=memory_mean,        ax=ax1[1][1], color='red')
sns.barplot(x='BRANCH_DESC_REORDER', y='queries_per_user', data=sql_text_mean,     ax=ax1[0][0], color='red')
sns.barplot(x='BRANCH_DESC_REORDER', y='Value',           data=run_mean_melted,    ax=ax1[0][1], hue='Metric')
add_labels(ax1[0][1], run_mean_melted, 'BRANCH_DESC_REORDER', 'Value')

for ax in ax1.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
plt.subplots_adjust(hspace=0.5)
plt.show()

# %%
heavy_branches = list(branch_run_mean['BRANCH_DESC'])

for branch in heavy_branches:
    print(f'\nbranch: {branch}')
    check_system = users_df[users_df['BRANCH_DESC'] == branch].reset_index(drop=True)
    print(f'  {len(check_system)} שאילתות כבדות, {check_system["group_key"].nunique()} ייחודיות')

    check_user_grp = check_system.groupby('emp_name_y').agg(
        CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
        group_key=('group_key', 'nunique'), used_memory=('used_memory', 'mean'),
        total_run_time=('total_run_time', 'median'), BRANCH_DESC=('BRANCH_DESC', 'max'),
        EMP_NUMBER=('EMP_NUMBER', 'max'), OCCUPATION_FULL_DESC=('OCCUPATION_FULL_DESC', 'max'),
        start_hour=('start_hour', 'median')
    ).sort_values('total_run_time', ascending=False).head(5).reset_index()

    for _, row in check_user_grp.iterrows():
        if row['EMP_NUMBER'] in emp_id_lst:
            continue
        emp_id_lst.append(row['EMP_NUMBER'])
        print(f'  {row["emp_name_y"]} ({row["OCCUPATION_FULL_DESC"]}): '
              f'{row["sql_text"]:.0f} שאילתות ({row["group_key"]:.0f} ייחודיות), '
              f'זמן ריצה חציוני {row["total_run_time"]:.0f} דקות, '
              f'שעה ~{2 + round(row["start_hour"]):.0f}')

# %% [markdown]
# # משתמשים כלליים

# %%
general_user_run_mean = users_df.groupby('login_name').agg(
    CPU_mean=('CPU', 'mean'), sql_text_count=('sql_text', 'count'),
    used_memory_mean=('used_memory', 'mean'),
    quantile_95_run_time=('total_run_time', lambda x: x.quantile(0.95)),
    run_time_median=('total_run_time', 'median'),
    EMP_NUMBER=('EMP_NUMBER', 'max')
).reset_index().sort_values('run_time_median', ascending=False).head(20)

cpu_mean_sql_count = users_df.groupby('login_name').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).sort_values('CPU', ascending=False).head(10).reset_index()

memory_mean = users_df.groupby('login_name').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).sort_values('used_memory', ascending=False).head(10).reset_index()

sql_text_mean = users_df.groupby('login_name').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'), used_memory=('used_memory', 'mean')
).sort_values('sql_text', ascending=False).head(20).reset_index()

run_mean_melted = general_user_run_mean.melt(
    id_vars=['login_name'],
    value_vars=['quantile_95_run_time', 'run_time_median'],
    var_name='Metric', value_name='Value'
)

fig, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
sns.barplot(x='login_name', y='CPU',       data=cpu_mean_sql_count, ax=ax1[1][0], color='blue')
sns.barplot(x='login_name', y='used_memory', data=memory_mean,      ax=ax1[1][1], color='red')
sns.barplot(x='login_name', y='sql_text',  data=sql_text_mean,      ax=ax1[0][0], color='red')
sns.barplot(x='login_name', y='Value',     data=run_mean_melted,    ax=ax1[0][1], hue='Metric')

for ax in ax1.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.subplots_adjust(hspace=0.5)
plt.show()

# %%
heavy_general_users    = list(general_user_run_mean['login_name'])
heavy_general_users_id = list(general_user_run_mean['EMP_NUMBER'])

for login, emp_id in zip(heavy_general_users, heavy_general_users_id):
    if emp_id in emp_id_lst:
        continue
    print(f'\nuser: {login}')
    check_system    = users_df[users_df['login_name'] == login].reset_index(drop=True)
    unique_que      = check_system['group_key'].nunique()
    median_run_time = check_system['total_run_time'].median()
    run_hour        = round(check_system['start_hour'].mean(), 0)
    print(f'  {len(check_system)} שאילתות כבדות, {unique_que} ייחודיות, '
          f'זמן ריצה חציוני {median_run_time:.0f} דקות, שעה ~{2 + run_hour:.0f}')

    check_system_grp = check_system.groupby('group_key').agg(
        CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
        used_memory=('used_memory', 'mean'), total_run_time=('total_run_time', 'median'),
        start_hour=('start_hour', 'mean')
    ).sort_values('total_run_time', ascending=False).head(2).reset_index()

    for _, row in check_system_grp.iterrows():
        print(f'  שאילתה {row["group_key"]}: {row["sql_text"]:.0f} הרצות, '
              f'זמן ריצה {row["total_run_time"]:.0f} דקות, שעה ~{2 + round(row["start_hour"]):.0f}')

# %% [markdown]
# # NOLOCK

# %%
full_no_lock_df = filterd_df_for_dates[filterd_df_for_dates['analyst_users'] == 1]

no_lock_mean = full_no_lock_df.groupby('EMP_USER_NAME').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
    used_memory=('used_memory', 'mean'), nolock_flag=('nolock_flag', 'mean')
).reset_index().sort_values('nolock_flag', ascending=True)

list_of_users = list(no_lock_mean[no_lock_mean['nolock_flag'] <= 0.4]['EMP_USER_NAME'])

no_lock_user_info = filterd_df_for_dates[
    filterd_df_for_dates['EMP_USER_NAME'].isin(list_of_users)
].groupby('EMP_USER_NAME').agg(
    CPU=('CPU', 'mean'), sql_text=('sql_text', 'count'),
    used_memory=('used_memory', 'mean'), nolock_flag=('nolock_flag', 'mean'),
    emp_name_y=('emp_name_y', 'max'), BRANCH_DESC=('BRANCH_DESC', 'max'),
    OCCUPATION_FULL_DESC=('OCCUPATION_FULL_DESC', 'max')
).sort_values('nolock_flag', ascending=True).reset_index()

no_lock_user_info.to_excel('nolock.xlsx')
print(f"Saved: nolock.xlsx ({len(no_lock_user_info)} users with <40% NOLOCK usage)")

# %%
print(f"\nFinal df shape: {df.shape}")
print(f"Date range: {df['start_date'].min()} → {df['start_date'].max()}")
