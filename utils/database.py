import os
import sqlite3

# TODO: add logging file writing

# Naming convention for the tables:
# - Singular names for tables
# - Singular names for columns
# - Schema name for tables prefix (E.g.: SchemeName.TableName)
# - Pascal casing (a.k.a. upper camel case)
# - all lower case for column names, words separated by _

def create_table_with_col_names(cursor, table_name: str):

    if table_name == "paper_info":
        cursor.execute(f'''
CREATE TABLE {table_name} (
    id INTEGER PRIMARY KEY,
    paper_identifier INTEGER NOT NULL, 
    paper_title TEXT NOT NULL, 
    paper_keywords TEXT NOT NULL
)
''')

    elif table_name == "software":
        cursor.execute(f'''
CREATE TABLE {table_name} (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    repo_description TEXT NOT NULL,
    stars INTEGER NOT NULL,
    language TEXT NOT NULL,
    paper_identifier INTEGER NOT NULL 
)
''')

    elif table_name == "operations_call":
        cursor.execute(f'''
CREATE TABLE {table_name} (
    shift_name VARCHAR(10) PRIMARY KEY,
    ops_call_date DATE NOT NULL,
    attendees TEXT,
    ops_pages TEXT,
    ops_intro TEXT,
    day_shift_report TEXT,
    hessiu_status TEXT,
    fc_status TEXT,
    daq_status TEXT,
    tracking TEXT,
    pointing TEXT,
    aob TEXT
)
''')

    else:
        raise NameError("Only paper_info, software, and operations_call are currently supported as table names.")

    return cursor

def database_creation(path_dir: str, database_name: str, table_name: str):

    assert " " not in table_name, "There can't be white spaces in the SQL table name..."
    assert database_name[-3:] == ".db", "Wrong database name"

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    database_path = os.path.join(path_dir, database_name)

    # Connection to database
    conn = sqlite3.connect(database_path)

    # Create a cursor object
    cursor = conn.cursor()

    # Drop the table if it exists
    cursor.execute('DROP TABLE IF EXISTS ' + table_name)

    # Create the table
    cursor = create_table_with_col_names(cursor, table_name=table_name)

    # BEWARE: YOU NEED TO CLOSE THE DATABASE CONNECTION ONCE YOU ARE DONE!
    return conn, cursor


def database_access(path_dir: str, database_name: str, table_name: str = "software"):

    assert " " not in table_name, "There can't be white spaces in the SQL table name..."
    assert database_name[-3:] == ".db", "Wrong database name"

    database_path = os.path.join(path_dir, database_name)

    conn = sqlite3.connect(database_path)

    cursor = conn.cursor()

    cursor.execute('SELECT * FROM ' + table_name)

    return conn, cursor
