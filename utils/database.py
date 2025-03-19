import os
import sqlite3

# TODO: add logging file writing

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
    else:
        raise NameError("Only paper_info and software are currently supported as table names.")

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
