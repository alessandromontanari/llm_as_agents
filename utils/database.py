import os
import sqlite3


def database_creation(path_dir, database_name):

    assert database_name[-2:] == ".db"

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    database_path = os.path.join(path_dir, database_name)

    # Connection to database
    conn = sqlite3.connect(database_path)

    # Create a cursor object
    cursor = conn.cursor()

    # TODO: fixed name for the table. Must be changed and choosable.
    # Drop the table if it exists
    cursor.execute('DROP TABLE IF EXISTS software')

    # Create the table
    cursor.execute('''
CREATE TABLE software (
    id INTEGER PRIMARY KEY,
    url TEST NOT NULL,
    repo_name TEXT NOT NULL,
    repo_description TEXT NOT NULL,
    stars INTEGER NOT NULL,
    language TEXT NOT NULL
)
''')

    # BEWARE: YOU NEED TO CLOSE THE DATABASE CONNECTION ONCE YOU ARE DONE!
    return conn, cursor


def database_access(path_dir, database_name):

    assert database_name[-2:] == ".db"

    database_path = os.path.join(path_dir, database_name)

    conn = sqlite3.connect(database_path)

    cursor = conn.cursor()

    # TODO: fixed name for the table. Must be changed and choosable.
    cursor.execute('SELECT * FROM software')

    return conn, cursor
