import sqlite3
import warnings
import logging
import os

warnings.filterwarnings("ignore")

log_file_path = "./logs/rag/"

if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)

full_path = __file__

script_name = os.path.basename(full_path)[:-3]  # remove .py from the end

logging.basicConfig(
    filename=log_file_path+script_name+".log",
    format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s: %(message)s",
    level=logging.INFO,
    filemode='w'
)

# TODO: implement logging output

def print_table_content(database_path, table_name, max_chars):

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")

    column_names = [description[0] for description in cursor.description]

    rows = cursor.fetchall()

    column_widths = []
    for i in range(len(column_names)):
        max_width = max(len(str(row[i])[:max_chars]) for row in rows)
        column_widths.append(max(max_width, len(column_names[i][:max_chars])))

    header = "| " + " | ".join(f"{name[:max_chars]:<{width}}" for name, width in zip(column_names, column_widths)) + " |"
    print(header)
    print("|-" + "-|-".join('-' * width for width in column_widths) + "-|")

    for row in rows:
        row_str = "| " + " | ".join(f"{str(value)[:max_chars]:<{width}}" for value, width in zip(row, column_widths)) + " |"
        print(row_str)

    conn.close()

def main():

    database_path = "./data/database_sql/hess/hess.db"
    table_name = 'operations_call'

    print_table_content(database_path=database_path, table_name=table_name, max_chars=16)


if __name__ == '__main__':
    main()