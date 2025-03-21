import unittest
from glob import glob
from utils.read_wiki_pages import process_ops_calls_for_database
from utils.database import database_creation


class TestCreateSQLDBOpsCalls(unittest.TestCase):
    def test_create_sql_db_ops_calls(self):

        db_connection, db_cursor = database_creation(path_dir="./data/database_sql/hess/", database_name="hess.db", table_name="operations_call")

        list_paths = glob("./data/hess_pages/ops_calls/*")

        ops_call_info = process_ops_calls_for_database(list_paths=list_paths)

        (
            ops_call_dates, shift_names, attendees, ops_intros, ops_pages, day_shift_reports, hessiu_statuses,
            fc_statuses, pointing_statuses, daq_statuses, tracking_statuses, aobs
        ) = ops_call_info

        data_for_db = list(zip(shift_names, ops_call_dates, attendees, ops_pages, ops_intros, day_shift_reports,
                               hessiu_statuses, fc_statuses, daq_statuses, tracking_statuses, pointing_statuses, aobs))
        db_cursor.executemany('''
        INSERT INTO operations_call (shift_name, ops_call_date, attendees, ops_pages, ops_intro, day_shift_report, 
        hessiu_status, fc_status, daq_status, tracking, pointing, aob) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_for_db)

        db_connection.commit()

        db_cursor.execute('SELECT * FROM operations_call')

        # Step 6: Fetch and print the data
        rows = db_cursor.fetchall()
        for row in rows:
            print(row)

        db_connection.close()




if __name__ == '__main__':
    unittest.main()
