import unittest
from utils.read_wiki_pages import process_meeting_minutes_ops_calls


class TestReadOPSCallPages(unittest.TestCase):
    def test_read_ops_calls_pages(self):

        file_path = './data/hess_pages/operations_call__25_04_2024.txt'  # Replace with your file path
        structured_data, hess_urls = process_meeting_minutes_ops_calls(file_path)

        for title, content in structured_data.items():
            print(f"{title}:\n{content}\n")

        # TODO: may need to think of some assertions here

if __name__ == '__main__':
    unittest.main()
