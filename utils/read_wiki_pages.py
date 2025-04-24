import re
from datetime import datetime

def process_meeting_minutes_ops_calls(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        meeting_minutes = file.read()

    structured_data = {}
    connection_details_pattern = r'## Connection Details\n.*?(?=\n##|\Z)'
    cleaned_minutes = re.sub(connection_details_pattern, '', meeting_minutes, flags=re.DOTALL)
    cleaned_minutes = re.sub(r'<span[^>]*>|</span>|<br>', '', cleaned_minutes)
    hess_url_pattern = r'/hess\S+'
    hess_urls = re.findall(hess_url_pattern, cleaned_minutes)
    sections = re.split(r'## |\n##', cleaned_minutes)

    for section in sections:
        if section.strip():
            title, *content = section.strip().split('\n', 1)

            if title != "Minutes":
                structured_data[title.strip()] = content[0].strip() if content else ''

            else:
                row_pattern = re.compile(r'\|(.*?)\|(.*?)\|(.*?)\|')  # for each entry in the Minutes table
                rows = row_pattern.findall(meeting_minutes)

                for agenda_item, presenter, minutes in rows[2:]:
                    agenda_item_cleaned = re.sub(r', \[slides\]\(.*?\)', '', agenda_item)
                    minutes_cleaned = re.sub(r'<span[^>]*>|</span>|<br>', '', minutes)

                    structured_data[agenda_item_cleaned.strip()] = f"{presenter.strip()}, {minutes_cleaned.strip()}" if minutes_cleaned else ''

    return structured_data, hess_urls


def find_dict_item_with_substring_key(dictionary, substring):
    for key in dictionary:
        if substring in key:
            return dictionary[key]
    return None

# TODO: these two functions could be merged into one, and one may just need to add an if condition
def find_substring_key(dictionary, substring):
    for key in dictionary:
        if substring in key:
            return key
    return None

def find_unexpected_keys(dictionary, allowed_keys):
    # TODO: something must be done here because a few things do not fit properly inside the filter
    unexpected_keys = [key for key in dictionary if key not in allowed_keys]
    return unexpected_keys

def try_filling_list(dictionary: dict, list_to_fill: list, shift_name: str, substring_for_keys: str) -> list:

    try:
        element_to_append = find_dict_item_with_substring_key(dictionary, substring_for_keys)
        list_to_fill.append(element_to_append)
    except Exception as e:
        list_to_fill.append(" ")
        print(f"skipped {shift_name}, because of {e}")

    return list_to_fill


def process_ops_calls_for_database(list_paths: list):

    ops_call_dates, shift_names, attendees, ops_pages, ops_intros, day_shift_reports = [], [], [], [], [], []
    day_shift_reports, hessiu_statuses, fc_statuses, pointing_statuses, daq_statuses, tracking_statuses = [], [], [], [], [], []
    aobs = []

    for path_to_ops_call_data in list_paths:
        structured_data, hess_urls = process_meeting_minutes_ops_calls(path_to_ops_call_data)

        attendees.append(structured_data["Attendees"])
        ops_pages.append(structured_data["Operations pages"])

        shift_link = structured_data["Operations pages"].split("Current shift workbook: ")[1].split("\n")[0]
        ops_call_date = "-".join(path_to_ops_call_data.split("operations_call__")[1].split(".txt")[0].split("_"))
        date_format = '%d-%m-%Y'

        ops_call_date = datetime.strptime(ops_call_date, date_format)
        ops_call_dates.append(ops_call_date)

        shift_name = shift_link.split("[Shift workbook - ")[1].split("](")[0]
        shift_names.append(shift_name)

        ops_intros = try_filling_list(structured_data, ops_intros, shift_name, "OPS intro")
        day_shift_reports = try_filling_list(structured_data, day_shift_reports, shift_name, "Day shift")
        hessiu_statuses = try_filling_list(structured_data, hessiu_statuses, shift_name, "HESSIU")
        fc_statuses = try_filling_list(structured_data, fc_statuses, shift_name, "FC")
        pointing_statuses = try_filling_list(structured_data, pointing_statuses, shift_name, "Pointing")
        tracking_statuses = try_filling_list(structured_data, tracking_statuses, shift_name, "Tracking")
        daq_statuses = try_filling_list(structured_data, daq_statuses, shift_name, "DAQ")

        try:
            aob = find_dict_item_with_substring_key(structured_data, "AOB")
            allowed_keys = ["OPS intro", "Day shift report", "HESSIU status", "DAQ status", "FC status", "Pointing",
                            "Tracking", "AOB", "Attendees", "Operations pages"]
            unexpected_keys = find_unexpected_keys(structured_data, allowed_keys)
            aob_to_append = aob + ", " + ", ".join([f"{key}: "+structured_data[key] for key in unexpected_keys])
            aobs.append(aob_to_append)
        except Exception as e:
            aobs.append(" ")
            print(f"skipped {shift_name}, because of {e}")

    sorted_indices = [index for index, _ in sorted(enumerate(ops_call_dates), key=lambda x: x[1])]
    ops_call_dates = sorted(ops_call_dates)
    shift_names, attendees = [shift_names[ii] for ii in sorted_indices], [attendees[ii] for ii in sorted_indices]
    ops_pages, ops_intros = [ops_pages[ii] for ii in sorted_indices], [ops_intros[ii] for ii in sorted_indices]
    day_shift_reports = [day_shift_reports[ii] for ii in sorted_indices]
    hessiu_statuses, fc_statuses = [hessiu_statuses[ii] for ii in sorted_indices], [fc_statuses[ii] for ii in sorted_indices]
    pointing_statuses, daq_statuses = [pointing_statuses[ii] for ii in sorted_indices], [daq_statuses[ii] for ii in sorted_indices]
    tracking_statuses = [tracking_statuses[ii] for ii in sorted_indices]
    aobs = [aobs[ii] for ii in sorted_indices]

    output = [
        ops_call_dates, shift_names, attendees, ops_intros, ops_pages, day_shift_reports, hessiu_statuses,
        fc_statuses, pointing_statuses, daq_statuses, tracking_statuses, aobs
    ]

    return output