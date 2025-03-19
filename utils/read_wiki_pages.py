import re

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
    structured_minutes = []

    for section in sections:
        if section.strip():
            title, *content = section.strip().split('\n', 1)
            structured_data[title.strip()] = content[0].strip() if content else ''
            if title == "Minutes":
                row_pattern = re.compile(r'\| (.*?) \| (.*?) \| (.*?) \|')  # for each entry in the Minutes table
                rows = row_pattern.findall(meeting_minutes)
                for agenda_item, presenter, minutes in rows[2:]:
                    agenda_item_cleaned = re.sub(r', \[slides\]\(.*?\)', '', agenda_item)
                    minutes_cleaned = re.sub(r'<span[^>]*>|</span>|<br>', '', minutes)
                    structured_minutes.append({
                        'Title': agenda_item_cleaned.strip(),
                        'Content': f"{presenter.strip()}, {minutes_cleaned.strip()}"
                    })

    return structured_data, hess_urls, structured_minutes