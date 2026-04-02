import csv


def write_csv_from_list(file_name, data_list):
    """
    Appends a list of data as a new row to a CSV file.

    Args:
        file_name (str): The name of the CSV file.
        data_list (list): The data to write as a row.

    Returns:
        None
    """
    with open(file_name, 'a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(data_list)
