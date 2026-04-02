import csv


def read_csv_to_list(file_path):
    """
    Reads a CSV file into a flattened list of cell values.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list containing all cell values from the CSV.
    """
    cell_list = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for cell in row:
                cell_list.append(cell)
    return cell_list

# cell_list = read_csv_to_list('data.csv')
# print(cell_list)


def read_csv_to_list_row(file_path):
    """
    Reads a CSV file into a list of rows, where each row is a list of cell values.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of lists, representing the rows of the CSV.
    """
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return data

# Example usage (this won't work here as we don't have access to files in this environment)
# data = read_csv_to_list("path_to_your_file.csv")
# print(data)
