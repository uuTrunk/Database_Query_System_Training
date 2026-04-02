from utils.paths import ASK_GRAPH_LOG_CSV
from utils.read_csv import read_csv_to_list_row


def process_output_list(output_file, verbose=False):
    """
    Process the output CSV file to calculate success rates and retry statistics for each question.

    Args:
        output_file (str): Path to the CSV file containing the output logs.

    Returns:
        dict: A dictionary where keys are questions and values are lists containing:
              - retry_list (list): Count of successful attempts for each retry count (0-5).
              - success_rate (float): The ratio of successful attempts to total attempts.
              - file (str): The filename associated with the successful attempt.
    """
    output_list = read_csv_to_list_row(output_file)
    if not output_list:
        return {}

    compact_rows = []
    for row in output_list:
        if len(row) < 9:
            continue
        compact_rows.append([row[1], row[5], row[6], row[7], row[8]])

    if not compact_rows:
        return {}

    unique_questions = list(set([row[0] for row in compact_rows]))
    outcome = {}
    for question in unique_questions:
        right = 0
        wrong = 0
        retry_list = [0 for _ in range(6)] # Initialize retry counts for 0 to 5 retries
        outputs = [row for row in compact_rows if row[0] == question]
        file = ""
        for result in outputs:
            # Check for success: status 200 or status 504 with valid content
            if result[1] == "200" or (result[1] == "504" and result[3] != ""):
                right = right+1
                try:
                    retry_idx = int(result[2])
                    if 0 <= retry_idx < len(retry_list):
                        retry_list[retry_idx] = retry_list[retry_idx] + 1
                except Exception:
                    pass
                file = result[4]
            else:
                wrong = wrong + 1
        # Calculate success rate and store results
        total = wrong + right
        success_rate = right / total if total else 0.0
        outcome[question] = [retry_list, success_rate, file]
    if verbose:
        print(outcome)
    return outcome


if __name__ == "__main__":
    process_output_list(ASK_GRAPH_LOG_CSV)
