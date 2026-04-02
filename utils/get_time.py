import time


def get_time():
    """
    Return the current local time as a formatted string.

    Args:
        None.

    Returns:
        str: Current time in ``YYYY-MM-DD HH:MM:SS`` format.
    """
    timestamp = time.time()
    readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    return readable_time
