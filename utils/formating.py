"""
Function, which formats a title to the following format
----------------------------
----------- title ----------
----------------------------
Used for better visual clarity of output
"""
def format_title_string(title: str, size_of_title: int) -> str:
    assert len(title) + 2 < size_of_title
    string = "-" * size_of_title + "\n"
    padding_size = int((size_of_title - len(title)) / 2)
    string += "-" * padding_size + " " + title + " " + "-" * padding_size + "\n"
    string += "-" * size_of_title + "\n"
    return string
