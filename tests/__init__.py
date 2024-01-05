import os
from pathlib import Path


def create_init_file():
    # Define the file path for __init__.py
    file_path = Path("tests/__init__.py")

    # Create the __init__.py file
    with open(file_path, "w") as f:
        pass  # Empty file

# Create the __init__.py file
create_init_file()
