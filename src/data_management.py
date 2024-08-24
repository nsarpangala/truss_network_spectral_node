import os
import shutil

def create_or_replace_directory(directory):
    """
    Create a directory if it doesn't exist, or ask for confirmation to replace it if it does exist.

    Parameters:
    directory (str): Path of the directory.

    Returns:
    None
    """
    if os.path.exists(directory):
        replace = input(f"The directory {directory} already exists. Do you want to replace it? (yes/no): ")
        if replace.lower() == "yes":
            shutil.rmtree(directory)
            os.makedirs(directory)
            print(f"The directory {directory} has been replaced.")
        else:
            print("The directory was not replaced.")
    else:
        os.makedirs(directory)
        print(f"The directory {directory} has been created.")