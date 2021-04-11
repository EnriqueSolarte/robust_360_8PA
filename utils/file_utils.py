import os
import csv


def write_file(file, line):
    """
    Writes in a file one specific line
    """
    with open(file, "a+") as f:
        writer = csv.writer(f)
        writer.writerow(line)
    f.close()

def create_file(file, delete_previous=True):
    """
    Creates a file given a path
    :param delete_previous:
    :param file:
    :return: Nothing
    """
    if not os.path.exists(file):
        file = open(file, "w+")
    else:
        if delete_previous:
            os.remove(file)
        file = open(file, "w+")
    return file
    
def create_dir(directory, delete_previous=True):
    """
    Creates a directory given a path
    :param delete_previous: 
    :param directory:
    :return: Nothing
    """
    import shutil
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if delete_previous:
            shutil.rmtree(directory)
            os.makedirs(directory)
    return directory
