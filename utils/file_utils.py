import os


def create_dir(directory, delete_previous=True):
    """
    Create a directory given a path
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