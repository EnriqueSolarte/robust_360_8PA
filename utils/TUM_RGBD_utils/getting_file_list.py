import os
from file_utilities import *

if __name__ == '__main__':
    path = '/home/kike/Documents/Dataset/TUM_Omnidirectional/T2/T2_orig'
    files_path = path + "/images"

    assert os.path.isdir(files_path)

    list_files = list_files(files_path, key=".png")
    list_names = [get_file_name(name) for name in list_files]
    out_put_file = path + "/images.txt"
    if os.path.isfile(out_put_file):
        os.remove(out_put_file)

    list_files = [[stamp[:-4], "images/{}".format(stamp)]
                  for _, stamp in list_names]
    list_files.sort()
    for stamp, filename in list_files:
        line = str(stamp) + " " + filename
        write_report(out_put_file, [line])
    print('done')
