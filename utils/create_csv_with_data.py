import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--load_img_dir', action="store", dest="load_img_dir", type=str)
parser.add_argument('--save_csv_path', action="store", dest="save_csv_path", type=str)
parser.add_argument('--save_full_path', action="store", default=False, dest="save_full_path", type=bool)


args = parser.parse_args()


def _getListOfFiles(dirName, patterns=('jpg', 'jpeg', 'png')):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + _getListOfFiles(fullPath)
        else:
            if str(fullPath).endswith(patterns):
                allFiles.append(fullPath)

    return allFiles


def getListOfFiles(dirName, patterns=('jpg', 'jpeg', 'png'), save_full_path=False):
    if dirName.endswith('/'):
        dirName = dirName[:-1]
    all_fully_paths = _getListOfFiles(dirName, patterns)
    if save_full_path:
        return all_fully_paths
    else:
        return [path[len(dirName)+1:] for path in all_fully_paths]


list_of_imgs = getListOfFiles(dirName=args.load_img_dir, save_full_path=args.save_full_path)
pd.DataFrame(list_of_imgs, columns=['imgName']).to_csv(args.save_csv_path)
print(f'find {len(list_of_imgs)} files!')
