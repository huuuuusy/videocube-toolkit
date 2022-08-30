from __future__ import absolute_import, division

import os

def makedir(path):
    isExists=os.path.exists(path)
    if not isExists:        
        os.makedirs(path)
        return True
    else:
        return False

def read_filename(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filenames.append(os.path.join(root, file))
    filenames.sort(key = lambda x:int(x[-10:-4]))
    return filenames

    