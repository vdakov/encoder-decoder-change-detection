



import os

os.chdir(os.path.join( '..', 'data', 'HRSCD', 'labels_land_cover_B'))

for filename in os.listdir('.'):
    if filename.__contains__(".png"):
        os.rename(filename, filename[:3] + filename[8:22] + filename[32:] )
        # 14-2012-0415-6890-LA93-0M50-E080_small.png
        # os.rename(filename, filename[:-11]  + filename[-10:])
        # 14-2012-0415-6890-LA93-0M50-E080_small.png
        # 14-2005-0435-6900-LA93_small 