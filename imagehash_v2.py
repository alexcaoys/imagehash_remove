import bisect
import glob
import os
import shutil
import argparse
from concurrent import futures
from operator import itemgetter

import hnswlib
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

lst_methods = ['ahash', 'phash', 'dhash', 'whash-haar', 'whash-db4', 'chash', 'crhash']


def imagehash_to_array(image_hash):
    binary_str = bin(int(str(image_hash), 16))
    float_arr = np.asarray(list(binary_str[2:])).astype('float')
    return float_arr


class ImageHashProcessor:
    def __init__(self, method='phash', tol=1, hnsw_ef=200, hnsw_m=16):
        assert (method in lst_methods)
        self.method = method
        self.tol = tol
        self.ef = hnsw_ef
        self.M = hnsw_m

    def imagehash_file(self, img_file):
        try:
            img = Image.open(img_file)
        except OSError:
            print("please check the file: " + img_file)
            return
        # find image size
        img_w, img_h = img.size
        # use different methods to hash
        if self.method == 'ahash':
            # Average hashing
            result = imagehash.average_hash(img)
        elif self.method == 'phash':
            # Perceptual hashing
            result = imagehash.phash(img)
        elif self.method == 'dhash':
            # Difference hashing
            result = imagehash.dhash(img)
        elif self.method == 'whash-haar':
            # Wavelet hashing - haar
            result = imagehash.whash(img)
        elif self.method == 'whash-db4':
            # Wavelet hashing - db4
            result = imagehash.whash(img, mode='db4')
        elif self.method == 'chash':
            # HSV color hashing
            result = imagehash.colorhash(img)
        elif self.method == 'crhash':
            # Crop-resistant hashing
            result = imagehash.crop_resistant_hash(img)
        else:
            result = None
        img.close()
        return {'file': img_file, 'width': img_w, 'height': img_h, 'hash': result}

    def multi_hashing(self, lst_imgs):
        # total number of images
        num_images = len(lst_imgs)

        result_lst = []

        with futures.ProcessPoolExecutor() as executor:
            fs = {
                # submit processes, image_hash(each file)
                executor.submit(self.imagehash_file, img_file):
                    img_file for img_file in lst_imgs
            }
            # tqdm progressbar
            for i, f in tqdm(enumerate(futures.as_completed(fs)), desc="Hashing images", total=num_images,
                             bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar:>10}'):
                # append result to dictionary
                res_dict = f.result()
                result_lst.append(res_dict)

        # concat all output df together
        return result_lst

    def brutal_force_group(self, lst_hashes):
        grouped_lst = []
        for img_dict in tqdm(lst_hashes, desc="Comparing hashes",
                             bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar:>10}'):
            flag_grouped = False
            for group in grouped_lst:
                group_hash = group[0]['hash']
                if abs(img_dict['hash'] - group_hash) < self.tol:
                    group.append(img_dict)
                    flag_grouped = True
                    break
            if flag_grouped:
                continue
            else:
                grouped_lst.append([img_dict])
        return grouped_lst

    def hnsw_query(self, lst_hashes):
        print('Indexing and Querying ...')
        data = np.asarray([imagehash_to_array(item_dict['hash']) for item_dict in lst_hashes])

        num_elements, dim = data.shape
        ids = np.arange(num_elements)

        # Declaring index
        p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

        # Initializing index - the maximum number of elements should be known beforehand
        p.init_index(max_elements=num_elements, ef_construction=self.ef, M=self.M)

        # Element insertion (can be called several times):
        p.add_items(data, ids)

        # Controlling the recall by setting ef:
        p.set_ef(num_elements)  # ef should always be > k

        k = num_elements
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        labels, distances = p.knn_query(data, k=k)
        
        return labels, distances
    
    def hnsw_group(self, lst_hashes):
        num_elements = len(lst_hashes)
        labels, distances = self.hnsw_query(lst_hashes)

        remaining_idx = list(range(num_elements))

        grouped_lst = []

        for i in tqdm(range(num_elements), desc="Comparing hashes",
                      bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar:>10}'):
            if i not in remaining_idx:
                continue
            cur_group_index = labels[i][:bisect.bisect_right(distances[i], self.tol)]
            filtered_idx = list(cur_group_index)
            for idx in cur_group_index:
                if idx in remaining_idx:
                    remaining_idx.remove(idx)
                else:
                    filtered_idx.remove(idx)
            grouped_lst.append([lst_hashes[j] for j in filtered_idx])

        return grouped_lst

    def hnsw_compare(self, lst_hashes, len_ori):
        labels, distances = self.hnsw_query(lst_hashes)
        
        delete_idx = []
        
        for i in tqdm(range(len_ori), desc="Comparing hashes",
                      bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar:>10}'):
            cur_group_index = labels[i][:bisect.bisect_right(distances[i], self.tol)]
            for idx in cur_group_index:
                if idx not in delete_idx and idx >= len_ori:
                    delete_idx.append(idx)
        
        delete_hashes = [lst_hashes[j] for j in delete_idx]
        return delete_hashes


def backup_folder(img_folder):
    if img_folder.endswith('/'):
        img_folder = img_folder[:-1]
    bak_folder = img_folder + '_bak'
    # if bak exist, rm first
    if os.path.isdir(bak_folder):
        input("bak folder already exist, press enter to delete and proceed, Ctrl+C to exit...")
        shutil.rmtree(bak_folder, ignore_errors=True)
    # make a backup
    shutil.copytree(img_folder, bak_folder)
    print('{} backed up to {}'.format(img_folder, bak_folder))


def file_operations(grouped_lst, img_folder, flag_del=True):
    print('Starting file operations ...')
    for i, group in enumerate(grouped_lst):
        if flag_del:
            sorted_group = sorted(group, key=itemgetter('width'), reverse=True)
            for img_dict in sorted_group[1:]:
                os.remove(img_dict['file'])
        else:
            res_grp_folder = os.path.join(img_folder, str(i))
            if not os.path.exists(res_grp_folder):
                os.makedirs(res_grp_folder)
            for img_dict in group:
                shutil.move(img_dict['file'], res_grp_folder)


def dup_remove(img_folder, ihp=ImageHashProcessor(), flag_bak=True, flag_del=True, img_suffixes=None):
    if img_suffixes is None:
        img_suffixes = ('jpg', 'JPG',
                        'jpeg', 'JPEG',
                        'png', 'PNG',
                        'gif', 'GIF',
                        'bmp', 'BMP')

    if flag_bak:
        backup_folder(img_folder)

    lst_img = [f for f_ in [glob.glob(os.path.join(img_folder, '**', "*." + e), recursive=True)
                            for e in img_suffixes]
               for f in f_]

    lst_res = ihp.multi_hashing(lst_img)
    grouped_lst = ihp.hnsw_group(lst_res)

    file_operations(grouped_lst, img_folder, flag_del)

    print('There are {}/{} duplicates'.format(len(lst_img) - len(grouped_lst), len(lst_img)))


def compare_folders(img_folder, cmp_folder, ihp=ImageHashProcessor(),
                    flag_bak=True, flag_del=True, img_suffixes=None):
    if img_suffixes is None:
        img_suffixes = ('jpg', 'JPG',
                        'jpeg', 'JPEG',
                        'png', 'PNG',
                        'gif', 'GIF',
                        'bmp', 'BMP')

    if flag_bak:
        backup_folder(img_folder)

    lst_img = [f for f_ in [glob.glob(os.path.join(img_folder, '**', "*." + e), recursive=True)
                            for e in img_suffixes]
               for f in f_]
    lst_cmp = [f for f_ in [glob.glob(os.path.join(cmp_folder, '**', "*." + e), recursive=True)
                            for e in img_suffixes]
               for f in f_]

    lst_all = lst_img + lst_cmp

    lst_res = ihp.multi_hashing(lst_all)
    delete_lst = ihp.hnsw_compare(lst_res, len(lst_img))

    print('Start file operations ...')
    if flag_del:
        for img_dict in delete_lst:
            os.remove(img_dict['file'])
    else:
        result_folder = os.path.join(cmp_folder, 'duplicates')
        if os.path.isdir(result_folder):
            input("duplicates folder already exist, press enter to delete and proceed, Ctrl+C to exit...")
            shutil.rmtree(result_folder, ignore_errors=True)
        os.makedirs(result_folder)
        for img_dict in delete_lst:
            shutil.move(img_dict['file'], result_folder)
    
    print('There are {}/{} duplicates in the target folder'.format(len(delete_lst), len(lst_cmp)))


def main():
    parser = argparse.ArgumentParser(
        description="Using different ImageHash method to find the duplicate images in a given folder")
    parser.add_argument("origin", type=str, help="the origin folder of all processing files")
    parser.add_argument("-c", "--compare", type=str, help="the folder need to compare with origin")
    parser.add_argument("-t", "--tolerance", type=float, default=1.,
                        help="Hashing difference tolerance (Default: 1)")
    parser.add_argument("-ef", "--hnsw_ef", type=int, default=200,
                        help="hnswlib ef_construction param, ADJUST WITH CAUTION (Default: 200)")
    parser.add_argument("-m", "--hnsw_m", type=int, default=16,
                        help="hnswlib M param, ADJUST WITH CAUTION (Default: 16)")
    parser.add_argument("-b", "--backup", action="store_false",
                        help="create backups for original images (Default: True)")
    parser.add_argument("-d", "--delete", action="store_false",
                        help="delete all duplicates (Default) (otherwise copied to subdirectories)")
    group_method = parser.add_mutually_exclusive_group()
    group_method.add_argument("-A", "--Average", action="store_true", help="Average hashing")
    group_method.add_argument("-P", "--Perceptual", action="store_true", help="Perceptual hashing (Default)")
    group_method.add_argument("-D", "--Difference", action="store_true", help="Difference hashing")
    group_method.add_argument("-WH", "--WaveHaar", action="store_true", help="Wavelet hashing - haar")
    group_method.add_argument("-WD", "--WaveDB4", action="store_true", help="Wavelet hashing - db4")
    group_method.add_argument("-C", "--Color", action="store_true", help="HSV color hashing")
    group_method.add_argument("-CR", "--Crop", action="store_true", help="Crop-resistant hashing (SLOW)")
    args = parser.parse_args()

    origin_folder = args.origin

    # tolerance for similarity (difference)
    if args.tolerance < 0:
        print('tolerance should > 0')
        exit(0)

    if args.hnsw_ef < 1:
        print("hnswlib ef_construction param should be an integer value between 1 and the size of the dataset")
        exit(0)

    if args.hnsw_m < 2:
        print("hnswlib M param can't be smaller than 2")
        exit(0)
    elif args.hnsw_m > 100:
        print("Reasonable range for hnswlib M param is 2-100")

    # applied methods (see image_hash function)
    if args.Average:
        applied_method = 'ahash'
    elif args.Perceptual:
        applied_method = 'phash'
    elif args.Difference:
        applied_method = 'dhash'
    elif args.WaveHaar:
        applied_method = 'whash-haar'
    elif args.WaveDB4:
        applied_method = 'whash-db4'
    elif args.Color:
        applied_method = 'chash'
    elif args.Crop:
        applied_method = 'crhash'
    else:
        # default
        applied_method = 'phash'

    # include all suffixes
    image_suffixes = ('jpg', 'JPG',
                      'jpeg', 'JPEG',
                      'png', 'PNG',
                      'gif', 'GIF',
                      'bmp', 'BMP')

    ihp = ImageHashProcessor(method=applied_method, tol=args.tolerance,
                             hnsw_ef=args.hnsw_ef, hnsw_m=args.hnsw_m)
    if args.compare:
        compare_folder = args.compare
        compare_folders(origin_folder, compare_folder, ihp, args.backup, args.delete, image_suffixes)
    else:
        dup_remove(origin_folder, ihp, args.backup, args.delete, image_suffixes)


if __name__ == "__main__":
    main()
