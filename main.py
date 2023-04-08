import os
import argparse
from tqdm import tqdm
import logging

from functools import partial
from PIL import Image, ImageChops
from multiprocessing import Pool, cpu_count
from numpy import array_split

from settings import IMAGE_FILETYPES

logging.basicConfig(filename='log.txt', level=logging.DEBUG)


def find_image_files(directory: str) -> list[str]:
    # recursively find all images in directory
    image_files = []
    for root, _, files in tqdm(os.walk(directory), desc='Finding images'):
        for file in files:
            if file.split('.')[-1].lower() in IMAGE_FILETYPES:
                image_files.append(os.path.join(root, file))
    return image_files


def get_date_taken(image: Image) -> str:
    try:
        return image._getexif()[36867]
    except Exception as e:
        logging.info(f"Could not get date taken: {e}")
        return ""


def image_content_equal(image_file1: str, image_file2: str) -> bool:
    try:
        with Image.open(image_file1).convert('RGB') as image1, Image.open(image_file2).convert('RGB') as image2:
            diff = ImageChops.difference(image1, image2)
            if diff.getbbox() is None:
                return True
    except Exception as e:
        logging.error(f"Could not read image: {e}")
    return False


def image_name_equal(image_file1: str, image_file2: str) -> bool:
    return image_file1.split('/')[-1] == image_file2.split('/')[-1]


def _map_duplicates(image_files_chunk: list[str], image_files: list[str]) -> dict[str, list[str]]:
    duplicates = {}
    for image_file1 in image_files_chunk:
        for image_file2 in image_files:
            if image_file1 == image_file2:
                continue
            if image_name_equal(image_file1, image_file2):
                if image_content_equal(image_file1, image_file2):
                    if image_file1 not in duplicates:
                        duplicates[image_file1] = []
                    duplicates[image_file1].append(image_file2)
    return duplicates


def get_duplicates(image_files: list[str]) -> set[str]:
    workers = cpu_count() - 1  # so you can use your computer while it's running
    image_files_split = array_split(image_files, workers)

    with Pool(workers) as p:
        duplicates_dict_list = list(tqdm(
            p.imap(
                partial(_map_duplicates, image_files=image_files),
                image_files_split
            ),
            desc='Comparing images', total=workers
        ))

    duplicates_dict = {}
    for duplicates_dict_ in duplicates_dict_list:
        duplicates_dict.update(duplicates_dict_)

    # clean duplicates_dict
    duplicates_dict_filtered = {}
    for key, values in duplicates_dict.items():
        for value in values:
            if value not in duplicates_dict.keys():
                duplicates_dict_filtered[key] = values

    # extract duplicates
    duplicates = set()
    for _, values in duplicates_dict_filtered.items():
        for value in values:
            duplicates.add(value)

    return duplicates


def _flatten_and_save(image_files: list[str], prefix: str, suffix: str, max_number_length: str, thread: int) -> None:
    for i, image_file in tqdm(enumerate(image_files), desc='Flattening and renaming images', total=len(image_files)):
        image = Image.open(image_file).convert('RGB')
        date_taken = get_date_taken(image)
        i = 0
        if date_taken:
            filename = date_taken.replace(':', '-').replace(' ', '-') + suffix
        else:
            filename = str(thread) + str(i).zfill(max_number_length) + suffix
        image.save(os.path.join(prefix, filename))


def flatten_and_save(image_files: list[str], prefix: str = 'out', suffix: str = '.jpg') -> None:
    # flatten images, rename and save them to the output directory
    max_number_length = len(str(len(image_files)))
    workers = cpu_count() - 1  # so you can use your computer while it's running
    image_files_split = array_split(image_files, workers)
    with Pool(workers) as p:
        p.starmap(
            _flatten_and_save,
            zip(image_files_split, [prefix] * workers, [suffix] * workers,
                [max_number_length] * workers, thread=range(workers))
        )


def main():
    # parse --dir argument to parse data from
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data', help='Directory containing the images')
    parser.add_argument('--out', type=str, default='out', help='Directory to save the output images')
    args = parser.parse_args()
    image_dir = args.dir
    out_dir = args.out

    image_files = find_image_files(image_dir)
    duplicates = get_duplicates(image_files)
    image_files_deduplicated = [image_file for image_file in image_files if image_file not in duplicates]
    flatten_and_save(image_files_deduplicated, prefix=out_dir)


if __name__ == '__main__':
    main()
