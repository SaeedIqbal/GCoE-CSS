import os
import sys
import wget
import tarfile
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path


class DatasetDownloader:
    """
    A unified downloader for common computer vision datasets:
    - Pascal VOC 2012
    - ADE20K
    - Cityscapes
    - MS COCO 2017
    - ImageNet (requires manual login)
    """

    def __init__(self, root_dir):
        self.root = Path(os.path.expanduser(root_dir))
        self.root.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def _download_with_progress(self, url, dest_path):
        """Download file with progress bar."""
        dest_path = Path(dest_path)
        if dest_path.exists():
            print(f"{dest_path.name} already exists. Skipping download.")
            return

        print(f"Downloading {url.split('/')[-1]}...")
        response = self._session.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(dest_path, 'wb') as f, \
             tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(block_size):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"Downloaded: {dest_path}")

    def _extract_tar(self, tar_path, extract_to=None):
        """Extract tar or tgz file."""
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_to or self.root)
        os.remove(tar_path)

    def _extract_zip(self, zip_path, extract_to=None):
        """Extract zip file."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_to or self.root)
        os.remove(zip_path)

    def download_voc(self):
        """Download Pascal VOC 2012 with augmented labels."""
        voc_dir = self.root / "VOC2012"
        if voc_dir.exists():
            print("Pascal VOC 2012 already downloaded.")
            return

        # Main dataset
        voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        voc_tar = self.root / "VOCtrainval_11-May-2012.tar"
        self._download_with_progress(voc_url, voc_tar)
        self._extract_tar(voc_tar, extract_to=self.root)

        # Augmented segmentation (Berkley)
        aug_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
        aug_tar = self.root / "benchmark.tgz"
        self._download_with_progress(aug_url, aug_tar)
        self._extract_tar(aug_tar, extract_to=self.root)

        # Rename to standard VOC2012
        (self.root / "VOCdevkit" / "VOC2012").replace(voc_dir)
        shutil.rmtree(self.root / "VOCdevkit", ignore_errors=True)
        print("✅ Pascal VOC 2012 downloaded and set up.")

    def download_ade20k(self):
        """Download ADE20K dataset."""
        ade_dir = self.root / "ADE20K"
        if ade_dir.exists():
            print("ADE20K already downloaded.")
            return

        url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        zip_path = self.root / "ADEChallengeData2016.zip"

        self._download_with_progress(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)

        # Rename to clean name
        extracted = self.root / "ADEChallengeData2016"
        extracted.replace(ade_dir)
        print("✅ ADE20K downloaded and set up.")

    def download_cityscapes(self):
        """Download Cityscapes dataset (requires login)."""
        city_dir = self.root / "cityscapes"
        if city_dir.exists():
            print("Cityscapes already downloaded.")
            return

        print("\n⚠️  Cityscapes requires manual registration:")
        print("1. Register at: https://www.cityscapes-dataset.com/register/")
        print("2. Login and download:")
        print("   - leftImg8bit_trainvaltest.zip (11.8 GB)")
        print("   - gtFine_trainvaltest.zip (241 MB)")
        print(f"3. Place both files in: {self.root}")
        print("4. We will handle extraction and organization.\n")

        input("Press Enter after downloading the files...")

        # Look for downloaded zip files
        img_zip = self.root / "leftImg8bit_trainvaltest.zip"
        gt_zip = self.root / "gtFine_trainvaltest.zip"

        if not img_zip.exists():
            raise FileNotFoundError(f"Missing: {img_zip}")
        if not gt_zip.exists():
            raise FileNotFoundError(f"Missing: {gt_zip}")

        # Extract
        self._extract_zip(img_zip, extract_to=city_dir)
        self._extract_zip(gt_zip, extract_to=city_dir)

        print("✅ Cityscapes downloaded and organized.")

    def download_coco(self):
        """Download MS COCO 2017 dataset."""
        coco_dir = self.root / "coco"
        coco_dir.mkdir(exist_ok=True)

        subsets = {
            "train2017": "http://images.cocodataset.org/zips/train2017.zip",
            "val2017": "http://images.cocodataset.org/zips/val2017.zip",
            "test2017": "http://images.cocodataset.org/zips/test2017.zip",
        }
        annot_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        # Download images
        for name, url in subsets.items():
            zip_path = self.root / f"{name}.zip"
            self._download_with_progress(url, zip_path)
            self._extract_zip(zip_path, extract_to=coco_dir / "images" / name)

        # Download annotations
        annot_zip = self.root / "annotations_trainval2017.zip"
        self._download_with_progress(annot_url, annot_zip)
        self._extract_zip(annot_zip, extract_to=coco_dir / "annotations")

        print("✅ MS COCO 2017 downloaded and set up.")

    def download_imagenet(self):
        """Provide instructions for ImageNet download."""
        imagenet_dir = self.root / "imagenet"
        if imagenet_dir.exists():
            print("ImageNet directory exists.")
            return

        print("\n⚠️  ImageNet requires academic account:")
        print("1. Request access at: https://www.image-net.org/download.php")
        print("2. You must have an academic email and agree to non-commercial use.")
        print("3. After approval, you'll get:")
        print("   - ILSVRC2012_img_train.tar (138 GB)")
        print("   - ILSVRC2012_img_val.tar (6.3 GB)")
        print("   - ILSVRC2012_devkit_t12.tar.gz (3.4 MB)")
        print(f"4. Place them in: {imagenet_dir}")
        print("5. Use extract_imagenet.py to organize.")

        imagenet_dir.mkdir(exist_ok=True)
        (imagenet_dir / "README.txt").write_text(
            "ImageNet requires manual download due to licensing.\n"
            "Visit https://www.image-net.org/download.php for access.\n"
            "Expected files:\n"
            "- ILSVRC2012_img_train.tar\n"
            "- ILSVRC2012_img_val.tar\n"
            "- ILSVRC2012_devkit_t12.tar.gz\n"
        )
        print("📄 ImageNet setup instructions generated.")


# Optional: Add helper to clean up
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download datasets for continual segmentation.")
    parser.add_argument("--root", type=str, default="./data", help="Root directory to store datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["voc", "ade20k", "cityscapes", "coco", "imagenet", "all"],
                        help="Dataset to download")
    args = parser.parse_args()

    downloader = DatasetDownloader(args.root)

    if args.dataset == "voc":
        downloader.download_voc()
    elif args.dataset == "ade20k":
        downloader.download_ade20k()
    elif args.dataset == "cityscapes":
        downloader.download_cityscapes()
    elif args.dataset == "coco":
        downloader.download_coco()
    elif args.dataset == "imagenet":
        downloader.download_imagenet()
    elif args.dataset == "all":
        downloader.download_voc()
        downloader.download_ade20k()
        downloader.download_coco()
        downloader.download_cityscapes()  # Manual
        downloader.download_imagenet()   # Instructions

if __name__ == "__main__":
    import shutil  # Import here to avoid top-level if not needed
    main()