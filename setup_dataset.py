import os
import argparse
from glob import glob
import pandas as pd
from tqdm import tqdm
import re
import h5py
from hashids import Hashids
from PIL import Image
import torchvision.transforms.functional as ttf
import numpy as np
import src.utils.validation as validation

parser = argparse.ArgumentParser()

parser.add_argument("dataset_root", type=validation.is_dir, help="path to dataset folder.")
parser.add_argument(
    "path_to_hdf", type=validation.new_hdf, help="path to hdf file to create or overwrite."
)
args = parser.parse_args()

dataset_root = args.dataset_root
hdf_file = args.path_to_hdf

# target names
targets = sorted(os.path.basename(p) for p in glob(os.path.join(dataset_root, "*")))

meta = []
idx = 0
for target in tqdm(targets):
    target_root = os.path.join(dataset_root, target)
    if not os.path.isdir(target_root):
        continue
    files = glob(os.path.join(target_root, "*"))
    df = pd.DataFrame(files, columns=["file"])
    df = df.assign(
        bundle=df.file.apply(
            lambda x: re.sub(r"([H,L]1)_([a-zA-Z0-9]{10})_.*.png", r"\1_\2", os.path.basename(x))
        )
    )
    for bundle, bf in df.groupby("bundle"):
        bundle_id = Hashids(min_length=6).encode(idx)
        idx += 1
        for file in bf.file:
            span = re.sub(r".*([0-9]\.[0-9]).png", r"\1", os.path.basename(file)).replace(".", "")
            uid = bundle_id + "_" + span
            assert uid not in meta
            meta.append(
                {
                    "unique_id": uid,
                    "file_path": file,
                    "target": target,
                    "bundle_id": bundle_id,
                    "span": span,
                }
            )

df = pd.DataFrame(meta)
df = df.groupby(["target", "bundle_id"], group_keys=False).apply(lambda d: d.sort_values("span"))
df = df.reset_index(drop=True)
df = df.reindex(columns=["unique_id", "file_path", "target", "bundle_id", "span"])


def file2img(file, target_size=(224, 224)):
    # read file and convert to gray scale
    img = Image.open(file).convert("L")
    img = ttf.crop(img, 66, 105, 466, 566)
    w, h = img.size
    tw, th = target_size
    if w > tw and h > th:
        ratio = max(tw, th) / min(w, h)
        img = ttf.resize(img, (int(h * ratio), int(w * ratio)))
    img = np.asarray(img, dtype=np.uint8)
    return img


def process(fp, df, dataset_id, **kwargs):
    images = []
    for index, row in df.iterrows():
        image = file2img(row["file_path"])
        images.append(image)
    images = np.stack(images)
    ds = fp.create_dataset(dataset_id, data=images, shape=images.shape, dtype="uint8")
    for k, v in kwargs.items():
        ds.attrs[k] = v


with h5py.File(hdf_file, mode="w") as fp:
    for target, tf in df.groupby("target"):
        tp = fp.create_group(target)
        print(f"Stroing {target}...")
        delayed_fns = []
        for bundle, bf in tqdm(tf.groupby("bundle_id")):
            bf = bf.sort_values("span")
            process(tp, bf, bundle, target=target)
        fp.flush()
