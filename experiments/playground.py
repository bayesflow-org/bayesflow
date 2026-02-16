import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import bayesflow as bf
    import keras

    import matplotlib.pyplot as plt
    import numpy as np

    import gdown
    import shutil
    from tqdm import tqdm

    return bf, gdown, np, os, shutil, tqdm


@app.cell
def _():
    return


@app.cell
def _(gdown, os, shutil, tqdm):
    dataset_dir = "datasets/celeba"
    os.makedirs(dataset_dir, exist_ok=True)

    # before downloading again check if files are already there
    if not os.path.exists(os.path.join(dataset_dir, "train")):
        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
        output = os.path.join(dataset_dir, "data.zip")
        gdown.download(url, output, quiet=False)
        shutil.unpack_archive(output, dataset_dir)
        os.remove(output)
        img_files = sorted([f for f in os.listdir("datasets/celeba/img_align_celeba") if f.endswith(".jpg")])
        print(f"Found {len(img_files)} images.")

        train_fnames = img_files[:162770]  # First 162770 for training
        valid_fnames = img_files[162770:182637]  # Next 19867 for
        test_fnames = img_files[182637:]  # Last 19962 for testing

        os.makedirs(os.path.join(dataset_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "valid"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "test"), exist_ok=True)

        for fname in tqdm(train_fnames, desc="Moving training images"):
            shutil.move(os.path.join("datasets/celeba/img_align_celeba", fname), os.path.join(dataset_dir, "train", fname))
        for fname in tqdm(valid_fnames, desc="Moving validation images"):
            shutil.move(os.path.join("datasets/celeba/img_align_celeba", fname), os.path.join(dataset_dir, "valid", fname))
        for fname in tqdm(test_fnames, desc="Moving test images"):
            shutil.move(os.path.join("datasets/celeba/img_align_celeba", fname), os.path.join(dataset_dir, "test", fname))

        os.remove("datasets/celeba/img_align_celeba")  # Remove the now-empty directory)
    return


@app.cell
def _(bf, np, os):
    from PIL import Image

    def load_fn(filepath: os.PathLike) -> any:
        with Image.open(filepath) as img:
            #img = img.convert("L")  # Convert to grayscale
            img = img.resize((64, 64), Image.Resampling.BILINEAR) # Resize to (64, 64)
            obs = img.convert("L") # Convert to grayscale

            img_array = np.array(img) / 255. # Convert to NumPy array
            obs_array = np.array(obs) / 255.
            obs_array = np.expand_dims(obs_array, axis=-1) # Add channel dimension for grayscale
            return {
                "img": img_array.astype(np.float32),
                "gray_img": obs_array.astype(np.float32),
            }

    adapter = (
        bf.adapters.Adapter()
        .standardize(
            include=["img", "gray_img"],
            mean=0.5,
            std=0.5,
        )
        .convert_dtype(np.float64, np.float32, include=["img", "gray_img"])
        .rename("img", "inference_variables")
        .rename("gray_img", "inference_conditions")
    )
    return adapter, load_fn


@app.cell
def _(adapter, bf, load_fn):
    # stage res: (64, 32, 16, 8)
    diffusion = bf.networks.DiffusionModel(
        subnet=bf.networks.UNet,
        subnet_kwargs=dict(
            widths=(64, 64, 128, 128),
            res_blocks=2,
            attn_stage=(False, False, True, True)
        )
    )
    workflow = bf.BasicWorkflow(
        simulator=None,
        adapter=adapter,
        inference_network=diffusion,
        summary_network=None,
        inference_variables="img",
        inference_conditions="gray_img",
        initial_learning_rate=2e-4,
    )

    dataset_test = bf.datasets.DiskDataset(
        root = "datasets/celeba/test_small/",
        pattern = "*.jpg",
        load_fn = load_fn,
        batch_size = 100,
        adapter = adapter,
    )

    history = workflow.fit_disk(
        root = "datasets/celeba/train",
        pattern = "*.jpg",
        batch_size = 64,
        load_fn = load_fn,
        epochs = 1,
        keep_optimizer = False,
        validation_data = dataset_test[0], 
        augmentations = None,
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
