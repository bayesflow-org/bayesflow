import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import bayesflow as bf
    import keras

    import matplotlib.pyplot as plt
    import numpy as np

    import gdown
    import shutil
    from tqdm import tqdm
    import gc

    from PIL import Image

    return Image, bf, gc, gdown, keras, np, os, plt, shutil, tqdm


@app.cell
def _(gdown, os, shutil, tqdm):
    dataset_dir = "datasets/celeba"
    os.makedirs(dataset_dir, exist_ok=True)

    train_dir = os.path.join(dataset_dir, "train")
    valid_dir = os.path.join(dataset_dir, "valid")
    test_dir = os.path.join(dataset_dir, "test")
    valid_small_dir = os.path.join(dataset_dir, "valid_small")

    # before downloading again check if files are already there
    if not os.path.exists(train_dir):
        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
        output = os.path.join(dataset_dir, "data.zip")

        gdown.download(url, output, quiet=False)
        shutil.unpack_archive(output, dataset_dir)
        os.remove(output)

        src_dir = os.path.join(dataset_dir, "img_align_celeba")
        img_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".jpg")])
        print(f"Found {len(img_files)} images.")

        train_fnames = img_files[:162770]              # First 162770 for training
        valid_fnames = img_files[162770:182637]        # Next 19867 for validation
        test_fnames = img_files[182637:]               # Last 19962 for testing

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(valid_small_dir, exist_ok=True)

        for fname in tqdm(train_fnames, desc="Moving training images"):
            shutil.move(os.path.join(src_dir, fname), os.path.join(train_dir, fname))

        for fname in tqdm(valid_fnames, desc="Moving validation images"):
            shutil.move(os.path.join(src_dir, fname), os.path.join(valid_dir, fname))

        for fname in tqdm(test_fnames, desc="Moving test images"):
            shutil.move(os.path.join(src_dir, fname), os.path.join(test_dir, fname))

        # copy first 100 images from validation into valid_small
        valid_sorted = sorted([f for f in os.listdir(valid_dir) if f.endswith(".jpg")])[:100]
        for fname in tqdm(valid_sorted, desc="Copying to valid_small"):
            src = os.path.join(valid_dir, fname)
            dst = os.path.join(valid_small_dir, fname)
            # copy2 keeps timestamps/metadata; copyfile would also be fine
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        # Remove the now-empty source directory (and any leftover non-jpg files)
        shutil.rmtree(src_dir, ignore_errors=True)
    return (valid_small_dir,)


@app.cell
def _(Image, bf, np, os):
    def load_fn(filepath: os.PathLike) -> any:
        with Image.open(filepath) as img:
            #img = img.convert("L")  # Convert to grayscale
            img = img.resize((64, 64), Image.Resampling.BILINEAR) # Resize to (64, 64)
            obs = img.convert("L") # Convert to grayscale

            img_array = np.array(img) / 255.
            obs_array = np.array(obs) / 255.
            obs_array = np.expand_dims(obs_array, axis=-1) # Add channel dimension for grayscale
            assert len(img_array.shape) == 3 and len(obs_array.shape) == 3, f"Expected 3D arrays, got {img_array.shape} and {obs_array.shape}"
            return {
                "img": img_array,
                "gray_img": obs_array,
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
    return


@app.cell
def _(gc, keras):
    # stage res: (64, 32, 16, 8)
    kwargs = {
        "UNet": {
            "widths": (64, 128, 256, 512),
            "res_blocks": 2,
            "attn_stage": None,
        },
        "UViT": {
            "widths": (64, 128, 256),
            "res_blocks": 3,
            "transformer_blocks": 3,
            "transformer_dropout": 0.2,
            "transformer_width": 1024,
        },
        "ResidualUViT": {
            "widths": (64, 128, 256),
            "res_blocks_up": 2,
            "res_blocks_down": 3,
            "transformer_blocks": 3,
            "transformer_dropout": 0.2,
            "transformer_width": 1024,
        }
    }

    def print_sublayers(layer, indent=0, visited=None):
        if visited is None:
            visited = set()
        if id(layer) in visited:
            return
        visited.add(id(layer))
        print("  " * indent + f"- {layer.name}: {layer.__class__.__name__}")
        for sub in layer._layers:  # tracked sublayers
            print_sublayers(sub, indent + 1, visited)

    for subnet_name, subnet_kwargs in kwargs.items():
        x = keras.Input(shape=(64, 64, 3))
        theta = keras.Input(shape=(64, 64, 1))
        t = keras.Input(shape=(1,))
        subnet = eval("bf.networks." + subnet_name)
        model_output = subnet(**subnet_kwargs)((theta, t, x))
        model = keras.Model(inputs=(theta, t, x), outputs=model_output)
        model.summary(expand_nested=True, show_trainable=True)

        print_sublayers(model)
        keras.utils.clear_session()
        del model
        gc.collect()
    return


@app.cell
def _():
    # samples_clipped = np.clip(samples, -1, 1)
    # data_inv_adapted = workflow.adapter.inverse(data={
    #     "inference_variables": np.concatenate([validation_data[0]["inference_variables"][0:1], samples_clipped], axis=0),
    #     "inference_conditions": np.concatenate([cond[:1], cond], axis=0)
    # })
    return


@app.cell
def _(plt):
    def plot_samples(data_inv_adapted):
        n_samples = data_inv_adapted["img"].shape[0] - 1
        fig, axs = plt.subplots(1, n_samples+2, figsize=(2*n_samples, 2))
        axs[0].imshow(data_inv_adapted["img"][0])
        axs[0].set_title("Ground Truth")
        axs[0].axis("off")


        axs[1].imshow(data_inv_adapted["gray_img"][0], cmap="gray")
        axs[1].set_title("Condition")
        axs[1].axis("off")
        for i in range(n_samples):
            axs[i+2].imshow(data_inv_adapted["img"][i+1])
            axs[i+2].set_title(f"Sample {i+1}")
            axs[i+2].axis("off")
        plt.tight_layout()
        return fig

    return (plot_samples,)


@app.cell
def _(Image, bf, gc, keras, np, os, plot_samples, valid_small_dir):
    def train_and_plot():
        def load_fn(filepath: os.PathLike) -> any:
            with Image.open(filepath) as img:
                #img = img.convert("L")  # Convert to grayscale
                img = img.resize((64, 64), Image.Resampling.BILINEAR) # Resize to (64, 64)
                obs = img.convert("L") # Convert to grayscale

                img_array = np.array(img) / 255.
                obs_array = np.array(obs) / 255.
                obs_array = np.expand_dims(obs_array, axis=-1) # Add channel dimension for grayscale
                assert len(img_array.shape) == 3 and len(obs_array.shape) == 3, f"Expected 3D arrays, got {img_array.shape} and {obs_array.shape}"
                return {
                    "img": img_array,
                    "gray_img": obs_array,
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

        kwargs = {
            "UViT": {
                "widths": (64, 128, 256),
                "res_blocks": 3,
                "transformer_blocks": 3,
                "transformer_dropout": 0.2,
                "transformer_width": 1024,
            },
            "ResidualUViT": {
                "widths": (64, 128, 256),
                "res_blocks_up": 2,
                "res_blocks_down": 3,
                "transformer_blocks": 3,
                "transformer_dropout": 0.2,
                "transformer_width": 1024,
            },
            "UNet": {
                "widths": (64, 128, 256, 512),
                "res_blocks": 2,
                "attn_stage": None,
            },

        }
        savedir = "checkpoints/celeba_diffusion"
        os.makedirs(savedir, exist_ok=True)

        validation_data = bf.datasets.DiskDataset(
            root=valid_small_dir,
            pattern="*.jpg",
            load_fn=load_fn,
            adapter=adapter,
            shuffle=False,
            batch_size=100,
        )
        cond = validation_data[0]["inference_conditions"][0:1].repeat(10, 0)
        for subnet_name, subnet_kwargs in kwargs.items():
            print(f"Training {subnet_name}...")
            subnet = eval("bf.networks." + subnet_name)
            diffusion = bf.networks.DiffusionModel(
                subnet=subnet,
                subnet_kwargs=subnet_kwargs,
                prediction_type="velocity",
                noise_schedule="cosine",
            )
            workflow = bf.BasicWorkflow(
                simulator=None,
                adapter=adapter,
                inference_network=diffusion,
                summary_network=None,
                inference_variables="img",
                inference_conditions="gray_img",
                initial_learning_rate=2e-4,
                checkpoint_filepath=savedir + f"/{subnet_name}_best", # <- should not be contain file when its a folder
            )
            history = workflow.fit_disk(
                root = "datasets/celeba/train",
                pattern = "*.jpg",
                batch_size = 64,
                load_fn = load_fn,
                epochs = 5,
                keep_optimizer = False,
                validation_data = None, 
                augmentations = None,
            )

            f = bf.diagnostics.plots.loss(history)
            f.savefig(os.path.join(savedir, f"{subnet_name}_loss.png"))

            n_samples = 10
            samples = workflow.approximator.inference_network._inverse(
                z=keras.random.normal(shape=(n_samples, 64, 64, 3)),
                conditions=cond,
                density=False,
                training=False,
            )
            samples = keras.ops.convert_to_numpy(samples)
            samples_clipped = np.clip(samples, -1, 1)
            val0 = keras.ops.convert_to_numpy(validation_data[0]["inference_variables"][0:1])
            cond0 = keras.ops.convert_to_numpy(cond[:1])
            cond_np = keras.ops.convert_to_numpy(cond)

            data_inv_adapted = workflow.adapter.inverse(data={
                "inference_variables": np.concatenate([val0, samples_clipped], axis=0),
                "inference_conditions": np.concatenate([cond0, cond_np], axis=0),
            })
            f = plot_samples(data_inv_adapted)
            f.savefig(os.path.join(savedir, f"{subnet_name}_samples.png"))

            keras.backend.clear_session()
            del subnet, diffusion, workflow, history, samples, samples_clipped, data_inv_adapted
            gc.collect()

    return (train_and_plot,)


@app.cell
def _(train_and_plot):
    train_and_plot()
    return


@app.cell
def _(keras):
    keras.backend.clear_session()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
