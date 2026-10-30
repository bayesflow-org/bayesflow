import keras

from bayesflow.utils.serialization import serializable


@serializable("bayesflow.experimental")
class MultilevelEncoder(keras.Layer):
    """
    Encodes the rows of a data node, so they can be pooled into conditions.

    Every layer transforms each row, and adds a transformation of the mean
    of every group the row belongs to, for each possible grouping.
    The encoding of a row therefore depends on the other rows of its groups,
    but not on their order. The number of rows in each group is added as a
    feature.

    Parameters
    ----------
    width : int, optional
        Number of features of the encoding of each row (default is 64).
    depth : int, optional
        Number of layers (default is 2).
    activation : str, optional
        Name of the activation function of every layer (default is "mish").
    **kwargs
        Additional keyword arguments passed to :py:class:`keras.Layer`.
    """

    def __init__(self, width: int = 64, depth: int = 2, activation: str = "mish", **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.activation = activation

    def build(self, observations_shape, groups_shape):
        num_groupings = len(groups_shape)
        dim = observations_shape[-1] + num_groupings

        self.kernels = []
        self.group_kernels = []
        self.biases = []

        for i in range(self.depth):
            self.kernels.append(self.add_weight(shape=(dim, self.width), name=f"kernel_{i}"))
            self.group_kernels.append(self.add_weight(shape=(num_groupings, dim, self.width), name=f"group_kernel_{i}"))
            self.biases.append(self.add_weight(shape=(self.width,), initializer="zeros", name=f"bias_{i}"))
            dim = self.width

    def call(self, observations, groups, training=False):
        """
        Parameters
        ----------
        observations : Tensor
            The rows of the data node, with shape `(num_rows, dim)`.
            The columns hold the data values and the values of the latent nodes
            each row depends on.
        groups : list[dict[str, Tensor]]
            One dictionary per grouping of the rows, with the group of every
            row under ``"ids"`` and the number of rows of every group under
            ``"counts"``.
        training : bool, optional
            Whether the layer is called during training (default is False).

        Returns
        -------
        Tensor
            The encoding of every row, with shape `(num_rows, width)`.
        """
        activation = keras.activations.get(self.activation)

        features = [observations]

        for group in groups:
            counts = keras.ops.cast(group["counts"], observations.dtype)
            log_counts = gather(keras.ops.log1p(counts), group["ids"])
            features.append(log_counts[:, None])

        hidden = keras.ops.concatenate(features, axis=-1)

        # every layer adds a transformation of the group means of each grouping to every row
        for kernel, group_kernel, bias in zip(self.kernels, self.group_kernels, self.biases):
            pooled = bias

            for i, group in enumerate(groups):
                num_groups = keras.ops.shape(group["counts"])[0]
                mean = segment_mean(hidden, group["ids"], num_groups)
                transformed = keras.ops.matmul(mean, group_kernel[i])
                pooled = pooled + gather(transformed, group["ids"])

            hidden = activation(keras.ops.matmul(hidden, kernel) + pooled)

        return hidden

    def get_config(self):
        return super().get_config() | {"width": self.width, "depth": self.depth, "activation": self.activation}


def gather(values, rows):
    return keras.ops.take(values, rows, axis=0)


def segment_count(ids, num_segments, dtype):
    ones = keras.ops.ones((keras.ops.shape(ids)[0], 1), dtype=dtype)

    return keras.ops.segment_sum(ones, ids, num_segments=num_segments)


def segment_mean(values, ids, num_segments):
    totals = keras.ops.segment_sum(values, ids, num_segments=num_segments)
    counts = segment_count(ids, num_segments, totals.dtype)

    return totals / keras.ops.maximum(counts, 1.0)


def sum_per_dataset(values, datasets, num_datasets):
    values = keras.ops.reshape(values, (keras.ops.shape(values)[0], -1))
    row_sums = keras.ops.sum(values, axis=-1)

    return keras.ops.segment_sum(row_sums, datasets, num_segments=num_datasets)
