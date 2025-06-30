from collections.abc import Callable, MutableSequence, Sequence, Mapping

import numpy as np

from bayesflow.utils.serialization import deserialize, serialize, serializable

from .transforms import (
    AsSet,
    AsTimeSeries,
    Broadcast,
    Concatenate,
    Constrain,
    ConvertDType,
    Drop,
    ExpandDims,
    FilterTransform,
    Group,
    Keep,
    Log,
    MapTransform,
    NNPE,
    NumpyTransform,
    OneHot,
    Rename,
    SerializableCustomTransform,
    Squeeze,
    Sqrt,
    Standardize,
    ToArray,
    Transform,
    Ungroup,
    RandomSubsample,
    Take,
    NanToNum,
)
from .transforms.filter_transform import Predicate


@serializable("bayesflow.adapters")
class Adapter(MutableSequence[Transform]):
    """
    Defines an adapter to apply various transforms to data.

    Where possible, the transforms also supply an inverse transform.

    Parameters
    ----------
    transforms : Sequence[Transform], optional
        The sequence of transforms to execute.
    """

    def __init__(self, transforms: Sequence[Transform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = list(transforms)

    @staticmethod
    def create_default(inference_variables: Sequence[str]) -> "Adapter":
        """Create an adapter with a set of default transforms.

        Parameters
        ----------
        inference_variables : Sequence of str
            The names of the variables to be inferred by an estimator.

        Returns
        -------
        An initialized Adapter with a set of default transforms.
        """
        return (
            Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .concatenate(inference_variables, into="inference_variables")
        )

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Adapter":
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self) -> dict:
        config = {
            "transforms": self.transforms,
        }

        return serialize(config)

    def forward(
        self, data: dict[str, any], *, stage: str = "inference", log_det_jac: bool = False, **kwargs
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Apply the transforms in the forward direction.

        Parameters
        ----------
        data : dict
            The data to be transformed.
        stage : str, one of ["training", "validation", "inference"]
            The stage the function is called in.
        log_det_jac: bool, optional
            Whether to return the log determinant of the Jacobian of the transforms.
        **kwargs : dict
            Additional keyword arguments passed to each transform.

        Returns
        -------
        dict | tuple[dict, dict]
            The transformed data or tuple of transformed data and log determinant of the Jacobian.
        """
        data = data.copy()
        if not log_det_jac:
            for transform in self.transforms:
                data = transform(data, stage=stage, **kwargs)
            return data

        log_det_jac = {}
        for transform in self.transforms:
            transformed_data = transform(data, stage=stage, **kwargs)
            log_det_jac = transform.log_det_jac(data, log_det_jac, **kwargs)
            data = transformed_data

        return data, log_det_jac

    def inverse(
        self, data: dict[str, np.ndarray], *, stage: str = "inference", log_det_jac: bool = False, **kwargs
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Apply the transforms in the inverse direction.

        Parameters
        ----------
        data : dict
            The data to be transformed.
        stage : str, one of ["training", "validation", "inference"]
            The stage the function is called in.
        log_det_jac: bool, optional
            Whether to return the log determinant of the Jacobian of the transforms.
        **kwargs : dict
            Additional keyword arguments passed to each transform.

        Returns
        -------
        dict | tuple[dict, dict]
            The transformed data or tuple of transformed data and log determinant of the Jacobian.
        """
        data = data.copy()
        if not log_det_jac:
            for transform in reversed(self.transforms):
                data = transform(data, stage=stage, inverse=True, **kwargs)
            return data

        log_det_jac = {}
        for transform in reversed(self.transforms):
            data = transform(data, stage=stage, inverse=True, **kwargs)
            log_det_jac = transform.log_det_jac(data, log_det_jac, inverse=True, **kwargs)

        return data, log_det_jac

    def __call__(
        self, data: Mapping[str, any], *, inverse: bool = False, stage="inference", **kwargs
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Apply the transforms in the given direction.

        Parameters
        ----------
        data : Mapping[str, any]
            The data to be transformed.
        inverse : bool, optional
            If False, apply the forward transform, else apply the inverse transform (default False).
        stage : str, one of ["training", "validation", "inference"]
            The stage the function is called in.
        **kwargs
            Additional keyword arguments passed to each transform.

        Returns
        -------
        dict | tuple[dict, dict]
            The transformed data or tuple of transformed data and log determinant of the Jacobian.
        """
        if inverse:
            return self.inverse(data, stage=stage, **kwargs)

        return self.forward(data, stage=stage, **kwargs)

    def __repr__(self):
        result = ""
        for i, transform in enumerate(self):
            result += f"{i}: {transform!r}"
            if i != len(self) - 1:
                result += " -> "

        return f"Adapter([{result}])"

    # list methods

    def append(self, value: Transform) -> "Adapter":
        """Append a transform to the list of transforms.

        Parameters
        ----------
        value : Transform
            The transform to be added.
        """
        self.transforms.append(value)
        return self

    def __delitem__(self, key: int | slice):
        del self.transforms[key]

    def extend(self, values: Sequence[Transform]) -> "Adapter":
        """Extend the adapter with a sequence of transforms.

        Parameters
        ----------
        values : Sequence of Transform
            The additional transforms to extend the adapter.
        """
        if isinstance(values, Adapter):
            values = values.transforms

        self.transforms.extend(values)

        return self

    def __getitem__(self, item: int | slice) -> "Adapter":
        if isinstance(item, int):
            return self.transforms[item]

        return Adapter(self.transforms[item])

    def insert(self, index: int, value: Transform | Sequence[Transform]) -> "Adapter":
        """Insert a transform at a given index.

        Parameters
        ----------
        index : int
            The index to insert at.
        value : Transform or Sequence of Transform
            The transform or transforms to insert.
        """
        if isinstance(value, Adapter):
            value = value.transforms

        if isinstance(value, Sequence):
            # convenience: Adapters are always flat
            self.transforms = self.transforms[:index] + list(value) + self.transforms[index:]
        else:
            self.transforms.insert(index, value)

        return self

    def __setitem__(self, key: int | slice, value: Transform | Sequence[Transform]) -> "Adapter":
        if isinstance(value, Adapter):
            value = value.transforms

        if isinstance(key, int) and isinstance(value, Sequence):
            if key < 0:
                key += len(self.transforms)

            key = slice(key, key + 1)

        self.transforms[key] = value

        return self

    def __len__(self):
        return len(self.transforms)

    # adapter methods

    add_transform = append

    def apply(
        self,
        include: str | Sequence[str] = None,
        *,
        forward: np.ufunc | str,
        inverse: np.ufunc | str = None,
        predicate: Predicate = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        """Append a :py:class:`~transforms.NumpyTransform` to the adapter.

        Parameters
        ----------
        forward : str or np.ufunc
            The name of the NumPy function to use for the forward transformation.
        inverse : str or np.ufunc, optional
            The name of the NumPy function to use for the inverse transformation.
            By default, the inverse is inferred from the forward argument for supported methods.
            You can find the supported methods in
            :py:const:`~bayesflow.adapters.transforms.NumpyTransform.INVERSE_METHODS`.
        predicate : Predicate, optional
            Function that indicates which variables should be transformed.
        include : str or Sequence of str, optional
            Names of variables to include in the transform.
        exclude : str or Sequence of str, optional
            Names of variables to exclude from the transform.
        **kwargs : dict
            Additional keyword arguments passed to the transform.
        """
        transform = FilterTransform(
            transform_constructor=NumpyTransform,
            predicate=predicate,
            include=include,
            exclude=exclude,
            forward=forward,
            inverse=inverse,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    def apply_serializable(
        self,
        include: str | Sequence[str] = None,
        *,
        forward: Callable[[np.ndarray, ...], np.ndarray],
        inverse: Callable[[np.ndarray, ...], np.ndarray],
        predicate: Predicate = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        """Append a :py:class:`~transforms.SerializableCustomTransform` to the adapter.

        Parameters
        ----------
        forward : function, no lambda
            Registered serializable function to transform the data in the forward pass.
            For the adapter to be serializable, this function has to be serializable
            as well (see Notes). Therefore, only proper functions and no lambda
            functions can be used here.
        inverse : function, no lambda
            Registered serializable function to transform the data in the inverse pass.
            For the adapter to be serializable, this function has to be serializable
            as well (see Notes). Therefore, only proper functions and no lambda
            functions can be used here.
        predicate : Predicate, optional
            Function that indicates which variables should be transformed.
        include : str or Sequence of str, optional
            Names of variables to include in the transform.
        exclude : str or Sequence of str, optional
            Names of variables to exclude from the transform.
        **kwargs : dict
            Additional keyword arguments passed to the transform.

        Raises
        ------
        ValueError
            When the provided functions are not registered serializable functions.

        Notes
        -----
        Important: The forward and inverse functions have to be registered with Keras.
        To do so, use the `@keras.saving.register_keras_serializable` decorator.
        They must also be registered (and identical) when loading the adapter
        at a later point in time.

        Examples
        --------

        The example below shows how to use the
        `keras.saving.register_keras_serializable` decorator to
        register functions with Keras. Note that for this simple
        example, one usually would use the simpler :py:meth:`apply`
        method.

        >>> import keras
        >>>
        >>> @keras.saving.register_keras_serializable("custom")
        >>> def forward_fn(x):
        >>>     return x**2
        >>>
        >>> @keras.saving.register_keras_serializable("custom")
        >>> def inverse_fn(x):
        >>>     return x**0.5
        >>>
        >>> adapter = bf.Adapter().apply_serializable(
        >>>     "x",
        >>>     forward=forward_fn,
        >>>     inverse=inverse_fn,
        >>> )
        """
        transform = FilterTransform(
            transform_constructor=SerializableCustomTransform,
            predicate=predicate,
            include=include,
            exclude=exclude,
            forward=forward,
            inverse=inverse,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    def as_set(self, keys: str | Sequence[str]):
        """Append an :py:class:`~transforms.AsSet` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to apply the transform to.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: AsSet() for key in keys})
        self.transforms.append(transform)
        return self

    def as_time_series(self, keys: str | Sequence[str]):
        """Append an :py:class:`~transforms.AsTimeSeries` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to apply the transform to.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: AsTimeSeries() for key in keys})
        self.transforms.append(transform)
        return self

    def broadcast(
        self,
        keys: str | Sequence[str],
        *,
        to: str,
        expand: str | int | tuple = "left",
        exclude: int | tuple = -1,
        squeeze: int | tuple = None,
    ):
        """Append a :py:class:`~transforms.Broadcast` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to apply the transform to.
        to : str
            Name of the data variable to broadcast to.
        expand : str or int or tuple, optional
            Where should new dimensions be added to match the number of dimensions in `to`?
            Can be "left", "right", or an integer or tuple containing the indices of the new dimensions.
            The latter is needed if we want to include a dimension in the middle, which will be required
            for more advanced cases. By default we expand left.
        exclude : int or tuple, optional
            Which dimensions (of the dimensions after expansion) should retain their size,
            rather than being broadcasted to the corresponding dimension size of `to`?
            By default we exclude the last dimension (usually the data dimension) from broadcasting the size.
        squeeze : int or tuple, optional
            Axis to squeeze after broadcasting.

        Notes
        -----
        Important: Do not broadcast to variables that are used as inference variables
        (i.e., parameters to be inferred by the networks). The adapter will work during training
        but then fail during inference because the variable being broadcasted to is not available.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = Broadcast(keys, to=to, expand=expand, exclude=exclude, squeeze=squeeze)
        self.transforms.append(transform)
        return self

    def clear(self):
        """Remove all transforms from the adapter."""
        self.transforms = []
        return self

    def concatenate(self, keys: str | Sequence[str], *, into: str, axis: int = -1):
        """Append a :py:class:`~transforms.Concatenate` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to concatenate.
        into : str
            The name of the resulting variable.
        axis : int, optional
            Along which axis to concatenate the keys. The last axis is used by default.
        """
        if isinstance(keys, str):
            transform = Rename(keys, to_key=into)
        else:
            transform = Concatenate(keys, into=into, axis=axis)
        self.transforms.append(transform)
        return self

    def convert_dtype(
        self,
        from_dtype: str,
        to_dtype: str,
        *,
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
    ):
        """Append a :py:class:`~transforms.ConvertDType` transform to the adapter.
        See also :py:meth:`~bayesflow.adapters.Adapter.map_dtype`.

        Parameters
        ----------
        from_dtype : str
            Original dtype
        to_dtype : str
            Target dtype
        predicate : Predicate, optional
            Function that indicates which variables should be transformed.
        include : str or Sequence of str, optional
            Names of variables to include in the transform.
        exclude : str or Sequence of str, optional
            Names of variables to exclude from the transform.
        """
        transform = FilterTransform(
            transform_constructor=ConvertDType,
            predicate=predicate,
            include=include,
            exclude=exclude,
            from_dtype=from_dtype,
            to_dtype=to_dtype,
        )
        self.transforms.append(transform)
        return self

    def constrain(
        self,
        keys: str | Sequence[str],
        *,
        lower: int | float | np.ndarray = None,
        upper: int | float | np.ndarray = None,
        method: str = "default",
        inclusive: str = "both",
        epsilon: float = 1e-15,
    ):
        """Append a :py:class:`~transforms.Constrain` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to constrain.
        lower: int or float or np.darray, optional
            Lower bound for named data variable.
        upper : int or float or np.darray, optional
            Upper bound for named data variable.
        method : str, optional
            Method by which to shrink the network predictions space to specified bounds. Choose from
            - Double bounded methods: sigmoid, expit, (default = sigmoid)
            - Lower bound only methods: softplus, exp, (default = softplus)
            - Upper bound only methods: softplus, exp, (default = softplus)
        inclusive : {'both', 'lower', 'upper', 'none'}, optional
            Indicates which bounds are inclusive (or exclusive).
            - "both" (default): Both lower and upper bounds are inclusive.
            - "lower": Lower bound is inclusive, upper bound is exclusive.
            - "upper": Lower bound is exclusive, upper bound is inclusive.
            - "none": Both lower and upper bounds are exclusive.
        epsilon : float, optional
            Small value to ensure inclusive bounds are not violated.
            Current default is 1e-15 as this ensures finite outcomes
            with the default transformations applied to data exactly at the boundaries.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform(
            transform_map={
                key: Constrain(lower=lower, upper=upper, method=method, inclusive=inclusive, epsilon=epsilon)
                for key in keys
            }
        )
        self.transforms.append(transform)
        return self

    def drop(self, keys: str | Sequence[str]):
        """Append a :py:class:`~transforms.Drop` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to drop.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = Drop(keys)
        self.transforms.append(transform)
        return self

    def expand_dims(self, keys: str | Sequence[str], *, axis: int | tuple):
        """Append an :py:class:`~transforms.ExpandDims` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to expand.
        axis : int or tuple
            The axis to expand.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: ExpandDims(axis=axis) for key in keys})
        self.transforms.append(transform)
        return self

    def group(self, keys: Sequence[str], into: str, *, prefix: str = ""):
        """Append a :py:class:`~transforms.Group` transform to the adapter.

        Groups the given variables as a dictionary in the key `into`. As most transforms do
        not support nested structures, this should usually be the last transform in the adapter.

        Parameters
        ----------
        keys : Sequence of str
            The names of the variables to group together.
        into : str
            The name of the variable to store the grouped variables in.
        prefix : str, optional
            An optional common prefix of the variable names before grouping, which will be removed after grouping.

        Raises
        ------
        ValueError
            If a prefix is specified, but a provided key does not start with the prefix.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = Group(keys=keys, into=into, prefix=prefix)
        self.transforms.append(transform)
        return self

    def ungroup(self, key: str, *, prefix: str = ""):
        """Append an :py:class:`~transforms.Ungroup` transform to the adapter.

        Ungroups the the variables in `key` from a dictionary into individual entries. Most transforms do
        not support nested structures, so this can be used to flatten a nested structure.
        The nesting can be re-established after the transforms using the :py:meth:`group` method.

        Parameters
        ----------
        key : str
            The name of the variable to ungroup. The corresponding variable has to be a dictionary.
        prefix : str, optional
            An optional common prefix that will be added to the ungrouped variable names. This can be necessary
            to avoid duplicate names.
        """
        transform = Ungroup(key=key, prefix=prefix)
        self.transforms.append(transform)
        return self

    def keep(self, keys: str | Sequence[str]):
        """Append a :py:class:`~transforms.Keep` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to keep.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = Keep(keys)
        self.transforms.append(transform)
        return self

    def log(self, keys: str | Sequence[str], *, p1: bool = False):
        """Append an :py:class:`~transforms.Log` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to transform.
        p1 : boolean
            Add 1 to the input before taking the logarithm?
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: Log(p1=p1) for key in keys})
        self.transforms.append(transform)
        return self

    def map_dtype(self, keys: str | Sequence[str], to_dtype: str):
        """Append a :py:class:`~transforms.ConvertDType` transform to the adapter.
        See also :py:meth:`~bayesflow.adapters.Adapter.convert_dtype`.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to transform.
        to_dtype : str
            Target dtype
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: ConvertDType(to_dtype) for key in keys})
        self.transforms.append(transform)
        return self

    def nnpe(
        self,
        keys: str | Sequence[str],
        *,
        spike_scale: float | None = None,
        slab_scale: float | None = None,
        per_dimension: bool = True,
        seed: int | None = None,
    ):
        """Append an :py:class:`~transforms.NNPE` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to transform.
        spike_scale : float or np.ndarray or None, default=None
            The scale of the spike (Normal) distribution. Automatically determined if None.
        slab_scale : float or np.ndarray or None, default=None
            The scale of the slab (Cauchy) distribution. Automatically determined if None.
        per_dimension : bool, default=True
            If true, noise is applied per dimension of the last axis of the input data.
            If false, noise is applied globally.
        seed : int or None
            The seed for the random number generator. If None, a random seed is used.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform(
            {
                key: NNPE(spike_scale=spike_scale, slab_scale=slab_scale, per_dimension=per_dimension, seed=seed)
                for key in keys
            }
        )
        self.transforms.append(transform)
        return self

    def one_hot(self, keys: str | Sequence[str], num_classes: int):
        """Append a :py:class:`~transforms.OneHot` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to transform.
        num_classes : int
            Number of classes for the encoding.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: OneHot(num_classes=num_classes) for key in keys})
        self.transforms.append(transform)
        return self

    def random_subsample(self, key: str, *, sample_size: int | float, axis: int = -1):
        """
        Append a :py:class:`~transforms.RandomSubsample` transform to the adapter.

        Parameters
        ----------
        key : str or Sequence of str
            The name of the variable to subsample.
        sample_size : int or float
            The number of samples to draw, or a fraction between 0 and 1 of the total number of samples to draw.
        axis: int, optional
            Which axis to draw samples over. The last axis is used by default.
        """

        if not isinstance(key, str):
            raise TypeError("Can only subsample one batch entry at a time.")

        transform = MapTransform({key: RandomSubsample(sample_size=sample_size, axis=axis)})

        self.transforms.append(transform)
        return self

    def rename(self, from_key: str, to_key: str):
        """Append a :py:class:`~transforms.Rename` transform to the adapter.

        Parameters
        ----------
        from_key : str
            Variable name that should be renamed
        to_key : str
            New variable name
        """
        self.transforms.append(Rename(from_key, to_key))
        return self

    def scale(self, keys: str | Sequence[str], by: float | np.ndarray):
        from .transforms import Scale

        if isinstance(keys, str):
            keys = [keys]

        self.transforms.append(MapTransform({key: Scale(scale=by) for key in keys}))
        return self

    def shift(self, keys: str | Sequence[str], by: float | np.ndarray):
        from .transforms import Shift

        if isinstance(keys, str):
            keys = [keys]

        self.transforms.append(MapTransform({key: Shift(shift=by) for key in keys}))
        return self

    def split(self, key: str, *, into: Sequence[str], indices_or_sections: int | Sequence[int] = None, axis: int = -1):
        from .transforms import Split

        if isinstance(into, str):
            transform = Rename(key, into)
        else:
            transform = Split(key, into, indices_or_sections, axis)

        self.transforms.append(transform)

        return self

    def squeeze(self, keys: str | Sequence[str], *, axis: int | Sequence[int]):
        """Append a :py:class:`~transforms.Squeeze` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to squeeze.
        axis : int or tuple
            The axis to squeeze. As the number of batch dimensions might change, we advise using negative
            numbers (i.e., indexing from the end instead of the start).
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: Squeeze(axis=axis) for key in keys})
        self.transforms.append(transform)
        return self

    def sqrt(self, keys: str | Sequence[str]):
        """Append an :py:class:`~transforms.Sqrt` transform to the adapter.

        Parameters
        ----------
        keys : str or Sequence of str
            The names of the variables to transform.
        """
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: Sqrt() for key in keys})
        self.transforms.append(transform)
        return self

    def standardize(
        self,
        include: str | Sequence[str] = None,
        *,
        predicate: Predicate = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        """Append a :py:class:`~transforms.Standardize` transform to the adapter.

        Parameters
        ----------
        predicate : Predicate, optional
            Function that indicates which variables should be transformed.
        include : str or Sequence of str, optional
            Names of variables to include in the transform.
        exclude : str or Sequence of str, optional
            Names of variables to exclude from the transform.
        **kwargs :
            Additional keyword arguments passed to the transform.
        """
        transform = FilterTransform(
            transform_constructor=Standardize,
            predicate=predicate,
            include=include,
            exclude=exclude,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    def take(
        self,
        include: str | Sequence[str] = None,
        *,
        indices: Sequence[int],
        axis: int = -1,
        predicate: Predicate = None,
        exclude: str | Sequence[str] = None,
    ):
        """
        Append a :py:class:`~transforms.Take` transform to the adapter.

        Parameters
        ----------
        include : str or Sequence of str, optional
            Names of variables to include in the transform.
        indices : Sequence of int
            Which indices to take from the data.
        axis : int, optional
            Which axis to take from. The last axis is used by default.
        predicate : Predicate, optional
            Function that indicates which variables should be transformed.
        exclude : str or Sequence of str, optional
            Names of variables to exclude from the transform.
        """
        transform = FilterTransform(
            transform_constructor=Take,
            predicate=predicate,
            include=include,
            exclude=exclude,
            indices=indices,
            axis=axis,
        )
        self.transforms.append(transform)
        return self

    def to_array(
        self,
        include: str | Sequence[str] = None,
        *,
        predicate: Predicate = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        """Append a :py:class:`~transforms.ToArray` transform to the adapter.

        Parameters
        ----------
        predicate : Predicate, optional
            Function that indicates which variables should be transformed.
        include : str or Sequence of str, optional
            Names of variables to include in the transform.
        exclude : str or Sequence of str, optional
            Names of variables to exclude from the transform.
        **kwargs : dict
            Additional keyword arguments passed to the transform.
        """
        transform = FilterTransform(
            transform_constructor=ToArray,
            predicate=predicate,
            include=include,
            exclude=exclude,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    def to_dict(self):
        from .transforms import ToDict

        transform = ToDict()
        self.transforms.append(transform)
        return self

    def nan_to_num(
        self,
        keys: str | Sequence[str],
        default_value: float = 0.0,
        return_mask: bool = False,
        mask_prefix: str = "mask",
    ):
        """
        Append :py:class:`~bf.adapters.transforms.NanToNum` transform to the adapter.

        Parameters
        ----------
        keys : str or sequence of str
            The names of the variables to clean / mask.
        default_value : float
            Value to substitute wherever data is NaN. Defaults to 0.0.
        return_mask : bool
            If True, encode a binary missingness mask alongside the data. Defaults to False.
        mask_prefix : str
            Prefix for the mask key in the output dictionary. Defaults to 'mask_'. If the mask key already exists,
            a ValueError is raised to avoid overwriting existing masks.
        """
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            self.transforms.append(
                NanToNum(key=key, default_value=default_value, return_mask=return_mask, mask_prefix=mask_prefix)
            )
        return self
