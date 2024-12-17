from collections.abc import Callable, Sequence

import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transforms import (
    AsSet,
    Broadcast,
    Concatenate,
    Constrain,
    ConvertDType,
    Drop,
    ExpandDims,
    ElementwiseTransform, # why wasn't this added before? 
    FilterTransform,
    Keep,
    LambdaTransform,
    MapTransform,
    OneHot,
    Rename,
    Standardize,
    ToArray,
    Transform,
)

from .transforms.filter_transform import Predicate


@serializable(package="bayesflow.adapters")
class Adapter:
    def __init__(self, transforms: Sequence[Transform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = transforms

    @staticmethod
    def create_default(inference_variables: Sequence[str]) -> "Adapter":
        return (
            Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .concatenate(inference_variables, into="inference_variables")
        )

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Adapter":
        return cls(transforms=deserialize(config["transforms"], custom_objects))

    def get_config(self) -> dict:
        return {"transforms": serialize(self.transforms)}

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        for transform in self.transforms:
            data = transform(data, **kwargs)

        return data

    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, any]:
        data = data.copy()

        for transform in reversed(self.transforms):
            data = transform(data, inverse=True, **kwargs)

        return data

    def __call__(self, data: dict[str, any], *, inverse: bool = False, **kwargs) -> dict[str, np.ndarray]:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    def __repr__(self):
        return f"Adapter([{' -> '.join(map(repr, self.transforms))}])"

    def __getitem__(self, index):

        if isinstance(index, slice): 
            if index.start > index.stop: 
                raise IndexError("Index slice must be positive integers such that a < b for adapter[a:b]")
            if index.stop < len(self.transforms): 
                # print("What is the slice?")
                # print(index)
                # print(type(index))
                # check that the slice is in range 
                sliced_transforms = self.transforms[index]
                # print("Are the sliced transforms a sequence")
                # print(isinstance(sliced_transforms, Sequence))
                # print("What is in the slice?")
                # print(sliced_transforms)
                new_adapter = Adapter(transforms = sliced_transforms)
                return new_adapter
            else: 
                raise IndexError("Index slice out of range")
                        
        elif isinstance(index, int): 
            if index < 0:
                index = index + len(self.transforms) # negative indexing 
            if index < 0 or index >= len(self.transforms): 
                raise IndexError("Adapter index out of range.")
            sliced_transforms = self.transforms[index]
            new_adapter = Adapter(transforms = sliced_transforms)
            return new_adapter
        else:
            raise TypeError("Invalid index type. Must be int or slice.")
        
    
    def __setitem__(self, index, new_value): 

        if not isinstance(new_value, Adapter): 
            raise TypeError("new_value must be an Adapter instance")
        
        
        new_transform = new_value.transforms 
        
        if len(new_transform) == 0: 
            raise ValueError("new_value is an Adapter instance without any specified transforms, new_value Adapter must contain at least one transform.")


        if isinstance(index, slice): 
            if index.start > index.stop: 
                raise IndexError("Index slice must be positive integers such that a < b for adapter[a:b]")
            
            if index.stop < len(self.transforms):
                self.transforms[index] = new_transform
            
            else: 
                raise IndexError("Index slice out of range")
            

        elif isinstance(index, int): 
            if index < 0: # negative indexing 
                index = index + len(self.transforms)
                
            if index < 0 or index >= len(self.transforms): 
                raise IndexError("Index out of range.")
                # could add that if the index is out of range, like index == len 
                # then we just add the transform 
            print("what is self.transforms[index]?")
            print(self.transforms[index])
            print("what is the value of the newvalue")
            print(new_transform)
            print(type(new_transform))
        
            self.transforms[index] = new_transform
        else: 
            raise  TypeError("Invalid index type. Must be int or slice.")
    
    def add_transform(self, transform: Transform):
        self.transforms.append(transform)
        return self

    def apply(
        self,
        *,
        forward: Callable[[np.ndarray, ...], np.ndarray],
        inverse: Callable[[np.ndarray, ...], np.ndarray],
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        transform = FilterTransform(
            transform_constructor=LambdaTransform,
            predicate=predicate,
            include=include,
            exclude=exclude,
            forward=forward,
            inverse=inverse,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    # Begin of transformed derived from transform classes 
    def as_set(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: AsSet() for key in keys})
        self.transforms.append(transform)
        return self

    def broadcast(
        self, keys: str | Sequence[str], *, to: str, expand: str | int | tuple = "left", exclude: int | tuple = -1
    ):
        if isinstance(keys, str):
            keys = [keys]

        transform = Broadcast(keys, to=to, expand=expand, exclude=exclude)
        self.transforms.append(transform)
        return self

    def clear(self):
        self.transforms = []
        return self

    def concatenate(self, keys: Sequence[str], *, into: str, axis: int = -1):
        if isinstance(keys, str):
            # this is a common mistake, and also passes the type checker since str is a sequence of characters
            raise ValueError("Keys must be a sequence of strings. To rename a single key, use the `rename` method.")

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
    ):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform(
            transform_map={key: Constrain(lower=lower, upper=upper, method=method) for key in keys}
        )
        self.transforms.append(transform)
        return self

    def drop(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = Drop(keys)
        self.transforms.append(transform)
        return self

    def expand_dims(self, keys: str | Sequence[str], *, axis: int | tuple):
        if isinstance(keys, str):
            keys = [keys]

        transform = ExpandDims(keys, axis=axis)
        self.transforms.append(transform)
        return self

    def keep(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = Keep(keys)
        self.transforms.append(transform)
        return self

    def one_hot(self, keys: str | Sequence[str], num_classes: int):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: OneHot(num_classes=num_classes) for key in keys})
        self.transforms.append(transform)
        return self

    def rename(self, from_key: str, to_key: str):
        self.transforms.append(Rename(from_key, to_key))
        return self

    def standardize(
        self,
        *,
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        transform = FilterTransform(
            transform_constructor=Standardize,
            predicate=predicate,
            include=include,
            exclude=exclude,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    def to_array(
        self,
        *,
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        transform = FilterTransform(
            transform_constructor=ToArray,
            predicate=predicate,
            include=include,
            exclude=exclude,
            **kwargs,
        )
        self.transforms.append(transform)
        return self
