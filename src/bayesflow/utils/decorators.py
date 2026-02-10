from collections.abc import Callable, Sequence
from functools import wraps
import inspect
from typing import overload, TypeVar
from bayesflow.types import Shape

Fn = TypeVar("Fn", bound=Callable[..., any])

# this can be done better, but not compactly in Python < 3.12
Decorator = Fn


def allow_args(fn: Decorator) -> Decorator:
    """Decorator to allow another decorator to be called with or without arguments."""

    @overload
    def wrapper(f: Fn) -> Fn: ...
    @overload
    def wrapper(*fargs: any, **fkwargs: any) -> Fn: ...
    @wraps(fn)
    def wrapper(*fargs: any, **fkwargs: any) -> Fn:
        if len(fargs) == 1 and not fkwargs and callable(fargs[0]):
            # called without arguments
            return fn(fargs[0])
        else:
            # called with arguments, bind
            return lambda f: fn(f, *fargs, **fkwargs)

    return wrapper


def alias(*aliases: str) -> Decorator:
    """Decorator to create aliases for keyword arguments"""
    aliases = list(set(aliases))

    def alias_wrapper(fn: Fn) -> Fn:
        nonlocal aliases

        signature = inspect.signature(fn)
        parameter_names = list(signature.parameters.keys())
        candidates = [name for name in aliases if name in parameter_names]

        if not candidates:
            raise ValueError("Found no valid argument candidates in the alias list.")
        if len(candidates) > 1:
            raise ValueError(f"Found multiple valid argument candidates in the alias list: {candidates!r}")

        argname = candidates[0]
        argpos = parameter_names.index(argname)

        aliases.remove(argname)

        del signature
        del parameter_names
        del candidates

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # check if multiple aliases are specified
            matches = [name for name in kwargs if name in aliases]

            if not matches:
                return fn(*args, **kwargs)

            if len(matches) > 1 or (len(matches) > 0 and len(args) > argpos):
                raise TypeError(
                    f"{fn.__name__}() got multiple values for argument {argname!r}.\n"
                    f"This argument is also aliased as {aliases!r}"
                )

            # map aliases to base name
            kwargs[argname] = kwargs.pop(matches[0])

            return fn(*args, **kwargs)

        return wrapper

    return alias_wrapper


def argument_callback(argname: str, callback: Callable[[any], any]) -> Decorator:
    """Decorator to apply a callback to an argument before passing it to the function"""

    def callback_wrapper(fn: Fn) -> Fn:
        argpos = list(inspect.signature(fn).parameters.keys()).index(argname)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if argname in kwargs:
                kwargs[argname] = callback(kwargs[argname])
            elif len(args) > argpos:
                args = list(args)
                args[argpos] = callback(args[argpos])

            return fn(*args, **kwargs)

        return wrapper

    return callback_wrapper


def allow_batch_size(fn: Callable):
    """Decorator to allow an integer batch_size argument in addition to a tuple batch_shape argument"""

    def callback(x):
        if isinstance(x, Sequence):
            return x

        return (x,)

    fn = argument_callback("batch_shape", callback)(fn)
    fn = alias("batch_shape", "batch_size")(fn)

    return fn


def sanitize_input_shape(fn: Callable):
    """Decorator to replace the first dimension in ..._shape arguments with a dummy batch size if it is None"""

    # The Keras functional API passes input_shape = (None, second_dim, third_dim, ...), which
    # causes problems when constructions like self.call(keras.ops.zeros(input_shape)) are used
    # in build. To alleviate those problems, this decorator replaces None with an arbitrary batch size.
    def callback(input_shape: Shape) -> Shape:
        if input_shape[0] is None:
            input_shape = list(input_shape)
            input_shape[0] = 1
            return tuple(input_shape)
        return input_shape

    args = inspect.getfullargspec(fn).args
    for arg in args:
        if arg.endswith("_shape"):
            fn = argument_callback(arg, callback)(fn)
    return fn
