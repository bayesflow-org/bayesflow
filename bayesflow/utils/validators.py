from bayesflow.types import Shape, ShapeLike


def validate_shape(shape: ShapeLike) -> Shape:
    if isinstance(shape, int):
        return (shape,)

    if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"Invalid shape: {shape}")

    return shape
