import pytest

from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("test", disable_module_check=True)
class Foo:
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        return {}


@serializable("test", disable_module_check=True)
class Bar:
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        return {}


@pytest.mark.parametrize("obj", [1, "1", True, False, None, ...])
def test_primitives(obj):
    assert deserialize(serialize(obj)) == obj


@pytest.mark.parametrize(
    "obj",
    [
        [1, 2, 3, "a", "b", None, True, ...],
        {"a": 1, "b": 2},
    ],
)
def test_collections_of_primitives(obj):
    assert deserialize(serialize(obj)) == obj


@pytest.mark.parametrize(
    "obj",
    [
        int,
        list,
    ],
)
def test_builtin_types(obj):
    assert deserialize(serialize(obj)) is obj


def test_custom_object():
    instance = Foo()

    assert isinstance(deserialize(serialize(instance)), Foo)


def test_custom_type():
    assert deserialize(serialize(Foo)) is Foo


@pytest.mark.parametrize(
    "obj",
    [
        [int, list, tuple, bool],
        {"a": int, "b": list, "c": tuple, "d": bool},
    ],
)
def test_collections_of_builtin_types(obj):
    assert deserialize(serialize(obj)) == obj


@pytest.mark.parametrize(
    "obj",
    [
        [Foo, Bar],
        {"a": Foo, "b": Bar},
    ],
)
def test_collection_of_custom_types(obj):
    assert deserialize(serialize(obj)) == obj
