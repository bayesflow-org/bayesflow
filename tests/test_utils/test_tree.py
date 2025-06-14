def test_map_dict():
    from bayesflow.utils.tree import map_dict

    input = {
        "a": {
            "x": [0, 1, 2],
        },
        "b": [0, 1],
        "c": "foo",
    }
    output = map_dict(len, input)
    for key, value in output.items():
        if key == "a":
            assert value["x"] == len(input["a"]["x"])
            continue
        assert value == len(input[key])
