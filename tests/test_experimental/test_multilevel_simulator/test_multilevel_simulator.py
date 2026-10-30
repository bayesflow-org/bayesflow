import networkx as nx
import numpy as np
import pytest

from bayesflow.experimental.multilevel_simulator import MultilevelSimulator
from bayesflow.experimental.multilevel_simulator.multilevel_simulator import from_table
from bayesflow.experimental.multilevel_simulator.samples import NodeSamples, concatenate, select, to_tensors

SIMULATORS = ["single_level_simulator", "two_level_simulator", "three_level_simulator", "crossed_design_irt_simulator"]


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_index_columns(request, simulator):
    simulator = request.getfixturevalue(simulator)
    simulation = simulator.sample(5)

    for node in simulator.graph:
        indices = simulation[node].indices
        ancestors = nx.ancestors(simulator.graph, node)

        assert set(indices) == {"dataset", node} | ancestors
        assert list(indices[node]) == list(range(len(indices["dataset"])))
        assert set(indices["dataset"]) <= set(range(5))


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_roots_have_one_row_per_dataset(request, simulator):
    simulator = request.getfixturevalue(simulator)
    simulation = simulator.sample(5)

    for node in simulator.graph:
        if simulator.graph.in_degree(node) == 0:
            assert list(simulation[node].indices["dataset"]) == list(range(5))


def test_fixed_reps(two_level_simulator):
    simulation = two_level_simulator.sample(5)

    assert len(simulation["hypers"].indices["dataset"]) == 5
    assert len(simulation["locals"].indices["dataset"]) == 5 * 6
    assert len(simulation["y"].indices["dataset"]) == 5 * 6 * 10


def test_meta_override(single_level_simulator):
    simulation = single_level_simulator.sample(5, meta={"N": 7})

    assert len(simulation["likelihood"].indices["dataset"]) == 5 * 7


def test_callable_reps():
    simulator = MultilevelSimulator()
    simulator.add_node("group", sample_fn=lambda: {"mu": np.random.normal()})
    simulator.add_node("obs", sample_fn=lambda mu: {"y": mu}, reps=lambda size: np.arange(size) + 1)
    simulator.add_edge("group", "obs")

    simulation = simulator.sample(4)

    assert list(np.bincount(simulation["obs"].indices["group"])) == [1, 2, 3, 4]


def test_reps_receive_combinations_and_meta():
    received = {}

    def even_items(combinations, num_items):
        received["combinations"] = combinations
        received["num_items"] = num_items

        # only even items are observed, so odd ones get no rows
        return (combinations["item"] % 2 == 0).astype(int)

    simulator = MultilevelSimulator(meta_fn=lambda: {"num_items": 4})
    simulator.add_node("group", sample_fn=lambda: {"mu": np.random.normal()}, reps=3)
    simulator.add_node("item", sample_fn=lambda: {"b": np.random.normal()}, reps="num_items")
    simulator.add_node("obs", sample_fn=lambda mu, b: {"y": mu + b}, reps=even_items)
    simulator.add_edge("group", "obs")
    simulator.add_edge("item", "obs")

    simulation = simulator.sample(2)

    # one combination for every group and item of the same dataset
    assert len(received["combinations"]["dataset"]) == 2 * 3 * 4
    assert np.all(received["num_items"] == 4)

    assert np.all(simulation["obs"].indices["item"] % 2 == 0)
    assert len(simulation["obs"].indices["dataset"]) == 2 * 3 * 2


def test_crossed_samples_join_within_school(crossed_design_irt_simulator):
    simulation = crossed_design_irt_simulator.sample(5)
    observations = simulation["observations"].indices

    # each observation pairs a question and a student of the same school
    question_school = simulation["questions"].indices["schools"][observations["questions"]]
    student_school = simulation["students"].indices["schools"][observations["students"]]

    assert np.array_equal(question_school, observations["schools"])
    assert np.array_equal(student_school, observations["schools"])


def sample_parent():
    return {"x": np.random.normal()}


def sample_parents(batch_size):
    return {"x": np.random.normal(size=batch_size)}


def copy_parent(x):
    return {"x_copy": x}


def copy_model(is_batched):
    # the child copies its parent's value, so we can see which parent each child belongs to
    simulator = MultilevelSimulator()

    if is_batched:
        simulator.add_node("parent", sample_fn=sample_parents, is_batched=True)
    else:
        simulator.add_node("parent", sample_fn=sample_parent)

    simulator.add_node("child", sample_fn=copy_parent, reps=3, is_batched=is_batched)
    simulator.add_edge("parent", "child")

    return simulator


@pytest.mark.parametrize("is_batched", [False, True])
def test_children_receive_their_parents_values(is_batched):
    simulation = copy_model(is_batched).sample(4)

    parent_row = simulation["child"].indices["parent"]
    expected = simulation["parent"].values["x"][parent_row]

    assert np.array_equal(simulation["child"].values["x_copy"], expected)


@pytest.mark.parametrize("is_batched", [False, True])
def test_vector_valued_variables(is_batched):
    simulator = MultilevelSimulator()

    if is_batched:
        simulator.add_node("prior", sample_fn=lambda batch_size: {"beta": np.zeros((batch_size, 3))}, is_batched=True)
    else:
        simulator.add_node("prior", sample_fn=lambda: {"beta": np.zeros(3)})

    assert simulator.sample(4)["prior"].values["beta"].shape == (4, 3)


def assert_same_simulation(simulation, expected):
    for node, samples in expected.items():
        assert list(simulation[node].indices) == list(samples.indices)

        for column, ids in samples.indices.items():
            np.testing.assert_array_equal(simulation[node].indices[column], ids)

        for variable, values in samples.values.items():
            np.testing.assert_array_equal(simulation[node].values[variable], values)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_select_and_concatenate(request, simulator):
    simulation = request.getfixturevalue(simulator).sample(4)

    parts = []
    for dataset in range(4):
        parts.append(select(simulation, np.array([dataset])))

    for part in parts:
        for node, samples in part.items():
            assert np.all(samples.indices["dataset"] == 0)
            np.testing.assert_array_equal(samples.indices[node], np.arange(len(samples.indices["dataset"])))

    assert_same_simulation(concatenate(parts), simulation)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_select_renumbers_in_increasing_order(request, simulator):
    simulation = request.getfixturevalue(simulator).sample(6)

    selected = select(simulation, np.array([4, 1, 3]))

    parts = []
    for dataset in [1, 3, 4]:
        parts.append(select(simulation, np.array([dataset])))

    assert_same_simulation(selected, concatenate(parts))


def test_to_tensors_shapes_and_values(two_level_simulator):
    simulation = two_level_simulator.sample(4)
    tensors = to_tensors(simulation)

    assert tensors["hyper_mean"].shape == (4, 1)
    assert tensors["local_mean"].shape == (4, 6, 1)
    assert tensors["y"].shape == (4, 60, 1)

    # dataset 1 holds rows 6 to 11 of the locals
    np.testing.assert_array_equal(tensors["local_mean"][1], simulation["locals"].values["local_mean"][6:12])


def test_to_tensors_raises_for_different_structure():
    # both datasets have 5 children, but split 3 + 2 and 2 + 3 between the parents
    roots = NodeSamples(
        indices={"dataset": np.array([0, 1]), "roots": np.arange(2)},
        values={"r": np.zeros((2, 1))},
    )
    parents = NodeSamples(
        indices={"dataset": np.array([0, 0, 1, 1]), "roots": np.array([0, 0, 1, 1]), "parents": np.arange(4)},
        values={"a": np.zeros((4, 1))},
    )
    children = NodeSamples(
        indices={
            "dataset": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "roots": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            "parents": np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3]),
            "children": np.arange(10),
        },
        values={"b": np.zeros((10, 1))},
    )
    simulation = {"roots": roots, "parents": parents, "children": children}

    with pytest.raises(ValueError, match="node 'children'"):
        to_tensors(simulation)

    assert to_tensors(simulation, datasets=1)["b"].shape == (1, 5, 1)
    assert to_tensors(simulation, nodes=["parents"])["a"].shape == (2, 2, 1)


def crossed_table(simulation):
    # one row per observation, with labels like a real data set would have
    observations = simulation["observations"].indices
    students = simulation["students"].indices

    # students are numbered from 0 within every school
    first_student = np.searchsorted(students["dataset"], students["dataset"])
    student_numbers = students["students"] - first_student

    return {
        "school": np.array(["A", "B", "C"])[observations["dataset"]],
        "question": np.char.add("q", np.char.zfill(observations["questions"].astype(str), 3)),
        "student": student_numbers[observations["students"]],
        "obs": simulation["observations"].values["obs"][:, 0],
    }


def test_from_table_matches_simulation(crossed_design_irt_simulator):
    simulation = crossed_design_irt_simulator.sample(3)
    table = crossed_table(simulation)

    groups = {"dataset": "school", "questions": "question", "students": "student"}
    observed = from_table(crossed_design_irt_simulator.graph, table, groups)

    assert set(observed) == set(simulation)
    for node, samples in simulation.items():
        assert set(observed[node].indices) == set(samples.indices)
        for column, ids in samples.indices.items():
            np.testing.assert_array_equal(observed[node].indices[column], ids)

    np.testing.assert_array_equal(observed["observations"].values["obs"], simulation["observations"].values["obs"])


def test_from_table_rows_in_any_order(crossed_design_irt_simulator):
    simulation = crossed_design_irt_simulator.sample(3)
    table = crossed_table(simulation)

    order = np.random.permutation(len(table["obs"]))
    shuffled = {}
    for column, values in table.items():
        shuffled[column] = values[order]

    groups = {"dataset": "school", "questions": "question", "students": "student"}
    observed = from_table(crossed_design_irt_simulator.graph, shuffled, groups)

    for node, samples in simulation.items():
        assert len(observed[node].indices[node]) == len(samples.indices[node])
        np.testing.assert_array_equal(np.sort(observed[node].indices["dataset"]), observed[node].indices["dataset"])

    # every (question, student) pair has the same observation, only the row order differs
    expected = simulation["observations"]
    expected_order = np.lexsort((expected.indices["students"], expected.indices["questions"]))

    actual = observed["observations"]
    actual_order = np.lexsort((actual.indices["students"], actual.indices["questions"]))

    np.testing.assert_array_equal(actual.values["obs"][actual_order], expected.values["obs"][expected_order])


def test_from_table_raises_without_label_column(crossed_design_irt_simulator):
    table = crossed_table(crossed_design_irt_simulator.sample(2))

    with pytest.raises(ValueError, match="Node 'students' is repeated"):
        from_table(crossed_design_irt_simulator.graph, table, groups={"dataset": "school", "questions": "question"})


def test_is_batched_default():
    simulator = MultilevelSimulator(is_batched=True)
    simulator.add_node("prior", sample_fn=sample_parents)
    simulator.add_node("unbatched", sample_fn=sample_parent, is_batched=False)

    assert simulator.graph.nodes["prior"]["is_batched"]
    assert not simulator.graph.nodes["unbatched"]["is_batched"]
    assert simulator.sample(4)["prior"].values["x"].shape == (4, 1)


def test_meta_fn_is_drawn_per_dataset():
    simulator = MultilevelSimulator(meta_fn=lambda: {"N": np.random.randint(1, 50)}, is_batched=True)
    simulator.add_node("hypers", sample_fn=lambda batch_size: {"mu": np.random.normal(size=batch_size)})
    simulator.add_node("locals", sample_fn=lambda mu: {"theta": mu}, reps="N")
    simulator.add_edge("hypers", "locals")

    np.random.seed(0)
    simulation = simulator.sample(200)
    counts = np.bincount(simulation["locals"].indices["dataset"], minlength=200)

    assert len(np.unique(counts)) > 10


def test_meta_fn_per_dataset_batched():
    received = {}

    def sample_locals(N, batch_size):
        received["N"] = N
        return {"theta": np.zeros(batch_size)}

    simulator = MultilevelSimulator(meta_fn=lambda: {"N": np.random.randint(1, 5)}, is_batched=True)
    simulator.add_node("hypers", sample_fn=lambda batch_size: {"mu": np.zeros(batch_size)})
    simulator.add_node("locals", sample_fn=sample_locals, reps="N")
    simulator.add_edge("hypers", "locals")

    simulation = simulator.sample(30)
    datasets = simulation["locals"].indices["dataset"]
    counts = np.bincount(datasets, minlength=30)

    assert np.array_equal(received["N"], counts[datasets])


def test_meta_fn_per_dataset_unbatched():
    received = []

    def sample_locals(N):
        received.append(N)
        return {"theta": 0.0}

    simulator = MultilevelSimulator(meta_fn=lambda: {"N": np.random.randint(1, 5)})
    simulator.add_node("hypers", sample_fn=lambda: {"mu": 0.0})
    simulator.add_node("locals", sample_fn=sample_locals, reps="N")
    simulator.add_edge("hypers", "locals")

    simulation = simulator.sample(10)
    datasets = simulation["locals"].indices["dataset"]
    counts = np.bincount(datasets, minlength=10)

    assert np.array_equal(np.array(received), counts[datasets])


def test_meta_override_is_shared():
    simulator = MultilevelSimulator(meta_fn=lambda: {"N": np.random.randint(1, 50)}, is_batched=True)
    simulator.add_node("hypers", sample_fn=lambda batch_size: {"mu": np.zeros(batch_size)})
    simulator.add_node("locals", sample_fn=lambda batch_size: {"theta": np.zeros(batch_size)}, reps="N")
    simulator.add_edge("hypers", "locals")

    simulation = simulator.sample(20, meta={"N": 8})

    assert np.all(np.bincount(simulation["locals"].indices["dataset"]) == 8)
