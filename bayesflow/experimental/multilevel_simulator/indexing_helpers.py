import numpy as np


def group_ids(columns: list[np.ndarray]) -> np.ndarray:
    # Speed optimized version of
    # np.unique(np.stack(columns, axis=-1), axis=0, return_inverse=True)[1]
    #
    # Assigns each row of the stacked columns a consecutive index from 0 to k-1,
    # where k is the number of groups

    ids = np.zeros(len(columns[0]), dtype=np.int64)

    for column in columns:
        ids = np.unique(ids * (int(np.max(column, initial=0)) + 1) + column, return_inverse=True)[1]

    return ids


def join(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Speed optimized version of
    # pd.DataFrame(left).merge(pd.DataFrame(right))

    shared = []
    for column in left:
        if column in right:
            shared.append(column)

    columns = []
    for column in shared:
        columns.append(np.concatenate([left[column], right[column]]))

    ids = group_ids(columns)
    left_ids = ids[: len(left[shared[0]])]
    right_ids = ids[len(left[shared[0]]) :]

    # sorting the right rows by id puts the rows of each id into one block
    order = np.argsort(right_ids, kind="stable")
    counts = np.bincount(right_ids, minlength=len(ids))
    starts = np.cumsum(counts) - counts

    # every left row is paired with each right row in the block of its id
    matches = counts[left_ids]
    left_rows = np.repeat(np.arange(len(left_ids)), matches)
    right_rows = order[concatenated_ranges(starts[left_ids], matches)]

    joined = {}
    for column in left:
        joined[column] = left[column][left_rows]

    for column in right:
        if column not in left:
            joined[column] = right[column][right_rows]

    return joined


def concatenated_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    # Speed optimized version of
    # np.concatenate([np.arange(start, start + count) for start, count in zip(starts, counts)])
    #
    # For example, concatenated_ranges([5, 0, 9], [2, 1, 3]) returns [5, 6, 1, 9, 10, 11]
    return np.repeat(starts - (np.cumsum(counts) - counts), counts) + np.arange(np.sum(counts))
