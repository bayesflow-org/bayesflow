def check_lengths_same(*args):
    if len(set(map(len, args))) > 1:
        raise ValueError(f"All tuple arguments must have the same length, but lengths are {tuple(map(len, args))}.")
