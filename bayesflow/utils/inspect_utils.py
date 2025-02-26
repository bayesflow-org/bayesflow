import inspect


def get_calling_frame_info(*, robust: bool = True):
    if not robust:
        return inspect.getouterframes(inspect.currentframe())[3]

    # could be in a special environment, loop through the whole stack
    stack = inspect.stack()

    idx = None
    for i, frame_info in enumerate(stack):
        if frame_info.filename == __file__ and frame_info.function == "get_calling_frame_info":
            idx = i + 2
            break

    if idx is None or idx >= len(stack):
        raise RuntimeError("Could not find calling frame.")

    return stack[idx]
