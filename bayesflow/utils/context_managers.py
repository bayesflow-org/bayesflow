import inspect


class monkey_patch:
    """
    Experimental, single-thread, stack-based monkey patching. Use with caution.

    Built for compatibility with recursive functions. Allows extending the functionality and base cases
    of recursive functions without modifying the original source. For example, adding type serialization
    to `keras.saving.deserialize_keras_object`, which acts recursively on collections.

    Usage:

    def replacement_fn():
        with monkey_patch(original_fn, replacement_fn) as original:
            # all calls to `original_fn` are replaced with calls to `replacement_fn`
            # the only exception is calls specifically to `original` in this scope
            return original()

        # original_fn is restored as soon as the scope exits
    """

    stack = []

    def __init__(self, original, replacement):
        if inspect.getmodule(original) == inspect.getmodule(replacement):
            raise ValueError("Cannot monkey patch the same module.")

        self.original = original
        self.replacement = replacement

        self.original_module = inspect.getmodule(original)
        self.original_module_dict = self.original_module.__dict__
        self.original_name = original.__name__

        self.is_redundant = None

    def __enter__(self):
        if self.stack:
            last_original, last_replacement = self.stack[-1]
            if self.original is last_original and self.replacement is last_replacement:
                self.is_redundant = True
                return self.original

        self.is_redundant = False
        self.stack.append((self.original, self.replacement))

        self.original_module_dict[self.original_name] = self.replacement
        return self.original

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_redundant:
            return

        self.original_module_dict[self.original_name] = self.original
        self.stack.pop()
