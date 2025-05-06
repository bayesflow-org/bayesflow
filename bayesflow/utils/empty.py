class empty:
    """
    Placeholder value for arguments left empty

    Usage:

    def f(x=empty):
        if x is empty:
            # we know the user did not pass x
        if x is None:
            # the user could have passed None explicitly

    """
