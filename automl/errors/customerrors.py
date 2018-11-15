"""Module to define custom errors classes."""


class AutoMLError(Exception):
    """Generic AutoML error.

    Used when no built-in exception seems useful but neither creating a new
    exception seems reasonable.

    Args:
        msg (str): Explanation of the error. Defaults to
            `Unexplained AutoMLError`.

    Attributes:
        msg (str): Explanation of the error.

    """

    def __init__(self, msg=None):
        """Constructor."""
        self.msg = msg
        super().__init__(msg)
    # def __init__(self, msg="Unexplained AutoMLError"):
    #     """Constructor."""
    #     Exception.__init__(self, msg)
    #     # self.msg = msg
    #     # super().__init__(self)


class AutoMLTestError(Exception):
    """Generic AutoML test error.

    Used when no built-in exception seems useful but neither creating a new
    exception looks reasonable.

    Args:
        msg (str): Explanation of the error. Defaults to
            `Unexplained AutoMLTestError`.
        test_name (str): Associated test (the one where the error raised from).
            Defaults to `None`.

    Attributes:
        msg (str): Explanation of the error.
        test_name (str): Associated test (the one where the error raised from).

    """

    def __init__(self, test_name=None, msg="Unexplained AutoMLTestError"):
        """Constructor."""
        self.test_name = test_name
        super().__init__(msg)


class CurrentlyNonSupportedError(Exception):
    """Error for logic that is still to be implemented.

    Args:
        msg (str): Explanation of the error. Defaults to
            `Unexplained CurrentlyNonSupportedError`.

    Attributes:
        msg (str): Explanation of the error.

    """

    def __init__(self, msg="Unexplained CurrentlyNonSupportedError"):
        """Constructor."""
        super().__init__(msg)
        # super().__init__(self.msg)
