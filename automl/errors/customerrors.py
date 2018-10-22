"""Module to define custom errors classes."""


class AutoMLError(Exception):
    """Generic AutoML error.

    Used when no built-in exception seems useful but neither creating a new
    exception looks reasonable.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error

    """

    def __init__(self, msg):
        """Constructor."""
        self.msg = msg
        Exception.__init__(self)


class AutoMLTestError(Exception):
    """Generic AutoML test error.

    Used when no built-in exception seems useful but neither creating a new
    exception looks reasonable.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error

    """

    def __init__(self, test_name, msg):
        """Constructor."""
        self.test_name = test_name
        self.msg = msg
        Exception.__init__(self)


class CurrentlyNonSupportedError(Exception):
    """Error for logic that is still to be implemented."""

    def __init__(self, msg):
        """Constructor."""
        self.msg = msg
        Exception.__init__(self)
