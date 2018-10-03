
class AutoMLError(Exception):
    """Generic AutoML error, used when no built-in exception seems useful but
    neither creating a new exception looks reasonable.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, msg):
        self.msg = msg


class AutoMLTestError(Exception):
    """Generic AutoML error, used when no built-in exception seems useful but
    neither creating a new exception looks reasonable.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, testName, msg):
        self.testName = testName
        self.msg = msg
