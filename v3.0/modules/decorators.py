from functools import wraps
from time import time
from typing import Callable


def timing(func=None, num_repeats: int = 1):
    from modules.utilities import timestamp  # must be inside due to circular import
    """
    Define parametrized decorator with arguments
    """
    if not (callable(func) or func is None):
        raise ValueError('The usage of timing decorator is "@timing", "@timing()", "@timing(num_repeats=integer)", '
                         'or "@timing(func=None, num_repeats=integer)"')

    if num_repeats < 1 or not isinstance(num_repeats, int):
        raise ValueError(f'"num_repeats" must be positive integer but is {num_repeats}')

    def _decorator(f: Callable):
        """
        Repeats execution "num_repeats" times and measures elapsed time
        """

        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            for _ in range(num_repeats - 1):
                f(*args, **kw)
            else:
                result = f(*args, **kw)
            te = time()

            elapsed_time = timestamp(te - ts, prec=3)
            if num_repeats == 1:
                print(f'Function "{f.__name__}" took {elapsed_time}.')
            else:
                elapsed_time_per_repetition = timestamp((te - ts) / num_repeats, prec=5)
                print(f'Function "{f.__name__}" took {elapsed_time} after {num_repeats} repetitions '
                      f'({elapsed_time_per_repetition} per repetition).')

            return result

        return wrap

    return _decorator(func) if callable(func) else _decorator


def reduce_like(func: Callable):
    @wraps(func)
    def _decorator(*args, **kw):
        args = list(args)
        arrays = args[0]
        result = arrays[0]

        for array in arrays[1:-1]:
            if args[1:]:
                new_args = [(result, array), *args[1:]]
            else:
                new_args = [(result, array)]
            result = func(*new_args, **kw)
        else:
            if args[1:]:
                new_args = [(result, arrays[-1]), *args[1:]]
            else:
                new_args = [(result, arrays[-1])]

            return func(*new_args, **kw)

    decorated_function = func
    decorated_function.reduce = _decorator

    return decorated_function
