from functools import wraps
from time import time
from typing import Callable


def timing(func=None, num_repeats: int = 1):
    """
    Define parametrized decorator with arguments
    """
    if not (callable(func) or func is None):
        raise ValueError('The usage of timing decorator is "@timing", "@timing()", "@timing(num_repeats=integer)", '
                         'or "@timing(func=None, num_repeats=integer)"')

    if num_repeats < 1 or not isinstance(num_repeats, int):
        raise ValueError('"num_repeats" must be positive integer but is {nrep}'.format(nrep=num_repeats))

    def _decorator(f: Callable):
        """
        Repeats execution 'num_repeats' times and measures elapsed time
        """

        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            for _ in range(num_repeats - 1):
                f(*args, **kw)
            else:
                result = f(*args, **kw)
            te = time()

            if num_repeats == 1:
                print('Function "{func}" took {elapsed_time:.3f} sec.'.format(func=f.__name__, elapsed_time=te - ts))
                # print("func: %r args: [%r, %r] took: %.3f sec".format(f.__name__, args, kw, te - ts))
            else:
                print('Function "{func}" took {elapsed_time:.3f} sec after {nrep} repetitions ({time_per_rep:.5f} '
                      'sec per repetition).'.format(func=f.__name__, nrep=num_repeats, elapsed_time=te - ts,
                                                    time_per_rep=(te - ts) / num_repeats))
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
                new_args = [(result, array), args[1:]]
            else:
                new_args = [(result, array)]
            result = func(*new_args, **kw)
        else:
            if args[1:]:
                new_args = [(result, arrays[-1]), args[1:]]
            else:
                new_args = [(result, arrays[-1])]

            return func(*new_args, **kw)

    _decorator.unwrapped = func

    return _decorator
