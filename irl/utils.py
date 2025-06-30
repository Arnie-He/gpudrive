from typing import Iterable, Iterator, Tuple, TypeVar
import itertools

T = TypeVar('T')

def endless_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Generator that endlessly yields elements from `iterable`.

    >>> x = range(2)
    >>> it = endless_iter(x)
    >>> next(it)
    0
    >>> next(it)
    1
    >>> next(it)
    0

    Args:
        iterable: The non-iterator iterable object to endlessly iterate over.

    Returns:
        An iterator that repeats the elements in `iterable` forever.

    Raises:
        ValueError: if iterable is an iterator -- that will be exhausted, so
            cannot be iterated over endlessly.
    """
    if iter(iterable) == iterable:
        raise ValueError("endless_iter needs a non-iterator Iterable.")

    _, iterable = get_first_iter_element(iterable)
    return itertools.chain.from_iterable(itertools.repeat(iterable))

def get_first_iter_element(iterable: Iterable[T]) -> Tuple[T, Iterable[T]]:
    """Get first element of an iterable and a new fresh iterable.

    The fresh iterable has the first element added back using ``itertools.chain``.
    If the iterable is not an iterator, this is equivalent to
    ``(next(iter(iterable)), iterable)``.

    Args:
        iterable: The iterable to get the first element of.

    Returns:
        A tuple containing the first element of the iterable, and a fresh iterable
        with all the elements.

    Raises:
        ValueError: `iterable` is empty -- the first call to it returns no elements.
    """
    iterator = iter(iterable)
    try:
        first_element = next(iterator)
    except StopIteration:
        raise ValueError(f"iterable {iterable} had no elements to iterate over.")

    return_iterable: Iterable[T]
    if iterator == iterable:
        # `iterable` was an iterator. Getting `first_element` will have removed it
        # from `iterator`, so we need to add a fresh iterable with `first_element`
        # added back in.
        return_iterable = itertools.chain([first_element], iterator)
    else:
        # `iterable` was not an iterator; we can just return `iterable`.
        # `iter(iterable)` will give a fresh iterator containing the first element.
        # It's preferable to return `iterable` without modification so that users
        # can generate new iterators from it as needed.
        return_iterable = iterable

    return first_element, return_iterable