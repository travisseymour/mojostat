"""
This module, MojoStat.StatUtil contains various useful helper functions and tools.
"""

import pandas as pd
from patsy import dmatrices  # pycharm says it doesn't exist, but works anyway?!
import itertools
from collections import namedtuple, Counter
from typing import Union, Sequence, Callable, Iterable, AnyStr
import psutil
import multiprocessing as mp
import numpy as np
from functools import partial

# >>>> CUSTOM TYPES
# ---------------------

EqnParseResult = namedtuple('EqnParseResult', 'data comparisons')

Number = Union[float, int]
ListOrTuple = Union[list, tuple]


class MojoStatsException(Exception):
    pass


# >>>> MODULE CODE
# ---------------------

def is_number_sequence(seq: Sequence[Number]) -> bool:
    """
    Verifies that input is a Sequence of Numbers
    :param seq: Some Sequence
    :return: True if input s a Sequence of Numbers
    """
    try:
        return all([is_numeric(x) for x in seq])
    except:
        return False


def is_str_sequence(seq: Sequence[str]) -> bool:
    """
    Verifies that input contains only strings
    :param seq: Some Sequence
    :return: True if inputs are all strings
    """
    if isinstance(seq, str):
        return False

    try:
        return all([isinstance(x, str) for x in seq])
    except:
        return False


def column_arrange(chunks: ListOrTuple) -> list:
    '''
    Returns the input strings arranged in columns.
    :param strings: list or tuple of strings
    :return: strings arranged in columns
    '''
    chunk_lists = [str(chunk).splitlines(keepends=False) for chunk in chunks]
    max_chunk_height = max([len(chunk_list) for chunk_list in chunk_lists])
    max_str_length = max([len(chunk) for chunk_list in chunk_lists for chunk in chunk_list])
    chunk_lists = [chunk_list + ([''] * (max_chunk_height - len(chunk_list))) for chunk_list in chunk_lists]
    rows = zip(*chunk_lists)
    output = []
    for words in rows:
        aline = ''
        for word in words:
            aline += f'{word:<{max_str_length}}  '
        output.append(aline)
    return output


def column_print(chunks: ListOrTuple):
    '''
    Prints the output of column_arrange
    '''
    output = column_arrange(chunks)
    print('\n'.join(output))


def is_numeric(x: Number) -> bool:
    """
    Returns True if argument can be safely cast as a float
    :param x: Something to test
    :return: Bool indicating whether x can be converted to a number
    """
    try:
        y = float(x)
        return True
    except:
        return False


def sqr(x: Number):
    """
    Returns square of argument or None if this isn't possible.
    :param x: Numeric
    :return: x squared or None if it's not possible
    """
    try:
        return x * x
    except:
        return None


def int_or_float(x: Union[int, float, str]):
    try:
        if isinstance(x, str):
            val = eval(x)
        else:
            val = x
        if type(val) in (int, float):
            return float(val)
        else:
            return np.nan
    except:
        return np.nan


def headtail(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise Exception('df must be type pandas.DataFrame')
    h = df.head()
    t = df.tail()
    if sorted(list(h.index)) == sorted(list(t.index)):
        return h
    else:
        return pd.concat([h, t])


def dump_nan_cols(df: pd.DataFrame, quiet: bool = True):
    """
    Inplace removal from a pandas DataFrame any columns containing only NaNs
    :param df: pandas DataFrame
    :param quiet: if False, prints list of column names that were dumped.
    :return: NA
    """
    if not isinstance(df, pd.DataFrame):
        raise MojoStatsException('df must be type pandas.DataFrame')
    dumped = []
    for col in df.columns:
        if is_nan_col(df[col]):
            del df[col]
            dumped.append(col)
    if not quiet:
        print('DUMPED NAN COLS:', dumped, sep='\n', end='\n\n')


def is_nan_col(alist: Sequence) -> bool:
    '''
    function to identify empty columns in a dataframe
    :param alist: any type comparable with typing.Sequence
    :return: True if xp.isnan(x) for all items in sequence
    '''
    try:
        return all([np.isnan(x) for x in list(alist)])
    except:
        return False


# FIXME: level scheme isn't super useful as written
def _title(msg: str, level: int = 1, frame_size: int = 0) -> str:
    fs = frame_size if frame_size else len(msg)
    fc = "=-"[level - 1] if level in (0, 1) else ""
    frame = fc * fs
    return f"{frame}\n{msg}\n{frame}"


def title(msg: str, level: int = 1, frame_size: int = 0):
    print(_title(msg, level, frame_size))


def unique_permutations(items: Iterable, r=None) -> list:
    perms = list(itertools.permutations(items, r))
    unique = list()
    for perm in perms:
        sp = sorted(perm)
        if sp not in unique:
            unique.append(sp)
    return unique


def make_var_formula(vars: Sequence, interactions: bool = False) -> str:
    var_formula = ''
    for var in vars:
        if not var_formula:
            var_formula += var
        else:
            glue = '+' if not interactions else '*'
            var_formula += f' {glue} C({var})'
    return var_formula


def parse_equation(formula: str, data: pd.DataFrame, min_groups: int = 1, max_groups: int = 0) -> EqnParseResult:
    y, x = dmatrices(formula, data, 1)
    x_fi = x.design_info.factor_infos
    y_fi = y.design_info.factor_infos

    assert list(y_fi.values())[0].num_columns == 1, "Formula Must Specify a 1-Dimensional Vector for Y"
    assert all([fi.type == 'numerical' for fi in y_fi.values()]), "Y Vector Must Be Entirely Numerical"
    assert all([fi.type == 'categorical' for fi in x_fi.values()]), "All X Factors Must Be Categorical"
    assert len(x_fi) >= min_groups, "Number of X Factors Must Be >= Than parameter min_groups"
    assert not max_groups or len(x_fi) <= max_groups, "Number of X Factors Must Be <= parameter max_groups"

    y = [float(value[0]) for value in y]
    x = [tuple(value) for value in x]

    groups = [ef.code for ef in x_fi.keys()]
    varname = list(y_fi.values())[0].factor.code
    data_vectors = dict()
    df = data.groupby(groups)
    for group_name, group_data in df:
        data_vectors[group_name] = list(group_data[varname])

    binary_comparisons = unique_permutations(data_vectors.keys(), 2)

    return EqnParseResult(data=data_vectors, comparisons=binary_comparisons)


def counter_to_df(c: Counter, sortkey: str = 'instances', sortorder: str = 'ascending') -> pd.DataFrame:
    """
    c: collections.Counter
    sortkey: str in ['terms', 'instances']
    sortorder: str in ['ascending', 'descending']
    """
    assert type(c) is Counter
    assert sortkey in ['terms', 'instances']
    d = dict(c)
    df = pd.DataFrame({'terms': list(d.keys()), 'instances': list(d.values())})
    order = int(sortorder == 'descending')
    df = df.sort_values([sortkey], ascending=[order])
    df.index = range(1, len(df) + 1)  # index is all jumbled now, recalc for sorted order
    return df


# todo: how am I supposed to test this?!
def parallelize_dataframe(df: pd.DataFrame, func: Callable, num_partitions: int = None) -> pd.DataFrame:
    """
    Run a function on a dataframe by splitting it across available CPU cores!
    :param df: pandas DataFrame object
    :param func: function or other callable
    :param num_partitions: If None, split will equal number of CPU cores.
    :return: altered pandas DataFrame object
    """
    if not isinstance(df, pd.DataFrame):
        raise MojoStatsException('df must be type pandas.DataFrame')
    if not isinstance(func, Callable):
        raise MojoStatsException('func must be type typing.Callable')
    num_cores = psutil.cpu_count() - 1  # number of cores on your machine minus 1
    df_split = np.array_split(df, num_partitions if num_partitions else num_cores)
    pool = mp.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def round_all(seq: Sequence[Number], digits: int = 2) -> ListOrTuple:
    """
    Rounds all the numbers in a sequence to _digits_ decimal places
    :param seq: Sequence of numbers
    :param digits: rounding precision
    :return: rounded sequence
    """

    if not is_number_sequence(seq):
        raise MojoStatsException('seq must be a sequence of numbers')

    rounded = map(partial(round, ndigits=digits), seq)
    if isinstance(seq, list):
        return list(rounded)
    else:
        return tuple(rounded)


# >>>> QUICK TESTING
# ---------------------

if __name__ == '__main__':
    from pprint import pprint

    print(round_all(seq=(22.3, 1.3, 58.44556, 5.98, .09), digits=1))

    # data_source = '/Users/nogard/Dropbox/Documents/python_coding/statworkflow/tests/data/data.csv'
    # print("rt ~ block")
    # res = parse_equation("rt ~ block", pd.read_csv(data_source))
    # print(res)
    # print()
    # print("rt ~ block * category")
    # res = parse_equation("rt ~ block * category", pd.read_csv(data_source))
    # pprint(res.data)
    # print()
    # pprint(res.comparisons)
