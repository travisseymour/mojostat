import pandas as pd
from patsy import dmatrices
import itertools
from typing import Iterable
from collections import namedtuple

EqnParseResult = namedtuple('EqnParseResult', 'data comparisons')


def unique_permutations(items: Iterable, r=None) -> list:
    perms = list(itertools.permutations(items, r))
    unique = list()
    for perm in perms:
        sp = sorted(perm)
        if sp not in unique:
            unique.append(sp)
    return unique


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


if __name__ == '__main__':
    from pprint import pprint

    data_source = '/Users/nogard/Dropbox/Documents/python_coding/statworkflow/tests/data/data.csv'
    print("rt ~ block")
    res = parse_equation("rt ~ block", pd.read_csv(data_source))
    print(res)
    print()
    print("rt ~ block * category")
    res = parse_equation("rt ~ block * category", pd.read_csv(data_source))
    pprint(res.data)
    print()
    pprint(res.comparisons)
