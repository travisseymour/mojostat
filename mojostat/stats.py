# >>>> BUILTIN MODULES
# --------------------

from collections import namedtuple
import multiprocessing as mp
import os
import time
import collections
import re
import logging
from functools import partial
from typing import Union, Sequence, Callable, Iterable
from io import StringIO  # py3.x
from io import BytesIO

# >>>> EXTERNAL MODULES
# ---------------------

import mojostat.template as template
import psutil
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.libqsturng import psturng
from scipy import stats

try:
    import mojostat.statutil as statutil
except:
    import statutil


class MojoStatsException(Exception):
    pass


# >>>> GLOBAL VARIABLES (really?!)
# ---------------------

num_partitions = 10  # number of partitions to split dataframe
num_cores = psutil.cpu_count() - 1  # number of cores on your machine minus 1

# >>>> NAMED TUPLES
# ---------------------

ttest_result = namedtuple('ttest_result', 'x_mean, x_stdev y_mean y_stdev low_ci hi_ci t_stat p_value deg_fred '
                                          'sig_flag comp alt_comp cohensd power x_name y_name summary oneliner')
PostHocResult = namedtuple('PostHocResults', 'title table pvalues')


# >>>> MODULE CODE
# ---------------------

def column_arrange(chunks: Union[list, tuple]) -> list:
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


def column_print(chunks: Union[list, tuple]):
    '''
    Prints the output of column_arrange
    '''
    output = column_arrange(chunks)
    print('\n'.join(output))


def _title(msg: str, level: int = 1, frame_size: int = 0) -> str:
    fs = frame_size if frame_size else len(msg)
    fc = "=-"[level - 1] if level in (0, 1) else ""
    frame = fc * fs
    return f"{frame}\n{msg}\n{frame}"


def title(msg: str, level: int = 1, frame_size: int = 0):
    print(_title(msg, level, frame_size))


def format_chi2(*args, **kwargs) -> str:
    if 'chi' in kwargs and 'df' in kwargs and 'p' in kwargs:
        return f"X2={chi:0.3f}, df={df}, p={p:0.6f}"
    elif type(args[0]) is tuple and len(args[0]) == 4:
        chi, p, df, _ = args[0]
        return f"X2={chi:0.3f}, df={df}, p={p:0.6f}"
    else:
        raise TypeError('format_ch2 expects either a 3+ length tuple, or explicit parameters for chi, df, and p')


def _describe(frame: pd.DataFrame, func: Iterable, numeric_only: bool, **kwargs) -> pd.DataFrame:
    def nullcount(ser):
        return ser.isnull().sum()

    if numeric_only:
        frame = frame.select_dtypes(include=np.number)
    return frame.agg(list(func) + [nullcount], **kwargs)


def describe(frame: pd.DataFrame,
             func: Iterable = ('count', 'sum', 'mean', 'median', 'min', 'max', 'quantile',
                               'std', 'var', 'sem', 'skew', 'kurt'),
             numeric_only: bool = True, **kwargs) -> pd.DataFrame:
    if type(frame) is pd.core.groupby.groupby.DataFrameGroupBy:
        dfs = list()
        for group, group_dat in frame:
            df = _describe(frame=group_dat, func=func, numeric_only=numeric_only, **kwargs)
            df['descrgrp'] = group
            dfs.append(df)
        res_df = pd.DataFrame(pd.concat(dfs))  # casting only to keep away pycharm warnings!
        res_df = res_df[['descrgrp'] + list(res_df.columns[:-1])]
    else:
        res_df = _describe(frame=frame, func=func, numeric_only=numeric_only, **kwargs)

    return res_df


def is_numeric(x: Union[int, float]) -> bool:
    """
    :param x: Numeric
    :return: Bool indicating whether x can be converted to a number
    """
    try:
        y = float(x)
        return True
    except:
        return False


def sqr(x: Union[int, float]):
    """
    :param x: Numeric
    :return: x squared or None if it's not possible
    """
    try:
        return x * x
    except:
        return None


def p_desc2(p: Union[int, float]) -> str:
    """
    :param p: Numeric statistical p-value
    :return: str indication of statistical significance level
    """

    if p < .001:
        return " ***"
    if p < .01:
        return " **"
    elif p < .05:
        return " *"
    elif p <= .06:
        return " ."
    else:
        return ""


# FIXME: Note: These are for t-tests, but not f-tests.
def d_desc(d: Union[int, float]) -> str:
    if d <= .01:
        return 'very small'  # Sawilowsky, 2009
    elif d <= .20:
        return 'small'  # Cohen, 1988
    elif d <= .50:
        return 'medium'  # Cohen, 1988
    elif d <= .80:
        return 'large'  # Cohen, 1988
    elif d <= 1.2:
        return 'very large'  # Sawilowsky, 2009
    else:
        return 'huge'  # Sawilowsky, 2009


def d_from_data(distribution1:Sequence, distribution2:Sequence)->float:
    """
    :param distribution1: list-like, numerical
    :param distribution2: list-like, numerical
    :return: Cohen's d
    """
    # from: http://www.socscistatistics.com/effectsize/Default3.aspx

    l1, l2 = 0, 0
    try:
        l1, l2 = len(distribution1), len(distribution2)
    except:
        return np.nan

    if l1 and l2:
        m1 = np.mean(distribution1)
        m2 = np.mean(distribution2)
        sd1 = np.std(distribution1)
        sd2 = np.std(distribution2)
        sd_pooled = np.sqrt((sqr(sd1) + sqr(sd2)) / 2)
        if m2 > m1:
            cohens_d = (m2 - m1) / sd_pooled
        else:
            cohens_d = (m1 - m2) / sd_pooled
        return cohens_d


# def my_ttest(res):
#     # res = pd.rpy.common.convert_robj(res)  # works at home with brew, but not on ltop w anaconda
#     res = com.convert_robj(res)
#     effect_size = d_from_t(t=res['statistic'].t, df=res['parameter'].df)
#     t_str = "{dname}\n t({df:0.1f})={t:0.3f}, p={p:0.4f}{outcome}, ({dirc}), CI=[{loconf:0.2f}, {hiconf:0.2f}], Cohen's d={cohen:0.2f} ({deval})."
#     d_str = d_desc(effect_size.d)
#     out_str = p_desc(res['p.value'][0])
#     return t_str.format(df=res['parameter'].df, t=res['statistic'].t, p=res['p.value'][0], outcome=out_str,
#                        dirc=res['alternative'][0].replace('.', '-'), loconf=res['conf.int'][0],
#                         hiconf=res['conf.int'][1], cohen=effect_size.d, deval=d_str,
#                         dname=res['data.name'][0].replace('and', 'vs.') + ':')


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


# http://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
def corr_pvalue(df: pd.DataFrame)->pd.DataFrame:
    from scipy.stats import pearsonr
    import numpy as np
    import pandas as pd

    numeric_df = df.dropna()._get_numeric_data()
    cols = numeric_df.columns
    mat = numeric_df.values

    arr = np.zeros((len(cols), len(cols)), dtype=object)

    for xi, x in enumerate(mat.T):
        for yi, y in enumerate(mat.T[xi:]):
            arr[xi, yi + xi] = list(map(lambda _: round(_, 4), pearsonr(x, y)))[1]
            arr[yi + xi, xi] = arr[xi, yi + xi]

    return pd.DataFrame(arr, index=cols, columns=cols)


def is_nan_col(alist: Sequence)->bool:
    '''
    function to identify empty columns in a dataframe
    :param alist: any type comparable with typing.Sequence
    :return: True if xp.isnan(x) for all items in sequence
    '''
    try:
        return all([np.isnan(x) for x in list(alist)])
    except:
        return False


# careful, this is an updater
def dump_nan_cols(df: pd.DataFrame, quiet: bool = True):
    if not isinstance(df, pd.DataFrame):
        raise MojoStatsException('df must be type pandas.DataFrame')
    dumped = []
    for col in df.columns:
        if is_nan_col(df[col]):
            del df[col]
            dumped.append(col)
    if not quiet:
        print('DUMPED NAN COLS:', dumped, sep='\n', end='\n\n')


# todo: this isn't sufficiently general!
def my_read_excel(fname: str, quiet: bool = True):
    # read in dataframe
    df = pd.read_excel(os.path.join('all_data_sources', fname))
    # strip out any spaces in the var names
    df.columns = [str(var).strip() for var in df.columns]
    # dump nan columns
    dump_nan_cols(df, quiet=quiet)
    return df


# todo: how am I supposed to test this?!
def parallelize_dataframe(df: pd.DataFrame, func: Callable):
    if not isinstance(df, pd.DataFrame):
        raise MojoStatsException('df must be type pandas.DataFrame')
    if not isinstance(func, Callable):
        raise MojoStatsException('func must be type typing.Callable')
    df_split = np.array_split(df, num_partitions)
    pool = mp.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# todo: does this even work? Where is it used??
def get_var_name(**kwargs):
    try:
        return kwargs.keys()[0]
    except:
        return None


def round_all(seq: Sequence, digits: int = 2)->list:
    def rnd(x, d):
        try:
            return round(x, d)
        except:
            return x

    return [rnd(seq, digits) for x in seq][0]


def table_pct(df: pd.DataFrame, colnames: Sequence[str], denom: Union[int, float])->pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise Exception('df must be type pandas.DataFrame')

    if isinstance(colnames, str):
        col_names = list(colnames)
    else:
        col_names = colnames
    for colname in col_names:
        df[f'{colname}_pct'] = df[colname] / denom * 100

    return df


def headtail(df: pd.DataFrame)->pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise Exception('df must be type pandas.DataFrame')
    h = df.head()
    t = df.tail()
    if sorted(list(h.index)) == sorted(list(t.index)):
        return h
    else:
        return pd.concat([h, t])

# FIXME: what is this parameter type? pd.DataFrame?? If so, say so.
def make_corr_p_table(corr_table)->str:
    p_table_html = corr_table.to_html().__str__()
    pat = re.compile(r'(<td>)(0\.\d+)(</td>)')
    matches = set(pat.findall(p_table_html))
    logging.debug(matches)
    for match in matches:
        val = match[1]
        fval = float(val)
        if fval < .06:
            if fval < .05:
                p_table_html = p_table_html.replace(val, f'<b>{val}</b>')
            else:
                p_table_html = p_table_html.replace(val, '<b><em>{val}</em></b>')
    return p_table_html


def make_var_formula(vars: Sequence, interactions: bool = False)->str:
    var_formula = ''
    for var in vars:
        if not var_formula:
            var_formula += var
        else:
            glue = '+' if not interactions else '*'
            var_formula += f' {glue} C({var})'
    return var_formula


# No longer included becuase of reliance on matplotlib, ...could consider having user supply 'plt'?
# def simple_regression(x_var: str, y_var: str, df: pd.DataFrame, fig_args: dict, winsorize: bool = False):
#     if winsorize:
#         df[y_var] = scipy.stats.mstats.winsorize(df[y_var], limits=(.05, .05))
#     est = ols(formula=f'{x_var} ~ {y_var}', data=df).fit()
#     fig, ax = plt.subplots(figsize=(fig_args['width'], fig_args['height']), dpi=fig_args['dpi'])
#     plt.scatter(df[x_var], df[y_var], alpha=0.6)
#     t = f"{x_var.title()} ~ {y_var.title()}{'*' if est.f_pvalue <= .05 else ''}" \
#         f"{chr(10)+'(winsorized)' if winsorize else ''}"
#     plt.title(t)
#     plt.xlabel(x_var.title())
#     plt.ylabel(y_var.title())
#
#     # fig_embed(fig)
#     # zprint(est.summary())
#
#     res = namedtuple('RegressionResult', 'figure, estimate')
#     return res(figure=fig, estimate=est)

def run_zprop(c: tuple)->str:
    """
    e.g., c = (c_s, c_l, 'Study', 'Leisure')
          (
          c_s = [hit, miss] for group1
          c_l = [hit, miss] for group2
          n1 = name of group 1
          n2 = name of group 2
          )
    :param c: (c_s, c_l, 'Study', 'Leisure')
    :return: formatted z test for proportions output
    """

    g1, g2, n1, n2 = c
    z, p = sm.stats.proportions_ztest([g1[0], g2[0]], [sum(g1), sum(g2)])
    # s = '{} ({:0.3f}%) vs {} ({:0.3f}%): Z({})={:0.2f}, p={:0.4f}'
    # s = s.format(n1, (g1[0] / sum(g1)) * 100, n2, (g2[0] / sum(g2)) * 100, sum(g1), z, p)
    s = f'{n1} ({(g1[0] / sum(g1)) * 100:0.3f}%) vs {n2} ({(g2[0] / sum(g2)) * 100:0.3f}%): ' \
        f'Z({sum(g1)})={z:0.2f}, p={p:0.4f}'
    return s


# todo: rewrite using patsy!
# todo: return a named tuple!
def run_anova(df: pd.DataFrame, dv: str, factors: list, html: bool = False)->tuple:
    formula = f'{dv} ~ {make_var_formula(factors, True)}'
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, type=2)
    extra = '<br>' if html else ''
    return formula + extra, table


def pvalues_from_tukeyhsd(table):
    return psturng(np.abs(table.meandiffs / table.std_pairs), len(table.groupsunique), table.df_total)


def anova_table_to_dataframe(table)->pd.DataFrame:
    df = pd.DataFrame(table)
    index = df.index.tolist()
    index = [element.replace('C(', '').replace(')', '').replace(':', '_') for element in index]
    df.index = pd.Index(index)

    def signify(row):
        p = row['PR(>F)']
        if pd.notnull(p):
            p = float(p)
            if p < .0001:
                row['sig'] = '****'
            elif p < .001:
                row['sig'] = '***'
            elif p < .01:
                row['sig'] = '**'
            elif p < .05:
                row['sig'] = '*'
            elif .05 < p < .07:
                row['sig'] = '.'
            else:
                row['sig'] = ''
        return row

    df['sig'] = ''  # without this, works but can reorder columns!
    df = df.apply(signify, axis=1)
    return df


def df_row_to_f_statement(df: pd.DataFrame, identifier: str)->str:
    """
    assumes df version of anovatable!
    should come from anova_table_to_dataframe()
    something like this: "condition:C(psychmajor)" is this "condition_psychmajor"
    """
    row = df.loc[identifier]
    df, sss, msq, f, p, sig = row
    sig_str = f' ({sig})' if sig else ''
    s = f"{identifier.replace('_', ' by ')}: F({df:0.1f})={f:0.2f}, p={p:0.5f}{sig_str}"
    return s


# def show_post_hocs(df: pd.DataFrame, dm: str, factor: str, *args, **kwargs):
#     zprint(f'Post-hoc T-Tests (TukeyHSD): {dm} ~ {factor}', *args, **kwargs)
#     table = sm.stats.multicomp.pairwise_tukeyhsd(df.dropna()[dm], df.dropna()[factor])
#     pvalues = pvalues_from_tukeyhsd(table)
#     zprint(table, '\n<br>', pvalues, *args, **kwargs)

def post_hocs(df: pd.DataFrame, dm: str, factor: str) -> PostHocResult:
    title = f'Post-hoc T-Tests (TukeyHSD): {dm} ~ {factor}'
    table = sm.stats.multicomp.pairwise_tukeyhsd(df.dropna()[dm], df.dropna()[factor])
    pvalues = pvalues_from_tukeyhsd(table)
    return PostHocResult(title, table, pvalues)


def get_post_hocs_as_df(df: pd.DataFrame, dm: str, factor: str, tag: str = '') -> pd.DataFrame:
    table = sm.stats.multicomp.pairwise_tukeyhsd(df.dropna()[dm], df.dropna()[factor])
    pvalues = pvalues_from_tukeyhsd(table)
    table_csv = table._results_table.as_csv()
    table_csv = table_csv[table_csv.find('\n') + 1:].replace(' ', '')
    df = pd.read_csv(StringIO(table_csv), delimiter=',')
    c = df.columns.tolist()
    df['tag'] = str(tag)
    df['p'] = [float(p) for p in pvalues] if type(pvalues) is not float else [float(str(pvalues))]
    df['dv'] = dm
    df['factor'] = factor
    df['sig'] = [p_desc2(p) for p in df['p']]
    df = df.reindex_axis(['tag', 'dv', 'factor'] + c + ['p', 'sig'], axis=1)
    df.drop(['reject'], axis=1, inplace=True)
    return df


def counter_to_df(c: collections.Counter, sortkey: str = 'instances', sortorder: str = 'ascending')->pd.DataFrame:
    """
    c: collections.Counter
    sortkey: str in ['terms', 'instances']
    sortorder: str in ['ascending', 'descending']
    """
    assert type(c) is collections.Counter
    assert sortkey in ['terms', 'instances']
    d = dict(c)
    df = pd.DataFrame({'terms': list(d.keys()), 'instances': list(d.values())})
    order = int(sortorder == 'descending')
    df = df.sort_values([sortkey], ascending=[order])
    df.index = range(1, len(df) + 1)  # index is all jumbled now, recalc for sorted order
    return df


# todo: x_name and y_name seem to do nothing!
# todo: Should put dv name in summary and oneliner!
# todo: Consider adding formula to comparison name
# todo: Allow list of alternatives!
def ttest(formula: str, data: pd.DataFrame, alternative: str = 'two-sided', usevar: str = 'unequal',
          x_name: str = 'Group1', y_name: str = 'Group2', html: bool = False, kind='ind')->list:
    assert kind in ('ind', 'rel'), f"MojoStat.Stats: kind parameter must be either 'ind' or 'rel', not '{kind}"

    data_groups, comparisons = statutil.parse_equation(formula, data, min_groups=1)

    results = list()

    StatResult = namedtuple('StatResult', 'comparison result')

    for comparison in comparisons:
        x_name, y_name = comparison
        x_name2 = x_name if type(x_name) is str else "|".join(x_name)
        y_name2 = y_name if type(y_name) is str else "|".join(y_name)
        comparison_name = f'[{formula.strip()}]: {x_name2} vs {y_name2}'
        x, y = data_groups[x_name], data_groups[y_name]
        x = [value for value, ok in zip(x, pd.notnull(x)) if ok]
        y = [value for value, ok in zip(y, pd.notnull(y)) if ok]
        # todo: should I add the data_group to the output to be used as a sort of title for the result?
        if kind == 'ind':
            res = r_ttest_ind(x, y, alternative, usevar, x_name2, y_name2, html)
        else:
            res = r_ttest_rel(x, y, alternative, x_name2, y_name2, html)
        results.append(StatResult(comparison=comparison_name, result=res))

    return results


def r_ttest_output(test_name: str, comp: str, alt_comp: str, alternative: str, x_name: str, y_name: str, x_mean: float,
                   y_mean: float, x_stdev: float, y_stdev: float, t_stat: float, deg_fred: float, p_value: float,
                   sig_flag: str, low_ci: float, hi_ci: float, cohens_d: float, power: float,
                   html: bool = False) -> ttest_result:
    if html:
        full = f"""\
        <p><b>{test_name}Two Sample T-Test</b><br>
        ===========================================<br>
        <font color=blue>Hypothesis</font>: {x_name} {comp} {y_name} ({alternative})<br>
        <font color=blue>Statistic</font>: t = {t_stat:0.4f}, df = {deg_fred:0.2f}, p-value = {p_value: 0.4f} <b>{sig_flag}</b><br>
        <font color=blue>Mean Group Difference</font>: {x_mean - y_mean:0.3f}<br>
        <font color=blue>Conffidence Interval (95%)</font>: [{low_ci:0.3f} to {hi_ci:0.3f}]<br>
        <font color=blue>Effect Size (Cohen's d)</font>: {cohens_d:0.2f}<br>
        <font color=blue>Post-hoc Power</font>: {power:0.3f}<br>
        <br>
        <font color=blue>Result</font>: {x_name} (M:{x_mean:0.2f} SD:{x_stdev:0.2f}) {alt_comp} {y_name} (M:{y_mean:0.2f}, SD:{y_stdev:0.2f})<br>
        </p>"""

        oneline = f"<p>{x_name} (M:{x_mean:0.2f} SD:{x_stdev:0.2f}) {alternative} {y_name} (M:{y_mean:0.2f}, " \
                  f"SD:{y_stdev:0.2f}), t({deg_fred:0.2f})={t_stat:0.3f}, p={p_value: 0.4f}, " \
                  f"CI=[{low_ci:0.2f}, {hi_ci:0.2f}], d={cohens_d:0.2f}.<br></p>"
    else:
        full = """\
        {test_name}Two Sample T-Test [{time_stamp}]
        ===========================================
        Hypothesis: {x_name} {comp} {y_name} ({alternative})
        Statistic: t = {t_stat:0.4f}, df = {deg_fred:0.2f}, p-value = {p_value: 0.4f} {sig_flag}
        Mean Group Difference: {x_mean - y_mean:0.3f}
        Conffidence Interval (95%): [{low_ci:0.3f} to {hi_ci:0.3f}]
        Effect Size (Cohen's d): {cohens_d:0.2f}
        Post-hoc Power: {power:0.3f}

        Result: {x_name} (M:{x_mean:0.2f} SD:{x_stdev:0.2f}) {alt_comp} {y_name} (M:{y_mean:0.2f}, SD:{y_stdev:0.2f})"""

        oneline = f"{x_name} (M:{x_mean:0.2f} SD:{x_stdev:0.2f}) {alternative} {y_name} (M:{y_mean:0.2f}, " \
                  f"SD:{y_stdev:0.2f}), t({deg_fred:0.2f})={t_stat:0.3f}, p={p_value: 0.4f}, " \
                  f"CI=[{low_ci:0.2f}, {hi_ci:0.2f}], d={cohens_d:0.2f}."

    res = ttest_result(x_mean, x_stdev, y_mean, y_stdev, low_ci, hi_ci, t_stat, p_value, deg_fred, sig_flag, comp,
                       alt_comp, cohens_d, power, x_name, y_name, full, oneline)

    return res


def r_ttest_ind(x: Sequence, y: Sequence, alternative: str = 'two-sided', usevar: str = 'unequal',
                x_name: str = 'Group1', y_name: str = 'Group2', html: bool = False) -> ttest_result:
    """
    >>> res = r_ttest_ind(x=list(np.arange(10,21)), y=list(np.arange(20.0,26.5,.5)), alternative='two-sided',
    ...             usevar='unequal', x_name='Group1', y_name='Group2')
    >>> res.t_stat
    -7.0390615210249585
    >>> res.deg_fred
    15.5795730883545
    >>> res.p_value
    3.2469535237571196e-06
    """
    try:
        x_mean = np.mean(x)
        x_stdev = np.std(x)
        y_mean = np.mean(y)
        y_stdev = np.std(y)
    except Exception as e:
        raise MojoStatsException(f'x and y must each be a sequence of valid numbers:\n\t{e}')

    assert alternative in ('larger', 'smaller',
                           'two-sided'), f"MojoStat.Stat: alternative must be 'larger', 'smaller' or 'two-sided', not {alternative}"

    time_stamp = time.strftime("%x %X")

    cm = sm.stats.CompareMeans(sm.stats.DescrStatsW(x), sm.stats.DescrStatsW(y))
    low_ci, hi_ci = cm.tconfint_diff(alpha=.05, alternative=alternative, usevar=usevar)

    t_stat, p_value, deg_fred = sm.stats.ttest_ind(x, y, alternative=alternative, usevar=usevar)

    sig_flag = p_desc2(p_value)
    test_name = ("Welch's Independent" if usevar == 'unequal' else "Idependent")
    cohens_d = d_from_data(x, y)
    power = sm.stats.tt_ind_solve_power(effect_size=cohens_d, nobs1=len(x), alpha=0.05, power=None,
                                        ratio=len(y) / len(x), alternative=alternative)
    if 'larger' in alternative:
        comp = '>'
        if p_value < .05:
            alt_comp = 'is significantly larger than'
        else:
            alt_comp = 'is not significantly larger than'
    elif 'smaller' in alternative:
        comp = '<'
        if p_value < .05:
            alt_comp = 'is significantly smaller than'
        else:
            alt_comp = 'is not significantly smaller than'
    else:
        comp = '!='
        if p_value < .05:
            if x_mean > y_mean:
                alt_comp = 'is significantly larger than'
            else:
                alt_comp = 'is significantly smaller than'
        else:
            alt_comp = 'is not significantly different than'

    res = r_ttest_output(test_name=test_name, comp=comp, alt_comp=alt_comp, alternative=alternative,
                         x_name=x_name, y_name=y_name, x_mean=x_mean, y_mean=y_mean, x_stdev=x_stdev, y_stdev=y_stdev,
                         t_stat=t_stat, deg_fred=deg_fred, p_value=p_value, sig_flag=sig_flag,
                         low_ci=low_ci, hi_ci=hi_ci, cohens_d=cohens_d, power=power, html=html)

    return res


def r_ttest_rel(x: Sequence, y: Sequence, alternative: str = 'two-sided', x_name: str = 'Group1',
                y_name: str = 'Group2', html: bool = False) -> ttest_result:
    try:
        x_mean = np.mean(x)
        x_stdev = np.std(x)
        y_mean = np.mean(y)
        y_stdev = np.std(y)
    except Exception as e:
        raise MojoStatsException(f'x and y must each be a sequence of valid numbers:\n\t{e}')

    assert alternative in ('larger', 'smaller',
                           'two-sided'), f"MojoStat.Stat: alternative must be 'larger', 'smaller' or 'two-sided', not {alternative}"

    xy_diff = [xy[0] - xy[1] for xy in zip(x, y)]
    time_stamp = time.strftime("%x %X")

    # cm = sm.stats.CompareMeans(sm.stats.DescrStatsW(x), sm.stats.DescrStatsW(y))
    # low_ci, hi_ci = cm.tconfint_diff(alpha=.05, alternative=alternative, usevar=usevar)

    # t_stat, p_value, deg_fred = sm.stats.ttest_ind(x, y, alternative=alternative, usevar=usevar)
    d = sm.stats.DescrStatsW(xy_diff)
    t_stat, p_value, deg_fred = d.ttest_mean(alternative=alternative)
    low_ci, hi_ci = d.tconfint_mean(alpha=0.05, alternative=alternative)

    sig_flag = p_desc2(p_value)
    # test_name = ("Welch's Related" if usevar == 'unequal' else "Related")
    test_name = "Related"
    # cohens_d = d_from_t(t_stat, deg_fred).d
    cohens_d = d_from_data(x, y)
    power = sm.stats.tt_ind_solve_power(effect_size=cohens_d, nobs1=len(x), alpha=0.05, power=None,
                                        ratio=len(y) / len(x), alternative=alternative)
    if 'larger' in alternative.lower():
        comp = '>'
        if p_value < .05:
            alt_comp = 'is significantly larger than'
        else:
            alt_comp = 'is not significantly larger than'
    elif 'smaller' in alternative.lower():
        comp = '<'
        if p_value < .05:
            alt_comp = 'is significantly smaller than'
        else:
            alt_comp = 'is not significantly smaller than'
    else:  # two-sided
        comp = '!='
        if p_value < .05:
            if x_mean > y_mean:
                alt_comp = 'is significantly larger than'
            else:
                alt_comp = 'is significantly smaller than'
        else:
            alt_comp = 'is not significantly different than'

    res = r_ttest_output(test_name=test_name, comp=comp, alt_comp=alt_comp, alternative=alternative,
                         x_name=x_name, y_name=y_name, x_mean=x_mean, y_mean=y_mean, x_stdev=x_stdev, y_stdev=y_stdev,
                         t_stat=t_stat, deg_fred=deg_fred, p_value=p_value, sig_flag=sig_flag,
                         low_ci=low_ci, hi_ci=hi_ci, cohens_d=cohens_d, power=power, html=html)

    return res


# >>>> QUICK TESTING
# ---------------------


if __name__ == '__main__':
    from pprint import pprint

    data_source = '/Users/nogard/Dropbox/Documents/python_coding/statworkflow/tests/data/data.csv'
    results = ttest(formula="rt ~ block", data=pd.read_csv(data_source),
                    alternative='two-sided', usevar='unequal',
                    x_name='Group1', y_name='Group2', html=False, kind='rel')
    pprint(results)
    print()
    for result in results:
        print(result.comparison)
        print(result.result.oneliner)

    results = ttest(formula="rt ~ block + category", data=pd.read_csv(data_source),
                    alternative='two-sided', usevar='unequal',
                    x_name='Group1', y_name='Group2', html=False, kind='ind')
    pprint(results)
    print()
    for result in results:
        print(result.comparison)
        print(result.result.oneliner)
        print('-----')

# if __name__ == '__main__':
# import pickle
#
# # visually verify that my_ttest is working
# res2_raw = None
# with open('ttest_res2.dat', 'rb') as infile:
#     res2_raw = infile.read()
# res2 = pickle.loads(res2_raw)
# # print(res2)
# print(my_ttest(res2))
#
# res4_raw = None
# with open('ttest_res4.dat', 'rb') as infile:
#     res4_raw = infile.read()
# res4 = pickle.loads(res4_raw)
# # print(res4)
# print(my_ttest(res4))

# res = r_ttest_ind(x=list(np.arange(10,21)), y=list(np.arange(20.0,26.5,.5)), alternative='two-sided', usevar='unequal',
#                   x_name='Group1', y_name='Group2')
# print(res.summary)
# print(res.t_stat, res.deg_fred, res.p_value)
