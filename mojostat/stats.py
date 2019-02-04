"""
This module, MojoStat.Stats, contains statistical analysis-related functions.
Travis L. Seymour, PhD - nogard@ucsc.edu - University of California Santa Cruz
"""

from collections import namedtuple
import re
from typing import Sequence, Iterable
from io import StringIO
# from io import BytesIO
# import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.libqsturng import psturng
import scipy.stats
import statistics

try:
    from mojostat.statutil import *
except:
    from statutil import *

# >>>> CUSTOM TYPES
# ---------------------

ttest_result = namedtuple('ttest_result', 'x_mean, x_stdev y_mean y_stdev low_ci hi_ci t_stat p_value deg_fred '
                                          'sig_flag comp alt_comp cohensd power x_name y_name summary oneliner')
PostHocResult = namedtuple('PostHocResults', 'title table pvalues')


# >>>> MODULE CODE
# ---------------------


def format_chi2(*args, **kwargs) -> str:
    if 'chi' in kwargs and 'df' in kwargs and 'p' in kwargs:
        return f"X2={chi:0.3f}, df={df}, p={p:0.6f}"
    elif type(args[0]) is tuple and len(args[0]) == 4:
        chi, p, df, _ = args[0]
        return f"X2={chi:0.3f}, df={df}, p={p:0.6f}"
    else:
        raise TypeError('format_ch2 expects either a 3+ length tuple, or explicit parameters for chi, df, and p')


def describe(frame: pd.DataFrame,
             func: Iterable = ('count', 'sum', 'mean', 'median', 'min', 'max', 'quantile',
                               'std', 'var', 'sem', 'skew', 'kurt'),
             numeric_only: bool = True, **kwargs) -> pd.DataFrame:
    """
    Calls _descirbe() with usable output whether input dataframe is a pandas
    DataFrame or a GroupBy object. Also provides a full set of descriptive stats.
    :param frame: pandas DataFrame to describe.
    :param func: sequence of descriptive statistics to calculate for each variable.
    :param numeric_only: If True, only describe numeric variables
    :param kwargs: Any other args you might pass to the pandas describe method.
    :return: pandas DataFrame containing descriptive stat report
    """

    def nullcount(ser):
        return ser.isnull().sum()

    def _describe(frame: pd.DataFrame, func: Iterable, numeric_only: bool, **kwargs) -> pd.DataFrame:
        """Custom pandas DataFrame describe method that adds null counter."""

        if numeric_only:
            frame = frame.select_dtypes(include=np.number)
        return frame.agg(list(func) + [nullcount], **kwargs)

    if type(frame) is pd.core.groupby.groupby.DataFrameGroupBy:
        dfs = list()
        for group, group_dat in frame:
            df = _describe(frame=group_dat, func=func, numeric_only=numeric_only, **kwargs)
            df['descrgrp'] = group
            dfs.append(df)
        res_df = pd.concat(dfs)
        res_df = res_df[['descrgrp'] + list(res_df.columns[:-1])]
    else:
        res_df = _describe(frame=frame, func=func, numeric_only=numeric_only, **kwargs)

    return res_df


def p_desc2(p: float, levels: Sequence[float] = (.001, .01, .05, .06)) -> str:
    """
    Returns a standard signifier of the passed p-value
    :param p: Numeric statistical p-value
    :param levels: sequence of floats representing the cutoffs for the
           '***', '**', '*', and '.' statistical level signifiers.
           Default is (.001, .01, .05, .06).
    :return: str indication of statistical significance level
    """
    assert isinstance(levels, Sequence), "MojoStat.Stat: levels must be a sequence"
    assert all([isinstance(x, float) for x in levels]), "MojoStat.Stat: levels must contain only floats"
    assert len(levels) == 4, "MojoStat.Stat: levels must contain exactly 4 floats."

    level = tuple(sorted(levels))
    if p < level[0]:
        return " ***"
    if p < level[1]:
        return " **"
    elif p < level[2]:
        return " *"
    elif p <= level[3]:
        return " ."
    else:
        return ""


# FIXME: Note: These are for t-tests, but not f-tests.
def d_desc(d: float, cutoffs:Sequence[float]=(.01, .2, .5, .8, 1.2)) -> str:
    """
    Returns a standard effect size label of the passed Cohen's d value
    :param d: Numeric effect size
    :param cutoffs: sequence of floats representing the cutoffs for the
           very small, small, medium, large, very large, and huge effect size levels.
           Default is (.01, .2, .5, .8, 1.2).
    :return: str label for effect size represented in argument
    """
    assert isinstance(cutoffs, Sequence), "MojoStat.Stat: cutoffs must be a sequence"
    assert all([isinstance(x, float) for x in cutoffs]), "MojoStat.Stat: cutoffs must contain only floats"
    assert len(cutoffs) == 5, "MojoStat.Stat: cutoffs must contain exactly 6 floats."

    cutoff = tuple(sorted(cutoffs))
    if d <= cutoff[0]:
        return 'very small'  # Sawilowsky, 2009
    elif d <= cutoff[1]:
        return 'small'  # Cohen, 1988
    elif d <= cutoff[2]:
        return 'medium'  # Cohen, 1988
    elif d <= cutoff[3]:
        return 'large'  # Cohen, 1988
    elif d <= cutoff[4]:
        return 'very large'  # Sawilowsky, 2009
    else:
        return 'huge'  # Sawilowsky, 2009


def d_from_data(distribution1: Sequence[Number], distribution2: Sequence[Number]) -> float:
    """
    Calculates a Cohen's d from 2 distributions of data
    :param distribution1: list-like, numerical
    :param distribution2: list-like, numerical
    :return: Cohen's d
    """
    # from: http://www.socscistatistics.com/effectsize/Default3.aspx

    l1, l2 = 0, 0
    try:
        l1, l2 = len(distribution1), len(distribution2)
    except:
        raise MojoStatsException('distribution1 and distribution2 must each be a Sequence of numbers')

    if l1 and l2:
        m1 = statistics.mean(distribution1)
        m2 = statistics.mean(distribution2)
        sd1 = statistics.stdev(distribution1)
        sd2 = statistics.stdev(distribution2)
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


# http://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
def corr_pvalue(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.dropna()._get_numeric_data()
    cols = numeric_df.columns
    mat = numeric_df.values

    arr = np.zeros((len(cols), len(cols)), dtype=object)

    for xi, x in enumerate(mat.T):
        for yi, y in enumerate(mat.T[xi:]):
            arr[xi, yi + xi] = list(map(lambda _: round(_, 4), scipy.stats.pearsonr(x, y)))[1]
            arr[yi + xi, xi] = arr[xi, yi + xi]

    return pd.DataFrame(arr, index=cols, columns=cols)


def table_pct(df: pd.DataFrame, colnames: Sequence[str], denom: Number):
    """
    Updater that creates new variables in a pandas data frame that represent existing variables as percentages when divided by demon.
    :param df: pandas DataFrame
    :param colnames: Sequence of strings representing existing variable names to process
    :param denom: denominator for creating percentage
    :return: NA - This is an updater
    """
    assert isinstance(df, pd.DataFrame), 'MojoStat.Stats: df must be type pandas.DataFrame'
    assert is_anystr_sequece(colnames), 'MojoStat.Stat: colnames must be a Sequence of strings'
    assert is_numeric(denom), 'MojoStat.Stat: denom must be a number'

    if isinstance(colnames, str):
        col_names = list(colnames)
    else:
        col_names = colnames
    for colname in col_names:
        # if colname in df.columns:
        df[f'{colname}_pct'] = df[colname] / denom * 100


# FIXME: what is this parameter type? pd.DataFrame?? If so, say so.
def make_corr_p_table(corr_table) -> str:
    p_table_html = corr_table.to_html().__str__()
    pat = re.compile(r'(<td>)(0\.\d+)(</td>)')
    matches = set(pat.findall(p_table_html))
    for match in matches:
        val = match[1]
        fval = float(val)
        if fval < .06:
            if fval < .05:
                p_table_html = p_table_html.replace(val, f'<b>{val}</b>')
            else:
                p_table_html = p_table_html.replace(val, '<b><em>{val}</em></b>')
    return p_table_html


def run_zprop(c: tuple) -> str:
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
def run_anova(df: pd.DataFrame, dv: str, factors: list, html: bool = False) -> tuple:
    formula = f'{dv} ~ {make_var_formula(factors, True)}'
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, type=2)
    extra = '<br>' if html else ''
    return formula + extra, table


def pvalues_from_tukeyhsd(table):
    return psturng(np.abs(table.meandiffs / table.std_pairs), len(table.groupsunique), table.df_total)


def anova_table_to_dataframe(table) -> pd.DataFrame:
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


def df_row_to_f_statement(df: pd.DataFrame, identifier: str) -> str:
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


# todo: x_name and y_name seem to do nothing!
# todo: Should put dv name in summary and oneliner!
# todo: Consider adding formula to comparison name
# todo: Allow list of alternatives!
def ttest(formula: str, data: pd.DataFrame, alternative: str = 'two-sided', usevar: str = 'unequal',
          x_name: str = 'Group1', y_name: str = 'Group2', html: bool = False, kind='ind') -> list:
    """
    Note: Currently, x_name and y_name seem to be ignored in some cases.
    """
    assert kind in ('ind', 'rel'), f"MojoStat.Stats: kind parameter must be either 'ind' or 'rel', not '{kind}"

    data_groups, comparisons = parse_equation(formula, data, min_groups=1)

    results = list()

    StatResult = namedtuple('StatResult', 'comparison result')

    for comparison in comparisons:
        x_name, y_name = comparison
        x_name2 = x_name if isinstance(x_name, str) else "|".join(x_name)
        y_name2 = y_name if isinstance(y_name, str) else "|".join(y_name)
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
    try:
        x_mean = statistics.mean(x)
        x_stdev = statistics.stdev(x)
        y_mean = statistics.mean(y)
        y_stdev = statistics.stdev(y)
    except Exception as e:
        raise MojoStatsException(f'x and y must each be a sequence of valid numbers:\n\t{e}')

    assert alternative in ('larger', 'smaller',
                           'two-sided'), f"MojoStat.Stat: alternative must be 'larger', 'smaller' or 'two-sided', not {alternative}"

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
        x_mean = statistics.mean(x)
        x_stdev = statistics.stdev(x)
        y_mean = statistics.mean(y)
        y_stdev = statistics.stdev(y)
    except Exception as e:
        raise MojoStatsException(f'x and y must each be a sequence of valid numbers:\n\t{e}')

    assert alternative in ('larger', 'smaller',
                           'two-sided'), f"MojoStat.Stat: alternative must be 'larger', 'smaller' or 'two-sided', not {alternative}"

    xy_diff = [xy[0] - xy[1] for xy in zip(x, y)]

    d = sm.stats.DescrStatsW(xy_diff)
    t_stat, p_value, deg_fred = d.ttest_mean(alternative=alternative)
    low_ci, hi_ci = d.tconfint_mean(alpha=0.05, alternative=alternative)

    sig_flag = p_desc2(p_value)
    test_name = "Related"
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

    data = pd.read_csv(data_source)

    title('testing table_pct()')
    table_pct(data, ('rt',), 1000)
    print(headtail(data))
    print()

    title('testing ttest')
    results = ttest(formula="rt ~ block", data=data,
                    alternative='two-sided', usevar='unequal',
                    x_name='Group1', y_name='Group2', html=False, kind='rel')
    pprint(results)
    print()
    for result in results:
        print(result.comparison)
        print(result.result.oneliner)

    results = ttest(formula="rt ~ block + category", data=data,
                    alternative='two-sided', usevar='unequal',
                    x_name='Group1', y_name='Group2', html=False, kind='ind')
    pprint(results)
    print()
    for result in results:
        print(result.comparison)
        print(result.result.oneliner)
        print('-----')

    quit()
