import sys
import os
from statworkflow import stats
import pytest
import random
from faker import Faker
import pandas as pd
from numpy import nan

"""
Uses pytest
"""

fake = Faker()

REPS = 25

data_categorical = {
    'subid': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107, 8: 108, 9: 109, 10: 110, 11: 111, 12: 112,
              13: 113, 14: 114, 15: 115, 16: 116, 17: 117, 18: 118, 19: 119, 20: 120, 21: 121, 22: 122, 23: 123,
              24: 124, 25: 125, 26: 126, 27: 127, 28: 128},
    'category': {0: 'Young', 1: 'Young', 2: 'Young', 3: 'Young', 4: 'Young', 5: 'Young', 6: 'Young', 7: 'Young',
                 8: 'Young', 9: 'Young', 10: 'Young', 11: 'Young', 12: 'Young', 13: 'Young', 14: 'Old', 15: 'Old',
                 16: 'Old', 17: 'Old', 18: 'Old', 19: 'Old', 20: 'Old', 21: 'Old', 22: 'Old', 23: 'Old', 24: 'Old',
                 25: 'Old', 26: 'Old', 27: 'Old', 28: 'Old'},
    'nada1': {0: nan, 1: nan, 2: nan, 3: nan, 4: nan, 5: nan, 6: nan, 7: nan, 8: nan, 9: nan, 10: nan, 11: nan, 12: nan,
              13: nan, 14: nan, 15: nan, 16: nan, 17: nan, 18: nan, 19: nan, 20: nan, 21: nan, 22: nan, 23: nan,
              24: nan,
              25: nan, 26: nan, 27: nan, 28: nan},
    'block': {0: 'Practice', 1: 'Practice', 2: 'Practice', 3: 'Practice', 4: 'Practice', 5: 'Practice', 6: 'Practice',
              7: 'Test', 8: 'Test', 9: 'Test', 10: 'Test', 11: 'Test', 12: 'Test', 13: 'Test', 14: 'Practice',
              15: 'Practice', 16: 'Practice', 17: 'Practice', 18: 'Practice', 19: 'Practice', 20: 'Practice',
              21: 'Practice', 22: 'Test', 23: 'Test', 24: 'Test', 25: 'Test', 26: 'Test', 27: 'Test', 28: 'Test'},
    'rt': {0: 911, 1: 439, 2: 718, 3: 431, 4: 275, 5: 797, 6: 777, 7: 553, 8: 558, 9: 671, 10: 300, 11: 287, 12: 657,
           13: 342, 14: 1094, 15: 1188, 16: 1301, 17: 1146, 18: 1246, 19: 1458, 20: 1455, 21: 1542, 22: 1623, 23: 1464,
           24: 1465, 25: 1443, 26: 1498, 27: 1655, 28: 1384},
    'nada2': {0: nan, 1: nan, 2: nan, 3: nan, 4: nan, 5: nan, 6: nan, 7: nan, 8: nan, 9: nan, 10: nan, 11: nan, 12: nan,
              13: nan, 14: nan, 15: nan, 16: nan, 17: nan, 18: nan, 19: nan, 20: nan, 21: nan, 22: nan, 23: nan,
              24: nan, 25: nan, 26: nan, 27: nan, 28: nan}}

data_numeric = {
    'subid': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107, 8: 108, 9: 109, 10: 110, 11: 111, 12: 112,
              13: 113, 14: 114, 15: 115, 16: 116, 17: 117, 18: 118, 19: 119, 20: 120, 21: 121, 22: 122, 23: 123,
              24: 124, 25: 125, 26: 126, 27: 127, 28: 128},
    'Test1': {0: 249, 1: 220, 2: 125, 3: 491, 4: 184, 5: 406, 6: 417, 7: 295, 8: 291, 9: 133, 10: 347, 11: 348, 12: 290,
              13: 178, 14: 117, 15: 156, 16: 300, 17: 148, 18: 307, 19: 272, 20: 165, 21: 464, 22: 151, 23: 288,
              24: 172, 25: 239, 26: 389, 27: 471, 28: 284},
    'Test2': {0: 332, 1: 324, 2: 471, 3: 222, 4: 478, 5: 432, 6: 378, 7: 451, 8: 479, 9: 305, 10: 102, 11: 321, 12: 310,
              13: 252, 14: 301, 15: 277, 16: 259, 17: 357, 18: 489, 19: 379, 20: 128, 21: 262, 22: 411, 23: 148,
              24: 339, 25: 396, 26: 291, 27: 454, 28: 375},
    'Test3': {0: 1152, 1: 1427, 2: 1340, 3: 1342, 4: 1278, 5: 1007, 6: 993, 7: 1481, 8: 899, 9: 1129, 10: 874, 11: 938,
              12: 1004, 13: 1055, 14: 1250, 15: 1374, 16: 955, 17: 1131, 18: 1184, 19: 925, 20: 829, 21: 1355, 22: 1086,
              23: 1002, 24: 1416, 25: 1274, 26: 904, 27: 1220, 28: 822}}


# def logAssert(test,msg):
#     if not test:
#         logging.error(msg)
#         assert test,msg

class TestIsNumericClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_numeric(self):
        assert stats.is_numeric(fake.pyint()) is True
        assert stats.is_numeric(fake.pyfloat()) is True
        assert stats.is_numeric(str(fake.pyfloat())) is True
        assert stats.is_numeric(fake.pybool()) is True

    def test_non_numeric(self):
        assert stats.is_numeric(fake.name()) is False
        assert stats.is_numeric(fake.ipv4(network=False)) is False
        assert stats.is_numeric(fake.pyiterable(2)) is False


class TestSqrClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_numeric(self):
        i, f, b = fake.pyint(), fake.pyfloat(), fake.pybool()
        assert stats.sqr(i) == i * i
        assert stats.sqr(f) == f * f
        assert stats.sqr(b) == b * b

    def test_non_numeric(self):
        assert stats.sqr(fake.name()) is None
        assert stats.sqr(fake.ipv4(network=False)) is None
        assert stats.sqr(fake.pyiterable(2)) is None


class TestPDesc2Class:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_p_values(self):
        # assert all([stats.p_desc2(p=random.uniform(.0001, .0009)) == " ***" for _ in range(REPS)]) is True
        # assert all([stats.p_desc2(p=random.uniform(.001, .009)) == " **" for _ in range(REPS)]) is True
        # assert all([stats.p_desc2(p=random.uniform(.01, .049)) == " *" for _ in range(REPS)]) is True
        # assert all([stats.p_desc2(p=random.uniform(.05, .059)) == " ." for _ in range(REPS)]) is True
        # assert all([stats.p_desc2(p=random.uniform(.06, 1.0)) == "" for _ in range(REPS)]) is True

        assert all([stats.p_desc2(p=random.uniform(.0001, .001)) == " ***" for _ in range(REPS)]) is True
        assert all([stats.p_desc2(p=random.uniform(.001, .01)) == " **" for _ in range(REPS)]) is True
        assert all([stats.p_desc2(p=random.uniform(.01, .05)) == " *" for _ in range(REPS)]) is True
        assert all([stats.p_desc2(p=random.uniform(.05, .06)) == " ." for _ in range(REPS)]) is True
        assert all([stats.p_desc2(p=random.uniform(.06, 1.1)) == "" for _ in range(REPS)]) is True

    def test_non_p_values(self):
        with pytest.raises(TypeError, message='Expecting TypeError'):
            stats.p_desc2('.05')
        with pytest.raises(TypeError, message='Expecting TypeError'):
            stats.p_desc2(fake.pystr())


class TestPDescClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_p_values(self):
        # todo: These ranges do not always work! problem with using .uniform for this?
        assert all([stats.d_desc(d=random.uniform(.00, .01001)) == "very small" for _ in range(REPS)]) is True
        assert all([stats.d_desc(d=random.uniform(.01, .2001)) == "small" for _ in range(REPS)]) is True
        assert all([stats.d_desc(d=random.uniform(.20, .5001)) == "medium" for _ in range(REPS)]) is True
        assert all([stats.d_desc(d=random.uniform(.50, .8001)) == "large" for _ in range(REPS)]) is True
        assert all([stats.d_desc(d=random.uniform(.80, 1.2001)) == "very large" for _ in range(REPS)]) is True
        assert all([stats.d_desc(d=random.uniform(1.2, 10.0)) == "huge" for _ in range(REPS)]) is True

    def test_non_p_values(self):
        with pytest.raises(TypeError, message='Expecting TypeError'):
            stats.d_desc('.20')
        with pytest.raises(TypeError, message='Expecting TypeError'):
            stats.d_desc(fake.pystr())


class TestDFromDataClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_known_input(self):
        seq1 = [476, 382, 366, 399, 341, 397, 438, 481, 453, 352, 414, 400, 344, 324, 361, 449, 379, 499, 426, 369]
        seq2 = [490, 496, 454, 598, 411, 404, 503, 441, 476, 566, 580, 593, 408, 504, 539, 417, 523, 492, 409, 431]
        assert stats.d_from_data(distribution1=seq1, distribution2=seq2) == 1.4887230013907178

    def test_good_input(self):
        seq1 = fake.pylist(10, False, fake.pyint())
        seq2 = fake.pylist(10, False, fake.pyint())
        assert isinstance(stats.d_from_data(distribution1=seq1, distribution2=seq2), float)

    def test_bad_input(self):
        assert stats.d_from_data(1, 2) is None
        assert stats.d_from_data([23, 34, 34], []) is None
        with pytest.raises(TypeError, message='Expecting TypeError'):
            stats.d_from_data('476, 382, 366, 399, 341, 397'.split(','), '490, 496, 454, 598, 411, 404'.split(','))


class TestIntOrFloatClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_convertable(self):
        assert stats.int_or_float('1') == 1.0
        assert stats.int_or_float('2.2') == 2.2

    def test_non_convertable(self):
        assert stats.int_or_float('3secs') is nan
        assert stats.int_or_float(None) is nan
        assert stats.int_or_float(nan) is nan
        assert stats.int_or_float('') is nan


class TestCorrPValueClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_me(self):
        df = pd.DataFrame().from_dict(data_numeric)
        correct_result = {'Test1': {'Test1': 0.0, 'Test2': 0.82430000000000003, 'Test3': 0.43859999999999999,
                                    'subid': 0.81489999999999996},
                          'Test2': {'Test1': 0.82430000000000003, 'Test2': 0.0, 'Test3': 0.1404,
                                    'subid': 0.51880000000000004},
                          'Test3': {'Test1': 0.43859999999999999, 'Test2': 0.1404, 'Test3': 0.0, 'subid': 0.2034},
                          'subid': {'Test1': 0.81489999999999996, 'Test2': 0.51880000000000004, 'Test3': 0.2034,
                                    'subid': 0.0}}
        assert stats.corr_pvalue(df).to_dict() == correct_result


class TestDumpNanColsClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_good_input(self):
        df = pd.DataFrame().from_dict(data_categorical)
        assert all([vname in df for vname in ('nada1', 'nada2')]) is True
        stats.dump_nan_cols(df)
        assert all([vname in df for vname in ('nada1', 'nada2')]) is False

    def test_bad_input(self):
        with pytest.raises(Exception, message='Expecting exception due to invalid df argument'):
            stats.dump_nan_cols({'Test1': [1, 2, 3], 'Test2': [4, 5, 6], 'Test3': [nan, nan, nan]})


class TestGetVarNameClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_me(self):
        assert stats.get_var_name(self) == 'test_me'
        assert stats.get_var_name('nada') is None

# todo: better test

class TestTablePctClass:
    @classmethod
    def setup_class(cls):
        print(f'\n****Testing {cls.__name__}\n')

    def test_me(self):
        from functools import partial
        df = pd.DataFrame().from_dict(data_numeric)
        df_pct = stats.table_pct(df, colnames=['Test1', 'Test2'], denom=1000)
        t1 = stats.round_all(tuple(df_pct.head().Test1_pct))
        t2 = stats.round_all(tuple(df_pct.head().Test2_pct))
        print('>>>', t1)
        assert t1 == (24.9, 22.0, 12.5, 49.1, 18.40)
        assert t2 == (33.20, 32.40, 47.10, 22.20, 47.80)

        with pytest.raises(Exception, message='Expecting exception due to invalid df argument'):
            df_pct = stats.table_pct([1, 2, 3], colnames=['Test1', 'Test2'], denom=1000)

        df_pct = stats.table_pct(df, colnames='Test1', denom=1000)

# import pandas as pd
# from functools import partial
# def halve(x):
#     return x/3
# data_numeric = {
#     'subid': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107, 8: 108, 9: 109, 10: 110, 11: 111, 12: 112,
#               13: 113, 14: 114, 15: 115, 16: 116, 17: 117, 18: 118, 19: 119, 20: 120, 21: 121, 22: 122, 23: 123,
#               24: 124, 25: 125, 26: 126, 27: 127, 28: 128},
#     'Test1': {0: 249, 1: 220, 2: 125, 3: 491, 4: 184, 5: 406, 6: 417, 7: 295, 8: 291, 9: 133, 10: 347, 11: 348, 12: 290,
#               13: 178, 14: 117, 15: 156, 16: 300, 17: 148, 18: 307, 19: 272, 20: 165, 21: 464, 22: 151, 23: 288,
#               24: 172, 25: 239, 26: 389, 27: 471, 28: 284},
#     'Test2': {0: 332, 1: 324, 2: 471, 3: 222, 4: 478, 5: 432, 6: 378, 7: 451, 8: 479, 9: 305, 10: 102, 11: 321, 12: 310,
#               13: 252, 14: 301, 15: 277, 16: 259, 17: 357, 18: 489, 19: 379, 20: 128, 21: 262, 22: 411, 23: 148,
#               24: 339, 25: 396, 26: 291, 27: 454, 28: 375},
#     'Test3': {0: 1152, 1: 1427, 2: 1340, 3: 1342, 4: 1278, 5: 1007, 6: 993, 7: 1481, 8: 899, 9: 1129, 10: 874, 11: 938,
#               12: 1004, 13: 1055, 14: 1250, 15: 1374, 16: 955, 17: 1131, 18: 1184, 19: 925, 20: 829, 21: 1355, 22: 1086,
#               23: 1002, 24: 1416, 25: 1274, 26: 904, 27: 1220, 28: 822}}
# df = pd.DataFrame().from_dict(data_numeric)
# print(df.head())
# df = df.applymap(halve)
# df = df.applymap(partial(round, ndigits=2))
# print(df.head())
