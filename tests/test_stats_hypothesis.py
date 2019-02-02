import sys
import os
from statworkflow import stats
import pytest
import random
from faker import Faker
from typing import Union
from hypothesis import given, example
import hypothesis.strategies as st

"""
Uses hypothesis and pytest to discover and run the tests
https://www.youtube.com/watch?v=jvwfDdgg93E
https://hypothesis.readthedocs.io/en/master/quickstart.html
"""

fake = Faker()

REPS = 25

Number = Union[int, float]


@given(st.integers())
def test_is_numeric_integer(num):
    assert stats.is_numeric(num) is True


@given(st.floats())
def test_is_numeric_float(num):
    assert stats.is_numeric(num) is True


@given(st.text())
def test_is_numeric_numeric_text(num):
    assert isinstance(stats.is_numeric(num), bool)

@given(st.text())
def test_is_numeric_numeric_text(num):
    assert isinstance(stats.is_numeric(num), bool)

@given(st.lists(st.floats()))
def test_is_numeric_list(num):
    result = stats.is_numeric(num)
    assert isinstance(result, bool)

