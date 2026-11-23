"""The synthetic reasoning task: answers must be correct and hop depth must be what it claims."""
import re

import numpy as np
import pytest

import prepare as P


def resolve(stmts):
    env = dict(s.split("=") for s in stmts.split(";"))
    def look(v, depth=0):
        x = env[v]
        return (int(x), depth) if x.isdigit() else look(x, depth + 1)
    return look


@pytest.mark.parametrize("hops", [0, 1, 2, 4, 6, 8, 10])
def test_problem_answer_and_depth(hops):
    rng = np.random.default_rng(hops)
    for _ in range(200):
        prompt, answer = P.make_problem(rng, hops)
        m = re.fullmatch(r"(.*);\?([a-z])=", prompt)
        assert m, prompt
        stmts, q = m.groups()
        value, depth = resolve(stmts)(q)
        assert str(value) == answer
        assert depth == hops


def test_variables_are_defined_before_use():
    rng = np.random.default_rng(0)
    for hops in range(0, 11):
        prompt, _ = P.make_problem(rng, hops)
        seen = set()
        for s in prompt.split(";")[:-1]:
            v, x = s.split("=")
            assert v not in seen
            assert x.isdigit() or x in seen
            seen.add(v)


def test_training_stream_answers_are_correct():
    data = P._reasoning_stream(seed=1, nbytes=20_000).tobytes().decode()
    lines = data.split("\n")[:-1]          # the last line is truncated
    assert len(lines) > 20
    for line in lines:
        stmts, queries = line.split(";?", 1)
        look = resolve(stmts)
        for q in ("?" + queries).split("?")[1:]:
            v, a = q.split("=")
            assert str(look(v)[0]) == a


def test_eval_set_is_fixed_and_disjoint_in_seed():
    a, b = P.reasoning_eval_set(), P.reasoning_eval_set()
    assert a == b
    assert set(a) == set(P.EVAL_HOPS_ID + P.EVAL_HOPS_OOD)
