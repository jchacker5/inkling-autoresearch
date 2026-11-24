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


def test_no_position_shortcut_and_fixed_size():
    """Regression for the v1 shortcut: the first digit scored ~.5 at every hop. v2 must stay near 1/#roots."""
    sets = P.reasoning_eval_set()
    for h, probs in sets.items():
        assert all(len(p.split(";")) - 1 == P.TASK_VARS for p, _ in probs)
        assert P.position_floor(probs) < 0.35, (h, P.position_floor(probs))   # ~1/#roots; scoring is by quad


def test_root_digits_are_distinct():
    rng = np.random.default_rng(3)
    for hops in range(0, 11):
        prompt, _ = P.make_problem(rng, hops)
        digits = re.findall(r"\d", prompt)
        assert len(digits) == len(set(digits))


@pytest.mark.parametrize("k", [1, 2, 4])
def test_eval_prompt_follows_newline_and_aligns_to_chunks(k):
    """Every eval prompt follows a newline (as in training); for CHUNK=k the query sits in a completed chunk."""
    for probs in P.reasoning_eval_set().values():
        for p, _ in probs[:20]:
            e = P.encode_prompt(p, k)
            assert e.endswith(p.encode()) and e[-len(p) - 1:-len(p)] == b"\n"
            assert (len(e) - 1) % k == 0
            q = e[-3:-1]                       # "?v" must lie strictly before the answer position's chunk
            assert (len(e) - 3) // k < (len(e) - 1) // k


@pytest.mark.parametrize("hops", [0, 2, 6, 10])
def test_twins_share_statements_have_equal_depth_and_differ(hops):
    rng = np.random.default_rng(10 + hops)
    for _ in range(100):
        (pa, aa), (pb, ab) = P.make_pair(rng, hops)
        sa, qa = re.fullmatch(r"(.*);\?([a-z])=", pa).groups()
        sb, qb = re.fullmatch(r"(.*);\?([a-z])=", pb).groups()
        assert sa == sb and qa != qb and aa != ab
        look = resolve(sa)
        assert look(qa)[1] == look(qb)[1] == hops


def test_query_blind_rule_scores_zero_pairs():
    rng = np.random.default_rng(5)
    for h in (2, 6, 10):
        probs = P.twin_problems(rng, h, 200)
        pairs = list(zip(probs[0::2], probs[1::2]))
        first_digit = [re.findall(r"\d", a[0])[0] for a, _ in pairs]
        assert sum(fd == a[1] and fd == b[1] for fd, (a, b) in zip(first_digit, pairs)) == 0


@pytest.mark.parametrize("hops", [1, 2, 4, 8, 10])
def test_crossed_twins_swap_answers_and_change_two_bytes(hops):
    rng = np.random.default_rng(20 + hops)
    probs = P.eval_problems(rng, hops, 400)
    assert len(probs) % 4 == 0
    for k in range(0, len(probs), 4):
        (pa, aa), (pb, ab), (ca, xa), (cb, xb) = probs[k:k + 4]
        assert (xa, xb) == (ab, aa) and aa != ab               # crossing exchanges the two answers
        assert len(pa) == len(ca) and sum(u != v for u, v in zip(pa, ca)) == 2
        sa = re.fullmatch(r"(.*);\?([a-z])=", ca).group(1)
        look = resolve(sa)
        assert str(look(re.fullmatch(r".*;\?([a-z])=", ca).group(1))[0]) == xa
        seen = set()                                           # crossed statements still define before use
        for st in sa.split(";"):
            v, y = st.split("=")
            assert y.isdigit() or y in seen
            seen.add(v)


def test_order_and_query_blind_rules_score_zero_quads():
    """Any rule that answers the original and crossed problems identically (it never reads the crossed hop)
    gets at most two of four right: e.g. 'answer the root that comes later for the later end'."""
    rng = np.random.default_rng(7)
    for h in (2, 4, 6, 8, 10):
        probs = P.eval_problems(rng, h, 400)
        for k in range(0, len(probs), 4):
            q = probs[k:k + 4]
            # a rule that ignores the crossed hop gives the crossed prompts the original answers
            rule = [q[0][1], q[1][1], q[0][1], q[1][1]]
            assert not all(r == a for r, (_, a) in zip(rule, q))
