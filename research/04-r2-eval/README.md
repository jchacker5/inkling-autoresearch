Debater 4, round 2: pure Python/numpy (no MLX, no model). Run from this folder with the repo venv:
    ../../.venv/bin/python audit_v2.py        # heuristic floors per hop (official set + 20k/hop), position oracle, filler
    ../../.venv/bin/python audit_v2b.py       # query-blind bare-path rule by lookup budget, decoy variant, training-format depth
    ../../.venv/bin/python pair_floor.py      # decoy-twin pair-accuracy floors;  pair_floor2.py: k lookups + position guess
    ../../.venv/bin/python sim_keep.py        # Monte Carlo of the current vs proposed keep/promotion rules;  sim_keep2.py: 3 vs 4 lanes
decoy_pairs.py is the proposed eval-only probe generator (make_pair, probe_sets, prompts). *.out are the outputs quoted in 04-debate-r2-evaluation.md.
