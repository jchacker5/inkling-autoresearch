import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests use tiny models on the CPU so they run in CI (no Metal) and never compete with a training run.
mx.set_default_device(mx.cpu)
