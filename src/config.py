"""Single source of truth for every experimental constant.

Nothing downstream may hard-code a seed, a path, or a model parameter. If a
number appears in the paper, it is either in this file or computed from it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _d in (PROC, RESULTS, RESULTS / "raw", FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

TRACES = {
    "conv": RAW / "AzureLLMInferenceTrace_conv_1week.csv",
    "code": RAW / "AzureLLMInferenceTrace_code_1week.csv",
}

# ---------------------------------------------------------------- workload ---
# The traces span one week. We evaluate on fixed-length windows rather than the
# full week: a week of arrivals is far more than is needed for stable tail
# statistics, and short windows let us report variability across windows instead
# of a single unrepeatable number.
WINDOW_MINUTES = 20
N_WINDOWS = 5           # one per seed; disjoint, drawn from busy periods
WARMUP_FRACTION = 0.1   # leading fraction discarded before measuring

# ----------------------------------------------------------- serving model ---
# Analytical roofline model of a continuous-batching (vLLM-style) engine with
# iteration-level scheduling. Parameters correspond to a Llama-2-13B-class model
# on a single A100-80GB. These are stated openly as model parameters, not as
# measurements from our own hardware, and every headline result is repeated
# under the sensitivity sweep below.
MODEL = dict(
    name="Llama-2-13B",
    n_layers=40,
    n_kv_heads=40,
    head_dim=128,
    dtype_bytes=2,
    gpu_memory_gb=80,
    weight_memory_gb=26.0,       # 13B params x 2 bytes
)

# Prefill cost is modelled as affine in prompt tokens; decode cost as affine in
# batch size. Coefficients are in milliseconds.
PERF = dict(
    prefill_ms_per_token=0.28,
    prefill_fixed_ms=8.0,
    decode_ms_base=12.0,
    decode_ms_per_request=0.42,
    max_batch_size=256,
)

# Multiplicative perturbations applied to PERF for the robustness sweep.
SENSITIVITY = [0.5, 0.75, 1.0, 1.5, 2.0]

# The published trace is fleet-aggregate traffic (~75-87 req/s), which is orders
# of magnitude above what one replica serves. We therefore rescale the arrival
# timeline so that offered load hits a target fraction of replica capacity,
# preserving the arrival process shape and the exact request mix. Reported
# utilisation is always the *measured* value, not this target.
TARGET_LOADS = [0.60, 0.75, 0.85, 0.92, 0.98]

# Requests simulated per run. Enough for stable P99s, small enough that the full
# matrix finishes in minutes rather than hours.
REQUESTS_PER_RUN = 20_000

# ---------------------------------------------------------------- SLO ------ #
# Time-to-first-token and time-between-tokens targets. Values in the range used
# by production serving literature.
SLO_TTFT_MS = 2000.0
SLO_TBT_MS = 100.0

# ------------------------------------------------------------- predictor ---- #
# Features available WITHOUT reading prompt content. This restriction is the
# whole point of the paper and is enforced in features.py.
CONTENT_BLIND_FEATURES = [
    "context_tokens",
    "log_context_tokens",
    "workload_is_code",
    "hour_of_day",
    "day_of_week",
    "recent_mean_output",
    "recent_p90_output",
    "recent_mean_context",
    "inter_arrival_ms",
    "recent_arrival_rate",
]

N_QUANTILE_BUCKETS = 4   # predictor emits a bucket, not a point estimate
TRAIN_FRACTION = 0.6     # chronological split; no shuffling across time
