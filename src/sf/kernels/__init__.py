"""Stage 3: GPU programming.  See ``_lang`` (what kernels call), ``_ir``
(the SF IR), ``_interp`` (host execution), ``_cgen`` (the C backend) and
``_runtime`` (targets, launch, autotune)."""
from ._lang import (arange, barrier, block_dim, block_idx, cdiv, constexpr, program_id,  # noqa: F401
                    load, predicate, shared, store, thread_idx, full, zeros)
from ._runtime import (HardwareAtreides, Kernel, LaunchResult, Limits, SimulatorAtreides,  # noqa: F401
                       autotune, jit, kernel, synchronize)
from ._sym import Ptr, Shared, Sym, is_traced  # noqa: F401
