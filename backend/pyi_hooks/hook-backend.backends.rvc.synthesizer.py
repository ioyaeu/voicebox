"""Bundle backend.backends.rvc.synthesizer with its .py source, not just .pyc.

synthesizer.py decorates ``fused_add_tanh_sigmoid_multiply`` with
``@torch.jit.script`` (and the generator exposes ``@torch.jit.export`` methods).
TorchScript compiles these lazily at first use via ``inspect.getsource``, which
needs the physical .py file on disk — with bytecode only, the frozen build
raises ``OSError: Can't get source for <fused_add_tanh_sigmoid_multiply>`` the
first time an RVC checkpoint is loaded. Same mechanism as the scipy/transformers
source hooks in this directory.
"""

module_collection_mode = "pyz+py"
