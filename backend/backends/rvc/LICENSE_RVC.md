# Vendored code attribution — RVC-Project

`synthesizer.py` in this directory is a vendored, inference-only subset of the
generator/network definitions from the **Retrieval-based Voice Conversion WebUI**
project, adapted from these upstream files:

- `infer/lib/infer_pack/models.py`
- `infer/lib/infer_pack/modules.py`
- `infer/lib/infer_pack/attentions.py`
- `infer/lib/infer_pack/commons.py`

Source: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

The tensor math and all `nn.Module` attribute/submodule names are preserved
verbatim so that community `.pth` checkpoints load without modification. Training-
only code (discriminators, losses, the training `forward` methods) and the
TorchScript `__prepare_scriptable__` helpers have been removed.

The upstream project is distributed under the MIT License, reproduced below.

---

MIT License

Copyright (c) 2023 liujing04
Copyright (c) 2023 源文雨
Copyright (c) 2023 Ftps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
