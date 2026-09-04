# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ._flash_attention_compat import fix_transformers_flash_attention_packed_sequence_detection

# Transformers can misclassify Qwen3.5's rank-3 position ids as packed-sequence
# metadata and send invalid cu_seqlens to FlashAttention's varlen kernels. Patch
# the packed-sequence probe before importing model implementations, while keeping
# the normal FlashAttention fast path enabled.
fix_transformers_flash_attention_packed_sequence_detection()
del fix_transformers_flash_attention_packed_sequence_detection

# Studio's multi-GPU loader asks for an Unsloth planned map but does not expose
# device_map_planner_kwargs. Install opt-in environment overrides before model
# implementations import the resolver, so spawned Studio training workers inherit
# the same planner hints without changing default placement for anybody else.
from ._device_map_env import install_device_map_environment_overrides

install_device_map_environment_overrides()
del install_device_map_environment_overrides

from .llama import FastLlamaModel
from .loader import FastLanguageModel, FastVisionModel, FastTextModel, FastModel
from .mistral import FastMistralModel
from .qwen2 import FastQwen2Model
from .qwen3 import FastQwen3Model
from .qwen3_moe import FastQwen3MoeModel
from .granite import FastGraniteModel
from .sentence_transformer import FastSentenceTransformer

try:
    from .falcon_h1 import FastFalconH1Model
except:
    # falcon_h1 absent before transformers 4.53.0; skip
    pass
from .dpo import PatchDPOTrainer, PatchKTOTrainer
from ._utils import is_bfloat16_supported, is_vLLM_available, __version__
from .rl import PatchFastRL, vLLMSamplingParams
