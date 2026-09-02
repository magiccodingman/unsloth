// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

const CHECKPOINT = "/mnt/world8/AI/Models/Qwen3.8-27B-Quark-AWQ-MXFP4-amd/";

test("the packed Quark training opt-in admits the exact local checkpoint", () => {
  for (const quantMethod of [undefined, "quark"] as const) {
    const support = classifyUnslothSupport({
      modelId: CHECKPOINT,
      quantMethod,
      deviceType: "cuda",
      allowExperimentalPackedQuarkTraining: true,
    });
    assert.deepEqual(support, { status: "supported", reason: null });
  }
});

test("the packed Quark exception remains training-only and narrowly scoped", () => {
  assert.deepEqual(
    classifyUnslothSupport({
      modelId: CHECKPOINT,
      deviceType: "cuda",
    }),
    { status: "unsupported", reason: "Detected AWQ quantization." },
  );

  for (const candidate of [
    "owner/Qwen3.8-27B-Quark-AWQ-MXFP4-amd",
    "/models/Some-Other-Quark-AWQ-MXFP4-model",
  ]) {
    assert.equal(
      classifyUnslothSupport({
        modelId: candidate,
        quantMethod: "quark",
        deviceType: "cuda",
        allowExperimentalPackedQuarkTraining: true,
      }).status,
      "unsupported",
      candidate,
    );
  }

  assert.deepEqual(
    classifyUnslothSupport({
      modelId: CHECKPOINT,
      quantMethod: "awq",
      deviceType: "cuda",
      allowExperimentalPackedQuarkTraining: true,
    }),
    { status: "unsupported", reason: "Detected AWQ quantization." },
  );
});
