# ONNX Runtime – Execution Provider Coverage Tester

<img src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" alt="ONNX Runtime Logo" width="500"/>

This open-source initiative, led by **[Graiphic](https://graiphic.io/)**, provides a comprehensive and practical coverage map of ONNX operator support for each **Execution Provider (EP)** in **[ONNX Runtime](https://github.com/microsoft/onnxruntime)**.

It is part of a broader effort to make AI deployment more accessible through **[SOTA](https://graiphic.io/download/)**, an ONNX-native orchestration framework designed for engineers, researchers, and industrial applications.

---

## Project Objectives

- Provide systematic and reproducible ONNX operator coverage per Execution Provider.
- Offer up-to-date insights for teams adopting ONNX Runtime in production or research settings.
- Help developers, maintainers, and hardware vendors prioritize missing or unstable operator implementations.

---

## What’s Tested

- Each ONNX operator is validated in isolation using a minimal single-node model.
- Status categories include: `SUCCESS`, `FALLBACK`, `FAIL`, `NOT TESTED`, `SKIPPED`, `UNKNOWN`.
- Per-EP datasets include logs, optimized models (when relevant), and a local README.

---

## How Testing Works

### Inference

Operators are tested using minimal ONNX graphs.  
For EPs such as OpenVINO or TensorRT, a graph *complexification* step may add a short chain of `Mul` or `And` nodes (depending on data type).  
This encourages the backend to compile more of the graph and exposes actual EP-level coverage rather than immediate fallbacks.

### Training

When ONNX Runtime Training is available:

- A trainable scalar `__train_C` is injected through a `Mul` on the first input (initialized to 1.0).
- A single optimization step (AdamW) is executed with an MSE loss on the first output.
- Operators successfully completing the step are marked **SUCCESS**.
- Known unsupported or intentionally excluded patterns are **SKIPPED**.
- All remaining cases are marked **FAIL**.

---

## Opset Dashboards

- [opset_20](./opset_20/)
- [opset_22](./opset_22/)

---

## Related Tools

For a complementary high-level view of backend compatibility, see the official  
**[ONNX Backend Scoreboard](https://onnx.ai/backend-scoreboard/)**.

This project complements the Scoreboard by providing operator-level behavior, fallback detection, and detailed EP execution insights.

---

## Maintainer

Maintained by **[Graiphic](https://graiphic.io/)** as part of the **[SOTA](https://graiphic.io/download/)** initiative.

Contributions and technical feedback are welcome.

---

### Contact

- **Email:** contact@graiphic.io  
- **Website:** https://graiphic.io  
- **Learn more about SOTA:** https://graiphic.io/download  
