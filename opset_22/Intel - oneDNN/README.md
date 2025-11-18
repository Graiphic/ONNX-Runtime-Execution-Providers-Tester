# ONNXRuntime Test Results — Provider: `DnnlExecutionProvider`

**Test Date:** 2025-11-18 14:39:59

## Test Methodology

### Inference
Each ONNX operator is tested individually using a minimal ONNX model containing only that specific node. This ensures a focused and isolated evaluation of operator support for the selected Execution Provider.

### Training
To validate training (backward) support with ONNX Runtime Training, we **inject a `Mul` node** just before the tested operator: the **first input** of the tested node is multiplied by a **trainable scalar** `__train_C` (initialized to **1.0** so the forward values remain unchanged). We focus on the first input because it generally carries the data flow; for symmetric binary ops (e.g., `Add`), if training works on the first path it usually works on the others as well.

We then generate ONNX Runtime **training artifacts** (AdamW), run an inference once to **patch output shapes** if needed, feed a **target equal to the model’s own output** (MSE loss on the first output), and perform **one optimization step**. A node is marked **SUCCESS** when this step completes; **NOT_TESTED** for explicitly skipped ops (e.g., some recurrent ops like GRU/LSTM) or unsupported input types for this method; otherwise it is **FAIL**.

### Test Configuration

- **ONNX Opset version:** 22
- **ONNX IR version:** 10
- **Data types:** Only one type is tested per node. This is usually `float32`, unless the node does not support it — in which case a compatible type is selected.

> **Note:** Some ONNX nodes may not be available on the selected Execution Provider (EP) for opset version 22. This can lead to fallback behavior even though these nodes were supported in earlier opset versions. This occurs because ONNX Runtime teams may not have implemented or updated certain operators for the latest opset. As a result, test outcomes can vary depending on both the ONNX opset version and the ONNX Runtime version used.

## Environment and Installation Details

- **ONNX version:** 1.18.0
- **ONNXRuntime version:** 1.23.0
- **Target provider:** DnnlExecutionProvider
- **Installation command:**
```bash
manual build from source (oneDNN included, no pre-install needed)
```
### Hardware and Software Versions

- **CPU:** Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

## Basic ONNX Nodes

| ONNX Node | Status |
|:---------:|:------:|
| [`Add`](https://onnx.ai/onnx/operators/onnx__Add.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`AffineGrid`](https://onnx.ai/onnx/operators/onnx__AffineGrid.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`And`](https://onnx.ai/onnx/operators/onnx__And.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ArgMax`](https://onnx.ai/onnx/operators/onnx__ArgMax.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ArgMin`](https://onnx.ai/onnx/operators/onnx__ArgMin.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`AveragePool`](https://onnx.ai/onnx/operators/onnx__AveragePool.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`BatchNormalization`](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Bernoulli`](https://onnx.ai/onnx/operators/onnx__Bernoulli.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`BitShift`](https://onnx.ai/onnx/operators/onnx__BitShift.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BitwiseAnd`](https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BitwiseNot`](https://onnx.ai/onnx/operators/onnx__BitwiseNot.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BitwiseOr`](https://onnx.ai/onnx/operators/onnx__BitwiseOr.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BitwiseXor`](https://onnx.ai/onnx/operators/onnx__BitwiseXor.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BlackmanWindow`](https://onnx.ai/onnx/operators/onnx__BlackmanWindow.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Cast`](https://onnx.ai/onnx/operators/onnx__Cast.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`CastLike`](https://onnx.ai/onnx/operators/onnx__CastLike.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`CenterCropPad`](https://onnx.ai/onnx/operators/onnx__CenterCropPad.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Clip`](https://onnx.ai/onnx/operators/onnx__Clip.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Col2Im`](https://onnx.ai/onnx/operators/onnx__Col2Im.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Compress`](https://onnx.ai/onnx/operators/onnx__Compress.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Concat`](https://onnx.ai/onnx/operators/onnx__Concat.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ConcatFromSequence`](https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Constant`](https://onnx.ai/onnx/operators/onnx__Constant.html) | ![UNKNOWN](https://img.shields.io/badge/UNKNOWN-AAAAAA?style=flat&logoColor=white) |
| [`ConstantOfShape`](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Conv`](https://onnx.ai/onnx/operators/onnx__Conv.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ConvInteger`](https://onnx.ai/onnx/operators/onnx__ConvInteger.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ConvTranspose`](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`CumSum`](https://onnx.ai/onnx/operators/onnx__CumSum.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`DFT`](https://onnx.ai/onnx/operators/onnx__DFT.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`DeformConv`](https://onnx.ai/onnx/operators/onnx__DeformConv.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`DepthToSpace`](https://onnx.ai/onnx/operators/onnx__DepthToSpace.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`DequantizeLinear`](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Det`](https://onnx.ai/onnx/operators/onnx__Det.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Div`](https://onnx.ai/onnx/operators/onnx__Div.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Dropout`](https://onnx.ai/onnx/operators/onnx__Dropout.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`DynamicQuantizeLinear`](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Einsum`](https://onnx.ai/onnx/operators/onnx__Einsum.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Equal`](https://onnx.ai/onnx/operators/onnx__Equal.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Expand`](https://onnx.ai/onnx/operators/onnx__Expand.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`EyeLike`](https://onnx.ai/onnx/operators/onnx__EyeLike.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Flatten`](https://onnx.ai/onnx/operators/onnx__Flatten.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GRU`](https://onnx.ai/onnx/operators/onnx__GRU.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Gather`](https://onnx.ai/onnx/operators/onnx__Gather.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GatherElements`](https://onnx.ai/onnx/operators/onnx__GatherElements.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GatherND`](https://onnx.ai/onnx/operators/onnx__GatherND.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Gelu`](https://onnx.ai/onnx/operators/onnx__Gelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Gemm`](https://onnx.ai/onnx/operators/onnx__Gemm.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GlobalAveragePool`](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GlobalMaxPool`](https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Greater`](https://onnx.ai/onnx/operators/onnx__Greater.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GreaterOrEqual`](https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GridSample`](https://onnx.ai/onnx/operators/onnx__GridSample.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GroupNormalization`](https://onnx.ai/onnx/operators/onnx__GroupNormalization.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`HammingWindow`](https://onnx.ai/onnx/operators/onnx__HammingWindow.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`HannWindow`](https://onnx.ai/onnx/operators/onnx__HannWindow.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`HardSigmoid`](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`HardSwish`](https://onnx.ai/onnx/operators/onnx__HardSwish.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Hardmax`](https://onnx.ai/onnx/operators/onnx__Hardmax.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Identity`](https://onnx.ai/onnx/operators/onnx__Identity.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`If`](https://onnx.ai/onnx/operators/onnx__If.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ImageDecoder`](https://onnx.ai/onnx/operators/onnx__ImageDecoder.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`InstanceNormalization`](https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`IsInf`](https://onnx.ai/onnx/operators/onnx__IsInf.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`IsNaN`](https://onnx.ai/onnx/operators/onnx__IsNaN.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`LRN`](https://onnx.ai/onnx/operators/onnx__LRN.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`LSTM`](https://onnx.ai/onnx/operators/onnx__LSTM.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`LayerNormalization`](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Less`](https://onnx.ai/onnx/operators/onnx__Less.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`LessOrEqual`](https://onnx.ai/onnx/operators/onnx__LessOrEqual.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Loop`](https://onnx.ai/onnx/operators/onnx__Loop.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`LpNormalization`](https://onnx.ai/onnx/operators/onnx__LpNormalization.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`LpPool`](https://onnx.ai/onnx/operators/onnx__LpPool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MatMul`](https://onnx.ai/onnx/operators/onnx__MatMul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`MatMulInteger`](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Max`](https://onnx.ai/onnx/operators/onnx__Max.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MaxPool`](https://onnx.ai/onnx/operators/onnx__MaxPool.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`MaxRoiPool`](https://onnx.ai/onnx/operators/onnx__MaxRoiPool.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`MaxUnpool`](https://onnx.ai/onnx/operators/onnx__MaxUnpool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Mean`](https://onnx.ai/onnx/operators/onnx__Mean.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MeanVarianceNormalization`](https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MelWeightMatrix`](https://onnx.ai/onnx/operators/onnx__MelWeightMatrix.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Min`](https://onnx.ai/onnx/operators/onnx__Min.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Mod`](https://onnx.ai/onnx/operators/onnx__Mod.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Mul`](https://onnx.ai/onnx/operators/onnx__Mul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Multinomial`](https://onnx.ai/onnx/operators/onnx__Multinomial.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`NegativeLogLikelihoodLoss`](https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`NonMaxSuppression`](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`NonZero`](https://onnx.ai/onnx/operators/onnx__NonZero.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Not`](https://onnx.ai/onnx/operators/onnx__Not.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Not`](https://onnx.ai/onnx/operators/onnx__Not.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`OneHot`](https://onnx.ai/onnx/operators/onnx__OneHot.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Optional`](https://onnx.ai/onnx/operators/onnx__Optional.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`OptionalGetElement`](https://onnx.ai/onnx/operators/onnx__OptionalGetElement.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`OptionalHasElement`](https://onnx.ai/onnx/operators/onnx__OptionalHasElement.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Or`](https://onnx.ai/onnx/operators/onnx__Or.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`PRelu`](https://onnx.ai/onnx/operators/onnx__PRelu.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Pad`](https://onnx.ai/onnx/operators/onnx__Pad.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Pow`](https://onnx.ai/onnx/operators/onnx__Pow.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`QLinearConv`](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`QLinearMatMul`](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`QuantizeLinear`](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`RNN`](https://onnx.ai/onnx/operators/onnx__RNN.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`RandomNormal`](https://onnx.ai/onnx/operators/onnx__RandomNormal.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RandomNormalLike`](https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RandomUniform`](https://onnx.ai/onnx/operators/onnx__RandomUniform.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RandomUniformLike`](https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Range`](https://onnx.ai/onnx/operators/onnx__Range.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ReduceL1`](https://onnx.ai/onnx/operators/onnx__ReduceL1.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceL2`](https://onnx.ai/onnx/operators/onnx__ReduceL2.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceLogSum`](https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceLogSumExp`](https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMax`](https://onnx.ai/onnx/operators/onnx__ReduceMax.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMean`](https://onnx.ai/onnx/operators/onnx__ReduceMean.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMin`](https://onnx.ai/onnx/operators/onnx__ReduceMin.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceProd`](https://onnx.ai/onnx/operators/onnx__ReduceProd.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceSum`](https://onnx.ai/onnx/operators/onnx__ReduceSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceSumSquare`](https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`RegexFullMatch`](https://onnx.ai/onnx/operators/onnx__RegexFullMatch.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Reshape`](https://onnx.ai/onnx/operators/onnx__Reshape.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Resize`](https://onnx.ai/onnx/operators/onnx__Resize.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ReverseSequence`](https://onnx.ai/onnx/operators/onnx__ReverseSequence.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`RoiAlign`](https://onnx.ai/onnx/operators/onnx__RoiAlign.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`STFT`](https://onnx.ai/onnx/operators/onnx__STFT.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Scan`](https://onnx.ai/onnx/operators/onnx__Scan.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ScatterElements`](https://onnx.ai/onnx/operators/onnx__ScatterElements.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ScatterND`](https://onnx.ai/onnx/operators/onnx__ScatterND.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceAt`](https://onnx.ai/onnx/operators/onnx__SequenceAt.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceConstruct`](https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceEmpty`](https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceErase`](https://onnx.ai/onnx/operators/onnx__SequenceErase.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceInsert`](https://onnx.ai/onnx/operators/onnx__SequenceInsert.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceLength`](https://onnx.ai/onnx/operators/onnx__SequenceLength.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceMap`](https://onnx.ai/onnx/operators/onnx__SequenceMap.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Shape`](https://onnx.ai/onnx/operators/onnx__Shape.html) | ![UNKNOWN](https://img.shields.io/badge/UNKNOWN-AAAAAA?style=flat&logoColor=white) |
| [`Size`](https://onnx.ai/onnx/operators/onnx__Size.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Slice`](https://onnx.ai/onnx/operators/onnx__Slice.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SoftmaxCrossEntropyLoss`](https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SpaceToDepth`](https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Split`](https://onnx.ai/onnx/operators/onnx__Split.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SplitToSequence`](https://onnx.ai/onnx/operators/onnx__SplitToSequence.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Squeeze`](https://onnx.ai/onnx/operators/onnx__Squeeze.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`StringConcat`](https://onnx.ai/onnx/operators/onnx__StringConcat.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`StringNormalizer`](https://onnx.ai/onnx/operators/onnx__StringNormalizer.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`StringSplit`](https://onnx.ai/onnx/operators/onnx__StringSplit.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Sub`](https://onnx.ai/onnx/operators/onnx__Sub.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Sum`](https://onnx.ai/onnx/operators/onnx__Sum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`TfIdfVectorizer`](https://onnx.ai/onnx/operators/onnx__TfIdfVectorizer.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Tile`](https://onnx.ai/onnx/operators/onnx__Tile.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`TopK`](https://onnx.ai/onnx/operators/onnx__TopK.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Transpose`](https://onnx.ai/onnx/operators/onnx__Transpose.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Trilu`](https://onnx.ai/onnx/operators/onnx__Trilu.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Unique`](https://onnx.ai/onnx/operators/onnx__Unique.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Unsqueeze`](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Where`](https://onnx.ai/onnx/operators/onnx__Where.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Xor`](https://onnx.ai/onnx/operators/onnx__Xor.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |

### Inference Summary
- **Total nodes tested:** 155
- **Executable directly (SUCCESS):** 37 (23.9%)
- **Executable directly (SUCCESS with complexification):** 0 (0.0%)
- **Executable via FALLBACK:** 105 (67.7%)
- **UNKNOWN (no Node event):** 2 (1.3%)
- **NOT TESTED:** 0 (0.0%)
- **SKIPPED:** 0 (0.0%)
- **FAIL:** 11 (7.1%)

![Pie Chart](./stats_DnnlExecutionProvider_basic.png)

## Microsoft Custom Nodes

| ONNX Node | Status |
|:---------:|:------:|
| [`com.microsoft.Attention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Attention) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.BeamSearch`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BeamSearch) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.BiasAdd`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasAdd) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BiasDropout`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasDropout) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BiasGelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasGelu) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.BiasSoftmax`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasSoftmax) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BiasSplitGelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasSplitGelu) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BifurcationDetector`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BifurcationDetector) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.BitmaskBiasDropout`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BitmaskBiasDropout) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BitmaskDropout`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BitmaskDropout) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.CDist`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.CDist) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.ComplexMul`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.ComplexMul) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.ComplexMulConj`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.ComplexMulConj) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.ConvTransposeWithDynamicPads`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.ConvTransposeWithDynamicPads) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.CropAndResize`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.CropAndResize) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.DecoderAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DecoderAttention) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DecoderMaskedMultiHeadAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DecoderMaskedMultiHeadAttention) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.DecoderMaskedSelfAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DecoderMaskedSelfAttention) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DequantizeBFP`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DequantizeBFP) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DequantizeLinear`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DequantizeLinear) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.DequantizeWithOrder`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DequantizeWithOrder) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DynamicQuantizeLSTM`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DynamicQuantizeLSTM) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.DynamicQuantizeMatMul`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DynamicQuantizeMatMul) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.DynamicTimeWarping`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DynamicTimeWarping) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.EmbedLayerNormalization`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.ExpandDims`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.ExpandDims) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.FastGelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FastGelu) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.FusedConv`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedConv) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.FusedGemm`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedGemm) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.FusedMatMul`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.FusedMatMulActivation`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMulActivation) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) |
| [`com.microsoft.GatedRelativePositionBias`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatedRelativePositionBias) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GatherBlockQuantized`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatherBlockQuantized) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) |
| [`com.microsoft.GatherND`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatherND) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Gelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Gelu) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.GemmFastGelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GemmFastGelu) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GemmFloat8`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GemmFloat8) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GemmaRotaryEmbedding`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GemmaRotaryEmbedding) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GreedySearch`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GreedySearch) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) |
| [`com.microsoft.GridSample`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GridSample) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.GroupNorm`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupNorm) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GroupQueryAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Inverse`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Inverse) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Irfft`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Irfft) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.LongformerAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.LongformerAttention) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MatMulBnb4`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulBnb4) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.MatMulFpQ4`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulFpQ4) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MatMulInteger16`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulInteger16) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.MatMulIntegerToFloat`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulIntegerToFloat) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.MatMulNBits`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.MaxpoolWithMask`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MaxpoolWithMask) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.MoE`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MoE) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MulInteger`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MulInteger) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MultiHeadAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MultiHeadAttention) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.MurmurHash3`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MurmurHash3) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.NGramRepeatBlock`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.NGramRepeatBlock) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.NhwcConv`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.NhwcConv) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.NhwcFusedConv`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.NhwcFusedConv) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) |
| [`com.microsoft.NhwcMaxPool`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.NhwcMaxPool) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.PackedAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.PackedAttention) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.PackedMultiHeadAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.PackedMultiHeadAttention) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Pad`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Pad) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QAttention) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QGemm`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QGemm) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearAdd`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAdd) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearAveragePool`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAveragePool) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearConcat`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearConcat) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearConv`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearConv) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearGlobalAveragePool`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearGlobalAveragePool) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearLeakyRelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearLeakyRelu) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearMul`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearMul) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearReduceMean`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearReduceMean) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QLinearSigmoid`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSigmoid) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearSoftmax`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSoftmax) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QLinearWhere`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearWhere) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QMoE`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QMoE) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QOrderedAttention) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedGelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QOrderedGelu) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedLayerNormalization`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QOrderedLayerNormalization) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedLongformerAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QOrderedLongformerAttention) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedMatMul`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QOrderedMatMul) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) |
| [`com.microsoft.QuantizeBFP`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QuantizeBFP) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QuantizeLinear`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QuantizeLinear) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.QuantizeWithOrder`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QuantizeWithOrder) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QuickGelu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QuickGelu) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Range`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Range) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.ReduceSumInteger`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.ReduceSumInteger) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.RelativePositionBias`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RelativePositionBias) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.RemovePadding`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RemovePadding) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.RestorePadding`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RestorePadding) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Rfft`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Rfft) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.RotaryEmbedding`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RotaryEmbedding) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.SampleOp`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SampleOp) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Sampling`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Sampling) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.SkipGroupNorm`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipGroupNorm) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.SkipLayerNormalization`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipLayerNormalization) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.SkipSimplifiedLayerNormalization`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipSimplifiedLayerNormalization) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Snpe`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Snpe) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.SparseAttention`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SparseAttention) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.SparseToDenseMatMul`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SparseToDenseMatMul) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Tokenizer`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Tokenizer) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.TorchEmbedding`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.TorchEmbedding) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.TransposeMatMul`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.TransposeMatMul) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Trilu`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Trilu) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.UnfoldTensor`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.UnfoldTensor) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`com.microsoft.Unique`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Unique) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.WhisperBeamSearch`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.WhisperBeamSearch) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) |

### Inference Summary
- **Total nodes tested:** 107
- **Executable directly (SUCCESS):** 6 (5.6%)
- **Executable directly (SUCCESS with complexification):** 0 (0.0%)
- **Executable via FALLBACK:** 52 (48.6%)
- **UNKNOWN (no Node event):** 0 (0.0%)
- **NOT TESTED:** 7 (6.5%)
- **SKIPPED:** 0 (0.0%)
- **FAIL:** 42 (39.3%)

![Pie Chart](./stats_DnnlExecutionProvider_ms.png)

## Nodes not tested

These nodes couldn't be tested due to lack of valid minimal ONNX model.

`com.microsoft.FusedMatMulActivation`, `com.microsoft.GatherBlockQuantized`, `com.microsoft.GreedySearch`, `com.microsoft.NhwcFusedConv`, `com.microsoft.QOrderedAttention`, `com.microsoft.QOrderedMatMul`, `com.microsoft.WhisperBeamSearch`

## README Generation

This file was generated automatically by `report.py`.

- Generated ONNX models: `models/<provider>/`
- Profiling JSON files: `profiling/<provider>/`
- Scripts: `main.py`, `report.py`, `utils.py`, `ops/*`
_End of README_