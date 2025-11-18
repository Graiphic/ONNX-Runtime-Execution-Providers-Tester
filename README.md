<div style="font-family:Arial, sans-serif; line-height:1.6; max-width:900px; margin:auto; padding:20px;">

<p align="center">
  <img src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" alt="ONNX Runtime Logo" width="320"/>
</p>

<h1>Welcome to the ONNX Runtime – Execution Provider Coverage Tester</h1>

<p>
  This open source initiative, led by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong>, provides 
  a detailed, real-world coverage map of ONNX operator support for each <strong>Execution Provider (EP)</strong> in 
  <strong><a href="https://github.com/microsoft/onnxruntime" target="_blank">ONNX Runtime</a></strong>.
</p>

<p>
  It is part of our broader effort to democratize AI deployment through 
  <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> — 
  an ONNX-native orchestration framework designed for engineers, researchers, and industrial use cases.
</p>

<h2>🎯 Project Objectives</h2>
<ul>
  <li>Systematically test and report ONNX operator coverage per Execution Provider.</li>
  <li>Deliver up-to-date insights to guide industrial and academic ONNX Runtime adoption.</li>
  <li>Help developers, maintainers, and hardware vendors prioritize missing or broken operator support.</li>
</ul>

<h2>🧪 What’s Tested</h2>
<ul>
  <li>Each ONNX operator is tested in isolation using a minimal single-node model.</li>
  <li>Status per operator: <code>SUCCESS</code>, <code>FALLBACK</code>, <code>FAIL</code>, <code>NOT TESTED</code>, <code>SKIPPED</code>, <code>UNKNOWN</code>.</li>
  <li>Per-EP datasets include logs, optimized models (when applicable), and a README.</li>
</ul>

<h2>📐 How’s Tested</h2>
<h3>Inference</h3>
<p>
  Each operator is tested with a minimal ONNX graph. For EPs like OpenVINO/TensorRT, a <em>complexification</em> pass can add a small chain
  of <code>Mul</code>/<code>And</code> nodes (type-dependent) to make the backend compile more of the graph and reveal actual EP coverage.
</p>
<h3>Training</h3>
<p>
  When ONNX Runtime Training is available, a trainable scalar <code>__train_C</code> is injected via a <code>Mul</code> on the first input of the tested node (initialized to 1.0).
  We generate artifacts (AdamW) and run a single optimization step with an MSE loss on the first output. Operators that complete this step are marked <strong>SUCCESS</strong>;
  explicitly skipped or unsupported patterns are <strong>SKIPPED</strong>; others are <strong>FAIL</strong>.
</p>

<p><em>For detailed results and EP lists, please navigate to the per-opset dashboards:</em></p>
<ul>
  <li><a href="./opset_20/" target="_blank">opset_20</a></li>
  <li><a href="./opset_22/" target="_blank">opset_22</a></li>
</ul>

<h2>🧭 Related Tools</h2>
<p>
  For a complementary and more aggregated perspective on backend compliance, we encourage you to also visit the official 
  <a href="https://onnx.ai/backend-scoreboard/" target="_blank"><strong>ONNX Backend Scoreboard</strong></a>.
</p>
<p>
  While the Scoreboard provides a high-level view of backend support based on ONNX's internal test suite, our initiative focuses 
  on operator-level validation and runtime behavior analysis — especially fallback detection — across Execution Providers. 
  Together, both efforts help build a clearer, more actionable picture of ONNX Runtime capabilities.
</p>

<h2>🤝 Maintainer</h2>
<p>
  This project is maintained by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong> 
  as part of the <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> initiative.
</p>
<p>
  We welcome collaboration, community feedback, and open contribution to make ONNX Runtime stronger and more widely adopted.
</p>

<p style="margin-top:20px;">
  📬 <strong>Contact:</strong> <a href="mailto:contact@graiphic.io">contact@graiphic.io</a><br>
  🌐 <strong>Website:</strong> <a href="https://graiphic.io/" target="_blank">graiphic.io</a><br>
  🧠 <strong>Learn more about SOTA:</strong> <a href="https://graiphic.io/download/" target="_blank">graiphic.io/download</a>
</p>

</div>