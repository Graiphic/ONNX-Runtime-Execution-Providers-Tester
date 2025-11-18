<div style="font-family:Arial, sans-serif; line-height:1.6; max-width:900px; margin:auto; padding:20px;">

<p align="center">
  <img src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" alt="ONNX Runtime Logo" width="320"/>
</p>

<h1>ONNX Runtime — EP Coverage (Opset 22)</h1>

<h2>🧪 What’s Tested</h2>
<ul>
  <li>Each ONNX operator is tested in isolation across all available EPs.</li>
  <li>Status per operator: <code>SUCCESS</code>, <code>FALLBACK</code>, <code>FAIL</code>, <code>NOT TESTED</code>, <code>SKIPPED</code>, <code>UNKNOWN</code>.</li>
  <li>Per-EP datasets include logs, optional optimized models, and a README with details.</li>
</ul>

<h2>📐 How’s Tested</h2>
<h3>Inference</h3>
<p>
  Minimal one-node ONNX model per op. A small “complexification” (e.g., extra <code>Mul</code>/<code>And</code>)
  can be added to trigger some compilers (OpenVINO/TensorRT) and reveal actual EP coverage.
</p>
<h3>Training</h3>
<p>
  When available (CPU/CUDA), a trainable scalar is injected before the tested node and a 1-step optimization (AdamW, MSE)
  validates basic backward. The training result appears only in the last column; it does not affect inference percentages.
</p>

<h2>📦 EPs with results in this opset</h2>
<ul>
<li><a href="./CPU/" target="_blank">CPU</a></li>
<li><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></li>
<li><a href="./Intel%20-%20oneDNN/" target="_blank">Intel - oneDNN</a></li>
<li><a href="./NVIDIA%20-%20CUDA/" target="_blank">NVIDIA - CUDA</a></li>
<li><a href="./NVIDIA%20-%20TensorRT/" target="_blank">NVIDIA - TensorRT</a></li>
<li><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></li>
</ul>

<h2>System / Versions</h2>
<ul>
  <li><strong>Test Date:</strong> 2025-11-18 14:43:06</li>
  <li><strong>CPU:</strong> Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz</li>
  <li><strong>GPU:</strong> NVIDIA GeForce RTX 2070</li>
  <li><strong>ONNX:</strong> 1.18.0 | <strong>ONNX Runtime:</strong> 1.23.0+cu125</li>
  <li><strong>ONNX Opset:</strong> 22 | <strong>ONNX IR:</strong> 10</li>
</ul>
<h3>ONNX Core Operators</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Execution Provider</th>
      <th>SUCCESS</th>
      <th>FALLBACK</th>
      <th>SUPPORTED</th>
      <th>FAIL</th>
      <th>NOT TESTED</th>
      <th>SKIPPED</th>
      <th>TRAINING</th>
    </tr>
  </thead>
  <tbody>
<tr><td><a href="./CPU/" target="_blank">CPU</a></td><td>142 (92%)</td><td>0 (0%)</td><td>142 (92%)</td><td>11 (7%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>40 (26%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></td><td>91 (59%)</td><td>50 (32%)</td><td>141 (91%)</td><td>12 (8%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20oneDNN/" target="_blank">Intel - oneDNN</a></td><td>37 (24%)</td><td>105 (68%)</td><td>142 (92%)</td><td>11 (7%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20CUDA/" target="_blank">NVIDIA - CUDA</a></td><td>71 (46%)</td><td>71 (46%)</td><td>142 (92%)</td><td>11 (7%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>28 (18%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20TensorRT/" target="_blank">NVIDIA - TensorRT</a></td><td>98 (63%)</td><td>50 (32%)</td><td>148 (95%)</td><td>5 (3%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></td><td>96 (62%)</td><td>45 (29%)</td><td>141 (91%)</td><td>12 (8%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
</tbody></table>
<h3>Microsoft Custom Operators</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Execution Provider</th>
      <th>SUCCESS</th>
      <th>FALLBACK</th>
      <th>SUPPORTED</th>
      <th>FAIL</th>
      <th>NOT TESTED</th>
      <th>SKIPPED</th>
      <th>TRAINING</th>
    </tr>
  </thead>
  <tbody>
<tr><td><a href="./CPU/" target="_blank">CPU</a></td><td>59 (55%)</td><td>0 (0%)</td><td>59 (55%)</td><td>41 (38%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>7 (7%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></td><td>15 (14%)</td><td>41 (38%)</td><td>56 (52%)</td><td>44 (41%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20oneDNN/" target="_blank">Intel - oneDNN</a></td><td>6 (6%)</td><td>52 (49%)</td><td>58 (54%)</td><td>42 (39%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20CUDA/" target="_blank">NVIDIA - CUDA</a></td><td>52 (49%)</td><td>34 (32%)</td><td>86 (80%)</td><td>14 (13%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>6 (6%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20TensorRT/" target="_blank">NVIDIA - TensorRT</a></td><td>6 (6%)</td><td>78 (73%)</td><td>84 (79%)</td><td>16 (15%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></td><td>27 (25%)</td><td>33 (31%)</td><td>60 (56%)</td><td>40 (37%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
</tbody></table>

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