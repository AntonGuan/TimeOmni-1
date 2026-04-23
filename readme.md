<div align="center">
<img src="figs/logo.png" alt="Logo" width="120"/>

<h1><b>
(ICLR'26) TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models
</b></h1>


<p align="left">
  <a href="https://arxiv.org/abs/2509.24803">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Paper-red?logo=arxiv&logoColor=red"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Paper on arXiv"
    />
  </a>

  <a href="https://huggingface.co/collections/anton-hugging/timeomni-1-from-4b-to-9b">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Model-yellow?logo=huggingface&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Model on Hugging Face"
    />
  </a>

  <a href="https://huggingface.co/datasets/anton-hugging/timeomni-1-testbed">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Dataset-orange?logo=huggingface&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Dataset on Hugging Face"
    />
  </a>

  <a href="https://huggingface.co/spaces/anton-hugging/TimeOmni-1">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Demo-blue?logo=huggingface&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Demo on Hugging Face Spaces"
    />
  </a>

  <a href="https://github.com/AntonGuan/TimeOmni-1" target="_blank" style="margin: 2px;">
    <img
      src="https://img.shields.io/badge/TimeOmni--1-Inference%20Code-536af5?logo=github&logoColor=white"
      style="display: inline-block; vertical-align: middle;"
      alt="TimeOmni-1 Inference Code on GitHub"
    />
  </a>
</p>

</div>

**This repository provides installation and usage scripts for TimeOmni-1.**

>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

---

## Updates/News:

🚩 **News** (Apr. 2026): We have released new post-trained versions based on Qwen3.5 on Hugging Face: [TimeOmni-1-9B](https://huggingface.co/TimeOmni-1/TimeOmni-1-9B) and [TimeOmni-1-4B](https://huggingface.co/TimeOmni-1/TimeOmni-1-4B). These new versions further scale up model performance (inference code coming soon).

🚩 **News** (Feb. 2026): Please find the open source model on Hugging Face: [TimeOmni-1-7B](https://huggingface.co/anton-hugging/TimeOmni-1-7B); see also our online demo: https://huggingface.co/spaces/anton-hugging/TimeOmni-1

🚩 **News** (Jan. 2026): TimeOmni-1 has been accepted to ICLR 2026! 🎉

## 📊 Benchmarks
**Table. Model Size Scaling Comparison**

<p style="margin-top:13px;font-size:11px;opacity:0.7">
* Note: All metrics below are computed only on valid responses. “–” indicates a success rate (SR) below 10%; in such cases, results are omitted due to insufficient statistical significance, and we therefore do not report them. For ACC, higher is better; for MAE, lower is better. <strong>Bold</strong> marks the best value in each ACC/MAE column. 
</p>


<table style="width:100%;border-collapse:collapse;font-size:13px">
  <thead>
    <tr>
      <th style="padding:10px 7px;text-align:left;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb"></th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task1 ID (ACC↑/SR)</th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task1 OOD (ACC↑/SR)</th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task2 ID (ACC↑/SR)</th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task2 OOD (ACC↑/SR)</th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task3 ID (MAE↓/SR)</th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task3 OOD (MAE↓/SR)</th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task4 ID (ACC↑/SR)</th>
      <th style="padding:10px 7px;text-align:center;font-weight:700;border-bottom:2px solid #2563eb;color:#2563eb;font-size:14px">Task4 OOD (ACC↑/SR)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" style="padding:8px 12px;font-weight:600;font-style:italic;color:#2563eb;border-bottom:1px solid rgba(37, 99, 235, 0.2);background:rgba(37, 99, 235, 0.1)">7B (Qwen2.5-Instruct)</td>
    </tr>
    <tr>
      <td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);color:#2563eb">Qwen2.5-Instruct-7B</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">48.5/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">42.8/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">21.6/99.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">26.3/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">23.28/53.1</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">146.12/55.5</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">25.5/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">24.9/100.0</td>
    </tr>
    <tr>
      <td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);color:#2563eb"><strong>TimeOmni-1-7B</strong></td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">90.7/97.5</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">87.7/98.3</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">69.3/99.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">64.0/99.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">14.30/93.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">145.53/82.3</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">47.9/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">58.9/100.0</td>
    </tr>
    <tr>
      <td colspan="9" style="padding:8px 12px;font-weight:600;font-style:italic;color:#2563eb;border-bottom:1px solid rgba(37, 99, 235, 0.2);background:rgba(37, 99, 235, 0.1)">4B (Qwen3.5)</td>
    </tr>
    <tr>
      <td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);color:#2563eb">Qwen-3.5-4B</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">0.0/16.5</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">5.9/17.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">28.3/12.4</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">35.4/12.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-/2.2</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-/9.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-/8.5</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-/9.2</td>
    </tr>
    <tr>
      <td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);color:#2563eb"><strong>TimeOmni-1-4B</strong></td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">91.5/99.5</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">91.2/98.4</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>71.1</strong>/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">66.1/99.9</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">13.68/97.6</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">170.41/86.1</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">58.5/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">72.0/100.0</td>
    </tr>
    <tr>
      <td colspan="9" style="padding:8px 12px;font-weight:600;font-style:italic;color:#2563eb;border-bottom:1px solid rgba(37, 99, 235, 0.2);background:rgba(37, 99, 235, 0.1)">9B (Qwen3.5)</td>
    </tr>
    <tr>
      <td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);color:#2563eb">Qwen-3.5-9B</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">91.2/51.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>93.5</strong>/46.1</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">43.3/12.1</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">36.3/12.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">17.56/14.1</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">-/0.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>64.2</strong>/28.2</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">72.0/32.2</td>
    </tr>
    <tr>
      <td style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);color:#2563eb"><strong>TimeOmni-1-9B</strong></td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>93.5</strong>/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">92.8/99.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">70.9/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>66.2</strong>/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>13.54</strong>/97.8</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>140.06</strong>/95.6</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)">59.6/100.0</td>
      <td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15)"><strong>75.6</strong>/99.6</td>
    </tr>
  </tbody>
</table>

## 🛠️ Installation
```bash
conda create -n timeomni python=3.10
conda activate timeomni
pip install -r requirements.txt
```

## 📦 Model Download
```bash
python install/download_hf_model.py
```
Default model path: `~/.cache/huggingface/hub`.

## 🧪 Dataset Download
```bash
python install/download_testbed.py
```
This creates:
- `data/timeomni1_id_test.json`
- `data/timeomni1_ood_test.json`

## 🚀 Inference (single question)
Default system prompt:
```
Output Format:
<think>Your step-by-step reasoning process that justifies your answer</think>
<answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>
```

Run:
```bash
python inference/inference.py \
  --model_dir "Local Model Path /models--anton-hugging--TimeOmni-1-7B/snapshots/<hash>" \
  --question "Your Question" \
  --system_prompt "Output Format:\n<think>Your step-by-step reasoning process that justifies your answer</think>\n<answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>"
```

## 📊 Evaluation
```bash
bash eval/run-timeomini_test.sh
```
Optional env overrides:
```bash
MODEL_DIR=anton-hugging/TimeOmni-1-7B \
ANS_ID_PATH=answer/timeomni1_test/your_id_outputs.json \
RES_ID_PATH=answer/timeomni1_test/your_id_results.json \
ANS_OOD_PATH=answer/timeomni1_test/your_ood_outputs.json \
RES_OOD_PATH=answer/timeomni1_test/your_ood_results.json \
bash eval/run-timeomini_test.sh
```

We report Success Rate (SR), defined as the proportion of model outputs that yield a valid and extractable answer. All other metrics are computed on valid cases only.

+ **Tasks 1, 2, 4:** model outputs a single uppercase letter (A/B/C/D). Metric: Accuracy (ACC).  
+ **Task 3:** model outputs a sequence (e.g., `[2, 20, 21, ..., 83]`). Metric: Mean Absolute Error (MAE).

## ✍️ Citation

```bibtex
@inproceedings{
guan2026timeomni,
title={TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models},
author={Tong Guan and Zijie Meng and Dianqi Li and Shiyu Wang and Chao-Han Huck Yang and Qingsong Wen and Zuozhu Liu and Sabato Marco Siniscalchi and Ming Jin and Shirui Pan},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=kOIclg7muL}
}
```
