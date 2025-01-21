+++
date = '2025-01-20T19:48:12+08:00'
draft = false
title = 'Z1-Coder: Unleashing the System-2 Reasoning Power of LLMs to Code Generation'
+++

**Author: Zhaojian Yu, Yilun Zhao, Arman Cohan, Xiao-Ping Zhang**

**Github: https://github.com/Z1-Coder/Z1-Coder**

{{<figure src="fig1.png">}}

- We introduce Z1-Coder, a series of fully open-source ([code](https://github.com/Z1-Coder/Z1-Coder), [weights](https://huggingface.co/Z1-Coder/Z1-Coder-7B), [data](https://huggingface.co/datasets/Z1-Coder/Z1Coder-Evol-CoT-110K)) LLMs that bridges reasoning capabilities with code generation.
- To train Z1-Coder, we curate reasoning trajectories on code-related datasets and propose self-invoking evolving to further refine models' reasoning behaviour in code generation.
- Z1-Coder model significantly outperforms other open-source models on different code generation benchmarks at a comparable size. Specifically, Z1-Coder-7B surpasses the best 7B code LLMs Qwen2.5-Coder-7B-Instruct, **with only 1% of its post-training data.**
- Z1-Coder-7B also achieves 20.7% pass@1 on LiveCodeBench(20240801-20241101) and 51.4% on BigCodeBench, which achieves comparable performance level compared to DeepseekCoder-33B-Instruct (21.5% and 51.1%) and LLaMA3.1-70B-Instruct (19.3% and 54.8%).

## Overview

System-2 Reasoning LLMs such as o1 and Gemini-2.0-flash-thinking have demonstrated remarkable progress in complex problem solving by producing a long internal chain of thought (CoT), especially in complex programming problems. However, the question about how they achieve such a great performance level are un-accessible, presenting a barrier to the participation of the academic and open-source communities.

In response, a number of initiatives have been launched to develop open-weight reasoning models such as [Sky-T1](https://novasky-ai.github.io/posts/sky-t1/) and [rStar-Math](https://arxiv.org/abs/2501.04519). In this work, we mainly focus on the field of coding and provide some new views to understand the bridge of coding and reasoning.

## **Recipes**

We train the base Qwen2.5-Coder-Base (1.5B and 7B) for two stages with two different reasoning trajectory dataset.


{{<figure src="pipeline.png">}}

**Data Curation for Stage 1**

We use QwQ-32B-Preview, an open-source model with reasoning capabilities comparable to o1-preview, to generate reasoning trajectory for the previous [🤗 Evol-Instruct dataset](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1). Evol-Instruct Dataset involves problems at different complexity by in-depth evolving and covers many code-related topics by in breadth evolving. After trajectory generation, we obtain [🤗 Z1Coder-Evol-CoT](https://huggingface.co/datasets/Z1-Coder/Z1Coder-Evol-CoT-110K) Dataset and train the base model with it.

**Data Curation for Stage 2**

For stage 2, we generate self-invoking code instructions from the open-source code according to the analysis of paper “[HumanEval Pro and MBPP Pro: Evaluating Large Language Models on Self-invoking Code Generation](https://arxiv.org/abs/2412.21199)”. Self-invoking problems are inherent complex programming task for LLM reasoning. Hence, we also use QwQ-32B-Preview to generation CoT trajectory for them and get [🤗 Z1Coder-SelfInvoking-CoT](https://huggingface.co/datasets/Z1-Coder/Z1Coder-SelfInvoking-CoT-20K) Dataset. We continually fine-tune the checkpoint from stage 1 and obtain Z1-Coder series model.

| Model                  | Trajectory Dataset Download       | Reference                      |
|------------------------|-----------------------------------|--------------------------------|
| SFT  *stage 1*  Data   | [🤗 Z1Coder-Evol-CoT-110K](https://huggingface.co/datasets/Z1-Coder/Z1Coder-Evol-CoT-110K)         |  https://github.com/nlpxucan/WizardLM  |
| SFT  *stage 2*  Data   | [🤗 Z1Coder-SelfInvoking-CoT-20K](https://huggingface.co/datasets/Z1-Coder/Z1Coder-SelfInvoking-CoT-20K)  | https://github.com/CodeEval-Pro/CodeEval-Pro                 |


**Training**

We train all the models with Fully Shard Data Parallel (FSDP) and set a global batch size to 1024 for 3 epochs using 2 NVIDIA A800-80G GPUs. We used greedy decoding for all results, with the maximum sequence length set to 1280. We use a learning rate of 5e-5 for the two training stages.

## Evaluation

{{<figure src="res1.png">}}

We achieve this with only 1% data resources compared with Qwen2.5-Coder. The following is a comparison of resource requirements between Z1-Coder-7B and Qwen2.5-Coder-7B-Instruct.

| **Model** | **Z1-Coder-7B** | **Qwen2.5-Coder-7B-Instruct** |
| --- | --- | --- |
| Base Model | Qwen2.5-Coder-7B-Base | Qwen2.5-Coder-7B-Base |
| SFT Data (Stage1) | 110K (open-source) | 10M+ (in-house and open-source) |
| SFT Data (Stage2) | 20K (open-source) | 1M+ (in-house) |
| RL | No | DPO |

## Future Work

Z1-Coder-1.5B and 7B only marks the start of our journey to develop open-sourced models bridging advanced reasoning capabilities and code generation. We will focus on more scalable and efficient way to maintain strong reasoning performance in code generation for future research. Stay tuned as we advance on these exciting projects.

## Citation

```latex
@misc{z1-coder,
  author       = {Z1-Coder Team},
  title        = {Z1-Coder: Unleashing the Reasoning Power of Large Language Models to Code Generation},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Z1-Coder/Z1-Coder}},
  note         = {Accessed: 2025-01-17},
  year         = {2025}
}
```