# GRPO policy ratio ignores rollout log probabilities

Training Qwen/Qwen2.5-0.5B-Instruct with GRPO on official GSM8K. The policy
loss uses current log probabilities in place of frozen rollout probabilities,
forcing the probability ratio to one and disabling clipping.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | algorithm |
| **Tags** | `verl` `grpo` `fsdp` `vllm` `gsm8k` |
| **Model** | Qwen/Qwen2.5-0.5B-Instruct |
| **Dataset** | openai/gsm8k main; first 256 train rows, 100 test questions sampled with seed 42 |
| **Framework** | verl 0.3.0.post1, FSDP, `scripts/train.sh` |
| **Environment** | `verl_policy_ratio_recomputes_old_logprob_clean_workspace` · 1 × A800 80 GB GPU · 40 GiB container RAM |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 repaired run, 30 steps each |

## Difficulty

In `verl/trainer/ppo/core_algos.py`, subtracting detached current log
probabilities makes the ratio one throughout four actor passes over each
rollout. The resulting loss diverges from the frozen behavior-policy objective.

| Measurement | Defective | Repaired |
| --- | ---: | ---: |
| Initial accuracy, 100 questions | 47% | 47% |
| Final accuracy, step 30 | 0% | 50% |
| Maximum behavior-objective loss error | 0.15077248 | 0 |

Each run uses batch size 16, group size 8 and four actor passes per step
(120 optimizer updates). Validation runs at steps 0, 10, 20 and 30 on the
same 100 questions. These results are from one pair of runs.

## Reference solution

- **issue**: identify the incorrect log-probability subtraction and explain
  the inactive clipping and behavior-objective mismatch.
- **solution**: use frozen rollout behavior log probabilities in the policy
  ratio and KL calculation and submit `report.md`; the harness then runs the
  repaired experiment for 30 steps. The reference
  patch and calibration evidence are hidden in the solution oracle.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires verdict = 1.0 and both root-cause and evidence scores > 0.8; the weighted score is reported separately.
- **solution**: complete the fixed 30-step training run with correct policy
  loss and KL, then achieve **greater than 45% accuracy** on all 100 validation
  questions at step 30 (at least 46 correct). The verifier independently
  scores the final answers and checks the training artifacts.
  Score: 1.0 if all checks pass, otherwise 0.0.

## Packaging maintenance

The clean-workspace image uses the same digest-pinned dependency stack as the
recorded run, clearing only the base image's unrelated `/workspace` contents
before the task is uploaded. This removes absolute example symlinks that Harbor
cannot safely archive. The original image recipe is retained unchanged.
Image build and archive checks are recorded separately from training validation;
the clean image was built in the 2026-10-31 packaging review; this does not establish fresh solution-training correctness.

## Review and execution guide

Scope: **issue and solution**. Run package validation from the repository root:

```bash
python -m autorlbench validate tasks/algorithm/verl_policy_ratio_recomputes_old_logprob
```

Evidence entry points: `workspace/observed_run/{metrics,objective,validation,worker_events}.jsonl`, `workspace/observed_run/resolved_full.yaml`, and `workspace/configs/experiment.yaml`. The symptom, mechanism and counterfactual
comparison are explained above; raw logs remain the source of truth. Historical
comparisons establish the documented case, not a guarantee over new seeds.

The task README, task card, issue oracle, solution oracle/reference patches and
verifier resources are reviewer material. Agents receive the level-specific
`instruction.md` and staged `workspace/`; do not place the reviewer README or
oracle in the workspace. Issue instructions use the common diagnosis scaffold.
Solution instructions describe the fixed workload, deliverables and acceptance
contract without specifying a repair. Upstream contribution policies and unrelated
utilities are excluded; the hidden round-two workspace audit records removals.
Production modules and frozen evidence are retained for diagnosis and import
compatibility; this is not a file-by-file minimal dependency extraction.

### Resource budget

One A800 80 GB, 40 GiB host RAM, 30 steps. The archived integration report records 1232 seconds for the repaired run and 1261 seconds for the defective run: approximately **0.34 / 0.35 GPU-hours**. These are historical measurements, not a new timing guarantee. The rerun timeout is 7200 seconds (**2 GPU-hours maximum**). The final 100-question evaluation remains required.

### Offline review, 2026-11-01

This review runs package validation, evidence/hash checks and applicable CPU
checks only. It does not run a model agent, hosted judge, GPU training, prepare
downloads, or checkpoint deserialization. GPU figures above are historical
measurements or configured timeout bounds; they are not measurements from this
review. Fresh end-to-end solution correctness remains unverified this round.
