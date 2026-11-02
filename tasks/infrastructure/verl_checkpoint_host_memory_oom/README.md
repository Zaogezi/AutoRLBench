# Checkpoint saving exhausts host memory at step 10

Training Qwen/Qwen2.5-0.5B-Instruct with GRPO on GSM8K. CPU-offloaded
Adam state leaves insufficient host-memory headroom when the native
checkpoint writer stages tensor data for disk at step 10.

| | |
| --- | --- |
| **Author** | Jinyang Wu |
| **Domain** | infrastructure |
| **Tags** | `verl` `grpo` `fsdp` `checkpoint` `optimizer-offload` `host-memory` |
| **Model** | Qwen/Qwen2.5-0.5B-Instruct |
| **Dataset** | openai/gsm8k main; all 7473 train rows, all 1319 official test rows for held-out validation |
| **Framework** | verl 0.3.0.post1, FSDP, `scripts/train.sh` |
| **Environment** | `verl_checkpoint_host_memory_oom_clean_workspace` · 1 × A800 80 GB GPU · 20 GiB container RAM |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 repaired run, 10-step budget; save at step 10 |

## Difficulty

`actor_rollout_ref.actor.fsdp_config.optimizer_offload` moves Adam state
to CPU after each update. Its moment tensors occupy 3.9GB,
leaving little host RAM for `torch.save` to stage GPU model tensors during
checkpoint serialization. All 10 actor updates complete before Ray's
memory monitor evicts the worker inside `actor_rollout_save_checkpoint`
at its 95% threshold. The cgroup kernel `oom_kill` counter stays at zero;
the recorded exception is host-memory pressure detected by Ray.

| Training and checkpoint outcome | Defective | Repaired |
| --- | --- | --- |
| Completed optimizer updates | 10 | 10 |
| Completed trainer-log steps | 9 | 10 |
| Step-10 checkpoint | Ray host-memory eviction | Saved and independently loaded |
| Fresh-process restore | Unavailable | Step 10, validation only |

Validation runs before training and after restoring the checkpoint;
the training loop saves at step 10 without an intervening validation pass.
Defective trainer logs end at step 9 because step 10 is logged after
saving.

## Reference solution

- **issue**: identify CPU-resident Adam state, the additional host-memory
  pressure during native checkpoint serialization and Ray's worker
  eviction. Explain why the trainer log stops before the tenth record.
- **solution**: set
  `actor_rollout_ref.actor.fsdp_config.optimizer_offload: false`, keeping
  optimizer state on the GPU, then complete a fresh 10-step run and save
  the full checkpoint. The reference configuration and calibration
  evidence are hidden in the solution oracle.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires verdict = 1.0 and both root cause
  and evidence scores > 0.8.
- **solution**: complete a fresh 10-step run within the fixed 20 GiB RAM
  budget, save the full step-10 checkpoint, and restore it in a new process
  for validation without further training. Score: 1.0 if all checks pass,
  otherwise 0.0.

## Submission checks

The packaged source and protected-evidence hashes were verified locally.
The frozen defective run passes the checkpoint-failure evidence checks;
the frozen repaired run passes the 10-step update and checkpoint-event checks.
These are offline evidence checks, not a new GPU reproduction. Docker image
building, model preparation, full checkpoint deserialization, and fresh GPU
training/restoration were not rerun during submission. The supplied integration
report records the original defective and repaired runs.

The solution verifier drives training and restoration itself, so no separate
`run_spec.json` is needed. The agent repairs the workspace and submits
`report.md` without launching training.

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
python -m autorlbench validate tasks/infrastructure/verl_checkpoint_host_memory_oom
```

Evidence entry points: `workspace/logs/{worker_events,checkpoint_events,host_memory,metrics}.jsonl`, `workspace/logs/{training.stderr.log,raylet.out,final_memory_events.txt}`, and `workspace/configs/experiment.yaml`. The symptom, mechanism and counterfactual
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

One A800 80 GB, 20 GiB host RAM. The verifier allows 1200 seconds for each of fresh training and fresh-process restore/validation: **at most 0.67 GPU-hours across those two commands**, plus container setup and artifact checks. No fresh runtime estimate is claimed. Checkpoint contents and the 10-step workload are unchanged.

### Offline review, 2026-11-01

This review runs package validation, evidence/hash checks and applicable CPU
checks only. It does not run a model agent, hosted judge, GPU training, prepare
downloads, or checkpoint deserialization. GPU figures above are historical
measurements or configured timeout bounds; they are not measurements from this
review. Fresh end-to-end solution correctness remains unverified this round.
