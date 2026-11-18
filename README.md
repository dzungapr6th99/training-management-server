# SmolVLA on OpenArm in Isaac Sim 6.0.1

```
 SmolVLA conda env (env\)                    Isaac Sim python (D:\Isaac\isaacsim)
 ┌───────────────────────────┐   TCP :5555   ┌───────────────────────────────────────┐
 │ policy_server.py          │ ◄──────────── │ openarm_smolvla_sim.py                │
 │  lerobot SmolVLAPolicy    │  state(8) +   │  OpenArm unimanual USD, table, cube   │
 │  models\smolvla_base      │  3 RGB 256²   │  cams: front / wrist / side (RTX)     │
 │  → action chunk (50 × 8)  │ ────────────► │  joint position targets @ 30 Hz       │
 └───────────────────────────┘               └───────────────────────────────────────┘
```

The two processes are kept separate so lerobot/transformers can't break Isaac Sim's bundled Python.
`wire.py` is a stdlib-only socket protocol shared by both sides.

## Run

```bat
scripts\run_all.bat                                   :: server window + Isaac Sim GUI
scripts\run_all.bat --task "pick up the red cube" --episodes 3
```
or in two terminals: `scripts\run_server.bat`, then `scripts\run_sim.bat [--headless] [--save-frames DIR]`.

Useful sim flags: `--exec-steps` (actions executed per chunk before replanning, default 25),
`--action-mode abs|delta`, `--max-step-rad` (safety rate limit, default 0.05 rad/step), `--probe`
(build the scene, save one set of camera frames, and exit).

Server self-test (no sim): `env\python.exe scripts\policy_server.py --selftest`.

## Robot ↔ model mapping

| | index | content |
|---|---|---|
| state / action | 0–6 | `openarm_joint1..7` (rad), clipped to USD joint limits |
| | 7 | gripper opening (m, 0 = closed … 0.044 = open), mirrored to both fingers |
| images | camera1 / camera2 / camera3 | front / wrist / side, 256×256 RGB |

## Important: base model vs. fine-tuned

`lerobot/smolvla_base` was pretrained on SO100 (6-DoF, different cameras and units). The server
rewrites its features to 8-D and uses identity normalization, so it **runs** on OpenArm, but it has never
seen this robot and its actions are not task-meaningful (in testing it swings the arm off to the side).
To get real behaviour, fine-tune on OpenArm demonstrations (e.g. teleop/mimic data recorded in this
scene as a LeRobot dataset) and point the server at the result:

```bat
scripts\run_server.bat --checkpoint D:\path\to\your\smolvla_openarm_ckpt
```

If a checkpoint's state/action dims are already 8, its own saved pre/post-processors (normalization
stats) are used unchanged. Cameras are matched to the checkpoint's image keys in order (front, wrist, side).

## Bimanual OpenArm + fine-tuned SmolVLA ("place the cube in the box")

```bat
scripts\run_bimanual_all.bat                 :: server window + Isaac Sim GUI, loops until closed
```

- Robot: `openarm_bimanual.usd` (two arms); the policy drives the **left** arm, the right arm hangs still.
- Model: `qualia-robotics/smolvla-openarm-cube-in-box-v2`, trained on 100 RL-teacher demos in a **MuJoCo** sim of
  the same robot (state 9 = left j1..7 + 2 fingers, action 8 = left j1..7 + gripper; 25 fps; ~1 s episodes;
  3 fisheye 480×480 cameras `base`, `wrist_left`, `wrist_right`).
- Scene (`scripts/openarm_bimanual_sim.py`) mirrors that data: each episode starts from a recorded training start
  pose (`scripts/data/qualia_cube_in_box.json`) with the open gripper around the cube; the table height follows the
  fingertips and the box sits where the teacher released. Fisheye cameras are emulated with RTX OpenCV-fisheye.
- Measured in Isaac (headless, 10 episodes): **1/10** cubes in the box; most episodes close + swing toward the box
  but the cube is dragged rather than lifted (MuJoCo → Isaac physics/visual gap).
- Tuning flags: `--grasp-depth` (cube between fingers, default 0.03 m), `--cube-friction`, `--start-index N`,
  `--box-pos X Y`, `--arm-stiffness/--arm-damping`, `--no-fisheye`, `--debug`.

## Data → train → evaluate pipeline (Isaac-native fine-tune)

1. **Generate demos** with the scripted IK expert (left arm: reach, top-down grasp, lift, carry, drop; only
   successful episodes are kept — 100/104 on the first run), then convert to LeRobot v3:
   ```bat
   scripts\gen_data.bat 100 --headless
   ```
   Raw: `datasets\raw\openarm_cube_in_box_isaac\*.npz` · LeRobot: `datasets\lerobot\openarm_cube_in_box_isaac`
   (100 episodes, 14,067 frames, 361 MB; converter uses NVENC + streaming encoding, ~4 min).
2. **Train** (fine-tune the qualia checkpoint, 20k steps, batch 8, ~2 h on the RTX 4070 Ti SUPER, loss 2.38 → 0.012):
   ```bat
   env\python.exe -m lerobot.scripts.lerobot_train --policy.path=models\smolvla_openarm_cube_in_box_v2 --policy.push_to_hub=false ^
     --dataset.repo_id=local/openarm_cube_in_box_isaac --dataset.root=datasets\lerobot\openarm_cube_in_box_isaac ^
     --dataset.video_backend=pyav --batch_size=8 --steps=20000 --save_freq=5000 ^
     --output_dir=outputs\smolvla_cube_in_box_isaac --wandb.enable=false
   ```
   `vendor\lerobot\...\common\train_utils.py` is patched to fall back to a directory junction for
   `checkpoints\last` (Windows refuses symlinks without Developer Mode, WinError 1314).
3. **Evaluate** from the demos' start condition (`--start-mode expert`), server on port 5556:
   ```bat
   scripts\eval_finetuned.bat                       :: GUI, loops until closed
   ```
   | Model | Cubes in the box |
   |---|---|
   | `smolvla_openarm_cube_in_box_v2` (MuJoCo-trained, before fine-tune) | 0 / 10 |
   | fine-tuned on the Isaac demos (step 20000) | **18 / 20** |

## Layout

- `env\`: conda env (Python 3.12, torch 2.11+cu128, lerobot editable from `vendor\lerobot`, `[smolvla]` extras)
- `models\smolvla_base`, `models\hf_home`: SmolVLA weights plus the SmolVLM2-500M backbone/tokenizer
- `vendor\`: lerobot, openarm_isaac_lab (USD assets), openarm_description (URDF)
- `logs\`: run logs and saved frames

Note: Isaac Sim logs errors about `isaacsim.robot.experimental.wheeled_robots` (a broken `jinja2`
import in the install). This is unrelated and harmless.
