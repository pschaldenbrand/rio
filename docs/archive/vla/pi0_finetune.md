# Finetuning a VLA policy with your custom data

The following example demonstrates the tuning of PI0.5 using the rollouts of a dummy policy collected in Libero.


### Collect the data using a Recorder Node

Run the `examples/libero/dummy_policy_libero_recording.py` script to collect data using a dummy policy. The script will save the collected data in a robodm format under `/tmp/collected_data/libero_demo.vla`.

```bash
python examples/libero_demos/dummy_policy_libero_recording.py
```
> *Note:* in this simple example with use the same data structure and naming as the libero dataset, so the training using pi0.5 will be straightforward requiring minimal modifications on third party code. If you are collecting data in a different format, you may need to modify the training scripts of *openpi* to match your data format.

TODO: add link for instructions on how to modify the training scripts for custom data formats.


> Optional: You can visualize the collected data with:
```bash
uv sync --all-extras --group rerun
python examples/libero_demos/replay_libero_recording.py
```

### Convert the data to LeRobot format

Ensure that the correct dependency installed match the versions required by openpi:
```bash
uv sync --all-extras --group openpi
```

Next, convert the collected robodm data to LeRobot format using the `examples/data/convert.py` script:

```bash
python examples/data/convert.py --input /tmp/collected_data/libero_demo.vla --repo-id 'your_hf_username/libero_demo' --format lerobot --robot-type panda
```
No output path is needed since Lerobot uses as default path `~/.cache/huggingface/lerobot/<repo_id>` to store datasets.

If you need to store your dataset in a different location, you can export the variable `HF_LEROBOT_HOME` to point to the desired directory. ```export HF_LEROBOT_HOME=/path/to/your/dataset/directory```. This will ensure that other dataloaders using Lerobot will be able to find the dataset in the alternative location.


### Finetune PI0.5 with the converted data

First we need to configure the new dataset for training in the openpi codebase.
in `third_party/openpi/src/openpi/training/config.py`, add a training configuration for the new dataset that points to the repo_id defined before, or edit the existing libero training config with *your repo_id*.

```python
    
    TrainConfig(
        name="pi05_libero_demo", # YOUR CONFIG NAME HERE
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_demo", # USER YOUR REPO_ID HERE
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32, # SMALLER BATCH_SIZE FOR TESTING
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
```
If you are reusing an existing config, make sure that the dataset keys match the expected naming conventions used in your dataset.

--- 

The openpi policies also require statistics of the dataset to be precomputed before finetuning.

To compute the statistics, run the following script:

```bash
cd third_party/openpi
uv sync --group rlds
source .venv/bin/activate
# use the same config name as defined in the training config above
uv run scripts/compute_norm_stats.py --config-name pi05_libero_demo
```

Finally, you can start the finetuning process using the following command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_demo --exp-name=my_experiment --overwrite
# Note: if using a config name other than pi05_libero, change it accordingly
```
