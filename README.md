<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/lerobot-logo-thumbnail.png" width="100%">
  <br/>
  <br/>
</p>

![ALOHA ACT 演示](media/act_aloha.gif)

- first, create same virtual environment by `environment.yml`
```bash
conda env create -f environment.yml
conda activate lerobot
```

- then, install `lerobot` by `pip`
```bash
cd lerobot
pip install -e ".[all]"
```

- run `test/sim/6_xarm_sys/6_env.py` to gather data using keyboard

- run `test/sim/6_xarm_sys/6_train_xarm.py` to train policy

- run `test/sim/6_xarm_sys/6_eval_xarm.py` to eval
 policy