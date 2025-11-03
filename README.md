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
cd mylerobot
pip install -e ".[all]"
```

- run `src/scripts_xarm/record_gui.py` to gather data using keyboard

- run `src/scripts_xarm/train.py` to train policy

- run `src/scripts_xarm/eval.py` to eval policy