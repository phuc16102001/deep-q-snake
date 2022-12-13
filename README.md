# Deep-Q Learning Snake

<div>
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="python">
</div>

## Introduction

This project is to train an snake which is controlled by Deep-Q Learning. It also consist of 4 different strategies to use by combining **AStar algorithm** with **DeepQNet**:
- AStar only
- DeepQ only
- Both (AStar is prioritized)
- Both (random choice between 2 algorithms)

## Gallery

<p align="center">
    <img src="res/train.png" alt="training" style="width: 50%">
</p>
<p align="center">
    Training process
</p>



## Usage

Firstly, to use the source, please change directory into `/src`:
```bash
cd src
```

### Configuration

```bash
python utils/gen_conf.py --size 8 \
    --speed 60 \
    --memory 100000 \
    --batch_size 1000 \
    --learning_rate 0.001 \
    --gamma 0.9
```

### Train models

```bash
python utils/train.py --input_path "../data/input2.txt" \
    --output_path "../models/current_model.pth"
```

### Run inference
```bash
python utils/inference.py --from_test 1 \
    --to_test 3 \
    --data_folder "../data" \
    --output_folder "../output" \
    --model_path "../models/best.pth"
```

## License
[MIT](LICENSE)
