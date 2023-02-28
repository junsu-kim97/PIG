# PIG

Implementation of [Imitating Graph-Based Planning with Goal-Conditioned Policies](https://openreview.net/forum?id=6lUEy1J5R7p) (ICLR 2023) in PyTorch.

Our code is based on official implementation of [Mapping State Space](https://github.com/FangchenLiu/map_planner).

## Instructions

Install dependencies

```angular2html
conda create -n pig python=3.6
conda activate pig
conda install pytorch=1.3.1 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

## Experiments

To reproduce our experiments, please run below scripts

### 2D Reach
```
source ./scripts/train_2dplane.sh {GPU} {SEED}
```
### Ant Maze
```
source ./scripts/train_antmaze.sh AntMazeL v1 {GPU} {SEED}  # L-shape
source ./scripts/train_antmaze.sh AntMaze v1 {GPU} {SEED}  # U-shape
source ./scripts/train_antmaze.sh AntMaze v0 {GPU} {SEED}  # Large U-shape
source ./scripts/train_antmaze.sh AntMazeS v1 {GPU} {SEED}  # S-shape
source ./scripts/train_antmaze.sh AntMazeW v1 {GPU} {SEED}  # W-shape
source ./scripts/train_antmaze.sh AntMazeP v1 {GPU} {SEED}  # Pi-shape
```
### Pusher
```
source ./scripts/train_pusher.sh {GPU} {SEED}
```
### Reacher
```
source ./scripts/train_reacher.sh {GPU} {SEED}
```

If you find this code useful, please reference in our paper:
```bibtex
@inproceedings{
  kim2023imitating,
  title={Imitating Graph-Based Planning with Goal-Conditioned Policies},
  author={Junsu Kim and Younggyo Seo and Sungsoo Ahn and Kyunghwan Son and Jinwoo Shin},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=6lUEy1J5R7p}
}
```
