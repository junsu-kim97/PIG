#!/usr/bin/env bash
GPU=$1
SEED=$2

CUDA_VISIBLE_DEVICES=${GPU} \
python train_ddpg.py \
--env-name GoalPlane-v0 \
--test GoalPlaneTest-v0 \
--device cuda:0 \
--gamma 0.99 \
--n-epochs 2000 \
--period 10 \
--fps \
--initial-sample 50 \
--clip-v -4 \
--jump \
--jump_temp 1 \
--lambda_goal_loss 1 \
--landmark 100 \
--seed ${SEED}