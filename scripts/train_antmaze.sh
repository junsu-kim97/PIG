#!/usr/bin/env bash
TASK=$1
VERSION=$2
GPU=$3
SEED=$4

CUDA_VISIBLE_DEVICES=${GPU} \
python train_ddpg.py \
--env-name "${TASK}-${VERSION}" \
--test "${TASK}Test-${VERSION}" \
--device cuda:0 \
--random-eps 0.2 \
--gamma 0.99 \
--n-epochs 5000 \
--period 3 \
--fps \
--landmark 400 \
--initial-sample 500 \
--clip-v -38 \
--seed ${SEED} \
--lambda_goal_loss 0.001 \
--jump \
--jump_temp 10