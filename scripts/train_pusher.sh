#!/usr/bin/env bash
GPU=$1
SEED=$2

CUDA_VISIBLE_DEVICES=${GPU} python train_ddpg.py \
--env-name "Pusher-v0" \
--test "Pusher-v0" \
--device cuda:0 \
--action-l2 0.01 \
--random-eps 0.1 \
--gamma 0.99 \
--n-epochs 5000 \
--period 3 \
--fps \
--landmark 80 \
--initial-sample 200 \
--clip-v -4 \
--goal-thr -5 \
--future-step 50 \
--seed ${SEED} \
--distance 0.25 \
--lambda_goal_loss 0.1 \
--jump \
--jump_temp 1