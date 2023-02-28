import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument(
        "--env-name", type=str, default="PointMaze-v1", help="the environment name"
    )
    parser.add_argument("--test", type=str, default="PointMaze-v1")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=14000,
        help="the number of epochs to train the agent",
    )
    parser.add_argument(
        "--n-batches", type=int, default=200, help="the times to update the network"
    )
    parser.add_argument("--seed", type=int, default=123, help="random seed")

    parser.add_argument(
        "--replay-strategy", type=str, default="future", help="the HER strategy"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_models/",
        help="the path to save the models",
    )

    parser.add_argument(
        "--noise-eps", type=float, default=0.2, help="noise factor for Gaussian"
    )
    parser.add_argument(
        "--random-eps", type=float, default=0.2, help="prob for acting randomly"
    )
    parser.add_argument(
        "--plan-eps",
        type=float,
        default=0.5,
        help="prob of using planner when training",
    )

    parser.add_argument(
        "--buffer-size", type=int, default=int(1e6), help="the size of the buffer"
    )
    parser.add_argument("--replay-k", type=int, default=5, help="ratio to be replaced")
    parser.add_argument(
        "--future-step", type=int, default=200, help="future step to be sampled"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="the sample batch size"
    )
    parser.add_argument("--gamma", type=float, default=0.98, help="the discount factor")
    parser.add_argument("--action-l2", type=float, default=0.5, help="l2 reg")
    parser.add_argument(
        "--lr-actor", type=float, default=0.0002, help="the learning rate of the actor"
    )
    parser.add_argument(
        "--lr-critic",
        type=float,
        default=0.0002,
        help="the learning rate of the critic",
    )
    parser.add_argument(
        "--polyak", type=float, default=0.99, help="the average coefficient"
    )
    parser.add_argument(
        "--n-test-rollouts", type=int, default=10, help="the number of tests"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="MLP",
        help="the metric for the distance embedding",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cuda device")

    parser.add_argument(
        "--lr-decay-actor", type=int, default=3000, help="actor learning rate decay"
    )
    parser.add_argument(
        "--lr-decay-critic", type=int, default=3000, help="critic learning rate decay"
    )
    parser.add_argument(
        "--layer", type=int, default=6, help="number of layers for critic"
    )

    parser.add_argument("--period", type=int, default=3, help="target update period")
    parser.add_argument(
        "--distance", type=float, default=0.1, help="distance threshold for HER"
    )

    parser.add_argument("--resume", action="store_true", help="resume or not")
    # Will be considered only if resume is True
    parser.add_argument("--resume-epoch", type=int, default=10000, help="resume epoch")
    parser.add_argument(
        "--resume-path",
        type=str,
        default="saved_models/AntMazeTest-v1_May23_04-14-15",
        help="resume path",
    )

    # args for the planner
    parser.add_argument("--fps", action="store_true", help="if use fps in the planner")
    parser.add_argument("--landmark", type=int, default=200, help="number of landmarks")
    parser.add_argument(
        "--initial-sample",
        type=int,
        default=1000,
        help="number of initial candidates for landmarks",
    )
    parser.add_argument(
        "--clip-v", type=float, default=-4.0, help="clip bound for the planner"
    )
    parser.add_argument("--goal-thr", type=float, default=-10)
    parser.add_argument("--heat", type=float, default=0.9)
    parser.add_argument("--schedule_n_nodes", action="store_true")
    parser.add_argument("--node_decay", type=float, default=0.5)
    parser.add_argument("--min_landmark", type=int, default=200)
    parser.add_argument("--eval-freq", type=int, default=10)

    # PIG
    parser.add_argument("--lambda_goal_loss", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--plan_budget", type=int, default=10)
    parser.add_argument("--jump", action="store_true")
    parser.add_argument("--jump_temp", type=float, default=0.1)

    args = parser.parse_args()
    return args
