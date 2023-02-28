import numpy as np


class her_sampler:
    def __init__(self, replay_strategy, replay_k, threshold, future_step=300):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == "future":
            self.future_p = 1 - (1.0 / (1 + replay_k))
        else:
            self.future_p = 0
        self.threshold = threshold
        self.furture_step = future_step
        print("Sample in future steps", self.furture_step)

    def reward_func(self, state, goal, info=None):
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        return -(dist > self.threshold).astype(np.float32)

    def reward_func_with_action_ref(self, state, goal, action, info=None):
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        reward_ctrl = 0.001 * -np.square(action).sum(axis=1)

        return -(dist > self.threshold).astype(np.float32) + reward_ctrl

    def sample_her_transitions(
        self, episode_batch, batch_size_in_transitions, fetch_task=False
    ):
        T = episode_batch["actions"].shape[1]
        rollout_batch_size = episode_batch["actions"].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        # cheat in her for large step length

        target_index = np.minimum(T, t_samples + self.furture_step)
        future_offset = np.random.uniform(size=batch_size) * (target_index - t_samples)

        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch["ag"][episode_idxs[her_indexes], future_t]
        transitions["g"][her_indexes] = future_ag
        # to get the params to re-compute reward
        if fetch_task:
            transitions["r"] = np.expand_dims(
                self.reward_func_with_action_ref(
                    transitions["ag_next"],
                    transitions["g"],
                    transitions["actions"],
                    None,
                ),
                1,
            )
        else:
            transitions["r"] = np.expand_dims(
                self.reward_func(transitions["ag_next"], transitions["g"], None), 1
            )
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }

        return transitions
