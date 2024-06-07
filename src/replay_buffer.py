import numpy as np
import collections
import torch

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'nxt_state', 'done'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, new_state, done)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones_np = np.array(dones, dtype=bool)
        dones = 1 - dones_np.astype(int)

        return (torch.tensor(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states), torch.tensor(dones, dtype=torch.float32))

if __name__ == "__main__":
    buffer = ReplayBuffer(2)
    buffer.add((1, 2, 3))
    buffer.add((4, 5, 6))
    buffer.add((7, 8, 9))
    print(buffer.buffer)
    print(buffer.sample(2))
    print(buffer.sample(2))
    print(buffer.sample(2))