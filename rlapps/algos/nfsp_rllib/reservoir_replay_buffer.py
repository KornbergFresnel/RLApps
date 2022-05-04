import collections
import logging
import platform
import random
from typing import List

import numpy as np

# Import ray before psutil will make sure we use psutil's bundled version
import ray  # noqa F401
import psutil  # noqa E402

from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.policy.sample_batch import (
    SampleBatch,
    MultiAgentBatch,
    DEFAULT_POLICY_ID,
)
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.window_stat import WindowStat
from ray.util.debug import log_once
from ray.util.iter import ParallelIteratorWorker

# Constant that represents all policies in lockstep replay mode.
_ALL_POLICIES = "__all__"

logger = logging.getLogger(__name__)


def warn_replay_buffer_size(*, item: SampleBatchType, num_items: int) -> None:
    """Warn if the configured replay buffer size is too large."""
    if log_once("replay_buffer_size"):
        item_size = item.size_bytes()
        psutil_mem = psutil.virtual_memory()
        total_gb = psutil_mem.total / 1e9
        mem_size = num_items * item_size / 1e9
        msg = (
            "Estimated max memory usage for replay buffer is {} GB "
            "({} batches of size {}, {} bytes each), "
            "available system memory is {} GB".format(
                mem_size, num_items, item.count, item_size, total_gb
            )
        )
        if mem_size > total_gb:
            raise ValueError(msg)
        elif mem_size > 0.2 * total_gb:
            logger.warning(msg)
        else:
            logger.info(msg)


@DeveloperAPI
class ReservoirReplayBuffer:
    @DeveloperAPI
    def __init__(self, size: int):
        """Create Prioritized Replay buffer.

        Args:
            size (int): Max number of timesteps to store in the reservoir buffer.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._hit_count = np.zeros(size)
        self._eviction_started = False
        self._num_timesteps_added = 0
        self._num_timesteps_added_wrap = 0
        self._num_timesteps_sampled = 0
        self._evicted_hit_stats = WindowStat("evicted_hit", 1000)
        self._est_size_bytes = 0
        self._add_calls = 0

    def __len__(self):
        return len(self._storage)

    def _sample_reservior_buffer_next_index(self):
        uniform_idx = np.random.randint(0, self._add_calls + 1)
        if uniform_idx < self._maxsize:
            self._next_idx = uniform_idx  # Reservoir Behavior
        else:
            self._next_idx = None

    @DeveloperAPI
    def add(self, item: SampleBatchType):
        warn_replay_buffer_size(item=item, num_items=self._maxsize / item.count)
        assert item.count == 1, item
        if self._next_idx is not None:

            self._num_timesteps_added += item.count

            if self._next_idx >= len(self._storage):
                self._storage.append(item)
                self._est_size_bytes += item.size_bytes()
            else:
                self._storage[self._next_idx] = item

            if self._num_timesteps_added >= self._maxsize:
                self._eviction_started = True
                self._sample_reservior_buffer_next_index()
            else:
                self._next_idx += 1

            if self._eviction_started:
                self._evicted_hit_stats.push(self._hit_count[self._next_idx])
                self._hit_count[self._next_idx] = 0
        else:
            assert self._add_calls >= self._maxsize
            self._sample_reservior_buffer_next_index()
        self._add_calls += 1

    def _encode_sample(self, idxes: List[int]) -> SampleBatchType:
        out = SampleBatch.concat_samples([self._storage[i] for i in idxes])
        out.decompress_if_needed()
        return out

    @DeveloperAPI
    def sample(self, num_items: int) -> SampleBatchType:
        """Sample a batch of experiences.

        Args:
            num_items (int): Number of items to sample from this buffer.

        Returns:
            SampleBatchType: concatenated batch of items.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(num_items)]
        self._num_timesteps_sampled += num_items

        out_batch = self._encode_sample(idxes)
        # assert out_batch.count == 128
        return out_batch

    @DeveloperAPI
    def stats(self, debug=False):
        data = {
            "added_count": self._num_timesteps_added,
            "sampled_count": self._num_timesteps_sampled,
            "est_size_bytes": self._est_size_bytes,
            "num_entries": len(self._storage),
        }
        if debug:
            data.update(self._evicted_hit_stats.stats())
        return data


# Visible for testing.
_local_replay_buffer = None


class LocalReservoirReplayBuffer(LocalReplayBuffer):
    """A replay buffer shard.

    Ray actors are single-threaded, so for scalability multiple replay actors
    may be created to increase parallelism."""

    def __init__(
        self,
        num_shards=1,
        learning_starts=1000,
        buffer_size=10000,
        replay_batch_size=1,
        replay_mode="independent",
        replay_sequence_length=1,
    ):
        self.replay_starts = learning_starts // num_shards
        self.buffer_size = buffer_size // num_shards
        self.replay_batch_size = replay_batch_size
        self.replay_mode = replay_mode
        self.replay_sequence_length = replay_sequence_length

        assert replay_sequence_length == 1, "NFSP debugging"
        if replay_sequence_length > 1:
            self.replay_batch_size = int(
                max(1, replay_batch_size // replay_sequence_length)
            )
            logger.info(
                "Since replay_sequence_length={} and replay_batch_size={}, "
                "we will replay {} sequences at a time.".format(
                    replay_sequence_length, replay_batch_size, self.replay_batch_size
                )
            )

        if replay_mode not in ["lockstep", "independent"]:
            raise ValueError("Unsupported replay mode: {}".format(replay_mode))

        def gen_replay():
            while True:
                yield self.replay()

        ParallelIteratorWorker.__init__(self, gen_replay, False)

        def new_buffer():
            return ReservoirReplayBuffer(size=self.buffer_size)

        self.replay_buffers = collections.defaultdict(new_buffer)

        # Metrics
        self.add_batch_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.update_priorities_timer = TimerStat()
        self.num_added = 0

        # Make externally accessible for testing.
        global _local_replay_buffer
        _local_replay_buffer = self
        # If set, return this instead of the usual data for testing.
        self._fake_batch = None

    @staticmethod
    def get_instance_for_testing():
        global _local_replay_buffer
        return _local_replay_buffer

    def get_host(self):
        return platform.node()

    def add_batch(self, batch):
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
        with self.add_batch_timer:
            if self.replay_mode == "lockstep":
                for s in batch.timeslices(self.replay_sequence_length):
                    self.replay_buffers[_ALL_POLICIES].add(s)
            else:
                for policy_id, b in batch.policy_batches.items():
                    for s in b.timeslices(self.replay_sequence_length):
                        self.replay_buffers[policy_id].add(s)
        self.num_added += batch.count

    def replay(self):
        # print(f"REPLAY!")
        if self._fake_batch:
            fake_batch = SampleBatch(self._fake_batch)
            return MultiAgentBatch({DEFAULT_POLICY_ID: fake_batch}, fake_batch.count)

        if self.num_added < self.replay_starts:
            return None

        with self.replay_timer:
            assert self.replay_mode == "independent"  # debugging nfsp
            if self.replay_mode == "lockstep":
                return self.replay_buffers[_ALL_POLICIES].sample(self.replay_batch_size)
            else:
                samples = {}
                for policy_id, replay_buffer in self.replay_buffers.items():
                    # print(replay_buffer.stats())
                    samples[policy_id] = replay_buffer.sample(self.replay_batch_size)
                return MultiAgentBatch(samples, self.replay_batch_size)

    def update_priorities(self, prio_dict):
        pass

    def stats(self, debug=False):
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3
            ),
        }
        for policy_id, replay_buffer in self.replay_buffers.items():
            stat.update(
                {"policy_{}".format(policy_id): replay_buffer.stats(debug=debug)}
            )
        return stat


ReservoirReplayActor = ray.remote(num_cpus=0)(LocalReservoirReplayBuffer)
