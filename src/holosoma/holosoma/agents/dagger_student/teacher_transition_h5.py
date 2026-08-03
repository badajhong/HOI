"""Bounded H5 export for teacher-executed DAgger transitions.

The on-disk datasets intentionally use the flat schema consumed by
``TeacherReplayBuffer``.  A fixed-size reservoir keeps a representative sample
without allowing a long student run to grow the file indefinitely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FASTSAC_TEACHER_BUFFER_FORMAT = "holosoma_fastsac_teacher_buffer"
FASTSAC_TEACHER_BUFFER_VERSION = 2

TEACHER_TRANSITION_FIELDS = (
    "observations",
    "critic_observations",
    "actions",
    "rewards",
    "dones",
    "truncations",
    "next_observations",
    "next_critic_observations",
)

_PENDING_GROUP = "__pending_reservoir_update__"


@dataclass(frozen=True)
class TeacherTransitionWriteStats:
    accepted: int
    seen: int
    saved: int


def infer_observation_mode(env: Any, actor_obs_keys: Sequence[str]) -> str:
    """Return the observation-interface marker stored in a teacher buffer.

    In particular, an ``ae_latent`` group is never described as compatible
    unless the environment actually has the corresponding AE checkpoint.
    """

    keys = tuple(str(key) for key in actor_obs_keys)
    if "ae_latent" in keys:
        if getattr(env, "di_pro_ae", None):
            return "di_pro_latent"
        if getattr(env, "di_ae", None):
            return "di_latent"
        if getattr(env, "ir_ae", None):
            return "ir_latent"
        raise ValueError(
            "The actor observation includes 'ae_latent', but no IR/DI/DI-pro AE checkpoint is configured. "
            "Refusing to label the exported transitions as FastSAC-compatible."
        )

    observation_manager = getattr(env, "observation_manager", None)
    groups = getattr(getattr(observation_manager, "cfg", None), "groups", {})
    actor_group = groups.get("actor_obs") if isinstance(groups, Mapping) else None
    actor_terms = getattr(actor_group, "terms", {})
    if keys == ("actor_obs",) and "interaction_representation" in actor_terms:
        return "direct_ir"
    return "flat"


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
    return np.asarray(value)


class TeacherTransitionH5Writer:
    """Crash-recoverable, fixed-capacity reservoir stored in HDF5.

    The file is opened, flushed, and closed for each rollout append.  A small
    pending group journals the selected reservoir rows so an interrupted append
    can be replayed on the next open.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_transitions: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        actor_obs_keys: Sequence[str],
        critic_obs_keys: Sequence[str],
        observation_mode: str,
        seed: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if int(max_transitions) <= 0:
            raise ValueError(f"teacher-buffer max_transitions must be positive, got {max_transitions}.")
        if min(int(actor_obs_dim), int(critic_obs_dim), int(action_dim)) <= 0:
            raise ValueError("Teacher-buffer observation and action dimensions must be positive.")

        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_transitions = int(max_transitions)
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.action_dim = int(action_dim)
        self.actor_obs_keys = tuple(str(key) for key in actor_obs_keys)
        self.critic_obs_keys = tuple(str(key) for key in critic_obs_keys)
        self.observation_mode = str(observation_mode)
        self.seed = int(seed)
        self.metadata = dict(metadata or {})
        self.seen = 0
        self.saved = 0

        # Import h5py only when export is enabled.
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise ImportError("h5py is required when DAgger teacher-buffer export is enabled.") from exc
        self._h5py = h5py
        self._initialize_or_validate_file()

    @property
    def _tail_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "observations": (self.actor_obs_dim,),
            "critic_observations": (self.critic_obs_dim,),
            "actions": (self.action_dim,),
            "rewards": (),
            "dones": (),
            "truncations": (),
            "next_observations": (self.actor_obs_dim,),
            "next_critic_observations": (self.critic_obs_dim,),
        }

    @staticmethod
    def _attr_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _expected_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "format": FASTSAC_TEACHER_BUFFER_FORMAT,
            "format_version": FASTSAC_TEACHER_BUFFER_VERSION,
            "source": "r1_student_teacher_executed",
            "teacher_only": True,
            "action_source": "executed_clipped_teacher_action",
            "observation_layout": "concatenated_observation_groups",
            "observation_mode": self.observation_mode,
            "actor_obs_keys": json.dumps(list(self.actor_obs_keys)),
            "critic_obs_keys": json.dumps(list(self.critic_obs_keys)),
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "action_dim": self.action_dim,
            "reservoir_sampling": True,
            "reservoir_capacity": self.max_transitions,
            "reservoir_seed": self.seed,
        }
        for key, value in self.metadata.items():
            attrs[f"source_{key}"] = (
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
            )
        return attrs

    def _initialize_or_validate_file(self) -> None:
        with self._h5py.File(self.path, "a") as output:
            is_new = "format" not in output.attrs and not output.keys()
            if is_new:
                for key, value in self._expected_attrs().items():
                    output.attrs[key] = value
                output.attrs["num_transitions"] = 0
                output.attrs["num_seen_transitions"] = 0
                rng = np.random.default_rng(self.seed)
                output.attrs["reservoir_rng_state"] = json.dumps(rng.bit_generator.state)
                chunk_rows = min(4096, self.max_transitions)
                for name, tail_shape in self._tail_shapes.items():
                    dtype = np.bool_ if name in {"dones", "truncations"} else np.float32
                    output.create_dataset(
                        name,
                        shape=(0, *tail_shape),
                        maxshape=(self.max_transitions, *tail_shape),
                        chunks=(chunk_rows, *tail_shape),
                        dtype=dtype,
                        compression="lzf",
                        shuffle=True,
                    )
                output.flush()

            self._validate_metadata(output)
            self._recover_pending_update(output)
            self._validate_datasets(output)
            self.seen = int(output.attrs["num_seen_transitions"])
            self.saved = int(output.attrs["num_transitions"])

    def _validate_metadata(self, output: Any) -> None:
        expected = self._expected_attrs()
        for name, expected_value in expected.items():
            if name not in output.attrs:
                raise ValueError(f"Existing teacher buffer is missing metadata attribute '{name}': {self.path}")
            actual = output.attrs[name]
            if isinstance(expected_value, str):
                matches = self._attr_text(actual) == expected_value
            elif isinstance(expected_value, bool):
                matches = bool(actual) is expected_value
            elif isinstance(expected_value, float):
                matches = bool(
                    np.isclose(float(actual), expected_value, rtol=0.0, atol=1e-12)
                )
            else:
                matches = int(actual) == int(expected_value)
            if not matches:
                raise ValueError(
                    f"Existing teacher-buffer metadata mismatch for '{name}': "
                    f"file={actual!r}, configured={expected_value!r}."
                )

    def _validate_datasets(self, output: Any) -> None:
        missing = [name for name in TEACHER_TRANSITION_FIELDS if name not in output]
        if missing:
            raise ValueError(f"Teacher buffer is missing datasets: {missing}")
        saved = int(output.attrs["num_transitions"])
        seen = int(output.attrs["num_seen_transitions"])
        if not 0 <= saved <= min(seen, self.max_transitions):
            raise ValueError(
                f"Invalid teacher-buffer counts: saved={saved}, seen={seen}, capacity={self.max_transitions}."
            )
        for name, tail_shape in self._tail_shapes.items():
            dataset = output[name]
            expected_shape = (saved, *tail_shape)
            if tuple(dataset.shape) != expected_shape:
                raise ValueError(
                    f"Teacher-buffer dataset '{name}' has shape {tuple(dataset.shape)}, expected {expected_shape}."
                )

    def _recover_pending_update(self, output: Any) -> None:
        if _PENDING_GROUP not in output:
            return
        pending = output[_PENDING_GROUP]
        required = {"slots", *TEACHER_TRANSITION_FIELDS}
        missing = sorted(required - set(pending.keys()))
        required_attrs = {"new_seen", "new_saved", "rng_state_after"}
        missing_attrs = sorted(required_attrs - set(pending.attrs.keys()))
        if missing or missing_attrs:
            raise RuntimeError(
                "Teacher-buffer pending journal is incomplete; refusing to guess after an interrupted H5 write. "
                f"missing datasets={missing}, missing attrs={missing_attrs}, file={self.path}."
            )

        new_saved = int(pending.attrs["new_saved"])
        for name in TEACHER_TRANSITION_FIELDS:
            output[name].resize((new_saved, *self._tail_shapes[name]))
        slots = np.asarray(pending["slots"][()], dtype=np.int64)
        if slots.size:
            for name in TEACHER_TRANSITION_FIELDS:
                output[name][slots] = pending[name][()]
        output.flush()
        output.attrs["num_seen_transitions"] = int(pending.attrs["new_seen"])
        output.attrs["num_transitions"] = new_saved
        output.attrs["reservoir_rng_state"] = self._attr_text(pending.attrs["rng_state_after"])
        output.flush()
        del output[_PENDING_GROUP]
        output.flush()

    def _normalize_batch(self, batch: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], int]:
        missing = sorted(set(TEACHER_TRANSITION_FIELDS) - set(batch))
        if missing:
            raise ValueError(f"Teacher-transition batch is missing fields: {missing}")

        normalized: dict[str, np.ndarray] = {}
        count: int | None = None
        for name, tail_shape in self._tail_shapes.items():
            value = _to_numpy(batch[name])
            if not tail_shape and value.ndim == 2 and value.shape[1] == 1:
                value = value[:, 0]
            expected_ndim = 1 + len(tail_shape)
            if value.ndim != expected_ndim or tuple(value.shape[1:]) != tail_shape:
                raise ValueError(
                    f"Teacher-transition field '{name}' has shape {tuple(value.shape)}, "
                    f"expected (N, {', '.join(map(str, tail_shape))})" if tail_shape else
                    f"Teacher-transition field '{name}' has shape {tuple(value.shape)}, expected (N,)."
                )
            if count is None:
                count = int(value.shape[0])
            elif int(value.shape[0]) != count:
                raise ValueError(
                    f"Teacher-transition field '{name}' has {value.shape[0]} rows, expected {count}."
                )
            dtype = np.bool_ if name in {"dones", "truncations"} else np.float32
            value = np.ascontiguousarray(value, dtype=dtype)
            if dtype == np.float32 and not np.isfinite(value).all():
                raise ValueError(f"Teacher-transition field '{name}' contains NaN or infinity.")
            normalized[name] = value
        return normalized, int(count or 0)

    def _select_reservoir_rows(
        self,
        *,
        seen: int,
        saved: int,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        slot_to_source: dict[int, int] = {}
        fill_count = min(count, self.max_transitions - saved)
        for source_index in range(fill_count):
            slot_to_source[saved + source_index] = source_index

        for source_index in range(fill_count, count):
            stream_index = seen + source_index
            replacement = int(rng.integers(0, stream_index + 1))
            if replacement < self.max_transitions:
                # Only the last write to a duplicate slot matters. Keeping it
                # here exactly matches sequential reservoir sampling.
                slot_to_source[replacement] = source_index

        slots = np.asarray(sorted(slot_to_source), dtype=np.int64)
        sources = np.asarray([slot_to_source[int(slot)] for slot in slots], dtype=np.int64)
        new_saved = min(self.max_transitions, saved + count)
        return slots, sources, new_saved

    def append(self, batch: Mapping[str, Any]) -> TeacherTransitionWriteStats:
        arrays, count = self._normalize_batch(batch)
        if count == 0:
            return TeacherTransitionWriteStats(accepted=0, seen=self.seen, saved=self.saved)

        with self._h5py.File(self.path, "a") as output:
            self._validate_metadata(output)
            self._recover_pending_update(output)
            self._validate_datasets(output)
            seen = int(output.attrs["num_seen_transitions"])
            saved = int(output.attrs["num_transitions"])
            rng = np.random.default_rng()
            rng.bit_generator.state = json.loads(self._attr_text(output.attrs["reservoir_rng_state"]))
            slots, source_indices, new_saved = self._select_reservoir_rows(
                seen=seen,
                saved=saved,
                count=count,
                rng=rng,
            )
            new_seen = seen + count

            pending = output.create_group(_PENDING_GROUP)
            pending.attrs["new_seen"] = new_seen
            pending.attrs["new_saved"] = new_saved
            pending.attrs["rng_state_after"] = json.dumps(rng.bit_generator.state)
            pending.create_dataset("slots", data=slots, dtype=np.int64)
            for name in TEACHER_TRANSITION_FIELDS:
                pending.create_dataset(name, data=arrays[name][source_indices])
            output.flush()

            # Use the same recovery path for the live commit and for restart
            # recovery, making the journal operation idempotent.
            self._recover_pending_update(output)
            self._validate_datasets(output)
            self.seen = int(output.attrs["num_seen_transitions"])
            self.saved = int(output.attrs["num_transitions"])

        return TeacherTransitionWriteStats(accepted=count, seen=self.seen, saved=self.saved)
