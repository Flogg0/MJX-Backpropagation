from typing import Tuple

import jax
import mujoco
import numpy as np
from brax.base import Motion
from brax.base import Transform
from brax.envs.base import Env
from brax.envs.base import State
from jax import numpy as jp
from mujoco import mjx

# @title MjxEnv


class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
    ):
        """Initializes MjxEnv.

        Args:
          mj_model: mujoco.MjModel
          physics_steps_per_control_step: the number of times to step the physics
            pipeline for each environment step
        """
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.device_put(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""

        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return (
                mjx.step(self.sys, data),
                None,
            )

        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def observation_size(self) -> int:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        return self.sys.nu

    @property
    def backend(self) -> str:
        return "mjx"

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[self.model.body_rootid[np.arange(1, self.model.nbody)]]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd


class Qube_mjx(MjxEnv):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        path = "environments/xml/qube.xml"
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 1
        mj_model.opt.ls_iterations = 6

        physics_steps_per_control_step = 5
        kwargs["physics_steps_per_control_step"] = kwargs.get(
            "physics_steps_per_control_step", physics_steps_per_control_step
        )

        super().__init__(mj_model=mj_model, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        """ min_z, max_z = self._healthy_z_range
        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
    is_healthy = jp.where(data.qpos[2] < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(
        data.qpos[2] > max_z, x=0.0, y=is_healthy
    )
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(data, action)
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
 """
        return state.replace(pipeline_state=data)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )
