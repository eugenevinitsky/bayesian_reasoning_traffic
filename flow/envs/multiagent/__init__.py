"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.envs.multiagent.base import MultiEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv

from flow.envs.multiagent.ring.accel import MultiAgentAccelEnv
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.multiagent.highway import MultiAgentHighwayPOEnv
from flow.envs.multiagent.bayesian_0_no_grid_env import Bayesian0NoGridEnv
from flow.envs.multiagent.bayesian_1_inference_env import Bayesian1InferenceEnv



__all__ = ['MultiEnv', 'MultiAgentAccelEnv', 'MultiWaveAttenuationPOEnv',
           'MultiTrafficLightGridPOEnv', 'MultiAgentHighwayPOEnv',
           'Bayesian0NoGridEnv', 'Bayesian1InferenceEnv']
