"""Contains all available networks in Flow."""

# base network class
from flow.networks.base import Network

# custom networks
from flow.networks.bay_bridge import BayBridgeNetwork
from flow.networks.bay_bridge_toll import BayBridgeTollNetwork
from flow.networks.bottleneck import BottleneckNetwork
from flow.networks.figure_eight import FigureEightNetwork
from flow.networks.traffic_light_grid import TrafficLightGridNetwork
from flow.networks.highway import HighwayNetwork
from flow.networks.ring import RingNetwork
from flow.networks.merge import MergeNetwork
from flow.networks.multi_ring import MultiRingNetwork
from flow.networks.minicity import MiniCityNetwork
from flow.networks.highway_ramps import HighwayRampsNetwork
from flow.networks.bayesian_0 import Bayesian0Network
from flow.networks.bayesian_1 import Bayesian1Network, Bayesian3Network, Bayesian4Network, L2ObscuredNetwork

__all__ = [
    "Network", "BayBridgeNetwork", "BayBridgeTollNetwork",
    "BottleneckNetwork", "FigureEightNetwork", "TrafficLightGridNetwork",
    "HighwayNetwork", "RingNetwork", "MergeNetwork", "MultiRingNetwork",
    "MiniCityNetwork", "HighwayRampsNetwork", "Bayesian0Network",
    "Bayesian1Network", "Bayesian3Network", "Bayesian4Network", "L2ObscuredNetwork"
]
