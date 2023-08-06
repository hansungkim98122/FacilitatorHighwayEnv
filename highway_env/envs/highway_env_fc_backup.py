from operator import xor
from typing import Dict, Text

import numpy as np
from typing import Tuple, Optional
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
import random
#from highway_env.prediction import  

Observation = np.ndarray


class HighwayEnvFacilitator(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    n_vehicles = 5 #number of vehicles to observe (surrounding vehicles)
    n_a = 5
    n_s = int(5*5)
    
    @classmethod
    def default_config(cls) -> dict:
        # print("HighwayEnvFacilitator Class has been correctly called".center(80, '*'))
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                "type": "DiscreteMetaAction",
                }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 5,
                # "features": ["x", "y", "vx", "vy", "heading", "long_off", "lat_off", "ang_off", "lane_index", "cos_h", "sin_h"],
                "features": ["x", "y", "vx", "vy", "heading"],
                "absolute": False,
                "see_behind": True,
                "observe_intentions": True
                }
            },
            "initial_lane_id":1,
            "lanes_count": 2,
            "vehicles_count": 10,
            "controlled_vehicles": 3,
            "initial_lane_id": 1,
            "duration": 40,  # [s]
            "policy_frequency":5,
            "ego_spacing": 0.4,
            "vehicles_density": 2,                                       # zero for other lanes.                                       # lower speeds according to config["reward_speed_range"].
            "reward_speed_range": [10, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "ego_initial_speed": 30, # [m/s]
            "COLLISION_REWARD": 200,  # default=200
            "SPEED_REWARD": 1,  # default=1
            "HEADWAY_COST": 5,  # default=10
            "HEADWAY_TIME": 0.5,  # default=0.5[s]
            "SEPARATION_COST": 4,
            "LANE_CHANGE_REWARD": 5,   # The reward received at each lane change action.
            "LC_TIME_RESIDUAL_COST": 2
        })
        return config

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              is_training: bool = True,testing_seeds: int = 0
    ) -> Tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration

        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through `options["config"]`
        :return: the observation of the reset state
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])

        if is_training:
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            np.random.seed(testing_seeds)
            random.seed(testing_seeds)

        self.update_metadata()        
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self.vehicle_speed = []
        self.vehicle_pos = []
        self.cav_d_thresh = 15
        self.target_lane_index = 0
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == 'human':
            self.render()
        return np.asarray(obs).reshape((len(obs), -1)), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        agent_info = []
        average_speed = 0
        obs, reward, terminated, truncated, info = super().step(action)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
            average_speed += v.speed
        average_speed = average_speed / len(self.controlled_vehicles)

        self.vehicle_speed.append([v.speed for v in self.controlled_vehicles])
        self.vehicle_pos.append(([v.position[0] for v in self.controlled_vehicles])) 

        info["agents_info"] = agent_info
        info["average_speed"] = average_speed #average CAV platoon speed

        self.steps += 1

        info["vehicle_speed"] = np.array(self.vehicle_speed)
        info["vehicle_position"] = np.array(self.vehicle_pos)

        for vehicle in self.controlled_vehicles:
            # print(vehicle.lane_index)
            if vehicle.lane_index[2] == self.target_lane_index:
                vehicle.lc_time = self.time

            vehicle.local_reward = self._agent_reward(action, vehicle)
            vehicle.cooperative_reward = self._reward(action) #type: float

        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        reward = np.array(list(info["agents_rewards"]))

        # regional reward (cooperative reward. All agents share the cost)
        info["regional_rewards"] = tuple(vehicle.cooperative_reward for vehicle in self.controlled_vehicles)
        truncated = self._is_truncated()
        terminated = self._is_terminated()

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, terminated, truncated, info
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            #Pre-defined spawning of the ego vehicles
            vehicle = Vehicle.create_random(
                self.road,
                speed=self.config["ego_initial_speed"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        for _ in range(self.config["vehicles_count"]):
            is_max_rand = np.random.randint(2, size=1)[0]
            
            vehicle = other_vehicles_type.create_random_fc(self.road, speed=self.config["ego_initial_speed"], spacing=1,is_max = is_max_rand)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _compute_headway_distance(self, vehicle, ):
        headway_distance = 60
        for v in self.road.vehicles:
            if (v.lane_index == vehicle.lane_index) and (v.position[0] > vehicle.position[0]):
                hd = v.position[0] - vehicle.position[0]
                if hd < headway_distance:
                    headway_distance = hd
        return headway_distance

    def _reward(self, action: int) -> float:
    # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
            / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        agent_cur_lane = vehicle.lane_index[-1]
        agent_cur_lane ^= 1 #reassign var with xor with 1

        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * forward_speed)) if forward_speed > 0 else 0
        
        #compute minimum distance with the platoon
        min_d = self._compute_intervehicle_distance(vehicle)
        
        # compute overall reward
        # print([self.config["COLLISION_REWARD"] * (-1 * float(vehicle.crashed)),(self.config["SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)),self.config["LANE_CHANGE_REWARD"] * agent_cur_lane,self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0),self.config["SEPARATION_COST"] * (-(min_d-self.cav_d_thresh) if min_d > self.cav_d_thresh else 0), self.config["LC_TIME_RESIDUAL_COST"] * - self._lc_time_residual()])
        reward = self.config["COLLISION_REWARD"] * (-1 * float(vehicle.crashed)) \
                 + (self.config["SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["LANE_CHANGE_REWARD"] * agent_cur_lane\
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)  \
                 + self.config["SEPARATION_COST"] * (-(min_d-self.cav_d_thresh) if min_d > self.cav_d_thresh else 0) \
                 + self.config["LC_TIME_RESIDUAL_COST"] * - self._lc_time_residual()
    
        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(reward,
        #                         [self.config["COLLISION_REWARD"],
        #                          self.config["SPEED_REWARD"] + self.config["LANE_CHANGE_REWARD"] + self.config["HEADWAY_COST"]],
        #                         [0, 1])       
        reward *= float(vehicle.on_road)
        return reward

    def _lc_time_residual(self) -> float:
        temp = [vehicle.lc_time for vehicle in self.controlled_vehicles if vehicle.lc_time > 0]
        if temp:
            mean = sum(temp) / len(temp) #float
        else: 
            mean = 0 #sum should be zero
        sum_ = sum([abs(mean - vehicle.lc_time) for vehicle in self.controlled_vehicles])
        return sum_/10
    
    def _compute_intervehicle_distance(self, ego_vehicle: Vehicle) -> float:
        min_d = 1e6
        for v in self.controlled_vehicles:
            if np.any(v.position != ego_vehicle.position): #if v is not ego_vehicle
                if self._euclidean_distance(ego_vehicle, v) <= min_d:
                    min_d = self._euclidean_distance(ego_vehicle,v)
        return min_d
    
    def _euclidean_distance(self, v1: Vehicle, v2: Vehicle) -> float:
        d = np.sqrt((v1.position[0] - v2.position[0])**2 + (v1.position[1] - v2.position[1])**2)
        return d

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                (self.config["offroad_terminal"] and not self.vehicle.on_road) or self.steps >= self.config["duration"] * self.config["policy_frequency"])

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnvFacilitator):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
