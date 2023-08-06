import numpy as np
from highway_env.vehicle.prediction import HDV_predict

 #Prediction model for vehicles

#if CAV -- use motion prediction

#If HDV -- use a simple constant velocity prediction model
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from typing import List, Tuple, Optional, Callable

class Obs2PredictionModule():
    def __init__(self,N,dt,ns) -> None:
        self.predictions = []
        self.N = N     #prediction horizon length
        self.dt = dt #sampling time
        self.ns = ns

    def predict(self,observations,env,pred_rel,is_absolute=None) -> List:
        hdv_prediction_module = HDV_predict(self.N,self.dt) #HDV predicition module (constat vel zero steering)

        observations_ = observations.reshape((-1,self.ns))
        n_obs_vehicle = env.config["observation"]["observation_config"]["vehicles_count"]
        if is_absolute is None:
            is_absolute = env.config["observation"]["observation_config"]["absolute"]
        else:
            is_absolute = True
        prediction_output = []
        
        for i in range(len(observations)):
            predictions = []
            is_ego_found = False
            ego_pred_not_found = True
            ego_prediction = []
            if not is_absolute:
                observations_[i*n_obs_vehicle+1:(i+1)*n_obs_vehicle,:] = observations_[i*n_obs_vehicle+1:(i+1)*n_obs_vehicle,:] + observations_[i*n_obs_vehicle,:]
            for observation in observations_[i*n_obs_vehicle:(i+1)*n_obs_vehicle,:]:
                is_cav = False

                #check for controlled vehicles (CAV)
                for vehicle in env.controlled_vehicles:
                    if np.all(abs(vehicle.position - observation[0:2]) < 0.001):
                        is_cav = True
                        is_ego_found = True
                        break                
                if is_cav: #if cav
                    prediction = vehicle.get_traj(self.N,self.dt,observation,is_absolute)
                    vehicle.update_traj(prediction)
                    if is_ego_found and ego_pred_not_found:
                        ego_prediction = prediction
                        ego_pred_not_found = False

                    predictions.append(prediction)
                else: #if not CAV then    
                    prediction = hdv_prediction_module.predict(observation,is_absolute=is_absolute)
                    # print('__________-')
                    # print(prediction)
                    predictions.append(prediction)
            if not is_absolute:
                for j, prediction in enumerate(predictions):
                    if j > 0: #ego index is 0 and it is always absolute
                        if pred_rel == "init":
                            predictions[j] = prediction -  observations_[i*n_obs_vehicle,:]
                        else:
                            predictions[j] = prediction - ego_prediction #If relative to ego traj
            predictions = np.array(predictions)
            d1,d2,d3 = predictions.shape
            prediction_output.append(np.transpose(predictions, (1, 0, 2) ).reshape((d2,-1)).flatten())
        self.predictions = np.array(prediction_output)
        # print(self.predictions)
        return self.predictions # 1xn vector