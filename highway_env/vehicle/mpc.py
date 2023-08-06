# Test
# import gymnasium as gym
# env = gym.make('fc-highway-v1', render_mode='rgb_array')
# obs, info = env.reset()
# from highway_env.vehicle.prediction_traj import Obs2PredictionModule
# pred_mod = Obs2PredictionModule(10,0.1,5)
# out = pred_mod.predict(obs,env,"init")

import casadi as ca
import time
import numpy as np
from utils.simulation_tools import safety_constraint_planner, virtual_car
from highway_env.vehicle.prediction import HDV_predict

class MPC():
	def __init__(self,
					N          = 15,     # timesteps in MPC Horizon
					DT         = 0.1,    # discretization time between timesteps (s)
					L_F        = 5/2, 		# distance from CoG to front axle (m)
					L_R        = 5/2, 		# distance from CoG to rear axle (m)
					V_MIN      = 10,    # min/max velocity constraint (m/s)
					V_MAX      = 40.0,     
					A_MIN      = -4,   # min/max acceleration constraint (m/s^2)
					A_MAX      =  2,     
					A_DOT_MIN  = -1,   # min/max jerk constraint (m/s^3)
					A_DOT_MAX  =  1,
					DF_MIN		= -np.pi / 3,
					DF_MAX		=	np.pi / 3,
					DF_DOT_MIN	=	-1,
					DF_DOT_MAX	=	1,
					d_min = 6,
					w = 2,
					Q = [1,1,30], #y, v, psi
					R = [1,1,1,1]):
		self.opti = ca.Opti()  
		self.eps = 0.001

		#Parameters
		for key in list(locals()):
			if key == 'self':
				pass
			elif key == 'Q':
				self.Q = ca.diag(Q)
			elif key == 'R':
				self.R = ca.diag(R)
			else:
				setattr(self, '%s' % key, locals()[key])

		self.laneWidth = 4
		self.road_right = 0
		self.numberlanes = 2
		# self.road_center = self.road_right - self.laneWidth
		self.road_left = self.road_right + self.numberlanes* self.laneWidth
		self.safety_dist_planner = d_min + (self.L_F + self.L_R)/2
		self.feasibility = True       
		#State Decision Variables
		self.nx = 4
		self.z_dv = self.opti.variable(self.N+1, self.nx)

		#Input Decision Variables
		self.nu = 2
		self.u_dv = self.opti.variable(self.N, self.nu)
		self.d = self.opti.parameter(self.N,1)
		self.z_curr = self.opti.parameter(4)
		self.u_prev = self.opti.parameter(2)

		#Slack variables
		self.sl = self.opti.variable(self.N+1,1)
		# self.sl = self.opti.variable(1)

		self.is_npc = False


	def _add_constraints(self,mode,ey_curr_lane,z_npc_arr_,vehicle_lengths_,z_platoon=np.array([]),platoon_ref=np.array([])):
		# print(z_platoon,platoon_ref)
		# State Bound Constraints
		self.opti.subject_to( self.opti.bounded(self.V_MIN, self.z_dv[:,3], self.V_MAX) )	

		# Initial State Constraints
		self.opti.subject_to( self.z_dv[0,0] == self.z_curr[0] )
		self.opti.subject_to( self.z_dv[0,1] == self.z_curr[1] )
		self.opti.subject_to( self.z_dv[0,2] == self.z_curr[2] )
		self.opti.subject_to( self.z_dv[0,3] == self.z_curr[3] )

		#slack
		self.opti.subject_to(0 <= self.sl)		
		
		#Safety constraint
		if self.is_npc and mode == 'LC':
			x_min, x_max, v_lc, bound = safety_constraint_planner(self.z0,z_npc_arr_, vehicle_lengths_, ey_curr_lane, self.ey_ref, self.N, self.DT,z_platoon=z_platoon,ref = platoon_ref, v_ref=self.v_ref) #THIS USED TO BE V_REF
			self.v_lc = v_lc
			# print(f'x_ego: {self.z0[0]}, x_min: {x_min[0]}, x_max: {x_max[0]}')
			# print(v_lc, bound)
		
		#Stage constraints
		for i in range(self.N):
			# State Dynamics Constraints
			self.beta = ca.atan((self.L_R/(self.L_F + self.L_R)) * ca.tan(self.u_dv[i,1]))
			self.opti.subject_to( self.z_dv[i+1,0]  == self.z_dv[i,0]   + self.DT*self.z_dv[i,3]*ca.cos(self.z_dv[i,2] + self.beta))
			self.opti.subject_to( self.z_dv[i+1,1]  == self.z_dv[i,1]   + self.DT*self.z_dv[i,3]*ca.sin(self.z_dv[i,2] + self.beta))
			self.opti.subject_to( self.z_dv[i+1,2]  == self.z_dv[i,2]   + self.DT*(self.z_dv[i,3]/self.L_R)*ca.sin(self.beta))
			self.opti.subject_to( self.z_dv[i+1,3]  == self.z_dv[i,3]   + self.DT*(self.u_dv[i,0]) )
			self.opti.subject_to(self.z_dv[i,1] + self.w/2 <= self.road_left) # road boundary constraint 
			self.opti.subject_to(-self.z_dv[i,1] + self.w/2 <= -self.road_right) # road boundary constraint
			
			if self.is_npc and mode == 'LC':
				#Safety constraint (Stage constraints)
				if x_min[i] + self.safety_dist_planner <= x_max[i] -self.safety_dist_planner:
					self.opti.subject_to( self.opti.bounded(x_min[i] + self.safety_dist_planner, self.z_dv[i,0], x_max[i] - self.safety_dist_planner))
				else:
					self.feasibility = False

				if i >= self.N-2: #previously it was N/2. LC was infeasible for short horizons as it is too demanding and physically impossible to satisfy conditions below given the dynamic constraints
				# if i == self.N:
					if bound == 'min':
						self.opti.subject_to( self.opti.bounded(v_lc, self.z_dv[i,3], self.V_MAX) )
					elif bound == 'max':
						self.opti.subject_to( self.opti.bounded(self.V_MIN, self.z_dv[i,3], v_lc) )
					elif bound is None:
						pass
					else: #bound == 'between'
						self.opti.subject_to( self.opti.bounded(v_lc[0], self.z_dv[i,3], v_lc[1]) )
					self.opti.subject_to( self.opti.bounded(-0.2,self.z_dv[i,1] - self.ey_ref,0.2) )

			else:
				#LK mode: maintain current lane
				self.opti.subject_to( self.opti.bounded(ey_curr_lane - 0.1,self.z_dv[i,1],ey_curr_lane + 0.1) )

		# Input Bound Constraints
		self.opti.subject_to( self.opti.bounded(self.A_MIN,  self.u_dv[:,0], self.A_MAX) )
		self.opti.subject_to( self.opti.bounded(self.DF_MIN,  self.u_dv[:,1], self.DF_MAX) )
		
		#Road bound - state constraints (Terminal)
		self.opti.subject_to(self.z_dv[self.N,1] + self.w/2 <= self.road_left) # road boundary constraint 
		self.opti.subject_to(-self.z_dv[self.N,1] + self.w/2 <= -self.road_right) # road boundary constraint

		# # # Input Rate Bound Constraints

		self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT, 
												 self.u_dv[0,0] - self.u_prev[0],
												 self.A_DOT_MAX*self.DT ) )

		self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT , 
												 self.u_dv[0,1]  - self.u_prev[1],
												 self.DF_DOT_MAX*self.DT ) )

		for i in range(self.N-1):
			self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT ,  
													 self.u_dv[i+1,0] - self.u_dv[i,0],
													 self.A_DOT_MAX*self.DT ))
			self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT , 
													 self.u_dv[i+1,1]  - self.u_dv[i,1],
													 self.DF_DOT_MAX*self.DT) )

		#Safety constraints
		if self.is_npc and mode == 'LC':
			if bound == 'min':
				self.opti.subject_to( self.opti.bounded(v_lc, self.z_dv[self.N,3], self.V_MAX) )
			elif bound == 'max':
				self.opti.subject_to( self.opti.bounded(self.V_MIN, self.z_dv[self.N,3], v_lc) )
			elif bound is None:
				pass
			else:
				self.opti.subject_to( self.opti.bounded(v_lc[0], self.z_dv[self.N,3], v_lc[1]) )
			self.opti.subject_to( self.opti.bounded(-0.2,self.z_dv[self.N,1] - self.ey_ref,0.2) )

		else:
			# pass
		# elif mode=='LK':
			#LK Mode (ACC)
			if hasattr(self, 'z_front_npc'):
				# print('z_front',self.z_front_npc)
				isfrontplatoon, ind = self._front_platoon(z_platoon)
				if isfrontplatoon:
					npc_x_pred = platoon_ref[ind][:,0]
					self.z_front_l  = 5
					for k in range(self.N):			
						self.opti.subject_to( self.safety_dist_planner < npc_x_pred[k]  - self.L_F/2  - self.z_dv[k,0] + self.sl[i])
				else:
					npc_x_pred = self.z_front_npc[0] + self.z_front_npc[3]*range(self.N+1)*self.DT	
					for k in range(self.N):			
						self.opti.subject_to( self.safety_dist_planner + 10 < npc_x_pred[k]  - self.L_F/2  - self.z_dv[k,0] + self.sl[i])
				if npc_x_pred[0] - self.z0[0] < 20:
					self.opti.subject_to(self.z_dv[self.N,3] <= self.z_front_vel)
			

		#Terminal constraints
		if mode == 'LC' and self.is_npc:
			if x_min[-1] + self.safety_dist_planner <= x_max[-1] -self.safety_dist_planner:
				self.opti.subject_to( self.opti.bounded(x_min[-1] + self.safety_dist_planner, self.z_dv[self.N,0], x_max[-1] -self.safety_dist_planner))
			else:
				# print('-----------------------------------')
				self.feasibility = False

	def _add_costs(self, mode):
		cost = 0
		for i in range(self.N+1):
			if mode == 'LC':
				cost += self.Q[0,0] * (self.z_dv[i,1]-self.ey_ref)**2 + self.Q[1,1] * (self.z_dv[i,3]-self.v_ref)**2 + self.Q[2,2]* (self.z_dv[i,2])**2
			elif mode == 'LK':
				cost += self.Q[0,0] * (self.z_dv[i,1]-self.ey_ref)**2 + self.Q[1,1] * (self.z_dv[i,3]-self.v_ref)**2 + self.Q[2,2]* (self.z_dv[i,2])**2 
				# if hasattr(self,'z_front_npc'):
				# 	if ((self.z_front_npc[0] - self.z0[0]) < 20) and (self.z_front_npc[3] - self.z0[3]) < 0:
				# 		if self.z_front_npc[0] + self.z_front_npc[3]*self.DT*i - self.z_dv[i,0] > self.z_front_l/2 - self.safety_dist_planner:
				# 			cost += 5*(self.z_front_npc[0] + self.z_front_npc[3]*self.DT*i - self.z_dv[i,0] - self.z_front_l/2 - self.safety_dist_planner)**2

			if i < self.N-1:
				for k in range(self.nu):
					cost += self.R[2*k+1,2*k+1]*(self.u_dv[i+1,k] - self.u_dv[i,k])**2 + self.R[2*k,2*k]*(self.u_dv[i,k])**2  

		cost += 2000*ca.sum1(self.sl) #+ 2000*ca.sum1(self.sl) )
		self.mode = mode 
		self.opti.minimize(cost)

	def _update_reference(self,ey_ref,v_ref):
		self.ey_ref = ey_ref
		self.v_ref = v_ref

	def _update_initial_condition(self, z0):
		self.z0 = z0
		self.opti.set_value(self.z_curr[0], z0[0])
		self.opti.set_value(self.z_curr[1], z0[1])  
		self.opti.set_value(self.z_curr[2], z0[2]) #heading
		self.opti.set_value(self.z_curr[3], z0[3]) #velocity
			
	def _update_prev_input(self,u_prev):
		self.opti.set_value(self.u_prev, u_prev)

	def _observe_environment(self,z_npc_arr,vehicle_lengths,z_platoon,y_other_lane,LC_P_F):
		v_ref = self.z0[3]
		minimum = self.z0 + np.array([100,0,0,self.v_ref-self.z0[3]])

		if LC_P_F:
			for z in z_platoon:
				if not (abs(self.z0[0] - z[0]) < self.eps and abs(self.z0[1] - z[1]) < self.eps) and not (abs(z[1]-y_other_lane) < 0.5):
					z_virtual_car = virtual_car(z,y_other_lane)
					z_npc_arr.append(z_virtual_car)
					vehicle_lengths.append(self.L_F * 2)
				elif not (abs(self.z0[0] - z[0]) < self.eps and abs(self.z0[1] - z[1]) < self.eps):
					z_npc_arr.append(np.array(z))
					vehicle_lengths.append(self.L_F *2)
		# else:
		# 	for z in z_platoon:
		# 		if not (abs(self.z0[0] - z[0]) < self.eps and abs(self.z0[1] - z[1]) < self.eps):
		# 			z_npc_arr.append(z)
		# 			vehicle_lengths.append(self.L_F *2)

		for ind, z_npc in enumerate(z_npc_arr):
			#Pick the front car
			if z_npc[0] > self.z0[0] and abs(z_npc[1] - self.z0[1]) < 2:
				if z_npc[0] < minimum[0]:
					minimum = z_npc
					self.z_front_l = vehicle_lengths[ind]

		if minimum[0] != self.z0[0] + 100:
			self.z_front_npc = minimum
			v_ref = minimum[3]
		return v_ref

	def _front_platoon(self,z_platoon):
		isfrontplatoon = False
		ind = -1
		for i, z in enumerate(z_platoon):
			
			if np.all(abs(np.array(self.z_front_npc) - np.array(z)) < 0.01):
				isfrontplatoon = True
				ind = i
		return isfrontplatoon, ind

	def _update_npc_states(self, z_npc, vehicle_lengths,z_platoon, y_other_lane, LC_P_F):
		self.is_npc = True
		self.z_npc = z_npc
		if len(z_npc) == 0:
			self.is_npc = False

		if self.is_npc:
			self.z_front_vel = self._observe_environment(z_npc,vehicle_lengths,z_platoon,y_other_lane,LC_P_F)

	def _solve(self):
		p_opts = dict(print_time=False, verbose=False)
		s_opts = dict(print_level=0)
		self.opti.solver('ipopt', p_opts, s_opts)
		st = time.time()
		if self.feasibility:
			try:
				sol = self.opti.solve()
				solve_time = time.time() - st
				# Optimal solution.
				u_mpc  = sol.value(self.u_dv)
				z_mpc  = sol.value(self.z_dv)
				is_opt = True
			except:
				if self.mode == 'LK' or self.mode =='GR':
					print(f'{self.mode}, {self.v_ref} CFTOCP Infeasible')
				else:
					print(f'{self.mode} CFTOCP Infeasible')
				# self.opti.debug.show_infeasibilities()
				# Suboptimal solution (e.g. timed out).
				u_mpc  = self.opti.debug.value(self.u_dv)
				z_mpc  = self.opti.debug.value(self.z_dv)
				# sl_mpc = self.opti.debug.value(self.sl_dv)
				is_opt = False
		else:
			is_opt = False
			solve_time  = -1
			u_mpc = np.array([[0,0]])
			pred_method = HDV_predict(self.N,self.DT)
			obs = [self.z0[0], self.z0[1], self.z0[3] * np.cos(self.z0[2]) , self.z0[3]* np.sin(self.z0[2]), self.z0[2]]
			z_mpc = pred_method.predict(obs,is_absolute=True)

		solve_time = time.time() - st
		
		sol_dict = {}
		sol_dict['u_control']  = u_mpc[0,:]     # control input to apply based on solution
		sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
		sol_dict['solve_time'] = solve_time  # how long the solver took in seconds
		sol_dict['u_mpc']      = u_mpc       # solution inputs (N by 2, see self.u_dv above) 
		sol_dict['z_mpc']      = z_mpc       # solution states (N+1 by 5, see self.z_dv above)
		# print(solve_time)
		return sol_dict
