import numpy as np

class HDV_predict():
    def __init__(self,N,dt) -> None:
        self.prediction = []
        self.N = N
        self.dt = dt

    def kinematic_bicycle_model(self, x,y,psi,v,delta_t,l_f,l_r,a,delta_f):
    # % x is position in longitudinal direction
    # % y is position in lateral direction
    # % psi is heading angle
    # % v is velocity (norm of velocity in x and y directions)
    # % delta_t is sampling time
    # % l_f is the length of the car from center of gravity to the front end
    # % l_r is the length of the car from center of gravity to the rear end
    # % a is acceleration which is control input
    # % delta_f is steering angle which is control input 

        beta = np.arctan(((l_r/(l_f+l_r))*np.tan(delta_f)))
        x_new = x + delta_t*v*np.cos(psi+beta) 
        y_new = y + delta_t*v*np.sin(psi+beta) 
        psi_new = psi + delta_t*(v*np.cos(beta)/(l_r+l_f)*np.tan(delta_f)) 
        v_new = v + delta_t*a

        return x_new,y_new,psi_new,v_new
    
    def predict(self, observation,is_absolute=True):
        '''
        observation is a flattened vector of size 1 x (number of states) (default: 5)
        '''
        prediction = np.zeros((self.N+1,len(observation)))
        prediction[0,:] = observation
        x_new, y_new, psi_new, v_new = observation[0],observation[1],observation[4],np.sqrt(observation[2]**2 + observation[3]**2)
        for i in range(self.N):
            x_new,y_new,psi_new,v_new = self.kinematic_bicycle_model(x_new, y_new, psi_new, v_new, self.dt, 2.5,2.5, 0, 0)
            prediction[i+1,:] = np.array([x_new,y_new,v_new*np.cos(psi_new),v_new*np.sin(psi_new),psi_new])
        return prediction
        
