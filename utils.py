import numpy as np

def euler2rotm(theta):

    R_x = np.array([[ 1,                     0,                     0 ],
                    [ 0, np.math.cos(theta[0]), -np.math.sin(theta[0])],
                    [ 0, np.math.sin(theta[0]),  np.math.cos(theta[0])]
                    ])
    R_y = np.array([[ np.math.cos(theta[1]), 0,  np.math.sin(theta[1])],
                    [                     0, 1,                      0],
                    [-np.math.sin(theta[1]), 0,  np.math.cos(theta[1])]
                    ])         
    R_z = np.array([[ np.math.cos(theta[2]), -np.math.sin(theta[2]), 0],
                    [ np.math.sin(theta[2]),  np.math.cos(theta[2]), 0],
                    [                     0,                      0, 1]
                    ])     
           
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R