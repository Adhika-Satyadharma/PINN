# ------------------------------------------------------
# Case:                 Simplified BFS (The domain is only the later part)
# Equation:             du/dx + dv/dy = 0
#                       u * du/dx + v * du/dy = -1/rho * dp/dx + nu * (d2u/dx2 + d2u/dy2)
#                       u * dv/dx + v * dv/dy = -1/rho * dp/dy + nu * (d2v/dx2 + d2v/dy2)
# Domain :              x, y = [0, 20], [-1, 1]
# Boundary Condition:   Inlet = Outlet = -(2y-1)2 + 1
#                       Wall (u = v = 0)
# Constant:             rho = 1, mew = 0.1
# Analytic solution:    -
# ------------------------------------------------------

# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from Case_Details import *
from NN_Class import *
from General_Functions import *


# Dynamic Params
DTYPE='float32'
N_i = 0          # Initial
N_b = 500          # Boundary
N_r = 4500          # Interior
N = 1000000         # Epoch
Backup_Interval = [200, 1000, 3000, 5000, 10000, 20000, 30000, 50000, 100000, 200000, 300000, 500000, 800000]
Speical_LF = []
BFGS_Criteria = ["iter", 100000]

NN_Neurons = [32, 32, 32, 32]
Load_NN = False
NN_Save_Name = "SBFS"
Data_Assimilation_File = []     #["", "" , ""]
Print_Every = 100   # Epoch
Rand_Seed = 0

# Some Initialization
print("Initializing")
tf.keras.backend.set_floatx(DTYPE)                      # Initialize type
CD = Channel_Flow(                           # Initialize Case
    [0., 30., -5., 1.],                                      # Domain (Main axis = x axis in image)
    [1., 4./300.],                                                   # Material / Constant
    [N_i, N_b, N_r],                                        # Points
    Data_Assimilation_File,                             # If there is some external data(s)
    DTYPE)
tf.random.set_seed(Rand_Seed)                           # Initialize Random Seed

# Derived
Create_Dir()
[lb, ub] = CD.Call_Boundary()
Case_Dim = CD.Call_Case_Dim()
Write_Case_Setup(CD, NN_Neurons, N, Rand_Seed)
PR = Print_Results(CD, lb, ub, DTYPE)

# Prepare Points
print("Generate Points & NN")
[xbi_data, ubi_data, xr_data] = CD.Generate_Points()
if Load_NN:
    NN_Model = tf.keras.models.load_model('Reports/'+NN_Save_Name)
else:
    NN_Model = PINN_NeuralNet([lb, ub], Case_Dim, NN_Neurons)
    NN_Model.build(input_shape=(None,Case_Dim[0]))
CD.Set_Model(NN_Model)
NN_Solver = PINN_Channel_Flow(NN_Model, CD, PR, xr_data, Backup_Interval, Speical_LF)     # PINN SOLVER

# Training
print("Training")
t0 = time()
Adam_Iter = NN_Solver.solve_with_TFoptimizer(xbi_data, ubi_data, N, Print_Every, t0, BFGS_Criteria)
print("Switching to L-BFGS")
print("L-BFGS iteration = " + str(N - Adam_Iter))
print(1.0 * np.finfo(float).eps)
NN_Solver.solve_with_ScipyOptimizer(xbi_data, ubi_data, method='L-BFGS-B',
                                options = {'maxiter': N - Adam_Iter,
                                           'maxfun': 50000,
                                           'maxcor': 50,
                                           'maxls': 50,
                                           'ftol': 1.0 * np.finfo(float).eps,
                                           'gtol': 1.0 * np.finfo(float).eps})

print('\nComputation time: {} seconds'.format(time()-t0))


# Print Results
print("Exporting Results")
NN_Save_Name2 = 'Reports/' + NN_Save_Name
if Load_NN:
    NN_Save_Name2 = 'Reports/'+NN_Save_Name+"_Next_Attempt"

Write_Array_Data("Reports/Hist_Data", NN_Solver.hist, 1, "NP")
PR.Execute_All(NN_Solver, "F", NN_Save_Name2)
