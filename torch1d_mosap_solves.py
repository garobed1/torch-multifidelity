from bluest import BLUEProblem
import numpy as np
import os
import pickle

"""
Script for solving the MOSAP problem for torch1D-only covariances
"""

home = os.getenv('HOME')
# in_dir = home + "/bedonian1/torch1d_resample_sens_r8/"

# include 4s when done
out_dirs = [home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse"]
# correspond to out_dirs order
out_names = ["1D_Fine", "1D_Mid", "1D_Coarse"]

# do single output for now
out_use = ["exit_X", 0]

# statistical error threshold
eps_fac = 0.1




n_models = len(out_dirs)


# load outputs
qoi_val = {}
fqoin = out_dirs[0] + '/qoi_list.pickle' 
with open(fqoin, 'rb') as f:
    qoi_list = pickle.load(f)

for i, out_dir in enumerate(out_dirs):

    fqoiv = out_dir + f'/qoi_samples.pickle' 

    with open(fqoiv, 'rb') as f:
        qoi_val[out_names[i]] = pickle.load(f)

qoi_sizes = {
    'exit_p': 1, 
    'exit_d': 2, 
    'exit_v': 1,
    'exit_T': 2,
    'exit_X': 5,
    'heat_dep': 1
}      

#Estimate Covariances
C = {}
Corr = {}
for qoi in qoi_list:
    
    C[qoi] = np.zeros([qoi_sizes[qoi], n_models, n_models])
    Corr[qoi] = np.zeros([qoi_sizes[qoi], n_models, n_models])

    #concatenate results
    for j in range(qoi_val[out_names[0]][qoi].shape[1]):
        qdata = np.zeros([n_models, qoi_val[out_names[0]][qoi].shape[0]])
        for i in range(n_models):
            qdata[i,:] = qoi_val[out_names[i]][qoi][:,j]

        C[qoi][j, :, :] = np.cov(qdata)
        Corr[qoi][j, :, :] = np.corrcoef(qdata)

        print(qoi + " " + str(j))
        print(Corr[qoi][j])
       
n_pilot = qoi_val[out_names[0]][qoi].shape[0]



class Problem_T1D_Only(BLUEProblem):
    def sampler(self, ls):
        L = len(ls)

        # use the case index as the input for now
        Z = np.random.choice(n_pilot, L, replace=False)
        return Z

    # just reading the output
    def evaluate(self, ls, samples):
        L = len(ls)
        out = [0 for i in range(L)]

        for i in range(L):
            out[i] = qoi_val[out_names[ls[i]]][out_use[0]][samples[i], out_use[1]]

        return [out]

# define costs somewhat arbitrarily
costs = np.array([15*60, 11*60, 7*60])

# precomputed covariances
problem = Problem_T1D_Only(n_models, costs=costs, C=C[out_use[0]][out_use[1]], verbose=True)

################################ PART 1 - BASIC USAGE ########################################

# Get covariance and correlation matrix
print("Covariance matrix:\n")
print(problem.get_covariance())
print("\nCorrelation matrix:\n")
print(problem.get_correlation())

# get cost vector
print("\nCost vector:\n")
print(problem.get_costs())

# define statistical error tolerance
eps = eps_fac*np.sqrt(problem.get_covariance()[0,0])

# solve with standard MC
sol_MC = problem.solve_mc(eps=eps)
print("\n\nStd MC\n")
print("Std MC solution: ", sol_MC[0], "\nTotal cost: ", sol_MC[2])


# Solve with MLBLUE. K denotes the maximum group size allowed.
MLBLUE_data = problem.setup_solver(K=n_models, eps=eps)
sol_MLBLUE = problem.solve(K=n_models, eps=eps)

print("\n\nMLBLUE\n")
print("MLBLUE data:\n")
for key, item in MLBLUE_data.items(): print(key, ": ", item)
print("MLBLUE solution: ", sol_MLBLUE[0])

breakpoint()