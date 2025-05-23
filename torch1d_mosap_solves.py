from bluest import BLUEProblem
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

"""
Script for solving the MOSAP problem for torch1D-only covariances
"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 18,
})

home = os.getenv('HOME')
# in_dir = home + "/bedonian1/torch1d_resample_sens_r8/"
# suffix = ''
# suffix = '_massflux'
# suffix = '_massflux_core'
# suffix = '_time_avg'
# suffix = '_trunc'
suffix = '_pres'
# out_dirs = [home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse"]
# out_dirs = [home + "/bedonian1/tps2d_mf_post_r1/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse", home + "/bedonian1/torch1d_post_r1_pilot_4s"]
# out_dirs = [home + "/bedonian1/tps2d_mf_post_r1_far/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse", home + "/bedonian1/torch1d_post_r1_pilot_4s"]
# out_dirs = [home + "/bedonian1/tps2d_mf_post_r1_far/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse"]#, home + "/bedonian1/torch1d_post_r1_pilot_4s"]
out_dirs = [home + "/bedonian1/tps2d_mf_post_r1_massflux/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse"]#, home + "/bedonian1/torch1d_post_r1_pilot_4s"]
# out_dirs = [home + "/bedonian1/tps2d_mf_post_r1_massflux_core/", home + "/bedonian1/torch1d_post_r1_pilot_fine_core", home + "/bedonian1/torch1d_post_r1_pilot_core", home + "/bedonian1/torch1d_post_r1_pilot_coarse_core"]#, home + "/bedonian1/torch1d_post_r1_pilot_4s"]
# out_dirs = [home + "/bedonian1/tps2d_mf_post_r1_time_avg/", home + "/bedonian1/torch1d_post_r1_pilot_fine", home + "/bedonian1/torch1d_post_r1_pilot", home + "/bedonian1/torch1d_post_r1_pilot_coarse"]#, home + "/bedonian1/torch1d_post_r1_pilot_4s"]
# correspond to out_dirs order
# out_names = ["1D_Fine", "1D_Mid", "1D_Coarse"]
# out_names = ["1D_Fine", "1D_Mid", "1D_Coarse", "1D_4Species"]
# out_names = ["2D_Axi", "1D_Fine", "1D_Mid", "1D_Coarse", "1D_4Species"]
out_names = ["2D_Axi", "1D_Fine", "1D_Mid", "1D_Coarse"]
# non-pilot samples, use these for the actual MLBLUE evaluation, not for the MOSAP
# provide as groups of directories
# G4: tps2d, t1d_coarse
# G2: t1d_coarse
# G1: t1d_coarse

# home + "/bedonian1/torch1d_post_r1_G3_fine",
comp_dirs = {
    'G1': [None, None, None, home + "/bedonian1/torch1d_post_r1_G1_coarse"],
    'G2': [None, None, None, home + "/bedonian1/torch1d_post_r1_G2_coarse"],
    'G3': [None, None, home + "/bedonian1/torch1d_post_r1_G3_mid", home + "/bedonian1/torch1d_post_r1_G3_coarse"],
    'G4': [home + "/bedonian1/tps2d_mf_post_r1_G4/", home + "/bedonian1/torch1d_post_r1_G4_fine", home + "/bedonian1/torch1d_post_r1_G4_mid", home + "/bedonian1/torch1d_post_r1_G4_coarse"]
}

# keep track of failed tps2d cases
c_exclude = {
   'G1': [],
   'G2': [],
   'G3': [16, 95, 114, 120, 127,  146, 149, 156, 160, 187, 228],
   'G4': [18, 71, 74]
}

# define costs
# costs = np.array([12*60*60, 15*60, 11*60, 7*60])
proc_fac = 112 # number of procs per tps run
# proc_fac = 1 # number of procs per tps run
costs = np.array([proc_fac*11.6*60*60, 17*60, 14*60, 11*60])

# contract all models and dependent arrays to the indices of this list
# restrict = [0, 3]
restrict = list(range(len(out_names)))


# statistical error threshold
eps_fac = 0.05

# max_sample = 0 # no limit
# max_sample = 64
# for pilot samples only
max_sample = 48
exclude = []
exclude = [60]



make_plots = True
# do single output for now
out_use = [['exit_d', 0],
           ['exit_d', 1],
           ['exit_v', 0],
           ['exit_T', 0],
           ['exit_X', 0]]
# out_use = [['exit_X', 0]]
qoi_list_plot = [['exit_v', 0], ['exit_T', 0], ['exit_X', 0]]
# 










##########################################################################################################

# apply restriction first
out_dirs = [out_dirs[x] for x in restrict]
out_names = [out_names[x] for x in restrict]
for key in comp_dirs.keys():
    comp_dirs[key] = [comp_dirs[key][x] for x in restrict]
costs = [costs[x] for x in restrict]

n_outputs = len(out_use)

##############################################################################

# function to pick an appropriate sample group

def getSampleGroup(ls, exact = True):
    # get a sample group that (exactly, optional) provides requested samples
    # if len(ls) > 1:
    #     breakpoint()

    found = False
    for key in comp_dirs.keys():
        carray = [comp_dirs[key][x] for x in ls]
        tarray = [gdir is not None for gdir in carray]

        if exact:
            ls2 = [x for x in range(n_models) if x not in ls]
            c2array = [comp_dirs[key][x] for x in ls2]
            farray = [gdir is None for gdir in c2array]

            if np.all(tarray) and np.all(farray):
                found = True
                break

        else:
            if np.all(tarray):
                found = True
                break

    if not found:
        raise Exception(f"No valid groups found for combo: {ls} (Exact = {exact})")

    return key




# flag to evaluate the square of output from MLBLUE, don't touch
eval_sq = False


n_models = len(out_dirs)


# load outputs
qoi_val = {}
fqoin = out_dirs[0] + '/qoi_list.pickle' 
with open(fqoin, 'rb') as f:
    qoi_list = pickle.load(f)

qoi_list = ['exit_p', 'exit_d', 'exit_v', 'exit_T', 'exit_X']

# pilot samples
for i, out_dir in enumerate(out_dirs):
    fqoiv = out_dir + f'/qoi_samples.pickle' 

    with open(fqoiv, 'rb') as f:
        qoi_val[out_names[i]] = pickle.load(f)

    if max_sample:
        glist = [p for p in range(max_sample) if p not in exclude]
        for qoi in qoi_list:
            if qoi_val[out_names[i]][qoi][:,:].shape[0] < max_sample:
                qoi_val[out_names[i]][qoi] = qoi_val[out_names[i]][qoi][:,:]
            else:
                qoi_val[out_names[i]][qoi] = qoi_val[out_names[i]][qoi][glist,:]
            
# computation samples
c_qoi_val = {}
c_qoi_num = {}
for sg in comp_dirs.keys():
    c_qoi_val[sg] = {}
    c_qoi_num[sg] = 1e10 # number of available samples in group

    for i, out_dir in enumerate(comp_dirs[sg]):
        if out_dir is None:
            c_qoi_val[sg][out_names[i]] = None
            continue
        fqoiv = out_dir + f'/qoi_samples.pickle' 

        with open(fqoiv, 'rb') as f:
            c_qoi_val[sg][out_names[i]] = pickle.load(f)

        glist = [p for p in range( c_qoi_val[sg][out_names[i]][qoi].shape[0]) if p not in c_exclude[sg]]
        for qoi in qoi_list:
            c_qoi_val[sg][out_names[i]][qoi] = c_qoi_val[sg][out_names[i]][qoi][glist,:]

        c_qoi_num[sg] = min(c_qoi_num[sg], c_qoi_val[sg][out_names[i]][qoi_list[0]].shape[0])

qoi_sizes = {
    'exit_p': 1, 
    'exit_d': 2, 
    'exit_v': 1,
    'exit_T': 2,
    'exit_X': 5,
    'heat_dep': 1
}      

Nplot = 0
# for key in qoi_list:
#     Nplot += qoi_sizes[key]
for key in qoi_list_plot:
    Nplot += 1

#Estimate Covariances
C = {}
C2 = {}
Corr = {}
Corr2 = {}
qdata = {}
if make_plots:

    fig, axs = plt.subplots(Nplot, 3, figsize=(15, 4.*Nplot))

cf = 0
for qoi in qoi_list:
    
    C[qoi] = np.zeros([qoi_sizes[qoi], n_models, n_models])
    Corr[qoi] = np.zeros([qoi_sizes[qoi], n_models, n_models])
    C2[qoi] = np.zeros([qoi_sizes[qoi], n_models, n_models])
    Corr2[qoi] = np.zeros([qoi_sizes[qoi], n_models, n_models])

    #concatenate results
    qdata[qoi] = {}
    for j in range(qoi_val[out_names[0]][qoi].shape[1]):
        qdata[qoi][j] = np.zeros([n_models, qoi_val[out_names[0]][qoi].shape[0]])
        for i in range(n_models):
            qdata[qoi][j][i,:] = qoi_val[out_names[i]][qoi][:,j]

        C[qoi][j, :, :] = np.cov(qdata[qoi][j])
        Corr[qoi][j, :, :] = np.corrcoef(qdata[qoi][j])

        # also get for C2
        C2[qoi][j, :, :] = np.cov(qdata[qoi][j]**2)
        Corr2[qoi][j, :, :] = np.corrcoef(qdata[qoi][j]**2)

        print(qoi + " " + str(j))
        print(Corr[qoi][j])
        # breakpoint()
        print("Squared")
        print(Corr2[qoi][j])
        if make_plots and np.any([(qoi == qoi_list_plot[x][0] and j == qoi_list_plot[x][1]) for x in range(len(qoi_list_plot))]):

            cax = axs[cf,0].matshow(Corr[qoi][j], vmin=0, vmax=1)
            for (m, n), o in np.ndenumerate(Corr[qoi][j]):
                axs[cf,0].text(n, m, '{:0.2f}'.format(o), ha='center', va='center')
                
            axs[cf,0].set_title(qoi + ' ' + str(j))
            # if cf == 0:
            fig.colorbar(cax)
            # plt.xticks(list(range(len(out_names))), out_names)
            # plt.yticks(list(range(len(out_names))), out_names)
            
            # breakpoint()
            axs[cf,0].set_xticks(list(range(len(out_names))))
            axs[cf,0].set_yticks(list(range(len(out_names))))
            axs[cf,0].set_xticklabels(out_names)
            axs[cf,0].set_yticklabels(out_names)
            
            if qoi == "exit_X":
                plt.ticklabel_format(scilimits = [-2,3])
            else:
                plt.ticklabel_format(scilimits = [-5,6])

            # then plot data, pair tps2d and the fine torch1d
            axs[cf,1].set_title(qoi + ' ' + str(j))
            # axs[cf,1].plot(qdata[qoi][j][0,:31], qdata[qoi][j][1,:], 'x')
            axs[cf,1].plot(qdata[qoi][j][0,:31], qdata[qoi][j][1,:31], 'x')
            axs[cf,1].plot(qdata[qoi][j][0,31:], qdata[qoi][j][1,31:], 'x')
            # axs[cf,1].plot()
            axs[cf,1].set_xlabel(out_names[0])
            axs[cf,1].set_ylabel(out_names[1])

            axs[cf,2].set_title(qoi + ' ' + str(j))
            # axs[cf,1].plot(qdata[qoi][j][0,:31], qdata[qoi][j][1,:], 'x')
            axs[cf,2].plot(qdata[qoi][j][0,:31], qdata[qoi][j][1,:31], 'x')
            axs[cf,2].plot(qdata[qoi][j][0,31:], qdata[qoi][j][1,31:], 'x')
            # x = y
            xy = np.linspace(min(np.min(qdata[qoi][j][0,:]), np.min(qdata[qoi][j][1,:31])),
                             max(np.max(qdata[qoi][j][0,:]), np.max(qdata[qoi][j][1,:31])))
            axs[cf,2].plot(xy, xy, '-k')

            axs[cf,2].set_xlabel(out_names[0])
            axs[cf,2].set_ylabel(out_names[1])

       

            cf += 1
if make_plots:
    fig.tight_layout()
    plt.savefig(f'model_corr_{n_models}{suffix}.png')
    plt.clf()

    quit()

n_pilot = qoi_val[out_names[0]][qoi].shape[0]

# breakpoint()
sample_cache = {}
flattened_groups = []

class Problem_Exp(BLUEProblem):

    # NOTE BAD HACK!
    def resetBC(self, groups):
        # self.BCOUNTER = 0
        self.BCOUNTER = {}
        for i , group in enumerate(groups):
            self.BCOUNTER[i] = 0


    def sampler(self, ls):
        L = len(ls)

        # use the case index as the input for now
        # sg = getSampleGroup(ls, exact = False)
        # Ns = c_qoi_num[sg]
        # Z = np.random.choice(n_pilot, L, replace=False)

        # NOTE BAD HACK FOR TESTING
        # HACK FIX THIS MONDAY
        # Z = [self.BCOUNTER%97]*L
        # self.BCOUNTER += 1

        # this should work
        fg = flattened_groups.index(ls)

        Z = sample_cache[fg][self.BCOUNTER[fg]%c_qoi_num[sg]]

        self.BCOUNTER[fg] += 1

        return Z

    # just reading the output
    def evaluate(self, ls, samples):
        L = len(ls)
        # out = [0 for i in range(L)]
        out = [[0 for i in range(L)] for n in range(n_outputs)]

        # find the appropriate group to draw samples from
        sg = getSampleGroup(ls, exact = False)

        for i in range(L):
            # list order is important
            for n, output in enumerate(out_use):
                if eval_sq:
                    # out[n][i] = qoi_val[out_names[ls[i]]][output[0]][samples[i], output[1]]**2
                    out[n][i] = c_qoi_val[sg][out_names[ls[i]]][output[0]][samples, output[1]]**2
                else:
                    out[n][i] = c_qoi_val[sg][out_names[ls[i]]][output[0]][samples, output[1]]

        return out

# class Problem_Exp_Sq(BLUEProblem):
#     def sampler(self, ls):
#         L = len(ls)

#         # use the case index as the input for now
#         Z = np.random.choice(n_pilot, L, replace=False)
#         return Z

#     # just reading the output
#     def evaluate(self, ls, samples):
#         L = len(ls)
#         # out = [0 for i in range(L)]
#         out = [[0 for i in range(L)] for n in range(n_outputs)]

#         for i in range(L):
#             # list order is important
#             for n, output in enumerate(out_use):
#                 out[n][i] = qoi_val[out_names[ls[i]]][output[0]][samples[i], output[1]]**2

#         return out

C_use = [C[output[0]][output[1]] for output in out_use]
# C_use_2 = [C2[output[0]][output[1]] for output in out_use]

# precomputed covariances
# problem = Problem_T1D_Only(n_models, costs=costs, C=C[out_use[0]][out_use[1]], verbose=True)
problem = Problem_Exp(n_models, n_outputs=n_outputs, costs=costs, C=C_use, verbose=True)

################################ PART 1 - Mean ########################################

# Get covariance and correlation matrix
print("Covariance matrix:\n")
print(problem.get_covariance())
print("\nCorrelation matrix:\n")
print(problem.get_correlation())

# get cost vector
print("\nCost vector:\n")
print(problem.get_costs())

# define statistical error tolerance
eps = [eps_fac*np.sqrt(problem.get_covariance(i)[0,0]) for i in range(n_outputs)]
# breakpoint()


# solve with standard MC
# sol_MC = problem.solve_mc(eps=eps)
# print("\n\nStd MC\n")
# print("Std MC solution: ", sol_MC[0], "\nTotal cost: ", sol_MC[2])


# Solve with MLBLUE. K denotes the maximum group size allowed.
# solves the opt problem
MLBLUE_data = problem.setup_solver(K=n_models, eps=eps)


# sol_MLBLUE = problem.solve(K=n_models, eps=eps)

print("\n\nMLBLUE\n")
print("MLBLUE data:\n")
for key, item in MLBLUE_data.items(): print(key, ": ", item)




# problem.resetBC()

flattened_groups = problem.MOSAP_output['flattened_groups']
sample_list = problem.MOSAP_output['samples']
# sums = [[] for n in range(problem.n_outputs)]
# for ls,N in zip(flattened_groups, sample_list):
#     if N == 0:
#         for n in range(problem.n_outputs):
#             sums[n].append([0 for l in range(len(ls))])
#         continue
#     sumse,_,_ = problem.blue_fn(ls, N, verbose=False)
#     for n in range(problem.n_outputs):
#         sums[n].append(sumse[n])

for i, fg in enumerate(flattened_groups):
    sg = getSampleGroup(fg, exact = False)
    arr = np.array(range(c_qoi_num[sg]))
    np.random.shuffle(arr)
    sample_cache[i] = arr

# breakpoint()
problem.resetBC(flattened_groups)
# compute mean and variance
# mus, Vs = problem.MOSAP.compute_BLUE_estimators(sums, sample_list)
sol_mu = problem.solve(K=n_models)
eval_sq = True
problem.resetBC(flattened_groups)
# mus_sq, Vs_sq = problem.MOSAP.compute_BLUE_estimators(sums, sample_list)
sol_sq = problem.solve(K=n_models)




print("MLBLUE solution: ", sol_mu[0])
print("\n")
print("Standard Deviation:")
stdv = []
for i, item in enumerate(sol_mu[0]):
    meansq = item**2 
    stdv.append(np.sqrt(sol_sq[0][i] - meansq))
print(stdv)


# print("\n")
# print("Solving for Variance ...")

# problem2 = Problem_Exp_Sq(n_models, n_outputs=n_outputs, costs=costs, C=C_use_2, verbose=True)

# print("Covariance matrix:\n")
# print(problem2.get_covariance())
# print("\nCorrelation matrix:\n")
# print(problem2.get_correlation())

# # same costs
# print(problem2.get_costs())

# # define statistical error tolerance
# eps2 = [eps_fac2*np.sqrt(problem2.get_covariance(i)[0,0]) for i in range(n_outputs)]
# # breakpoint()
# # solve with standard MC
# sol_MC2 = problem2.solve_mc(eps=eps2)
# print("\n\nStd MC\n")
# print("Std MC Variance solution: ", sol_MC2[0], "\nTotal cost: ", sol_MC2[2])

# MLBLUE_data_2 = problem2.setup_solver(K=n_models, eps=eps2)
# sol_MLBLUE_2 = problem2.solve(K=n_models, eps=eps2)

# print("\n\nMLBLUE\n")
# print("MLBLUE data:\n")
# for key, item in MLBLUE_data_2.items(): print(key, ": ", item)
# print("MLBLUE solution: ", sol_MLBLUE_2[0])



breakpoint()