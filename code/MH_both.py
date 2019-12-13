import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import math
import os
from scipy.stats import multivariate_normal

# --------------------------------------------------------------------------
# Control exercise execution
# --------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_EXERCISE = True

# --------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------
random.seed(500)
DATA_ROOT = os.path.join('..', 'data')
PATH_TO_INPUTS_DATA = os.path.join(DATA_ROOT, 'inputs.csv')
PATH_TO_CAMERA_1_DATA = os.path.join(DATA_ROOT, 'points_2d_camera_1.csv')
PATH_TO_CAMERA_2_DATA = os.path.join(DATA_ROOT, 'points_2d_camera_2.csv')

FIGURES_ROOT = os.path.join('..', 'figures')

n = 50000
M = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
M2 = np.array([[0,0,1,-5], [0,1,0,0], [-1,0,0,5]])
meanL = np.array([0, 0, 4])
covL = 0.1 * np.identity(3) 
cov = (0.05)**2 * np.identity(2)

# Pre-allocate for speed
pi = np.zeros((n, 3))
pf = np.zeros((n, 3))
post = np.zeros((n, 1))
pri = np.zeros((n, 1))
prf = np.zeros((n, 1))
ratio = np.zeros((n, 1))
acceptedpi = np.zeros((n, 3))
acceptedpf = np.zeros((n, 3))
acceptedpost = np.zeros((n, 1))

def read_camera(path, d=','):

    arr = np.genfromtxt(path, delimiter=d, dtype=None)
    length = len(arr)
    r = np.zeros(shape=(length, 2))
    for i, (x1, x2) in enumerate(arr):
        r[i, 0] = x1
        r[i, 1] = x2

    return r

def read_t_inputs(path):

    t = np.genfromtxt(path)
    return t

def calculate_q(pi, pf, M):

    pi_array = np.array([[pi[0]], [pi[1]], [pi[2]], [1]])
    pf_array = np.array([[pf[0]], [pf[1]], [pf[2]], [1]])
    
    qi_cell = M.dot(pi_array)
    qi = qi_cell[0:2]/qi_cell[2, 0]
    
    qf_cell = M.dot(pf_array)
    qf = qf_cell[0:2]/qf_cell[2, 0]
    
    return qi.T[0], qf.T[0]

def Posterior_func(pij, pfj, r, r2, t, pi, pf, covL, cov, M, M2):
    
    qi, qf = calculate_q(pij, pfj, M)
    likelihood = np.zeros(len(r))
    
    qi2, qf2 = calculate_q(pij, pfj, M2)
    likelihood2 = np.zeros(len(r2))
    
    for i in range(len(t)):
        qs = qi + (qf - qi)*t[i]
        likelihood[i] = multivariate_normal.logpdf(r[i], mean=qs, cov=cov)
    
        qs2 = qi2 + (qf2 - qi2)*t[i]
        likelihood2[i] = multivariate_normal.logpdf(r2[i], mean=qs2, cov=cov)
        
    pi_prior = multivariate_normal.logpdf(pij, mean=pi, cov=covL)
    pf_prior = multivariate_normal.logpdf(pfj, mean=pf, cov=covL)
    posterior = sum(likelihood2) + pi_prior + pf_prior + sum(likelihood)
    #print("Posterior is ", posterior)
    return posterior

def Proposed_func(meanL, covL):
    return np.random.multivariate_normal(meanL, covL)

def MetroHast(M, M2, meanL, covL, r, r2, t, cov, n):
    # Store in a list
    # Initialize
    pi[0] = Proposed_func(meanL, covL)
    pf[0] = Proposed_func(meanL, covL)
    post[0] = Posterior_func(pi[0], pf[0], r, r2, t, meanL, meanL, covL, cov, M, M2)
    count = 0
    for i in range(1, n):
        print("Running %d of %d iteration"% (i, n)) 
        pi[i] = Proposed_func(pi[i-1], covL)
        pf[i] = Proposed_func(pf[i-1], covL)
        post[i] = Posterior_func(pi[i], pf[i], r, r2, t, pi[i-1], pf[i-1], covL, cov, M, M2) 
        ratio[i-1] = np.exp(post[i] - post[i-1])
        #print(new_Post, " ", previous_Post)
        alpha = min(1, ratio[i-1])
        
        if random.uniform(0, 1) <= alpha:
            acceptedpi[count] = pi[i]
            acceptedpf[count] = pf[i]
            acceptedpost[count] = post[i]
            count += 1
        else:
            pi[i] = pi[i-1]
            pf[i] = pf[i-1]
    
    MAP = np.amax(post)
    print(f"MAP: {MAP}")
    print("The accpetance rate : ", count/n)
    return acceptedpi[~np.all(acceptedpi==0, axis=1)], acceptedpf[~np.all(acceptedpf==0, axis=1)], acceptedpost[~np.all(acceptedpost==0, axis=1)]

def plot_p(pi, pf, save_path):
    
    plt.figure()
    plt.plot(range(len(pi)), pi[:, 0], label="x")
    plt.plot(range(len(pi)), pi[:, 1], label="y")
    plt.plot(range(len(pi)), pi[:, 2], label="z")
    plt.legend(loc="upper left")
    plt.title("Accepted Pi")
    plt.xlabel("Count")
    plt.ylabel("coordinate")
    plt.savefig(save_path + "_Pi", fmt='png')
    
    plt.figure()
    plt.plot(range(len(pf)), pf[:, 0], label="x")
    plt.plot(range(len(pf)), pf[:, 1], label="y")
    plt.plot(range(len(pf)), pf[:, 2], label="z")
    plt.legend(loc="upper left")
    plt.title("Accepted Pf")
    plt.xlabel("Count")
    plt.ylabel("coordinate")
    plt.savefig(save_path + "_Pf", fmt='png')
 
def plot_map(post, pi, pf, r, r2, M, M2, save_path):
    
    max_n = np.argmax(post)
    MAP = post[max_n]
    MAP_pi = pi[max_n]
    MAP_pf = pf[max_n]
    print(f"MAP estimates: {MAP}")
    print(f"MAP for pi {MAP_pi} and pf {MAP_pf}")

    qi, qf = calculate_q(MAP_pi, MAP_pf, M)
    print(f"MAP for qi {qi} and qf {qf} from camera 1")
    plt.figure()
    plt.plot([qi[0], qf[0]], [qi[1], qf[1]], label = "projection line")
    plt.scatter(r[:, 0], r[:, 1], label = "camera points")
    plt.legend(loc="upper left")
    plt.title("MAP for Camera 1 Image")
    plt.xlabel("u coordinate")
    plt.ylabel("v coordinate")
    plt.savefig(save_path + "_MAP_c1", fmt='png')

    qi2, qf2 = calculate_q(MAP_pi, MAP_pf, M2)
    print(f"MAP for qi {qi2} and qf {qf2} from camera 2") 
    plt.figure()
    plt.plot([qi2[0], qf2[0]], [qi2[1], qf2[1]], label = "projection line")
    plt.scatter(r2[:, 0], r2[:, 1], label = "camera points")
    plt.legend(loc="upper right")
    plt.title("MAP for Camera 2 Image")
    plt.xlabel("u coordinate")
    plt.ylabel("v coordinate")
    plt.savefig(save_path + "_MAP_c2", fmt='png')


def run_MH(meanL, covL, cov, M, M2, n, camera_path1, camera_path2, t_path, figures_root, data_name):

    r = read_camera(camera_path1)
    r2 = read_camera(camera_path2)
    t = read_t_inputs(t_path)
    save_path = os.path.join(figures_root, f'MH_{data_name}')

    pi, pf, post= MetroHast(M, M2, meanL, covL, r, r2, t, cov, n)
    plot_p(pi, pf, save_path=save_path)
    plot_map(post, pi, pf, r, r2, M, M2, save_path=save_path)

 # -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if __name__ == "__main__":
    if RUN_EXERCISE:
        run_MH(meanL, covL, cov, M, M2, n, camera_path1=PATH_TO_CAMERA_1_DATA, camera_path2=PATH_TO_CAMERA_2_DATA, t_path=PATH_TO_INPUTS_DATA, figures_root=FIGURES_ROOT, data_name='both')
        plt.show()
