import matplotlib.pyplot as plt
from data_helper import read_csv, read_exp_csv
from montecarlo2 import *
from scipy.optimize import minimize
import csv

xs_min = 1.10 + 0.6 #cm, min distance + thumbscrew thickness
xd_min = 3.66 #cm

exp_data = read_exp_csv('data.csv')
l = len(exp_data[0])

def fit_data(data1, data2, S0):

    xd1 = data1[0] + xd_min
    xs1 = data1[1] + xs_min
    f1 = data1[2]
    u_f1 = data1[5]
    x1 = np.column_stack((xd1, xs1))

    xd2 = data2[0] + xd_min
    xs2 = data2[1] + xs_min
    f2 = data2[2]
    u_f2 = data2[5]
    x2 = np.column_stack((xd2, xs2))

    # Define chi-square function
    def chi_square(s):
        chi2 = np.sum(((f1 - s * f2) ** 2) / (u_f1**2 + (s*u_f2) ** 2))
        return chi2
    
    
    # Minimize chi-square to find best scaling factor S

    result = minimize(chi_square, x0=[S0], bounds=[(0, 1e10)])
    S_best = result.x[0]
    return chi_square(S_best), S_best  # Initial guess for S

def plot_final(dfs, Ns):

    total_N = sum(Ns)
    data2 = read_csv(dfs[0])
    if len(dfs) > 1:
        data2[2,:] = sum(read_csv(dfs[i])[2,:]*Ns[i] for i in range(len(dfs)))/total_N
        data2[5,:] = sum(read_csv(dfs[i])[5,:]*Ns[i] for i in range(len(dfs)))/total_N

    data1 = exp_data

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    S, u_S, x2 = fit_data(data1, data2, 50)
    print(S)
    print(u_S)
    print(x2)

    d0_data1 = data1[0:, 0:29]
    d5_data1 = data1[0:, 29:44]
    d10_data1 = data1[0:, 44:59]
    s0_data1 = data1[0:,59:73]
    s5_data1 = data1[0:,73:87]
    s10_data1 = data1[0:, 87:]

    ax1.errorbar(d0_data1[0, :] + xd_min, d0_data1[2, :], xerr=d0_data1[3,:], yerr=d0_data1[5,:], fmt='.',  lw = 1, color='red', alpha=0.5, label=r'$x_{s}$ = ' + str(0+np.round(xs_min,2)))
    ax1.errorbar(d5_data1[0, :]+ xd_min, d5_data1[2, :],  xerr=d5_data1[3,:], yerr=d5_data1[5,:],fmt='.', lw = 1, color='green', alpha=0.5, label=r'$x_{s}$ = ' + str(5+xs_min))
    ax1.errorbar(d10_data1[0, :]+ xd_min, d10_data1[2, :],  xerr=d10_data1[3,:], yerr=d10_data1[5,:], fmt='.',lw = 1, color='blue', alpha=0.5, label=r'$x_{s}$ = ' + str(10+xs_min))

    ax2.errorbar(s0_data1[1, :] + xs_min, s0_data1[2, :], xerr=s0_data1[4,:], yerr=s0_data1[5,:], fmt='.',lw = 1, color='red', alpha=0.5, label=r'$x_{d}$ = ' + str(0+xd_min))
    ax2.errorbar(s5_data1[1, :] + xs_min, s5_data1[2, :], xerr=s5_data1[4,:], yerr=s5_data1[5,:], fmt='.', lw = 1, color='green', alpha=0.5, label=r'$x_{d}$ = ' + str(5+xd_min))
    ax2.errorbar(s10_data1[1, :] + xs_min, s10_data1[2, :], xerr=s10_data1[4,:], yerr=s10_data1[5,:], fmt='.',lw = 1, color='blue', alpha=0.5, label=r'$x_{d}$ = ' + str(10+xd_min))

    
    data2[2,:] = data2[2,:] *S
    data2[5,:] = data2[5,:] *S

    d0_data2 = data2[0:, 0:29]
    d5_data2 = data2[0:, 29:44]
    d10_data2 = data2[0:, 44:59]
    s0_data2 = data2[0:,59:73]
    s5_data2 = data2[0:,73:87]
    s10_data2 = data2[0:, 87:]

    ax1.plot(d0_data2[0, :]+ xd_min, d0_data2[2, :],   lw = 1, color='red', alpha=0.5, label='Simulation')
    ax1.plot(d5_data2[0, :]+ xd_min, d5_data2[2, :],   lw = 1, color='green', alpha=0.5 )
    ax1.plot(d10_data2[0, :]+ xd_min, d10_data2[2, :],   lw = 1, color='blue', alpha=0.5)

    ax2.plot(s0_data2[1, :] + xs_min, s0_data2[2, :],  lw = 1, color='red', alpha=0.5, label='Simulation')
    ax2.plot(s5_data2[1, :] + xs_min, s5_data2[2, :],   lw = 1, color='green', alpha=0.5)
    ax2.plot(s10_data2[1, :] + xs_min, s10_data2[2, :],  lw = 1, color='blue', alpha=0.5)

    ax1.legend()
    ax2.legend()

    ax1.set_xlabel(r'$x_{d}$')
    ax2.set_xlabel(r'$x_{s}$')
    ax1.set_ylabel('Rate (Hz)')

    z = S*z_expected/(activity_expected*A_expected)

    fig.suptitle('Hit Rate vs Target Foil Position, Simulation vs Experiment')
    ax1.text(7, 20, 'S = ' + str(round(S,2)) + '± ' + str(round(u_S,2)) + '\n'+r'$χ^2$ = ' + str(np.round(x2,2)) + '\nN = ' + str(total_N) + '\nz = ' + str(round(z,3)) + ' cm', bbox=dict(facecolor='red', alpha=0.5))

    
    print("S " + str(S))
    print(str())
    print("S expected " + str(activity_expected*A_expected))
    print("z " + str(S*z_expected/(activity_expected*A_expected)))

    plt.show()


def rutherford_opt(N, A):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simop_rutherford_fast(x[0],x[1],N,A) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    return data


def rutherford_cos_opt(N, A):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simop_rutherford_cos_fast(x[0],x[1],N,A) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    return data

def optimize_mc(N,A0):
    S_list = []
    S_var_list = []
    chi_list =[]
        
        
    def model_chi(param):
        A= param[0]
        S0 = A0*activity_expected
        print("Simulating A = " + str(A) + ", for N = " + str(N) + "...")
        chi, S = fit_data(exp_data, rutherford_opt(N,A), S0)
        S_list.append(S)
        chi_list.append(chi)
        print("Chi squared " + str(chi))
        print("S " + str(S))
        return chi
    
    result = minimize(model_chi, x0=[A0], bounds=[(1e-10,1/(2*(1/np.sin(pi/4))**2-2))])
    A_best = result.x[0]
    S_best = S_list[-1]

    data_best = rutherford_opt(N*5,A_best)
    csv_name = 'rutherford_mc_fast_opt ' + str(N*5) +str(A_best)+ '.csv'
    np.savetxt(csv_name, data_best.transpose(), delimiter=',',fmt='%.10e')
    print("Saved " + 'rutherford_mc_fast_opt ' + str(N*5) +str(A_best)+ '.csv')

    print(A_best)
    print(S_best)
    print(chi_list[-1])

    plot_final([csv_name],[N*5])

optimize_mc(10000000,3.82e-5)
