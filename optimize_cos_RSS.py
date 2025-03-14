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

    # Define RSS function
    def rss(S):
        return np.sum((f1 - S * f2) ** 2)
    
    
    # Minimize rss to find best scaling factor S
    options = {'xatol':u_S_calc}
    result = minimize(rss, x0=[S0], bounds=[(0, 1e10)], method='Nelder-Mead', options=options)
    S_best = result.x[0]
    return rss(S_best), S_best  # Initial guess for S

def plot_final(df, N, A):
    
    data2 = read_csv(df)
    data1 = exp_data

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    rss, S = fit_data(data1, data2, 50)

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

    z = A * (mass_Au/density) * (4*E_alpha/(k*q*Q))**2
    u_z = z * np.sqrt( (u_A_calc/A)**2 + (u_E_squared/E_squared)**2)

    act = (S/A)/uCi

    u_act = act* np.sqrt( (u_S_calc/S)**2 + (u_A_calc/A)**2 )

    fig.suptitle('Hit Rate vs Target Foil Position, Simulation vs Experiment')

    
    print("S " + str(S))
    print(str())
    print("act " + str(act))
    print("u_act " + str(u_act))
    print("z " + str(z))
    print("u_z " + str(u_z))

    plt.show()
    plt.savefig("optimal_cos_rss.png")

    ax1.text(7, 20, 'S = ' + str(round(S,2)) + '\nRSS = ' + str(np.round(rss,2)) + '\nN = ' + str(N) + '\nz = ' + str(round(z,3)) + ' ± ' + str(u_z) + ' cm' + '\nAct = ' + str(act) + ' ± ' + str(u_act) + 'uCi', bbox=dict(facecolor='red', alpha=0.5))
    plt.show()
    plt.savefig("optimal_cos_rss_text.png")



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
        
    def model_rss(param):
        A= param[0]
        S0 = A0*activity_expected
        print("Simulating A = " + str(A) + ", for N = " + str(N) + "...")
        rss, S = fit_data(exp_data, rutherford_cos_opt(N,A), S0)
        print("RSS " + str(rss))
        print("S " + str(S))
        return rss
    
    options=  {'xatol':u_A_calc}
    result = minimize(model_rss, x0=[A0], bounds=[(A_min,A_max)], method='Nelder-Mead', options=options)
    A_best = result.x[0]

    data_best = rutherford_cos_opt(N*5,A_best)
    csv_name = 'rutherford_cos_mc_fast_opt ' + str(N*5/2) +str(A_best)+ '.csv'
    np.savetxt(csv_name, data_best.transpose(), delimiter=',',fmt='%.10e')
    
    print("Saved " + 'final_cos ' + str(N*5/2) +str(A_best)+ '.csv')
    print("A best"  + str(A_best))

    plot_final(csv_name,N*5)

optimize_mc(20000000,3.82e-5)
