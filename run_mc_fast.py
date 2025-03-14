from montecarlo2 import simulate_flat_fast, simulate_flat__cos_fast, simulate_rutherford_fast, simulate_spherical__cos_fast, simulate_spherical_fast, simulate_rutherford_cos_fast, simop_rutherford_cos_fast, simop_rutherford_fast
from data_helper import *
import os

exp_data = read_exp_csv('data.csv')
l = len(exp_data[0])

flat_data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
flat_cos_data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])


def rutherford_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_rutherford_fast(x[0],x[1],N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('rutherford_mc_fast ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'rutherford_mc_fast ' + str(N) + '.csv')

def rutherford_cos_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_rutherford_cos_fast(x[0],x[1],N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('rutherford_cos_mc_fast ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'rutherford_cos_mc_fast ' + str(N) + '.csv')

def rutherford_ang_exp(N, ang):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_rutherford_fast(x[0],x[1],N,ang) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('rutherford_ang_mc_fast ' + str(N) + str(ang)+'.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'rutherford_ang_mc_fast ' + str(N) + str(ang)+'.csv')

def rutherford_ang_cos_exp(N,ang):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_rutherford_cos_fast(x[0],x[1],N,ang) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('rutherford_ang_cos_mc_fast ' + str(N) + str(ang)+'.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'rutherford_ang_cos_mc_fast ' + str(N) +str(ang)+ '.csv')

def flat_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_flat_fast(x[0],x[1],N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('flat_mc_fast ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'flat_mc_fast ' + str(N) + '.csv')

def flat_cos_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_flat__cos_fast(x[0],x[1],N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('flat_cos_mc_fast ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'flat_cos_mc_fast ' + str(N) + '.csv')

def spherical_fast_cos_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_spherical__cos_fast(x[0],x[1],N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('spherical_cos_mc_fast ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'spherical_cos_mc_fast ' + str(N) + '.csv')

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


for n in [50000000]:
    np.savetxt('final_fastmc_opt.csv', rutherford_opt(n,3.82e-5).transpose(), delimiter=',',fmt='%.10e')
    print("saved final_fastmc_opt.csv" )
    
    np.savetxt('final_fastmc_cos_opt.csv', rutherford_cos_opt(n,3.82e-5).transpose(), delimiter=',',fmt='%.10e')
    print("saved final_fastmc_opt.csv" )

    rutherford_ang_exp(n,np.pi/2)
    rutherford_ang_cos_exp(n,np.pi/4)
    rutherford_ang_cos_exp(n,np.pi/2)
    rutherford_ang_cos_exp(n,np.pi)


