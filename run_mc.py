from montecarlo import simulate_flat, simulate_flat_cos, simulate_rutherford, simulate_spherical, simulate_spherical_cos, simulate_spherical_fast, simulate_spherical_fast_cos
from data_helper import *

exp_data = read_csv('data.csv')
l = len(exp_data[0])

flat_data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
flat_cos_data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])


def rutherford_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_rutherford(x,N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('rutherford_mc ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'rutherford_mc ' + str(N) + '.csv')

def flat_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_flat(x,N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('flat_mc ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'flat_mc ' + str(N) + '.csv')

def flat_cos_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_flat_cos(x,N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('flat_cos_mc ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'flat_cos_mc ' + str(N) + '.csv')

def spherical_fast_cos_exp(N):
    data = np.array([exp_data[0], exp_data[1], np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)])
    xx = np.transpose(data[0:2])
    results = np.transpose(np.array([simulate_spherical_fast_cos(x,N) for x in xx]))
    data[2] = (results[0].astype(float))/float(N)
    data[5] = (results[1].astype(float))/float(N)
    np.savetxt('spherical_fast_cos_mc ' + str(N) + '.csv', data.transpose(), delimiter=',',fmt='%.10e')
    print("saved " + 'spherical_fast_cos_mc ' + str(N) + '.csv')

for n in [10000000, 50000000]:
    rutherford_exp(n)

