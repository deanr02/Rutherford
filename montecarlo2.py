
import numpy as np
import random

uCi = 3.7 * (10**4) 
year = 3.154*(10**7)
pi = np.pi

activity_0 = 57 * uCi
u_activity_0 = 0.5 * uCi
half_life = 87.7 * year
u_half_life = 0.05 * year
age = 54 * year
u_age = (1/24)*year

activity_expected = activity_0 * np.exp(-np.log(2)*age/half_life)
u_activity_expected = activity_expected * np.sqrt((u_activity_0/activity_0)**2 + (u_age/age)**2 + (u_half_life/half_life)**2)

inch = 2.54 #cm
amu = 1.6605390689*(10**-24)

density = 19.283 #g/cm3
mass_Au = 196.96657 * amu #g

min_theta = 1e-6

e_0 = 55.26349406 * (10**6) * 10000 #e2 MeV-1 cm-1
k = 1/(4*pi*e_0)
q = 2 #e
Q = 79 #e
E = 5 #MeV
u_E = 0.5


E_squared = E**2
u_E_squared = E_squared*2*u_E/E

z_expected = 0.0002 * inch
u_z_expected = 0.00005 * inch

n_tar = density*z_expected/mass_Au
u_n_tar = density*u_z_expected/mass_Au

A_expected = n_tar*((k*q*Q/(4))**2)/E_squared
u_A_expected = A_expected* np.sqrt((u_n_tar/n_tar)**2 + (u_E_squared/E_squared)**2)

xs_min = 1.10 + 0.6 #cm, min distance + thumbscrew thickness
xd_min = 3.66 #cm
z_Au = 0.0005 #cm, or 0.0002 in
d_Au = 0.1 #cm, or 0.040 in
d_Pu = 1.3 #cm
d_det = 0.7 #cm


c = 3 * 10**10 #cm/s
m_alpha = (3.727379 * 10**3)/(c*c) #MeV/c2
E_alpha = 5 #MeV
dE_alpha = 1 #MeV, energy lost in gold


#Calculating minimum tolerance on A basd on z = 5 micrometer
micrometer = 10**-4
u_z_calc = 2 * micrometer
u_n_calc = density*u_z_calc/mass_Au

u_A_calc = A_expected*np.sqrt((u_n_calc/n_tar)**2 + (u_E_squared/E_squared)**2)

u_S_calc = A_expected*activity_expected*( (uCi/activity_expected)**2  + (u_A_calc/A_expected)**2 )

print("ah")
print(u_S_calc)

z_min = 1*micrometer/1000
n_min = density*z_min/mass_Au
A_min = n_min*((k*q*Q/(4))**2)/E_squared

z_max = 1/10
n_max = density*z_max/mass_Au
A_max = n_max*((k*q*Q/(4))**2)/E_squared

print((A_max-A_min)/100)

# returns speed of particle in cm/s from kinetic energy E (MeV) and mass (MeV/c2)
def speed(E, m):
    return np.sqrt(2*E/m)

s_alpha = speed(E_alpha, m_alpha)
s2_alpha = speed(E_alpha-dE_alpha, m_alpha)
ds_alpha = s2_alpha-s_alpha

print("poo")
print(u_S_calc)

# Alpha partilce object. Contains kinematic variables needed for simulation
class Alpha:
    def __init__(self, t, x, v):
        self.t = t
        self.x = x
        self.v = v

    # returns radial position
    def r(self):
        return np.sqrt(self.x[0]**2 + self.x[1]**2)
    
    # return spherical components of velocity: speed, theta, phi
    def v_sph(self):
        s = np.linalg.norm(self.v)
        theta = np.arccos(self.v[2]/s)
        phi = np.arccos(self.v[0]/(s*np.sin(theta)))
        return np.array([s, theta, phi])

    # given z position and diameter of target (centered at r = 0), returns True and updates position+time if alpha hits target 
    def hit(self, z_targ, d_targ):
        if self.v[2] <= 0:
            return False
        delt = (z_targ-self.x[2])/self.v[2]
        self.t += delt
        self.x = np.add(self.x, delt*self.v)
        return self.r() < d_targ/2
    
    def update(self, t):
        delt = t - self.t
        self.x = np.add(self.x, delt*self.v)
        self.t = t
    
    # given change in speed and scattering angle, updates velocity
    def deflect(self, ds, dtheta, dphi):
        #create basis around self.v
        s = np.linalg.norm(self.v)
        th = self.v_sph()[1]
        v = self.v/s
        d = np.array([1,0,0])
        u = np.cross(v,np.add(v,d))
        u = u/np.linalg.norm(u)
        w = np.cross(v,u)

        # Create basis transformations
        B = np.array([u, w, v]).transpose() #Change of basis matrix v-basis -> standard

        # Create vector in v basis in spherical coordiantes with rotations
        v_B = np.array([np.sin(dtheta) * np.cos(dphi), np.sin(dtheta) * np.sin(dphi), np.cos(dtheta)])

        # transform back to standard basis and update v
        self.v = (s+ds) * np.matmul(B, v_B)
        
        

# Creates alpah particle with random position (uniform in area of disc-shaped source) and direction (uniform in +z direction)
def generate(t):
    r = (d_Pu/2) * np.sqrt(random.random())
    x_phi = (2*pi) * random.random()
    v_theta = np.arccos(random.random())
    v_phi = (2*pi) * random.random()
    x = np.array([r*np.cos(x_phi), r*np.sin(x_phi), 0])
    v = s_alpha * np.array([np.sin(v_theta)*np.cos(v_phi), np.sin(v_theta)*np.sin(v_phi), np.cos(v_theta)])
    return Alpha(t, x, v)

# checks if alphas hit foil, scatters alphas flat in theta, then checks if alphas hit detector
def flat_scatter(a, xs, xd):
    dtheta, itheta = -pi/4, -pi/4
    hit_detector = False
    hit_foil = a.hit(xs_min + xs, d_Au)
    if hit_foil:
        itheta = a.v_sph()[1]
        dtheta =  pi * random.random()
        dphi = (2*pi) * random.random()
        a.deflect(ds_alpha, dtheta, dphi)
        hit_detector = a.hit(xs_min + xs + xd_min + xd, d_det)
    return np.array([hit_foil, hit_detector, dtheta, itheta])

# checks if alphas hit foil, scatters alphas uniformly in solid angle, then checks if alphas hit detector
def spherical_scatter(a, xs, xd):
    dtheta, itheta = -pi/4, -pi/4
    hit_detector = False
    hit_foil = a.hit(xs_min + xs, d_Au)
    if hit_foil:
        itheta = a.v_sph()[1]
        dtheta =  np.arccos(random.uniform(-1,1))
        dphi = (2*pi) * random.random()
        a.deflect(ds_alpha, dtheta, dphi)
        hit_detector = a.hit(xs_min + xs + xd_min + xd, d_det)
    return [hit_foil, hit_detector, dtheta, itheta]

# checks if alphas hit foil, scatters alphas uniformly in solid angle, then checks if alphas hit detector
# instead of choosing random scattering angle, chooses random outgoing angle, and calculates scattering angle after the fact
def fast_spherical_scatter(a, xs, xd):
    dtheta, itheta = -pi/4, -pi/4
    hit_detector = False
    hit_foil = a.hit(xs_min + xs, d_Au)
    if hit_foil:
        itheta = a.v_sph()[1]
        theta =  np.arccos(random.uniform(-1,1))
        phi = (2*pi) * random.random()
        new_v = s2_alpha * np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        dtheta = np.arccos(np.dot(a.v, new_v)/(s_alpha*s2_alpha))
        a.v = new_v
        hit_detector = a.hit(xs_min + xs + xd_min + xd, d_det)
    return [hit_foil, hit_detector, dtheta, itheta]



# checks if alphas hit foil, scatters alphas with rutherford distribution (given normalization), then checks if alphas hit detectors
def rutherford_scatter(a, xs, xd):
    dtheta, itheta = -pi/4, -pi/4
    hit_detector = False
    hit_foil = a.hit(xs_min + xs, d_Au)
    if hit_foil:
        itheta = a.v_sph()[1]
        # mininum scattering angle (min_theta) determiend by normalization of differential cross section from min_theta to pi, with factor of A
        min_f = -2*10**12 #f(min_theta)
        max_f = -2 #f(pi)
        f = random.uniform(min_f, max_f) #flat distribution in f(theta) = -2csc^2(x/2), integral of (sinx)/(sin^4(x/2))
        dtheta =  2*np.arcsin(np.sqrt(-2/f)) # generates random theta between min_theta and pi, weighted by (sinx)/(sin^4(x/2)) 
        dphi = random.uniform(0, 2*pi)
        a.deflect(ds_alpha, dtheta, dphi)
        hit_detector = a.hit(xs_min + xs + xd_min + xd, d_det)
    return [hit_foil, hit_detector, dtheta, itheta]


# simulates generation, scattering, and detection of N alpha particles given xs, xd
# models generation of alphas every 1 second (actiivty = 1 Hz)
# models flat scattering, then weights number of scattered particles by rutherford * spherical distribution 

def simulate_flat(x, N):
    u_N_detected = 0
    N_scattered=0
    N_hit = 0
    N_detected =0
    results = np.array([flat_scatter(generate(t),x[1],x[0]) for t in range(N)])
    #    results = np.array(list(map(lambda t: flat_scatter(generate(t),x[1],x[0]), range(N))))
    hit_foil = results[:,0].astype(bool)
    scatters = results[hit_foil]
   
    hit_detector, dtheta = scatters[:,1].astype(bool), scatters[:,2]
    N_incident = len(scatters)
    if N_incident >0:
        N_scattered = sum(np.sin(dtheta)/((np.sin(dtheta/2))**4))/N_incident
        N_hit = sum(hit_detector)
        N_detected = sum(hit_detector * np.sin(dtheta)/((np.sin(dtheta/2))**4))/N_incident
    if N_hit > 0:
        u_N_detected = 0.5*N_detected/N_hit
    return N_detected, u_N_detected

#modified
def simulate_flat_cos(x, N):
    u_N_detected = 0
    N_scattered=0
    N_hit = 0
    N_detected =0
    results = np.array([flat_scatter(generate(t),x[1],x[0]) for t in range(N)])
    
    hit_foil = results[:,0].astype(bool)
    scatters = results[hit_foil]
   
    hit_detector, dtheta, itheta = scatters[:,1].astype(bool), scatters[:,2], scatters[:,3]

    N_incident = len(scatters)
    if N_incident >0:
        N_scattered = sum(np.sin(dtheta)/(np.cos(itheta)*(np.sin(dtheta/2))**4))/N_incident
        N_hit = sum(hit_detector)
        N_detected = sum(hit_detector * np.sin(dtheta)/(np.cos(itheta)*(np.sin(dtheta/2))**4))/N_incident
    if N_hit > 0:
        u_N_detected = 0.5*N_detected/N_hit
    return N_detected, u_N_detected


# simulates generation, scattering, and detection of N alpha particles given xs, xd
# models generation of alphas every 1 second (actiivty = 1 Hz)
# models spherical scattering, then weights number of scattered particles by rutherford  distribution 

def simulate_spherical(x, N):
    u_N_detected = 0
    N_scattered=0
    N_hit = 0
    N_detected =0
    results = np.array([fast_spherical_scatter(generate(t),x[1],x[0]) for t in range(N)])
    
    hit_foil = results[:,0].astype(bool)
    scatters = results[hit_foil]
   
    hit_detector, dtheta = scatters[:,1].astype(bool), scatters[:,2]
    N_incident = len(scatters)
    if N_incident >0:
        
        N_scattered = sum(1/((np.sin(dtheta/2))**4))/N_incident
        N_hit = sum(hit_detector)
        N_detected = sum(hit_detector/((np.sin(dtheta/2))**4))/N_incident
    if N_hit > 0:
        u_N_detected = 0.5*N_detected/N_hit
    return N_detected, u_N_detected


def simulate_spherical_cos(x, N):
    u_N_detected = 0
    u_N_detected=0
    N_scattered=0
    N_hit = 0
    N_detected =0
    results = np.array([fast_spherical_scatter(generate(t),x[1],x[0]) for t in range(N)])
    
    hit_foil = results[:,0].astype(bool)
    scatters = results[hit_foil]
   
    hit_detector, dtheta, itheta = scatters[:,1].astype(bool), scatters[:,2], scatters[:,3]
    N_incident = len(scatters)
    if N_incident >0:

        N_scattered = sum(1/(np.cos(itheta)*(np.sin(dtheta/2))**4))/N_incident
        N_hit = sum(hit_detector)
        N_detected = sum(hit_detector/(np.cos(itheta)*(np.sin(dtheta/2))**4))/N_incident
    if N_hit > 0:
        u_N_detected = 0.5*N_detected/N_hit
    return N_detected, u_N_detected

def simulate_rutherford(x, N):
    u_N_detected=0
    N_scattered=0
    N_hit = 0
    N_detected =0
    results = np.array([rutherford_scatter(generate(t),x[1],x[0]) for t in range(N)])
    N_scattered = sum(results[:,0])
    N_detected = sum(results[:,1])
    u_N_detected = 0.5*N_detected
    return N_detected, u_N_detected



def simulate_spherical_fast(xd, xs, N):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.random(N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    dtheta = np.arccos(rng.uniform(-1, 1, size=np.sum(hit_foil)))
    #dtheta = rng.uniform(0, pi, size=np.sum(hit_foil))
    dphi = rng.uniform(0, 2 * np.pi, size=np.sum(hit_foil))
    ds = s2_alpha - s_alpha

    itheta = np.arccos(v[hit_foil,2]/s_alpha)

    # Update velocities after scattering
    v_scattered = np.column_stack((
        np.sin(dtheta) * np.cos(dphi),
        np.sin(dtheta) * np.sin(dphi),
        np.cos(dtheta)
    )) * s2_alpha

    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]


    # Count results
    N_scattered = 0
    N_detected = 0

    N_hit_foil = np.sum(hit_foil)
    N_hit_det = np.sum(hit_detector)
    if N_hit_foil >0:
        N_scattered = np.sum((1/(np.sin(dtheta/2))**4))/(N_hit_foil)
        #N_detected = np.sum((np.sin(dtheta[hit_detector]))*(1/(np.sin(dtheta[hit_detector]/2))**4)/np.cos(itheta[hit_detector]))/(N_scattered)
        #N_detected = (N_hit_det/N_hit_foil)*N_scattered
        N_detected = np.sum((1/(np.sin(dtheta[hit_detector]/2))**4))/(N_hit_foil)

    if N_hit_det > 0:
        u_N_detected = 0.5 * N_detected / N_hit_det
    else:
        u_N_detected = 0
    return N_detected, u_N_detected

def simulate_flat_fast(xd, xs, N):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.random(N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    dtheta = rng.uniform(0, pi, size=np.sum(hit_foil))
    #dtheta = rng.uniform(0, pi, size=np.sum(hit_foil))
    dphi = rng.uniform(0, 2 * np.pi, size=np.sum(hit_foil))
    ds = s2_alpha - s_alpha

    itheta = np.arccos(v[hit_foil,2]/s_alpha)

    # Update velocities after scattering
    v_scattered = np.column_stack((
        np.sin(dtheta) * np.cos(dphi),
        np.sin(dtheta) * np.sin(dphi),
        np.cos(dtheta)
    )) * s2_alpha

    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]


    # Count results
    N_scattered = 0
    N_detected = 0

    N_hit_foil = np.sum(hit_foil)
    N_hit_det = np.sum(hit_detector)
    if N_hit_foil >0:
        N_scattered = np.sum((np.sin(dtheta)/(np.sin(dtheta/2))**4))/(N_hit_foil)
        #N_detected = np.sum((np.sin(dtheta[hit_detector]))*(1/(np.sin(dtheta[hit_detector]/2))**4)/np.cos(itheta[hit_detector]))/(N_scattered)
        #N_detected = (N_hit_det/N_hit_foil)*N_scattered
        N_detected = np.sum((np.sin(dtheta[hit_detector])/(np.sin(dtheta[hit_detector]/2))**4))/(N_hit_foil)

    if N_hit_det > 0:
        u_N_detected = 0.5 * N_detected / N_hit_det
    else:
        u_N_detected = 0
    return N_detected, u_N_detected


def simulate_spherical__cos_fast(xd, xs, N):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.random(N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    dtheta = np.arccos(rng.uniform(-1, 1, size=np.sum(hit_foil)))
    #dtheta = rng.uniform(0, pi, size=np.sum(hit_foil))
    dphi = rng.uniform(0, 2 * np.pi, size=np.sum(hit_foil))
    ds = s2_alpha - s_alpha

    itheta = np.arccos(v[hit_foil,2]/s_alpha)

    # Update velocities after scattering
    v_scattered = np.column_stack((
        np.sin(dtheta) * np.cos(dphi),
        np.sin(dtheta) * np.sin(dphi),
        np.cos(dtheta)
    )) * s2_alpha

    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]


    # Count results
    N_scattered = 0
    N_detected = 0

    N_hit_foil = np.sum(hit_foil)
    N_hit_det = np.sum(hit_detector)
    if N_hit_foil >0:
        N_scattered = np.sum((1/((np.cos(itheta))*(np.sin(dtheta/2))**4)))/(N_hit_foil)
        #N_detected = np.sum((np.sin(dtheta[hit_detector]))*(1/(np.sin(dtheta[hit_detector]/2))**4)/np.cos(itheta[hit_detector]))/(N_scattered)
        #N_detected = (N_hit_det/N_hit_foil)*N_scattered
        N_detected = np.sum((1/((np.cos(itheta[hit_detector]))*(np.sin(dtheta[hit_detector]/2))**4)))/(N_hit_foil)

    if N_hit_det > 0:
        u_N_detected = 0.5 * N_detected / N_hit_det
    else:
        u_N_detected = 0
    return N_detected, u_N_detected

def simulate_flat__cos_fast(xd, xs, N):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.random(N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    dtheta = rng.uniform(0, pi, size=np.sum(hit_foil))
    #dtheta = rng.uniform(0, pi, size=np.sum(hit_foil))
    dphi = rng.uniform(0, 2 * np.pi, size=np.sum(hit_foil))
    ds = s2_alpha - s_alpha

    itheta = np.arccos(v[hit_foil,2]/s_alpha)

    # Update velocities after scattering
    v_scattered = np.column_stack((
        np.sin(dtheta) * np.cos(dphi),
        np.sin(dtheta) * np.sin(dphi),
        np.cos(dtheta)
    )) * s2_alpha

    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]


    # Count results
    N_scattered = 0
    N_detected = 0

    N_hit_foil = np.sum(hit_foil)
    N_hit_det = np.sum(hit_detector)
    if N_hit_foil >0:
        N_scattered = np.sum((np.sin(dtheta)/((np.cos(itheta))*(np.sin(dtheta/2))**4)))/(N_hit_foil)
        #N_detected = np.sum((np.sin(dtheta[hit_detector]))*(1/(np.sin(dtheta[hit_detector]/2))**4)/np.cos(itheta[hit_detector]))/(N_scattered)
        #N_detected = (N_hit_det/N_hit_foil)*N_scattered
        N_detected = np.sum((np.sin(dtheta[hit_detector])/((np.cos(itheta[hit_detector]))*(np.sin(dtheta[hit_detector]/2))**4)))/(N_hit_foil)

    if N_hit_det > 0:
        u_N_detected = 0.5 * N_detected / N_hit_det
    else:
        u_N_detected = 0
    return N_detected, u_N_detected


def simulate_rutherford_fast(xd, xs, N, ang):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.uniform(np.cos(ang), np.cos(0), size=N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    min_f = -2/(np.sin(min_theta/2))**2
    max_f = -2 #f(pi)
     #flat distribution in f(theta) = -2csc^2(x/2), integral of (sinx)/(sin^4(x/2))
    f = rng.uniform(min_f, max_f, size=np.sum(hit_foil))
    dtheta =  2*np.arcsin(np.sqrt(-(2)/f)) # generates random theta between min_theta and pi, weighted by (sinx)/(sin^4(x/2)) 
    dphi = rng.uniform(0, 2*np.pi, size=np.sum(hit_foil))

    #create basis around self.v
    itheta = np.arccos(v[hit_foil,2]/s_alpha)
    n = v[hit_foil] / np.linalg.norm(v[hit_foil], axis=1, keepdims=True)  # Normalize original velocity vectors

       # Create an arbitrary perpendicular vector `u` that is not parallel to n
    d = np.where(np.abs(n[:, 0:1]) < 0.99, [1, 0, 0], [0, 1, 0])  
    u = np.cross(n, d)  # Perpendicular to n
    u /= np.linalg.norm(u, axis=1, keepdims=True)  # Normalize u

    # Create the second perpendicular vector `w`
    w = np.cross(n, u)  # Perpendicular to both u and n
    w /= np.linalg.norm(w, axis=1, keepdims=True)  # Normalize w

    # Construct the rotated velocity in the local frame
    v_scattered = (
        np.sin(dtheta)[:, None] * (np.cos(dphi)[:, None] * u + np.sin(dphi)[:, None] * w) +
        np.cos(dtheta)[:, None] * n
    ) *s2_alpha
 
    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]

    # Count results
    N_scattered = np.sum(hit_foil)
    N_hit = np.sum(hit_detector)
    N_detected = np.sum(hit_detector)

    return N_detected, 0.5

# def simulate_rutherford_fast(xd, xs, N):
#     return simulate_rutherford_fast(xd,xs,N,pi/2)

def simulate_rutherford_cos_fast(xd, xs, N, ang):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.uniform(np.cos(ang), np.cos(0), size=N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    min_f = -2/(np.sin(min_theta/2))**2
    max_f = -2 #f(pi)
     #flat distribution in f(theta) = -2csc^2(x/2), integral of (sinx)/(sin^4(x/2))
    f = rng.uniform(min_f, max_f, size=np.sum(hit_foil))
    dtheta =  2*np.arcsin(np.sqrt(-(2)/f)) # generates random theta between min_theta and pi, weighted by (sinx)/(sin^4(x/2)) 
    dphi = rng.uniform(0, 2*np.pi, size=np.sum(hit_foil))

    #create basis around self.v
    itheta = np.arccos(v[hit_foil,2]/s_alpha)
    n = v[hit_foil] / np.linalg.norm(v[hit_foil], axis=1, keepdims=True)  # Normalize original velocity vectors

       # Create an arbitrary perpendicular vector `u` that is not parallel to n
    d = np.where(np.abs(n[:, 0:1]) < 0.99, [1, 0, 0], [0, 1, 0])  
    u = np.cross(n, d)  # Perpendicular to n
    u /= np.linalg.norm(u, axis=1, keepdims=True)  # Normalize u

    # Create the second perpendicular vector `w`
    w = np.cross(n, u)  # Perpendicular to both u and n
    w /= np.linalg.norm(w, axis=1, keepdims=True)  # Normalize w

    # Construct the rotated velocity in the local frame
    v_scattered = (
        np.sin(dtheta)[:, None] * (np.cos(dphi)[:, None] * u + np.sin(dphi)[:, None] * w) +
        np.cos(dtheta)[:, None] * n
    ) *s2_alpha
 
    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]

    # Count results
    N_scattered = np.sum(hit_foil)
    N_hit = np.sum(hit_detector)
    N_detected = np.sum(1/np.cos(itheta[hit_detector]))

    return N_detected, 0.5*N_detected/N_hit

# def simulate_rutherford_cos_fast(xd, xs, N):
#     return simulate_rutherford_cos_fast(xd, xs, N, pi/2)



def simop_rutherford_fast(xd, xs, N,A):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.random(N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    min_f = -2 - 1/A
    max_f = -2 #f(pi)
     #flat distribution in f(theta) = -2csc^2(x/2), integral of (sinx)/(sin^4(x/2))
    f = rng.uniform(min_f, max_f, size=np.sum(hit_foil))
    dtheta =  2*np.arcsin(np.sqrt(-(2)/f)) # generates random theta between min_theta and pi, weighted by (sinx)/(sin^4(x/2)) 
    dphi = rng.uniform(0, 2*np.pi, size=np.sum(hit_foil))

    #create basis around self.v
    itheta = np.arccos(v[hit_foil,2]/s_alpha)
    n = v[hit_foil] / np.linalg.norm(v[hit_foil], axis=1, keepdims=True)  # Normalize original velocity vectors

       # Create an arbitrary perpendicular vector `u` that is not parallel to n
    d = np.where(np.abs(n[:, 0:1]) < 0.99, [1, 0, 0], [0, 1, 0])  
    u = np.cross(n, d)  # Perpendicular to n
    u /= np.linalg.norm(u, axis=1, keepdims=True)  # Normalize u

    # Create the second perpendicular vector `w`
    w = np.cross(n, u)  # Perpendicular to both u and n
    w /= np.linalg.norm(w, axis=1, keepdims=True)  # Normalize w

    # Construct the rotated velocity in the local frame
    v_scattered = (
        np.sin(dtheta)[:, None] * (np.cos(dphi)[:, None] * u + np.sin(dphi)[:, None] * w) +
        np.cos(dtheta)[:, None] * n
    ) *s2_alpha
 
    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]

    # Count results
    N_scattered = np.sum(hit_foil)
    N_hit = np.sum(hit_detector)
    N_detected = np.sum(hit_detector)

    return N_detected, 0.5

def simop_rutherford_cos_fast(xd, xs, N,A):
    rng = np.random.default_rng()  # Use NumPy's faster random number generator
    t = np.arange(0,N)  # Generate all times at once

    # Generate all particles at once
    r = (d_Pu / 2) * np.sqrt(rng.random(N))  # Radial position
    x_phi = 2 * np.pi * rng.random(N)  # Azimuthal angle
    v_theta = np.arccos(rng.random(N))  # Polar angle
    v_phi = 2 * np.pi * rng.random(N)  # Azimuthal velocity direction

    x = np.column_stack((r * np.cos(x_phi), r * np.sin(x_phi), np.zeros(N)))  # Initial positions
    v = s_alpha * np.column_stack((
        np.sin(v_theta) * np.cos(v_phi),
        np.sin(v_theta) * np.sin(v_phi),
        np.cos(v_theta)
    ))  # Initial velocities

    # Compute hitting foil
    hit_time = (xs_min + xs - x[:, 2]) / v[:, 2]
    hit_positions = x + hit_time[:, np.newaxis] * v
    hit_radii = np.sqrt(hit_positions[:, 0] ** 2 + hit_positions[:, 1] ** 2)
    hit_foil = (v[:,2] > 0) * (hit_radii < d_Au / 2)
    t = t[hit_foil] + hit_time[hit_foil]

    # Scattering
    min_f = -2 -1/A
    max_f = -2 #f(pi)
     #flat distribution in f(theta) = -2csc^2(x/2), integral of (sinx)/(sin^4(x/2))
    f = rng.uniform(min_f, max_f, size=np.sum(hit_foil))
    dtheta =  2*np.arcsin(np.sqrt(-(2)/f)) # generates random theta between min_theta and pi, weighted by (sinx)/(sin^4(x/2)) 
    dphi = rng.uniform(0, 2*np.pi, size=np.sum(hit_foil))

    #create basis around self.v
    itheta = np.arccos(v[hit_foil,2]/s_alpha)
    n = v[hit_foil] / np.linalg.norm(v[hit_foil], axis=1, keepdims=True)  # Normalize original velocity vectors

       # Create an arbitrary perpendicular vector `u` that is not parallel to n
    d = np.where(np.abs(n[:, 0:1]) < 0.99, [1, 0, 0], [0, 1, 0])  
    u = np.cross(n, d)  # Perpendicular to n
    u /= np.linalg.norm(u, axis=1, keepdims=True)  # Normalize u

    # Create the second perpendicular vector `w`
    w = np.cross(n, u)  # Perpendicular to both u and n
    w /= np.linalg.norm(w, axis=1, keepdims=True)  # Normalize w

    # Construct the rotated velocity in the local frame
    v_scattered = (
        np.sin(dtheta)[:, None] * (np.cos(dphi)[:, None] * u + np.sin(dphi)[:, None] * w) +
        np.cos(dtheta)[:, None] * n
    ) *s2_alpha
 
    # Compute hitting detector
    hit_time_det = (xs_min + xs + xd_min + xd - hit_positions[hit_foil, 2]) / v_scattered[:, 2]
    hit_positions_det = hit_positions[hit_foil] + hit_time_det[:, np.newaxis] * v_scattered
    hit_radii_det = np.sqrt(hit_positions_det[:, 0] ** 2 + hit_positions_det[:, 1] ** 2)
    hit_detector = (v_scattered[:, 2] > 0) * (hit_radii_det < d_det / 2)
    t = t[hit_detector] + hit_time_det[hit_detector]

    # Count results
    N_scattered = np.sum(hit_foil)
    N_hit = np.sum(hit_detector)
    N_detected = np.sum(1/np.cos(itheta[hit_detector]))
    if N_detected == 0:
        u_N_detected =0
    else:
        u_N_detected = 0.5*N_detected/N_hit

    return N_detected, u_N_detected
