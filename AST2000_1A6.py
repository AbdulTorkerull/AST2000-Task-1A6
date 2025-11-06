# IKKE KODEMAL
# SKREVET EGEN KODE :)
# (Python 3.9)

# Import neccesary packages
from time import time 
start_time = time() # The time of when the code starts running

# Packages used for computations
import numpy as np
import random as r

# Packages from ast2000tools that we need
import ast2000tools.utils as ast
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem

# Packages for visualization of the simulation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Constants
m_sun:float = const.m_sun # Solar mass
k:float = const.k_B # Boltzman constant
m_H2:float = const.m_H2 # mass of hydrogen gass
G:float = const.G

# Find escape velocity for my home planet with ast2000tools
seed:int = ast.get_seed('abdulaah')
solarsystem:SolarSystem = SolarSystem(seed)

planet_r:float = solarsystem.radii[0]*1000 # Home planets radius
planet_m:float = m_sun*solarsystem.masses[0] # Mass of our home planey
planet_v:float = np.sqrt(2*6.67408e-11*planet_m/planet_r) # Escape velocity for out home planet (m/s)

satellite_m:float = 1100 # Mass of our satelitte (kg)

# Initial values
L:float = 1e-6 # Size of cube (m)
T:float = 1e5 # Temperature (K)
frames:int = 1000 # Amount of steps calculated
deltaT:float = 1e-12 # Length of each step (s)
timeframe:float = 60*20 # How much time our sattelite has unntil it is supposed to have reached escape velocity
N:int = 100000 # Amount of particles

# Class that will be used to represent a gas and its behaviour
class particles:
    # Simualets the initial condition of a gass
    def __init__(self, m:float, T:float, N:int) -> None:
        '''Constructor method: N is amount of particles, m is mass of particles and T is temperature of the system '''

        self.N = N # Amount of particles in gass
        self.m = m # Mass of particles in gass
        self.sigma:float = np.sqrt((k*T)/m) # Sigma value for gauss distribution
        self.mu:int = 0 # Mu value for gauss distribution

        self.positions:np.ndarray = np.zeros((N, 3)) # Array to store the particles xyz-values 
        self.speeds:np.ndarray = np.zeros((N, 3)) # Array to store the particles VxVyVz-values 

        self.momentum_walls:float = 0 # Momentum that will be used to calculate pressure
        self.momentum_escaped:float = 0 # Momentum that will be used to calculate uppwards thrust
        self.lost:int = 0 # Amount of lost particles to calculate fuel consumption

        # Give all N particles randomized position and randomized speed
        for i in range(N):
            self.positions[i, 0], self.positions[i, 1], self.positions[i, 2] = (r.uniform(-L/2, L/2),
                                                                                r.uniform(-L/2, L/2),
                                                                                r.uniform(-L/2, L/2))
            self.speeds[i, 0], self.speeds[i, 1], self.speeds[i, 2] = (r.gauss(self.mu, self.sigma),
                                                                       r.gauss(self.mu, self.sigma),
                                                                       r.gauss(self.mu, self.sigma))

    # Method to simualte how the gas moves with time
    def nextFrame(self, timeStep:float) -> None:
        '''Method that updates the system 1 frame'''

        self.positions += self.speeds*timeStep

    # Method that makes the gas collide with a given boundry
    def collison(self, L:float) -> None:
        '''Method for checking and redirecting particles that are outside of the given bounds'''

        out_of_bounds = abs(self.positions) >= L/2 # Checks if particles are outside of the cube
        index = np.where(out_of_bounds == True) # Finds the indexes of particles that were out of bounds
        # loops only over the particles that were out of bounds (save a lot of time rather than checking every particle)
        # this change is the most noticable change from the code described in the flow chart
        # we do not loop over every particle, only particles that were found to be outside the boundry
        for j in range(len(index[0])):
            self.speeds[index[0][j], index[1][j]] = -self.speeds[index[0][j], index[1][j]]
            # Checks collision on one of the walls to calculate pressure on the box
            # The pressure calculated from this should not be influenced too much by the hole in the box, since these collisons are perpenducilar to the hole
            if (self.positions[index[0][j], 0] >= L/2):
                self.momentum_walls += 2*abs(self.speeds[index[0][j], 0])*self.m

            # Checks if particle escapes through the hole
            if ((self.positions[index[0][j], 1] <= -L/2)
                and (self.positions[index[0][j], 0] >= -L/4 and self.positions[index[0][j], 0] <= L/4)
                and (self.positions[index[0][j], 2] >= -L/4 and self.positions[index[0][j], 2] <= L/4)):
                # If a particle escape through the hole, its position and speed will be randomized again to simulate it being replaced (ensures constan pressure)
                self.positions[index[0][j], 0], self.positions[index[0][j], 1], self.positions[index[0][j], 2] = (r.uniform(-L/2, L/2),
                                                                                                                  r.uniform(-L/2, L/2),
                                                                                                                  r.uniform(-L/2, L/2))
                self.speeds[index[0][j], 0], self.speeds[index[0][j], 1], self.speeds[index[0][j], 2] = (r.gauss(self.mu, self.sigma),
                                                                                                         r.gauss(self.mu, self.sigma),
                                                                                                         r.gauss(self.mu, self.sigma))
                self.momentum_escaped += 2*abs(self.speeds[index[0][j], 1])*self.m # Momentum used to calculate uppwards force exerted on satelitte
                self.lost += 1 # Stores how many particles have escaped

            
gas = particles(m_H2, T, N) # Create an object that represent our cube filled with hydrogen gass

mean_K = 0 # Mean kinetic enery of particles
for i in range(N):
    mean_K += (1/2)*gas.m*(gas.speeds[i,0]**2 + gas.speeds[i,1]**2 + gas.speeds[i,2]**2)
mean_K = mean_K/N

data = np.zeros((frames, N, 3)) # A 3D array that stores all N particles posistion in xyz-coordinates for every single frame included in the simulation
    
# Simulation of the gas
for i in range(frames):
    gas.collison(L) # Makes sure that particles has collision with walls
    gas.nextFrame(deltaT) # Moves the gas
    data[i] = gas.positions # Stores all position for every frame in a 3D array

# Further computations to find out how many of these simulated cubes is required to launch our sattelite
force_walls = (gas.momentum_walls)/(deltaT*frames) # Constant force on the walls caused by particle collision
force_escaped = (gas.momentum_escaped)/(deltaT*frames) # Uppward force caused by particles escaping

pressure = (force_walls)/L**2 # Pressure inside the cube 
acceleration = force_escaped/satellite_m # Uppwards accelaration caused by particles escaping the cube
speed_per_cube = acceleration*timeframe # total speed 1 cube will generate for the satelite in the timeframe
amount_of_cubes = planet_v/speed_per_cube # Amount of cubes needed to reach espace velocity
amount_of_fuel = (gas.lost)*gas.m*amount_of_cubes*(timeframe/(deltaT*frames)) # Amount of fuel required

theoretic_mean_K = (3/2)*k*T
theoretic_Pressure = (N*k*T)/(L**3)

# Relevant calculations
print(force_escaped)
print(f'Initial conditions: Temperature = \033[1m{T}\033[0m K, Amount of particles = \033[1m{N}\033[0m, Length of cube sides = \033[1m{L}\033[0m m')
print()
print(f'Escape velocity: \033[1m{planet_v}\033[0m m/s (Task 1)')
print(f'Average kinetik energy of particles = \033[1m{mean_K}\033[0m J, Theoretical kinetik energy = \033[1m{theoretic_mean_K}\033[0m J (Task 3)')
print(f'Pressure in every cube = \033[1m{pressure}\033[0m Pa, Theoretical pressure = \033[1m{theoretic_Pressure}\033[0m Pa (Task 5)')
print(f'minimum amounts of cubes required: \033[1m{int(amount_of_cubes)}\033[0m (Task 8)')
print(f'minimum amounts of fuel required: \033[1m{amount_of_fuel}\033[0m kg of pure hyrdrogen gass (Task 9)')
print()

end_time = time() # The time of when the code stops running
print(f'The code used \033[1m{end_time - start_time}\033[0m seconds to fully execute')
print()

# Code for plotting the Guassian distrubution used for determining speed of particles.
'''v = np.linspace(gas.mu - 4*gas.sigma, gas.mu + 4*gas.sigma, 1000)
gaussian = (1/(gas.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v - gas.mu) / gas.sigma)**2)*100
    
plt.plot(v, gaussian, label=f'$\mu$ = {gas.mu}, $\sigma$ = {gas.sigma:.0f}')
plt.title('Gaussian (Normal) Distribution')
plt.xlabel('Particle speed (m/s)')
plt.ylabel('Probability (%)')
plt.legend()
plt.grid(True)
plt.show()'''

# Animation to visualize in 3D (Slow if N has a order of magnitude larger than 1e2)
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatters = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2])
ax.set_xlim([-L/2, L/2])
ax.set_ylim([-L/2, L/2])
ax.set_zlim([-L/2, L/2])

def update(frame):
    scatters._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
    return scatters,

ani = FuncAnimation(fig, update, frames = frames, interval = 1, blit=False, repeat=False)
plt.show()'''