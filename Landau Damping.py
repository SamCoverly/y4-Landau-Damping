#!/usr/bin/env python
#
# Electrostatic PIC code in a 1D cyclic domain

from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt # Matplotlib plotting library

try:
    import matplotlib.gridspec as gridspec  # For plot layout grid
    got_gridspec = True
except:
    got_gridspec = False

# Need an FFT routine, either from SciPy or NumPy
try:
    from scipy.fftpack import fft, ifft
except:
    # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft

def rk4step(f, y0, dt, args=()):
    """ Takes a single step using RK4 method """
    k1 = f(y0, *args)
    k2 = f(y0 + 0.5*dt*k1, *args)
    k3 = f(y0 + 0.5*dt*k2, *args)
    k4 = f(y0 + dt*k3, *args)

    return y0 + (k1 + 2.*k2 + 2.*k3 + k4)*dt / 6.

def calc_density(position, ncells, L):
    """ Calculate charge density given particle positions

    Input
      position  - Array of positions, one for each particle
                  assumed to be between 0 and L
      ncells    - Number of cells
      L         - Length of the domain

    Output
      density   - contains 1 if evenly distributed
    """
    # This is a crude method and could be made more efficient

    density = zeros([ncells])
    nparticles = len(position)

    dx = L / ncells       # Uniform cell spacing
    for p in position / dx:    # Loop over all the particles, converting position into a cell number
        plower = int(p)        # Cell to the left (rounding down)
        offset = p - plower    # Offset from the left
        density[plower] += 1. - offset
        density[(plower + 1) % ncells] += offset
    # nparticles now distributed amongst ncells
    density *= float(ncells) / float(nparticles)  # Make average density equal to 1
    return density

def periodic_interp(y, x):
    """
    Linear interpolation of a periodic array y at index x

    Input

    y - Array of values to be interpolated
    x - Index where result required. Can be an array of values

    Output

    y[x] with non-integer x
    """
    ny = len(y)
    if len(x) > 1:
        y = array(y) # Make sure it's a NumPy array for array indexing
    xl = floor(x).astype(int) # Left index
    dx = x - xl
    xl = ((xl % ny) + ny) % ny  # Ensures between 0 and ny-1 inclusive
    return y[xl]*(1. - dx) + y[(xl+1)%ny]*dx

def fft_integrate(y):
    """ Integrate a periodic function using FFTs
    """
    n = len(y) # Get the length of y

    f = fft(y) # Take FFT
    # Result is in standard layout with positive perioduencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]

    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )
    k = 2.*pi*k/n

    # Modify perioduencies by dividing by ik
    f[1:] /= (1j * k[1:])
    f[0] = 0. # Set the arbitrary zero-perioduency term to zero

    return ifft(f).real # Reverse Fourier Transform


def pic(f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    nparticles = len(f)/2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L

    # Calculate number density, normalised so 1 when uniform
    density = calc_density(pos, ncells, L)

    # Subtract ion density to get total charge density
    rho = density - 1.

    # Calculate electric field
    E = -fft_integrate(rho)*dx

    # Interpolate E field at particle locations
    accel = -periodic_interp(E, pos/dx)

    # Put back into a single array
    return concatenate( (vel, accel) )

####################################################################

def run(pos, vel, L, ncells=None, out=[], output_times=linspace(0,20,100), cfl=0.5):

    if ncells == None:
        ncells = int(sqrt(len(pos))) # A sensible default

    dx = L / float(ncells)

    f = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)

    time = 0.0
    for tnext in output_times:
        # Advance to tnext
        stepping = True
        while stepping:
            # Maximum distance a particle can move is one cell
            dt = cfl * dx / max(abs(vel))
            if time + dt >= tnext:
                # Next time will hit or exceed required output time
                stepping = False
                dt = tnext - time
            #print "Time: ", time, dt
            f = rk4step(pic, f, dt, args=(ncells, L))
            time += dt

        # Extract position and velocities
        pos = ((f[0:nparticles] % L) + L) % L
        vel = f[nparticles:]

        # Send to output functions
        for func in out:
            func(pos, vel, ncells, L, time)

    return pos, vel

####################################################################
#
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):

        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])

        # Plot initial positions
        if got_gridspec:
            self.fig = plt.figure()
            self.gs = gridspec.GridSpec(4, 4)
            ax = self.fig.add_subplot(self.gs[0:3,0:3])
            self.phase_plot = ax.plot(pos, vel, '.')[0]
            ax.set_title("Phase space")

            ax = self.fig.add_subplot(self.gs[3,0:3])
            self.density_plot = ax.plot(linspace(0, L, ncells), d)[0]

            ax = self.fig.add_subplot(self.gs[0:3,3])
            self.vel_plot = ax.plot(vhist, vbins)[0]
        else:
            self.fig = plt.figure()
            self.phase_plot = plt.plot(pos, vel, '.')[0]

            self.fig = plt.figure()
            self.density_plot = plt.plot(linspace(0, L, ncells), d)[0]

            self.fig = plt.figure()
            self.vel_plot = plt.plot(vhist, vbins)[0]
        plt.ion()
        plt.show()
        
    def __call__(self, pos, vel, ncells, L, t):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])

        #self.phase_plot.set_data(pos, vel) # Update the plot
        #self.density_plot.set_data(linspace(0, L, ncells), d)
        #self.vel_plot.set_data(vhist, vbins)
        plt.draw()
        plt.pause(0.05)


class Summary:
    def __init__(self):
        self.t = []
        self.firstharmonic = []

    def __call__(self, pos, vel, ncells, L, t):
        # Calculate the charge density
        d = calc_density(pos, ncells, L)

        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)

        print "Time:", t, "First:", fh

        self.t.append(t)
        self.firstharmonic.append(fh)

####################################################################
#
# Functions to create the initial conditions
#

def landau(npart, L, alpha=0.2):
    """
    Creates the initial conditions for Landau damping

    """
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    pos0 = pos.copy()
    k = 2.*pi / L
    for i in range(10): # Adjust distribution using Newton iterations
        pos -= ( pos + alpha*sin(k*pos)/k - pos0 ) / ( 1. + alpha*cos(k*pos) )

    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)

    return pos, vel

def twostream(npart, L, vbeam=2):
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)

    np2 = int(npart / 2)
    vel[:np2] += vbeam  # Half the particles moving one way
    vel[np2:] -= vbeam  # and half the other

    return pos,vel

####################################################################

if __name__ == "__main__":
    # Generate initial condition
    #
    if False:
        # 2-stream instability
        L = 100
        ncells = 20
        pos, vel = twostream(10000, L, 3.)
    else:
        # Landau damping
        L = 4.*pi
        ncells = 20
        npart = 20000
        pos, vel = landau(npart, L)

    # Create some output classes
    p = Plot(pos, vel, ncells, L) # This displays an animated figure
    s = Summary()                 # Calculates, stores and prints summary info

    # Run the simulation
    pos, vel = run(pos, vel, L, ncells,
                   out=[p, s],                      # These are called each output
                   output_times=linspace(0.,40,100)) # The times to output

    # Summary stores an array of the first-harmonic amplitude
    # Make a semilog plot to see exponential damping
    gradient = [] #stores gradient of first harmonic
    gradient.append(1)
    turn_t = []
    turn_amp = []
    
    for i in range(1,len(s.firstharmonic)-1,1):
        gradient.append((s.firstharmonic[i+1]-s.firstharmonic[i]))
        if (gradient[i]*gradient[i-1] < 0):
            turn_t.append(s.t[i])
            turn_amp.append(s.firstharmonic[i])
            
    if (turn_amp[0]>turn_amp[1]):
        del turn_amp[1::2]
        del turn_t[1::2]
    else:
        del turn_amp[0::2]
        del turn_t[0::2]
        
    period = []
    period_max =0
    period_avg=0
    period_min = 100 
    noise_amp = []
    noise_t = []
      
    for i in range(1,len(turn_amp),1):
        period.append(turn_t[i]-turn_t[i-1])
        if turn_amp[i]>turn_amp[i-1]:
            noisetime = turn_t[i]
            for j in range(i,len(turn_amp),1):
                noise_amp.append(turn_amp[j])
                noise_t.append(turn_t[j])
            break
    '''for i in range(0,len(period),1):
        if (period[i] > periodmax):
            periodmax = period[i]
        if (period[i] < periodmin):
            periodmin = period[i]
        periodavg+=period[i]
        count += 1
    periodavg = periodavg/count
    perioderror = (periodmax-periodmin)/2
        '''
    period_max = max(period)
    period_min = min(period)
    period_avg = 2*sum(period)/(len(period))
    period_error = (period_max-period_min)/2
    freq = 2*pi*(1/period_avg)
    
        
    plt.figure()
    plt.plot(s.t, s.firstharmonic)
    plt.plot(turn_t,turn_amp, 'gx')
    plt.xlabel("Time [Normalised]")
    plt.ylabel("First harmonic amplitude [Normalised]")
    plt.yscale('log')

    print "Frequency = ",freq,"+- ",period_error," , noisetime = ",noisetime
    print "periodmax = ",period_max, ", periodmin = ",period_min
    
    plt.ioff() # This so that the windows stay open
    plt.show()
