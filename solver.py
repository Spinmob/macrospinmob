import ctypes            as _c
import numpy             as _n
import os                as _os
import spinmob           as _s
import spinmob.egg       as _egg; _g = _egg.gui
import pyqtgraph.opengl  as _gl
import pyqtgraph         as _pg
from sys import platform as _platform
import traceback         as _t
import time              as _time
_p = _t.print_last

# Find the path to the compiled c-code (only Windows and Linux supported so far.)
if   _platform in ['win32']:  _path_dll = _os.path.join(_os.path.split(__file__)[0],'engine-windows.dll')
elif _platform in ['darwin']: _path_dll = _os.path.join(_os.path.split(__file__)[0],'engine-osx.so')
else:                         _path_dll = _os.path.join(_os.path.split(__file__)[0],'engine-linux.so')

# Windows currently can't handle the 3d plots for some reason.
if _platform == 'win32': _3d_enabled = False
else:                    _3d_enabled = True

# Get the engine.
_engine = _c.cdll.LoadLibrary(_path_dll)

# Constants
pi   = _n.pi
u0   = 1.25663706212e-6 # Vacuum permeability [H/m | N/A^2]
ec   = 1.60217662e-19   # Elementary charge [Coulomb]
me   = 9.1093837015e-31 # Electron mass [kg]
hbar = 1.0545718e-34    # [m^2 kg/s | J s]
uB   = 9.274009994e-24  # Bohr magneton [J/T]
c    = 299792458.0      # Speed of light [m/s]
kB   = 1.380649e-23   # Boltzmann constant [J/K]

## TO DO: Get check boxes working!

def _to_pointer(numpy_array): 
    """
    Converts the supplied numpy_array (assumed to be the usual 64-bit float)
    to a pointer, allowing it ot be "connected" to the C-code engine. If None
    is supplied, this returns None.
    
    SUPER IMPORTANT
    ---------------
    Make sure you assign your array to a variable name prior to calling this.
    If you do not keep a handle on the numpy array, garbage collection can
    delete it while the simulation is running!
    
    Parameters
    ----------
    numpy_array
        1D ndarray of 64-bit floats.
        
    Returns
    -------
    C-pointer to the first element, or None if numpy_array=None
    
    Examples
    --------
    my_solver.Bxs = _to_pointer(my_Bxs).
    
    """
    if numpy_array is None: return None
    return numpy_array.ctypes.data_as(_c.POINTER(_c.c_double))

def set_log_level(self, level=0):
    """
    Sets the log level for the engine.
    """
    _c.c_int.in_dll(_engine, 'log_level').value = level

class _domain(_c.Structure):
    """
    Structure for sending all the simulation parameters to the c library.
    The user should not interact with this for the most part.
    """
        
    # NOTE: The order here and structure here REALLY has to match the struct 
    # in the C-code! This class will be sent by reference, and the c-code 
    # will expect everything to be in its place.
    
    # We use underscores on the array pointers, so that the solver_api can 
    # store the numpy arrays without them for easy user interfacing.
    _fields_ = [
        
        # Index of valid Langevin field
        ('n_langevin_valid', _c.c_long),
        
        # Temperature [K]
        ('T', _c.c_double), ("_Ts", _c.POINTER(_c.c_double)),
        
        # Volume of domain [m^3]
        ('V', _c.c_double), ("_Vs", _c.POINTER(_c.c_double)),
        
        # Magnitude of the gyromagnetic ratio [radians / (sec T)]
        ('gyro', _c.c_double), ("_gyros", _c.POINTER(_c.c_double)),
        
        #Gilbert damping parameter [unitless]
        ('M', _c.c_double), ("_Ms", _c.POINTER(_c.c_double)),
        
        # Gilbert damping
        ('damping', _c.c_double), ('_dampings', _c.POINTER(_c.c_double)),
        
        # Exchange-like field strength [T], applied in the direction of the other domain's unit vector
        ('X', _c.c_double), ('_Xs', _c.POINTER(_c.c_double)),
        
        # Spin transfer torque (rate) parallel to other domain [rad / s]
        ('STT', _c.c_double), ('_STTs', _c.POINTER(_c.c_double)),
        
        # Other torQues (rate) unrelated to either domain [rad / s]
        ('Qx', _c.c_double), ('_Qxs', _c.POINTER(_c.c_double)),
        ('Qy', _c.c_double), ('_Qys', _c.POINTER(_c.c_double)),
        ('Qz', _c.c_double), ('_Qzs', _c.POINTER(_c.c_double)),
        
        # Externally applied field
        ('Bx', _c.c_double), ('_Bxs', _c.POINTER(_c.c_double)),
        ('By', _c.c_double), ('_Bys', _c.POINTER(_c.c_double)),
        ('Bz', _c.c_double), ('_Bzs', _c.POINTER(_c.c_double)),
        
        # Anisotropy tensor elements [unitless], defined such that Nxx+Nyy+Nzz=1 for an aligned ellipsoid
        ('Nxx', _c.c_double), ('_Nxxs', _c.POINTER(_c.c_double)),
        ('Nxy', _c.c_double), ('_Nxys', _c.POINTER(_c.c_double)),
        ('Nxz', _c.c_double), ('_Nxzs', _c.POINTER(_c.c_double)),
    
        ('Nyx', _c.c_double), ('_Nyxs', _c.POINTER(_c.c_double)),
        ('Nyy', _c.c_double), ('_Nyys', _c.POINTER(_c.c_double)),
        ('Nyz', _c.c_double), ('_Nyzs', _c.POINTER(_c.c_double)),
        
        ('Nzx', _c.c_double), ('_Nzxs', _c.POINTER(_c.c_double)),
        ('Nzy', _c.c_double), ('_Nzys', _c.POINTER(_c.c_double)),
        ('Nzz', _c.c_double), ('_Nzzs', _c.POINTER(_c.c_double)),
    
        # Dipole tensor [unitless], representing the fraction of the other layer's saturation magnetization
        ('Dxx', _c.c_double), ('_Dxxs', _c.POINTER(_c.c_double)),
        ('Dxy', _c.c_double), ('_Dxys', _c.POINTER(_c.c_double)),
        ('Dxz', _c.c_double), ('_Dxzs', _c.POINTER(_c.c_double)),
        
        ('Dyx', _c.c_double), ('_Dyxs', _c.POINTER(_c.c_double)),
        ('Dyy', _c.c_double), ('_Dyys', _c.POINTER(_c.c_double)),
        ('Dyz', _c.c_double), ('_Dyzs', _c.POINTER(_c.c_double)),
        
        ('Dzx', _c.c_double), ('_Dzxs', _c.POINTER(_c.c_double)),
        ('Dzy', _c.c_double), ('_Dzys', _c.POINTER(_c.c_double)),
        ('Dzz', _c.c_double), ('_Dzzs', _c.POINTER(_c.c_double)),
        
        # Solver mode: 0=disabled, 1=LLG
        ('mode', _c.c_int),
        
        # Initial conditions
        ('x0',   _c.c_double),
        ('y0',   _c.c_double),
        ('z0',   _c.c_double),
        
        # Solution arrays
        ('_x', _c.POINTER(_c.c_double)),
        ('_y', _c.POINTER(_c.c_double)),
        ('_z', _c.POINTER(_c.c_double)),
        
        # Calculated Langevin fields
        ('_Lx', _c.POINTER(_c.c_double)),
        ('_Ly', _c.POINTER(_c.c_double)),
        ('_Lz', _c.POINTER(_c.c_double)),
        
    ] # End of _data structure
    
    def keys(self):
        """
        Returns a list of keys that can be used in set() or get().
        """
        return ['T', 'V', 'x0', 'y0', 'z0', 'gyro', 'M', 'damping', 'X', 'STT',
                'Qx', 'Qy', 'Qz', 'Bx', 'By', 'Bz',
                'Nxx', 'Nxy', 'Nxz', 'Nyx', 'Nyy', 'Nyz', 'Nzx', 'Nzy', 'Nzz',
                'Dxx', 'Dxy', 'Dxz', 'Dyx', 'Dyy', 'Dyz', 'Dzx', 'Dzy', 'Dzz',
                'mode']
    
    def set(self, key, value): 
        """
        Sets the specified parameter (key) to the specified value. Specifically,
        will be set by evaluating self.keyword = value. In the case of an array, 
        it will convert it to a pointer and (64-bit float) use an underscore, as needed, 
        saving a "local" copy of the supplied value, such that garbage 
        collection doesn't automatically delete it.

        Parameters
        ----------
        key:
            Parameter name to set, e.g. 'Bx'. 
        value:
            Value to set it to. Can be a number or ndarray.
        
        Example
        -------
        my_solver.a.set(By=numpy.linspace(0,5,my_solver.steps), gyro=27)

        Returns
        -------
        self

        """
    
        # Store the "local" copy, to make sure we keep it from getting
        # deleted by the garbage collection.
        exec('self.'+key+"s=v", dict(self=self,v=value))
        
        # If it's an array, convert it before setting and use _*s
        if type(value)==list: value = _n.ndarray(value)
        if type(value)==_n.ndarray:
            exec('self._'+key+"s=v", dict(self=self,v=_to_pointer(value)))
        
        # Otherwise it's a number, so we need to kill the old pointer and
        # array
        else: 
            exec('self.' +key+'=v', dict(self=self,v=value))
            exec('self.' +key+'s=None', dict(self=self)) # Local copy
            exec('self._'+key+"s=None", dict(self=self)) # Converted copy
            
        return self
    
    __setitem__ = set
    
    def set_multiple(self, **kwargs):
        """
        Sends all keyword arguments to self.set().
        """
        for k in kwargs: self[k] = kwargs[k]
    
    __call__ = set_multiple
    
    def get(self, key='Bx'):
        """
        Returns the specified parameter. Will return the array (e.g., Bxs) 
        if there is one, and the value if there is not.
        """
        # If it's an array, return that
        if hasattr(self, key+'s'):
            x = eval('self.'+key+'s', dict(self=self))
            if x is not None: return x
        
        # Otherwise just return the value.
        return eval('self.'+key, dict(self=self))
    
    __getitem__ = get
    
    def clear_arrays(self):
        """
        This sets all array pointers to NULL (None).
        """
        self['gyro'] = self.gyro
        self['M']     = self.M
        self['damping'] = self.damping
        self['X']     = self.X
        self['STT']   = self.STT
        
        self['Bx']    = self.Bx
        self['By']    = self.By
        self['Bz']    = self.Bz
        
        self['Nxx']   = self.Nxx
        self['Nxy']   = self.Nxy
        self['Nxz']   = self.Nxz
        self['Nyx']   = self.Nyx
        self['Nyy']   = self.Nyy
        self['Nyz']   = self.Nyz
        self['Nzx']   = self.Nzx
        self['Nzy']   = self.Nzy
        self['Nzz']   = self.Nzz
        
        self['Dxx']   = self.Dxx
        self['Dxy']   = self.Dxy
        self['Dxz']   = self.Dxz
        self['Dyx']   = self.Dyx
        self['Dyy']   = self.Dyy
        self['Dyz']   = self.Dyz
        self['Dzx']   = self.Dzx
        self['Dzy']   = self.Dzy
        self['Dzz']   = self.Dzz
    



class solver_api():
    """
    Scripted interface for the solver engine. 
    """
    
    _solver_keys  = ['dt', 'steps', 'continuous']
    
    def __init__(self, **kwargs):

        # Store the default run parameters
        self.dt             = 1e-12
        self.steps          = 1e3
        self.continuous     = True
        self.valid_solution = False
        self.log_level      = 0
                
        # Create the settings structure, and set the default values.
        self.a = _domain()
        self.b = _domain()

        # Store the default magnetic parameters
        self['T']       = 0
        self['gyro']    = 1.76085963023e11
        self['M']       = 1
        self['V']       = 1000e-27
        self['damping'] = 0.01
        self['X']       = 0
        self['STT']     = 0
        self['Bx']      = 0
        self['By']      = 0
        self['Bz']      = 1
        
        # Initial conditions
        self.a.x0   = 1.0
        self.a.y0   = 0.0
        self.a.z0   = 0.0
        self.b.x0   = 1.0
        self.b.y0   = 0.0
        self.b.z0   = 0.0
        
        # Solution arrays
        self.a.x = self.a.y = self.a.z = None
        self.b.x = self.b.y = self.b.z = None
        self.a.Lx = self.a.Ly = self.a.Lz = None
        self.b.Lx = self.b.Ly = self.b.Lz = None
        
        # Index up to which the langevin field has been calculated
        self.a.n_langevin_valid = -1
        self.b.n_langevin_valid = -1

        # Null all the array pointers, just to be safe 
        # (different platforms, Python versions, etc...)
        self.a._Ts       = self.b._Ts       = None
        self.a._Vs       = self.b._Vs       = None
        self.a._gyros    = self.b._gyros    = None
        self.a._Ms       = self.b._Ms       = None
        self.a._dampings = self.b._dampings = None
        self.a._Xs       = self.b._Xs       = None
        self.a._STTs     = self.b._STTs     = None
        
        self.a._Bxs    = self.b._Bxs    = None
        self.a._Bys    = self.b._Bys    = None
        self.a._Bzs    = self.b._Bzs    = None
        
        self.a._Qxs    = self.b._Qxs    = None
        self.a._Qys    = self.b._Qys    = None
        self.a._Qzs    = self.b._Qzs    = None
        
        self.a._Nxxs   = self.b._Nxxs   = None
        self.a._Nxys   = self.b._Nxys   = None
        self.a._Nxzs   = self.b._Nxzs   = None
        self.a._Nyxs   = self.b._Nyxs   = None
        self.a._Nyys   = self.b._Nyys   = None
        self.a._Nyzs   = self.b._Nyzs   = None
        self.a._Nzxs   = self.b._Nzxs   = None
        self.a._Nzys   = self.b._Nzys   = None
        self.a._Nzzs   = self.b._Nzzs   = None
        
        self.a._Dxxs   = self.b._Dxxs   = None
        self.a._Dxys   = self.b._Dxys   = None
        self.a._Dxzs   = self.b._Dxzs   = None
        self.a._Dyxs   = self.b._Dyxs   = None
        self.a._Dyys   = self.b._Dyys   = None
        self.a._Dyzs   = self.b._Dyzs   = None
        self.a._Dzxs   = self.b._Dzxs   = None
        self.a._Dzys   = self.b._Dzys   = None
        self.a._Dzzs   = self.b._Dzzs   = None
        
        # By default, enable the two domains.
        self.a.mode=1
        self.b.mode=1
        
        # No solution arrays initially
        self.a._x  = self.a._y  = self.a._z  = None
        self.a._Lx = self.a._Ly = self.a._Lz = None
        self.b._x  = self.b._y  = self.b._z  = None
        self.b._Lx = self.b._Ly = self.b._Lz = None
        
        # Send any supplied parameters
        self.set_multiple(**kwargs)
        
    def set(self, key, value): 
        """
        Sets a parameter for the solver. Magnetic parameters without a domain
        specified, e.g., 'gyro' (not 'a/gyro') will be applied to both domains.

        Parameters
        ----------
        key:
            Parameter name to set, e.g. 'Bx'. 
        value:
            Value to set it to. Can be a number or ndarray.
        
        Example
        -------
        my_solver.a.set(By=numpy.linspace(0,5,my_solver.steps), gyro=27)

        Returns
        -------
        self

        """
        s = key.split('/')
        
        # If it's a property of the solver [dt, steps], store it in the solver
        if key in self._solver_keys:
            exec('self.'+key+"=v", dict(self=self,v=value))
    
        # If we've specified a domain
        elif s[0] == 'a': self.a.set(s[-1], value)
        elif s[0] == 'b': self.b.set(s[-1], value)
        
        # Otherwise, if it's something that we send to both a and b.
        elif key in self.a.keys():
            self.a.set(key, value)
            self.b.set(key, value)
        
        else: 
            print('OOPS solver_api.set(): Cannot find key "'+key+'"')
        return self
    
    __setitem__ = set
    
    def set_multiple(self, **kwargs):
        """
        Sends all keyword arguments to self.set().
        """
        for k in kwargs: self[k] = kwargs[k]
        return self
    
    __call__ = set_multiple
    
    def get(self, key='a/Bx'):
        """
        Returns the specified parameter. Will return the array (e.g., Bxs) 
        if there is one, and the value if there is not.
        
        Parameters
        ----------
        key='a/Bx'
            Key of item to retrieve. Specify domain 'a' or 'b' with a '/' as in
            the example for domain-specific items, and just the parameter for
            the solver itself, e.g., 'steps'.
        
        Returns
        -------
        The value (or array if present).
        """
        
        # First split by '/'
        s = key.split('/')
        
        # If length is 1, this is a solver parameter
        if len(s) == 1: return eval('self.'+key)
        
        # Otherwise, it's a domain parameter
        domain = eval('self.'+s[0])
        key    = s[1]
        
        return domain.get(key)
    
    __getitem__ = get
    
    def create_solution_arrays(self):
        """
        If no solution arrays exist, or their length does not match self['steps'],
        create new ones and specify self['valid_solution'] = False, so we know
        they are not valid solutions.
        """
        
        # If no solution arrays exist or they are the wrong length, 
        # create new ones (and specify that there is no valid solution yet!)
        if self.a.x is None                 \
        or not type(self.a.x) == _n.ndarray \
        or not  len(self.a.x) == self.steps:
            
            # Create a bunch of zeros arrays for the solver to fill.
            # Also get the pointer to these arrays for the solver engine.
            self.a.x = _n.zeros(self.steps); self.a._x = _to_pointer(self.a.x)
            self.a.y = _n.zeros(self.steps); self.a._y = _to_pointer(self.a.y)
            self.a.z = _n.zeros(self.steps); self.a._z = _to_pointer(self.a.z)
            self.b.x = _n.zeros(self.steps); self.b._x = _to_pointer(self.b.x)
            self.b.y = _n.zeros(self.steps); self.b._y = _to_pointer(self.b.y)
            self.b.z = _n.zeros(self.steps); self.b._z = _to_pointer(self.b.z)
            
            # Only create langevin arrays if the temperature is nonzero or 
            self.a.Lx  = _n.zeros(self.steps); self.a._Lx = _to_pointer(self.a.Lx)
            self.a.Ly  = _n.zeros(self.steps); self.a._Ly = _to_pointer(self.a.Ly)
            self.a.Lz  = _n.zeros(self.steps); self.a._Lz = _to_pointer(self.a.Lz)
            self.a.n_langevin_valid = -1
            
            self.b.Lx = _n.zeros(self.steps); self.b._Lx = _to_pointer(self.b.Lx)
            self.b.Ly = _n.zeros(self.steps); self.b._Ly = _to_pointer(self.b.Ly)
            self.b.Lz = _n.zeros(self.steps); self.b._Lz = _to_pointer(self.b.Lz)
            self.b.n_langevin_valid = -1
        
            # Remember that we don't have a valid solution yet.
            self.valid_solution = False
        
            
        # Otherwise, just use the arrays we've been using.
        
        return self
    
    def reset(self):
        """
        Resets the magnetization to the specified initial conditions.
        
        Specifically, this creates solution arrays if necessary, and then 
        sets the first element as per self['a/x0'], self['a/y0'], etc.
        Then sets n_langevin_valid = -1 for both domains, ensuring that 
        the Langevin field will be calculated for index n=0 onward. 
        Finally, sets self.valid_solution = False, because now the
        current solution is not valid.
        """
        # If we don't have solution arrays, create them.
        self.create_solution_arrays()
        
        # User-specified initial conditions
        self.a.x[0] = self.a.x0
        self.a.y[0] = self.a.y0
        self.a.z[0] = self.a.z0
        self.b.x[0] = self.b.x0
        self.b.y[0] = self.b.y0
        self.b.z[0] = self.b.z0
        
        # Make sure to calculate the first value of the langevin field
        self.a.n_langevin_valid = -1
        self.b.n_langevin_valid = -1
        
        # We don't have a valid solution
        self.valid_solution = False
        
        return self
    
    def transfer_last_to_initial(self):
        """
        Sets the initial values of the magnetizations and Langevin fields
        to the last element of the previously calculated arrays (if we have
        a valid solution!). Also sets self.n_langevin_valid=0 so that the 
        engine doesn't re-calculate the zero'th element of the Langevin field.
        """
        if self.valid_solution:
            
            # Set the initial magnetization to the final value
            self.a.x[0] = self.a.x[-1]
            self.a.y[0] = self.a.y[-1]
            self.a.z[0] = self.a.z[-1]
            self.b.x[0] = self.b.x[-1]
            self.b.y[0] = self.b.y[-1]
            self.b.z[0] = self.b.z[-1]
                        
            # Do the same for Langevin components
            self.a.Lx[0] = self.a.Lx[-1]
            self.a.Ly[0] = self.a.Ly[-1]
            self.a.Lz[0] = self.a.Lz[-1]
            self.a.n_langevin_valid = 0 # Engine won't calculate for n=0
                
            self.b.Lx[0] = self.b.Lx[-1]
            self.b.Ly[0] = self.b.Ly[-1]
            self.b.Lz[0] = self.b.Lz[-1]
            self.a.n_langevin_valid = 0 # Engine won't calculate for n=0
        
        else:
            print("ERROR in transfer_last_to_initial(): no valid solution!")
        
        return self
    
    def run(self):
        """
        Creates the solution arrays (self.a.x, self.a.Lx, etc) and runs 
        the solver to fill them up. Afterward, the initial conditions are 
        (by default) set to the last value of the solution arrays.
        """
        self.steps = int(self.steps)
        
        # Create solution arrays
        self.create_solution_arrays()
            
        # If we have a valid previous solution and are in continuous mode,
        # Set the initial values to the last calculated values
        if self.continuous and self.valid_solution: 
            self.transfer_last_to_initial()
        
        # Otherwise, we just reset the solver
        else: self.reset()
        
        # Really large exponents seem to slow this thing way down!
        # I think 100 digits of precision is a bit much anyway, but...
        roundoff = 1e-200
        if abs(self.a.x[0]) < roundoff: self.a.x[0] = 0.0
        if abs(self.a.y[0]) < roundoff: self.a.y[0] = 0.0
        if abs(self.a.z[0]) < roundoff: self.a.z[0] = 0.0
        if abs(self.b.x[0]) < roundoff: self.b.x[0] = 0.0
        if abs(self.b.y[0]) < roundoff: self.b.y[0] = 0.0
        if abs(self.b.z[0]) < roundoff: self.b.z[0] = 0.0

        # Solve it.
        _engine.solve_heun.restype = None
        _engine.solve_heun(_c.byref(self.a), _c.byref(self.b), 
                           _c.c_double(self.dt), _c.c_long(self.steps),
                           _c.c_int(self.log_level))
        
        # Let future runs know there is a valid solution for initialization
        self.valid_solution = True
            
        return self


class solver():
    """
    Graphical and scripted interface for the solver_api. Creating an instance
    should pop up a graphical interface.
    """    
    
    def __init__(self):
        
        # For benchmarking
        self._t0  = 0
        self._t05 = 0.5
        self._t1  = 1
        self._t2  = 2
        self._t3  = 3
        
        # Solver application programming interface.
        self.api = solver_api()
        self.a = self.api.a
        self.b = self.api.b
        
        # Previous values for the thermal field
        self._a_thermal_Bx_last = None
        self._a_thermal_By_last = None
        self._a_thermal_Bz_last = None
        
        self._b_thermal_Bx_last = None
        self._b_thermal_By_last = None
        self._b_thermal_Bz_last = None
        
        # Previous parameters relevant to thermal field
        self._dt       = None
        self._T        = None
        
        self._a_M      = None
        self._a_volume = None
        self._a_gyro  = None
        self._a_damping  = None
        
        self._b_M      = None
        self._b_volume = None
        self._b_gyro  = None
        self._b_damping  = None
        
        # Set up the GUI
        self._build_gui()
    
    def _build_gui(self):
        """
        Creates the window, puts all the widgets in there, and then shows it.
        """
        
        # Graphical interface
        self.window = _g.Window(title='Macrospin(mob)', autosettings_path='solver.window.txt', size=[1000,550])
        
        # Top row controls for the "go" button, etc
        self.grid_top          = self.window  .add(_g.GridLayout(False), alignment=1) 
        self.button_run         = self.grid_top.add(_g.Button('Go!', True))
        self.label_iteration   = self.grid_top.add(_g.Label(''))
        
        # Bottom row controls for settings and plots.
        self.window.new_autorow()
        self.grid_bottom  = self.window     .add(_g.GridLayout(False), alignment=0)
        
        # Settings
        self.settings     = self.grid_bottom.add(_g.TreeDictionary(autosettings_path='solver.settings.txt')).set_width(210)
        
        self.settings.add_parameter('solver/iterations', 0, limits=(0,None))
        self.settings.add_parameter('solver/dt',    1e-12, dec=True,                  siPrefix=True, suffix='s')
        self.settings.add_parameter('solver/steps', 5000,  dec=True, limits=(2,None), siPrefix=True, suffix='steps')
        self.settings.add_parameter('solver/T',      0.0,  limits=(0,None), step=10,  siPrefix=True, suffix='K')
        self.settings.add_parameter('solver/continuous', True)
        
        self.settings.add_parameter('a/mode', 1, limits=(0,1), tip='0=disabled, 1=LLG')
        
        self.settings.add_parameter('a/initial_condition/x0', 1.0, tip='Initial magnetization direction (will be normalized to unit length)')
        self.settings.add_parameter('a/initial_condition/y0', 1.0, tip='Initial magnetization direction (will be normalized to unit length)')
        self.settings.add_parameter('a/initial_condition/z0', 0.0, tip='Initial magnetization direction (will be normalized to unit length)')
        
        self.settings.add_parameter('a/material/gyro',      1.760859644e11, siPrefix=True, suffix='rad/(s*T)', tip='Magnitude of gyromagnetic ratio')
        self.settings.add_parameter('a/material/M',          1.0,  siPrefix=True, suffix='T', tip='Saturation magnetization (u0*Ms)')
        self.settings.add_parameter('a/material/volume', 100*50*3, bounds=(1e-3, None), siPrefix=False, suffix=' nm^3', tip='Volume of domain (nm^3). Relevant only for thermal and STT.')
        self.settings.add_parameter('a/material/damping',      0.01, step=0.01, tip='Gilbert damping parameter')        
        
        self.settings.add_parameter('a/applied_field', True)
        self.settings.add_parameter('a/applied_field/Bx', 0.0, siPrefix=True, suffix='T', tip='Externally applied magnetic field')
        self.settings.add_parameter('a/applied_field/By', 0.0, siPrefix=True, suffix='T', tip='Externally applied magnetic field')
        self.settings.add_parameter('a/applied_field/Bz', 0.0, siPrefix=True, suffix='T', tip='Externally applied magnetic field')
        
        self.settings.add_parameter('a/other_torques', True)
        self.settings.add_parameter('a/other_torques/X', 0.0, siPrefix=True, suffix='T', tip='Exchange field parallel to domain b\'s magnetization')
        self.settings.add_parameter('a/other_torques/STT', 0.0, siPrefix=True, suffix='rad/s', tip='Spin-transfer-like torque, parallel to domain b\'s magnetization')
        
        self.settings.add_parameter('a/other_torques/Qx', 0.0, siPrefix=True, suffix='rad/s', tip='Other externally applied torque')
        self.settings.add_parameter('a/other_torques/Qy', 0.0, siPrefix=True, suffix='rad/s', tip='Other externally applied torque')
        self.settings.add_parameter('a/other_torques/Qz', 0.0, siPrefix=True, suffix='rad/s', tip='Other externally applied torque')
        
        self.settings.add_parameter('a/anisotropy', True)
        self.settings.add_parameter('a/anisotropy/Nxx', 0.01, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nyy', 0.10, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nzz', 0.89, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nxy', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nxz', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nyx', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nyz', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nzx', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('a/anisotropy/Nzy', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        
        self.settings.add_parameter('a/dipole', True)
        self.settings.add_parameter('a/dipole/Dxx', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dyy', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dzz', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dxy', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dxz', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dyx', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dyz', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dzx', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('a/dipole/Dzy', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        
        self.settings.add_parameter('b/mode', 1, limits=(0,1), tip='0=disabled, 1=LLG')
        
        self.settings.add_parameter('b/initial_condition/x0',-1.0, tip='Initial magnetization direction (will be normalized to unit length)')
        self.settings.add_parameter('b/initial_condition/y0', 1.0, tip='Initial magnetization direction (will be normalized to unit length)')
        self.settings.add_parameter('b/initial_condition/z0', 0.0, tip='Initial magnetization direction (will be normalized to unit length)')
        
        self.settings.add_parameter('b/material/gyro', 1.760859644e11, siPrefix=True, suffix='rad/(s*T)', tip='Magnitude of gyromagnetic ratio')
        self.settings.add_parameter('b/material/M',     1.0, siPrefix=True, suffix='T', tip='Saturation magnetization (u0*Ms)')
        self.settings.add_parameter('b/material/volume', 100*50*3, bounds=(1e-3, None), siPrefix=False, suffix=' nm^3', tip='Volume of domain (nm^3). Relevant only for thermal and STT.')
        self.settings.add_parameter('b/material/damping', 0.01, step=0.01, tip='Gilbert damping parameter')        
        
        self.settings.add_parameter('b/applied_field', True)
        self.settings.add_parameter('b/applied_field/Bx', 0.0, siPrefix=True, suffix='T', tip='Externally applied magnetic field')
        self.settings.add_parameter('b/applied_field/By', 0.0, siPrefix=True, suffix='T', tip='Externally applied magnetic field')
        self.settings.add_parameter('b/applied_field/Bz', 0.0, siPrefix=True, suffix='T', tip='Externally applied magnetic field')
        
        self.settings.add_parameter('b/other_torques', True)
        self.settings.add_parameter('b/other_torques/X', 0.0, siPrefix=True, suffix='T', tip='Exchange field parallel to domain b\'s magnetization')
        self.settings.add_parameter('b/other_torques/STT', 0.0, siPrefix=True, suffix='rad/s', tip='Spin-transfer-like torque, parallel to domain b\'s magnetization')
        
        self.settings.add_parameter('b/other_torques/Qx', 0.0, siPrefix=True, suffix='rad/s', tip='Other externally applied torque')
        self.settings.add_parameter('b/other_torques/Qy', 0.0, siPrefix=True, suffix='rad/s', tip='Other externally applied torque')
        self.settings.add_parameter('b/other_torques/Qz', 0.0, siPrefix=True, suffix='rad/s', tip='Other externally applied torque')
        
        self.settings.add_parameter('b/anisotropy', True)
        self.settings.add_parameter('b/anisotropy/Nxx', 0.01, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nyy', 0.10, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nzz', 0.89, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nxy', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nxz', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nyx', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nyz', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nzx', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        self.settings.add_parameter('b/anisotropy/Nzy', 0.0, tip='Anisotropy matrix (diagonal matrix has values adding to 1)')
        
        self.settings.add_parameter('b/dipole', True)
        self.settings.add_parameter('b/dipole/Dxx', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dyy', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dzz', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dxy', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dxz', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dyx', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dyz', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dzx', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        self.settings.add_parameter('b/dipole/Dzy', 0.0, tip='Dipolar field matrix exerted by domain b, expressed\nas a fraction of b\'s saturation magnetization.')
        
        # Plot tabs
        self.tabs = self.grid_bottom.add(_g.TabArea(autosettings_path='solver.tabs.txt'), alignment=0)
        
        # Inspection plot for all arrays
        self.tab_inspect  = self.tabs.add_tab('Inspect')
        self.plot_inspect = self.tab_inspect.add(_g.DataboxPlot(autoscript=6, autosettings_path='solver.plot_inspect.txt'), alignment=0)
        self.initialize_plot_inspect()
        self.plot_inspect.after_clear = self.initialize_plot_inspect
        self.plot_inspect.autoscript_custom = self._user_script
        
        # Analysis of inspection
        self.tab_process_inspect = self.tabs.add_tab('Process')
        self.process_inspect = self.tab_process_inspect.add(_g.DataboxProcessor('process_inspect', self.plot_inspect), alignment=0)
        
        self.tab_3d = self.tabs.add_tab('3D')
        self.button_3d_a   = self.tab_3d.add(_g.Button('a',   checkable=True, checked=True)) 
        self.button_3d_b   = self.tab_3d.add(_g.Button('b',   checkable=True, checked=False)) 
        self.button_3d_sum = self.tab_3d.add(_g.Button('Sum', checkable=True, checked=False)) 
        self.button_plot_3d = self.tab_3d.add(_g.Button('Update Plot'))
        self.tab_3d.new_autorow()
        
        # Make the 3D plot window
        self._widget_3d = _gl.GLViewWidget()
        self._widget_3d.opts['distance'] = 50
        
        # Make the grids
        self._gridx_3d = _gl.GLGridItem()
        self._gridx_3d.rotate(90,0,1,0)
        self._gridx_3d.translate(-10,0,0)
        self._widget_3d.addItem(self._gridx_3d)
        self._gridy_3d = _gl.GLGridItem()
        self._gridy_3d.rotate(90,1,0,0)
        self._gridy_3d.translate(0,-10,0)
        self._widget_3d.addItem(self._gridy_3d)
        self._gridz_3d = _gl.GLGridItem()
        self._gridz_3d.translate(0,0,-10)
        self._widget_3d.addItem(self._gridz_3d)
        
        # Trajectories
        color_a = _pg.glColor(100,100,255)
        color_b = _pg.glColor(255,100,100)
        color_n = _pg.glColor(50,255,255)
        self._trajectory_a_3d   = _gl.GLLinePlotItem(color=color_a, width=2.5, antialias=True)
        self._trajectory_b_3d   = _gl.GLLinePlotItem(color=color_b, width=2.5, antialias=True)
        self._trajectory_sum_3d = _gl.GLLinePlotItem(color=color_n, width=2.5, antialias=True)
        self._widget_3d.addItem(self._trajectory_a_3d)
        self._widget_3d.addItem(self._trajectory_b_3d)
        self._widget_3d.addItem(self._trajectory_sum_3d)
        
        # Other items
        self._start_dot_a_3d   = _gl.GLScatterPlotItem(color=color_a, size=7.0, pos=_n.array([[10,0,0]]))
        self._start_dot_b_3d   = _gl.GLScatterPlotItem(color=color_b, size=7.0, pos=_n.array([[-10,0,0]]))
        self._start_dot_sum_3d = _gl.GLScatterPlotItem(color=color_n, size=7.0, pos=_n.array([[-10,0,0]]))
        self._widget_3d.addItem(self._start_dot_a_3d)
        self._widget_3d.addItem(self._start_dot_b_3d)
        self._widget_3d.addItem(self._start_dot_sum_3d)
        self._update_start_dots()
        
        # Add the 3D plot window to the tab
        self.tab_3d.add(self._widget_3d, column_span=4, alignment=0)
        self.tab_3d.set_column_stretch(3)
        
        self.button_3d_a   .signal_clicked.connect(self._button_plot_3d_clicked)
        self.button_3d_b   .signal_clicked.connect(self._button_plot_3d_clicked)
        self.button_3d_sum .signal_clicked.connect(self._button_plot_3d_clicked)
        self.button_plot_3d.signal_clicked.connect(self._button_plot_3d_clicked)
         
        # Create the test GUI
        self._build_gui_test()

        # Connect the other controls
        self.button_run     .signal_clicked.connect(self._button_run_clicked)
        self.button_run_test.signal_clicked.connect(self._button_run_test_clicked)
        
        # Transfer the solver data based on the check boxes.
        self.settings.emit_signal_changed('a/applied_field')        
        self.settings.emit_signal_changed('a/other_torques')        
        self.settings.emit_signal_changed('a/anisotropy')        
        self.settings.emit_signal_changed('a/dipole')        
        self.settings.emit_signal_changed('b/applied_field')        
        self.settings.emit_signal_changed('b/other_torques')        
        self.settings.emit_signal_changed('b/anisotropy')        
        self.settings.emit_signal_changed('b/dipole')        
        
        # Let's have a look!
        self.window.show()

    def _build_gui_test(self):
        """
        Builds the test tab.
        """        
        
        # Test tab
        self.tab_test        = self.tabs.add_tab('Test')
        self.button_run_test = self.tab_test.add(_g.Button('Run Test!', checkable=True))
        self.label_test      = self.tab_test.add(_g.Label(''))
        self.label_test_info = self.tab_test.add(_g.Label(''))
        self.tab_test.new_autorow()
        
        self.settings_test  = self.tab_test.add(_g.TreeDictionary(autosettings_path='solver.settings_test.txt')) 
        self.settings_test.add_parameter('test', 
                                         ['thermal_noise',
                                          'field_sweep'], 
                                         tip='Which test to perform')
        self.settings_test.add_parameter('iterations', 10, limits=(1,None), tip='How many test iterations')
        
        self.settings_test.add_parameter('thermal_noise/bins',      100, limits=(1,None), tip='How many bins for the histogram')
        
        self.settings_test.add_parameter('field_sweep/steps',        100, limits=(1, None), dec=True)
        self.settings_test.add_parameter('field_sweep/B_start',      0.0, suffix='T',   tip='Start value.')
        self.settings_test.add_parameter('field_sweep/B_stop',       0.0, suffix='T',   tip='Stop value.')
        self.settings_test.add_parameter('field_sweep/theta_start', 90.0, suffix=' deg', tip='Start value of spherical coordinates angle from z-axis.')
        self.settings_test.add_parameter('field_sweep/theta_stop',  90.0, suffix=' deg', tip='Stop value of spherical coordinates angle from z-axis.')
        self.settings_test.add_parameter('field_sweep/phi_start',    0.0, suffix=' deg', tip='Start value of spherical coordinates angle from x-axis.')
        self.settings_test.add_parameter('field_sweep/phi_stop',     0.0, suffix=' deg', tip='Stop value of spherical coordinates angle from x-axis.')
        self.settings_test.add_parameter('field_sweep/solver_iterations', 10, limits=(1,None), dec=True, tip='How many times to push "Go!" per step.')
        
        self.plot_test   = self.tab_test.add(_g.DataboxPlot(autoscript=1, autosettings_path='solver.plot_test.txt'), alignment=0, column_span=10)
        self.tab_test.set_column_stretch(7)
        
        # Used for verifying that the continuous thermal field code is working properly.
        self._previous_a_thermal_Bx = self._previous_a_thermal_By = self._previous_a_thermal_Bz = None
        self._previous_b_thermal_Bx = self._previous_b_thermal_By = self._previous_b_thermal_Bz = None
        
        
        
        

    def _user_script(self):
        """
        Creates a "useful" default script.
        """
        return """x = None
xlabels = 'Step'
y = []
ylabels = []
for n in range(1, len(d.ckeys)):
  y      .append(d[n])
  ylabels.append(d.ckeys[n])
"""

    def initialize_plot_inspect(self):
        """
        Clears the Inspect plot and populates it with empty arrays.
        """
        self.plot_inspect.clear()
        # self.plot_inspect['t']  = []
        # self.plot_inspect['ax'] = []
        # self.plot_inspect['ay'] = []
        # self.plot_inspect['az'] = []
        # self.plot_inspect['bx'] = []
        # self.plot_inspect['by'] = []
        # self.plot_inspect['bz'] = []
        

    def _update_start_dots(self):
        """
        Gets the initial condition from the settings and updates the start
        dot positions.
        """
        if _3d_enabled:
            ax = self.settings['a/initial_condition/x0']
            ay = self.settings['a/initial_condition/y0']
            az = self.settings['a/initial_condition/z0']
            an = 1.0/_n.sqrt(ax*ax+ay*ay+az*az)
            ax = ax*an
            ay = ay*an
            az = az*an
            
            
            bx = self.settings['b/initial_condition/x0']
            by = self.settings['b/initial_condition/y0']
            bz = self.settings['b/initial_condition/z0']
            bn = 1.0/_n.sqrt(bx*bx+by*by+bz*bz)
            bx = bx*bn
            by = by*bn
            bz = bz*bn
            
            self._start_dot_a_3d  .setData(pos=10*_n.array([[ax, ay, az]]))
            self._start_dot_b_3d  .setData(pos=10*_n.array([[bx, by, bz]]))
            self._start_dot_sum_3d.setData(pos=10*_n.array([[ax+bx, ay+by, az+bz]]))

    def _button_plot_3d_clicked(self, *a):
        """
        Plot 3d button pressed: Update the plot!
        """
        if _3d_enabled:
            d = self.plot_inspect
            
            # If we have data to plot, plot it; otherwise, disable it.
            if 'ax' in d.ckeys: 
                self._trajectory_a_3d.setData(pos=10*_n.vstack([d['ax'],d['ay'],d['az']]).transpose())
            else:               
                self.button_3d_a.set_checked(False, True)
                
            if 'bx' in d.ckeys: 
                self._trajectory_b_3d.setData(pos=10*_n.vstack([d['bx'],d['by'],d['bz']]).transpose())
            else:               
                self.button_3d_b.set_checked(False, True)
            
            if self.button_3d_a.is_checked() and self.button_3d_b.is_checked():
                self._trajectory_sum_3d.setData(pos=10*_n.vstack([d['ax']+d['bx'],d['ay']+d['by'],d['az']+d['bz']]).transpose())
            else:
                self.button_3d_sum.set_checked(False, True)
            
            self._trajectory_a_3d  .setVisible(self.button_3d_a  .is_checked())
            self._trajectory_b_3d  .setVisible(self.button_3d_b  .is_checked())
            self._trajectory_sum_3d.setVisible(self.button_3d_sum.is_checked())
            
            self._start_dot_a_3d  .setVisible(self.button_3d_a  .is_checked())
            self._start_dot_b_3d  .setVisible(self.button_3d_b  .is_checked())
            self._start_dot_sum_3d.setVisible(self.button_3d_sum.is_checked())
            
            self.window.process_events()
    
    def _same_thermal_settings(self):
        """
        Checks the current thermal field settings against previous ones (if any)
        and returns true if they are the same.
        """
        # print(self._dt,       self['dt'])
        # print(self._T,        self['T'])
        # print(self._a_M,      self['a/M'])
        # print(self._a_volume, self['a/volume'])
        # print(self._a_gyro,  self['a/gyro'])
        # print(self._a_damping,  self['a/damping'])
        # print(self._b_M,      self['b/M'])
        # print(self._b_volume, self['b/volume'])
        # print(self._b_gyro,  self['b/gyro'])
        # print(self._b_damping,  self['b/damping'])
        return self._dt       == self['dt']       and \
               self._T        == self['T']        and \
               self._a_M      == self['a/M']      and \
               self._a_volume == self['a/volume'] and \
               self._a_gyro  == self['a/gyro']  and \
               self._a_damping  == self['a/damping']  and \
               self._b_M      == self['b/M']      and \
               self._b_volume == self['b/volume'] and \
               self._b_gyro  == self['b/gyro']  and \
               self._b_damping  == self['b/damping']
       

    def _button_run_clicked(self, *a):
        """
        Go button pressed: Run the simulation!
        """

        n = 0
        N = self['iterations']
        while (n < N or N < 1) and self.button_run.is_checked():
            
            # End time of previous run
            old_t3 = self._t3
            
            # Update the user and run it.
            self.run()
            
            # Provide some user information
            self.label_iteration.set_text(
                'Iteration ' + str(n+1) + 
                ': Duty Cycle = %.2f, %.3f seconds/iteration' % 
                ( (self._t2-self._t05)/(self._t3-old_t3), self._t3-old_t3 ))
            n += 1
        
        self.button_run.set_checked(False)

    def _button_run_test_clicked(self, *a):
        """
        Go button pressed: Run the simulation!
        """

        n = 1
        while n <= self.settings_test['iterations'] \
        or self.settings_test['iterations'] < 1 \
          and self.button_run_test.is_checked():
            
            # Clear the test plot
            self.plot_test.clear()
            
            # Send the settings in as header information
            self.settings     .send_to_databox_header(self.plot_test)
            self.settings_test.send_to_databox_header(self.plot_test)
    
            # Send iteration number to the header
            self.plot_test.h(n = n)
            
            # Autosave if checked
            self.plot_test.autosave()
            
            # Update the gui iteration number
            self.label_test.set_text('Iteration ' + str(n))

            # RUN THE TEST!
            exec('self.test_'+self.settings_test['test']+'()')
            
            n += 1
            
        self.button_run_test.set_checked(False)

    def run(self):
        """
        Run the specified simulation.
        
        IMPORTANT: See self.api.run()'s documentation for a concern about 
        setting continuous=True.
        
        Returns
        -------
        self
        """
        self._t0 = _time.time()
        
        # Clear the api arrays, and transfer all the values from the 
        # TreeDictionary to the api (API). This skips non-api entries 
        # like 'solver', 'a', 'b', and 'solver/T', and will transfer arrays
        # like 'a/Bx' if they exist in the plot_inspect.
        self.api.a.clear_arrays()
        self.api.b.clear_arrays()
        self._transfer_all_to_api()
        
        self._t05 = _time.time()
        
        # If we have a Langevin field, calculate and add it in.
        thermal_enabled = self['T'] > 0
        if thermal_enabled:
            
            # If domain a is enabled
            if self['a/mode']:
                
                # Generate new langevin field arrays
                self.a_thermal_Bx = self.thermal_field('a')
                self.a_thermal_By = self.thermal_field('a')
                self.a_thermal_Bz = self.thermal_field('a')
                
                # If thermal field-relevant parameters are the same
                # and we're doing a continuous simulation, and we have
                # previous thermal number, set it!
                if  self._same_thermal_settings() \
                and self['continuous'] \
                and self._a_thermal_Bx_last is not None:
                    
                    self.a_thermal_Bx[0] = self._a_thermal_Bx_last
                    self.a_thermal_By[0] = self._a_thermal_By_last
                    self.a_thermal_Bz[0] = self._a_thermal_Bz_last
                
                # Now add the thermal field to the existing field.
                self.api['a/Bx'] += self.a_thermal_Bx
                self.api['a/By'] += self.a_thermal_By
                self.api['a/Bz'] += self.a_thermal_Bz
            
            # If domain b is enabled
            if self['b/mode']:
                
                # Generate new langevin field arrays
                self.b_thermal_Bx = self.thermal_field('b')
                self.b_thermal_By = self.thermal_field('b')
                self.b_thermal_Bz = self.thermal_field('b')
                
                # If thermal field-relevant parameters are the same
                # and we're doing a continuous simulation, and we have
                # previous thermal number, set it!
                if  self._same_thermal_settings() \
                and self['continuous'] \
                and self._b_thermal_Bx_last is not None:
                    
                    self.b_thermal_Bx[0] = self._b_thermal_Bx_last
                    self.b_thermal_By[0] = self._b_thermal_By_last
                    self.b_thermal_Bz[0] = self._b_thermal_Bz_last
                
                # Now add the thermal field to the existing field.
                self.api['b/Bx'] += self.b_thermal_Bx
                self.api['b/By'] += self.b_thermal_By
                self.api['b/Bz'] += self.b_thermal_Bz
        
        self._t1 = _time.time()
        
        # Run it.
        self.api.run(self['continuous'])
        
        self._t2 = _time.time()
        
        # If we're not resetting (continuous mode), keep track of the last
        # thermal field component. 
        if thermal_enabled:
            if self['a/mode']: # IF the layer is enabled.
                self._a_thermal_Bx_last = self.a_thermal_Bx[-1]
                self._a_thermal_By_last = self.a_thermal_By[-1]
                self._a_thermal_Bz_last = self.a_thermal_Bz[-1]

            if self['b/mode']:
                self._b_thermal_Bx_last = self.b_thermal_Bx[-1]
                self._b_thermal_By_last = self.b_thermal_By[-1]
                self._b_thermal_Bz_last = self.b_thermal_Bz[-1]
                
        # Otherwise, None those things out so we know not to mess with them
        else:
            self._a_thermal_Bx_last = None
            self._a_thermal_By_last = None
            self._a_thermal_Bz_last = None
            self._b_thermal_Bx_last = None
            self._b_thermal_By_last = None
            self._b_thermal_Bz_last = None
        
        # Transfer to the initial condition from the solver to the tree
        if(self['a/mode']):
            self.settings['a/initial_condition/x0'] = self.api.a.x0
            self.settings['a/initial_condition/y0'] = self.api.a.y0
            self.settings['a/initial_condition/z0'] = self.api.a.z0
        
        if(self['b/mode']):
            self.settings['b/initial_condition/x0'] = self.api.b.x0
            self.settings['b/initial_condition/y0'] = self.api.b.y0
            self.settings['b/initial_condition/z0'] = self.api.b.z0
        
        # Transfer the results to the inspector
        self.plot_inspect['t']  = self.api.dt*_n.array(range(self.api.steps))
        
        if self['a/mode']:
            self.plot_inspect['ax'] = self.api.ax
            self.plot_inspect['ay'] = self.api.ay
            self.plot_inspect['az'] = self.api.az
        else:
            self.plot_inspect.pop_column('ax', True)
            self.plot_inspect.pop_column('ay', True)
            self.plot_inspect.pop_column('az', True)
        
        if self['b/mode']:
            self.plot_inspect['bx'] = self.api.bx
            self.plot_inspect['by'] = self.api.by
            self.plot_inspect['bz'] = self.api.bz
        else:
            self.plot_inspect.pop_column('bx', True)
            self.plot_inspect.pop_column('by', True)
            self.plot_inspect.pop_column('bz', True)
            
        # Update the plot and analyze the result
        self.plot_inspect.plot()
        self.process_inspect.run()
        if _3d_enabled: self._button_plot_3d_clicked()
            
        self.window.process_events()
        
        self._t3 = _time.time()
        
        # Store this run's thermal settings for comparing with the next run
        self._dt       = self['dt']
        self._T        = self['T']
        self._a_M      = self['a/M']
        self._a_volume = self['a/volume']
        self._a_gyro  = self['a/gyro']
        self._a_damping  = self['a/damping']
        self._b_M      = self['b/M']
        self._b_volume = self['b/volume']
        self._b_gyro  = self['b/gyro']
        self._b_damping  = self['b/damping']
        
        return self

    def _api_key_exists(self, key):
        """
        See if the key (from the TreeDictionary only!) exists in the API.
        """
        s = key.split('/')
        
        # Single-element splits implie categories (solver, a, b)
        if len(s) == 1: return False
        
        # Solver parameters
        if s[-1] in self.api._solver_keys: return True
        
        # Otherwise, see if it's in a (b has the same structure, so this works)
        return s[-1] in self.api.a.keys()
        

    def _transfer(self, key):
        """
        Transfers the domain's parameter to the solver data.
        """
        
        # Ignore some keys that aren't in the api (defined above)
        if not self._api_key_exists(key): return        
        
        # s[0] is the domain, s[-1] is the parameter name
        s = key.split('/')

        # Array values from the plotter, normal values from dictionary
        short_key = s[0]+'/'+s[-1]
        if short_key in self.plot_inspect.ckeys: 
              value = self.plot_inspect[short_key]    
        else: value = self.settings[key]
        
        # If it's under an unchecked category, zero it.
        if s[1] in ['applied_field', 'other_torques', 'anisotropy', 'dipole']:
            
            # If this is the category itself, break.
            if len(s) == 2: return
            
            # Otherwise it's a parameter. Zero the value if it's in an 
            # Unchecked category.
            elif not self.settings[s[0]+'/'+s[1]]: value = 0
                
        # Come up with the command for sending the parameter to the api
        if s[0]=='solver': command =            'self.api.set_multiple('+s[-1]+'= value)'
        else:              command = 'self.api.'+s[0]+  '.set_multiple('+s[-1]+'= value)'   
        
        # Try it!
        try:    exec(command, dict(self=self, value=value))
        except: print('FAIL: "'+command+'"')
    
    def _transfer_all_to_api(self):
        """
        Loops over the settings keys and transfers them to the api.
        """
        for key in self.settings.keys(): self._transfer(key)
        
        # Update the start dots in the 3D plot
        self._update_start_dots()

    def _elongate_domain_key(self, key):
        """
        Returns a key that is the long form (inserting any sub-headings) to 
        make it work on settings. Assumes it's of the form 'a/something' or
        'b/something'
        """
        split = key.split('/')
        
        for k in self.settings.keys():
            s = k.split('/')
            if s[0] == split[0] and s[-1] == split[-1]: return k
        
        print('UH OH, could not elongate "'+key+'"')
        return key
        
    def _get_root(self, key):
        """
        Returns the first root found for the specified short key, or None
        """
        for k in self.settings.keys():
            s = k.split('/')
            if s[-1] == key: return s[0]
            
        return None    
    
    def set(self, key, value):
        """
        Sets self.settings[key] = value, unless value is an array. In that case
        it sets self.plot_inspect[key] = value. 
        
        You can also skip the sub-heading, so self.set('a/x0',0.5) will work 
        the same as self.set('a/initial_condition/x0', 0.5)

        Also, if you skip the root, it will assume either 'solver' or 'a' by
        default, so 'Qx' is the same as 'a/Qx'.

        Parameters
        ----------
        key : string
            Parameter key to set.
        value: string
            Parameter value to set.
            
        Returns
        -------
        self

        """
        s = key.split('/')
        
        # If we're using a shortcut key, like 'dt'
        if len(s) == 1 and s[0] not in ['solver', 'a', 'b']:
            
            # Insert the first root we find, if it's simple.
            s.insert(0,self._get_root(s[0]))
            key = '/'.join(s)
        
        # By this stage it better be length 2 or we're hosed.
        if len(s) < 2: 
            print('ERROR: Cannot set', key)
            return
        
        # Arrays go to the inspect plotter, values go to settings
        if type(value) == _n.ndarray: 
            s = key.split('/')
            self.plot_inspect[s[0]+'/'+s[-1]] = value
        
        # Otherwise it's just a value. Update the tree.
        else:                 
            self.settings[self._elongate_domain_key(key)] = value
        
        # Update plot
        self.plot_inspect.plot()
        self.window.process_events()
        
        return self
    
    __setitem__ = set    
    
    def set_multiple(self, **kwargs):
        """
        Sends all keyword arguments to self.set().
        """
        for k in kwargs: self[k] = kwargs[k]
    
    __call__ = set_multiple
    
    def get(self, key):
        """
        Returns the value (or array if available) for the specified key. Key
        can be "short", not including the sub-heading, e.g. 'a/x0' is the same
        as 'a/initial_condition/x0'. You can also specify any of the Inspect
        plot's columns to get the data (i.e., any of self.plot_inspect.ckeys).

        Parameters
        ----------
        key : string
            Key of parameter to retrieve.

        Returns
        -------
        value of parameter or array if available.

        """
        # For solver settings, return them with simple keys
        if key in ['dt', 'steps', 'continuous', 'T', 'iterations']: return self.settings['solver/'+key]
        
        # If the key is right there in the plot_inspect, return that
        if key in self.plot_inspect.ckeys: return self.plot_inspect[key]
        
        # s[0] is the domain, s[-1] is the parameter name
        s = key.split('/')

        # Don't do anything for the roots
        if len(s) < 2: 
            print('WHOOPS. WHOOPS. "'+key+'" is invalid or not specific enough (forgot the domain?)')
            return
        
        # Array value
        short_key = s[0]+'/'+s[-1]
        if short_key in self.plot_inspect.ckeys: return self.plot_inspect[short_key]
        else: return self.settings[self._elongate_domain_key(key)]
    
    __getitem__ = get
    
    def ns(self, n=0):
        """
        Returns an array of integer indices for the n'th simulation iteration.
        
        Specifically, it returns 
        
        _n.array(range(self['steps'])) + n*(self.settings['steps']-1)
          
        The '-1' in the above is because the last step of the previous iteration
        is used to initialize the first step of the following iteration.
          
        Parameters
        ----------
        n=0 [integer]
            Which simulation iteration this corresponds to.
        """
        return _n.array(range(self.settings['solver/steps'])) \
               + n*(self.settings['solver/steps']-1)

    def ts(self, n=0):
        """
        Returns a time array for the n'th iteration as per the solver settings.
        
        Specifically, it returns self.ns()*self.settings['solver/dt']
        
        Parameters
        ----------
        n=0 [integer]
            Which iteration of the simulation
        """
        return self.ns(n)*self.settings['solver/dt']

    def zeros(self):
        """
        Returns a self['steps']-length array of zeros.
        """
        return _n.zeros(self['steps'])

    def ones(self):
        """
        Returns a self['steps']-length array of ones.
        """
        return _n.ones(self['steps'])

    def pulse(self, n0, n1):
        """
        Returns an array of length self['steps'] that is zero everywhere except
        from n0 to n1-1.
        """
        z = self.zeros()
        z[n0:n1] = 1.0
        return z
    
    def sin(self, frequency_GHz=1.0, phase_rad=0.0, n=0):
        """
        Returns an appropriately sized array of sinusoidal oscillations at
        frequency f_GHz [GHz] with phase p_rad [radians]. The integer n adds
        the appropriate phase for the n'th iteration of the simulation, allowing
        you to generate a continuous sinusoid over many iterations.
        
        Specifically, the returned array is 
        sin(2*pi*frequency_GHz*self.ts(n) + phase_rad)
        
        Parameters
        ----------
        frequency_GHz [GHz]
            Frequency of oscillation in GHz.
        phase_rad [radians]
            Phase in radians.
        n=0 [integer]
            Which iteration this is. You can use this integer to make a 
            continuous sinusoid over multiple iterations of the solver. 
        """
        return _n.sin(2*_n.pi*frequency_GHz*1e9*self.ts(n)+phase_rad)
    
    def cos(self, frequency_GHz=1.0, phase_rad=0.0, n=0):
        """
        Returns an appropriately sized array of sinusoidal oscillations at
        frequency f_GHz [GHz] with phase p_rad [radians]. The integer n adds
        the appropriate phase for the n'th iteration of the simulation, allowing
        you to generate a continuous sinusoid over many iterations.
        
        Specifically, the returned array is 
        cos(2*pi*frequency_GHz*self.ts(n) + phase_rad)
        
        Parameters
        ----------
        f_GHz=1.0 [GHz]
            Frequency of oscillation in GHz.
        p_rad=0.0 [radians]
            Phase in radians.
        n=0 [integer]
            Which iteration this is. You can use this integer to make a 
            continuous sinusoid over multiple iterations of the solver. 
        """
        return _n.cos(2*_n.pi*frequency_GHz*1e9*self.ts(n)+phase_rad)
            
    def spin_torque_per_mA(self, domain='a', efficiency=1.0, volume_nm3=100*100*10):
        """
        Returns the torque per milliamp [rad/(s*mA)] that would be applied when
        spin polarization is perpendicular to a magnetization 
        self[domain+'/M'] [T] filling the specified volume_nm3 [nm^3]. This can 
        basically be used to calculate the prefactor on the a x (a x s) term in
        the LLG equation, where a is the domain unit vector, and s is the 
        spin polarization unit vector, which, for this simulation has units of
        [rad/s].
        
        Parameters
        ----------
        domain='a'
            Which domain receives the spin transfer torque. The only domain-
            specific parameters used are gyro and M.
            
        efficiency=1.0 [unitless]
            How electron spins are deposited on average per passing electron.
            
        volume_nm3=100*100*10 [nm^3]
            Total volume of the magnetic domain in cubic nanometers.
        
        Returns
        -------
        The torque per mA [rad/(s*mA)] applied to a unit vector.
        """
        return efficiency*self[domain+'/gyro']*hbar*1e-3 / \
               (2*ec*(self[domain+'/M']/u0)*volume_nm3*1e-27)
    
    def test_thermal_noise(self):
        """
        Runs the simulation in continuous mode, binning the results by
        energy and comparing to a Boltzmann distribution in the Test tab.

        Returns
        -------
        self
        """

        # Make sure the temperature is not zero (otherwise we're hosed)
        if self['T'] == 0: self['T'] = 293; # Why not.
    
        # Make sure it's in continuous mode, with one iteration
        self['continuous'] = True
        self['iterations'] = 1
                
        # Set up the plot style
        self.plot_test.styles = [dict(pen=(0,2)),
                                 dict(symbol='+', symbolPen='w', pen=None)]
        
        ##################
        # RUN SIMULATION #
        ##################
                
        self.button_run.click()
        
        ###############################
        # MAGNETIC ENERGY CALCULATION #
        ###############################
        
        # Calculate the magnetic energy for each point.
        if self['a/mode']:
            
            if self['ax'] is None: 
                print('ERROR: No data from simulation?')
                return
            
            # Formula is 0.5*Ms*V*B*cos(theta), so calculate some constants
            # to keep the code looking reasonably clean
            
            # Magnetic volume
            MsV = (self['a/M']/u0) * (self['a/volume']*1e-27) 
            
            # Applied field
            B0x = self['a/Bx']
            B0y = self['a/By']
            B0z = self['a/Bz']
            
            # Anisotropy
            BAx = -self['a/M'] * (self['a/Nxx']*self['ax'] + self['a/Nxy']*self['ay'] + self['a/Nxz']*self['az'])
            BAy = -self['a/M'] * (self['a/Nyy']*self['ay'] + self['a/Nyz']*self['az'] + self['a/Nyx']*self['ax'])
            BAz = -self['a/M'] * (self['a/Nyz']*self['az'] + self['a/Nzx']*self['ax'] + self['a/Nzy']*self['ay'])
            
            # Get the other domain's vector
            if self['b/mode']:
                bx = self['bx']
                by = self['by']
                bz = self['bz']
            else:
                bx = self['b/x0']
                by = self['b/y0']
                bz = self['b/z0']
                norm = (bx*bx+by*by+bz*bz)**(-0.5)
                bx *= norm
                by *= norm
                bz *= norm
                
            # Dipole
            BDx = -self['b/M'] * (self['a/Dxx']*bx + self['a/Dxy']*by + self['a/Dxz']*bz)
            BDy = -self['b/M'] * (self['a/Dyy']*by + self['a/Dyz']*bz + self['a/Dyx']*bx)
            BDz = -self['b/M'] * (self['a/Dyz']*bz + self['a/Dzx']*bx + self['a/Dzy']*by)
        
            # Exchange
            BXx = self['a/X'] * bx
            BXy = self['a/X'] * by
            BXz = self['a/X'] * bz
            
            # Total field
            Bx = B0x + BAx + BDx + BXx
            By = B0y + BAy + BDy + BXy
            Bz = B0z + BAz + BDz + BXz
            
            # Energy!
            self.plot_inspect['Ua'] = -0.5*MsV*(self['ax']*Bx + self['ay']*By + self['az']*Bz)
            
            # Update the plot and wait for events
            self.plot_inspect.plot()
            self.window.process_events()
        
        # Bin everything from this run.
        Na, bins = _n.histogram(self.plot_inspect['Ua'], self.settings_test['thermal_noise/bins'])
            
        # Get x-axis from the midpoints of the bin edges
        Ta = 0.5*(bins[1:]+bins[0:len(bins)-1])/kB
        Ta = Ta-Ta[0]
        
        # Theory
        Ba = _n.exp(-Ta/self['T'])
        
        # Populate the test tab
        self.settings.send_to_databox_header(self.plot_test)
        
        self.plot_test['Ua_K'] = Ta
        self.plot_test['Ba']   = Ba / _n.sum(Ba)
        self.plot_test['Pa']   = Na / _n.sum(Na)
        
        self.plot_test.plot()
        self.window.process_events()
                
        
        ##################################
        # CONTINUOUS THERMAL FIELD CHECK #
        ##################################
            
        # Test that the previous last thermal field matches the first
        # value of the new array, but only do this after the first iteration.
        if self['a/mode']:
            
            if self.plot_test.h('n') > 0:
            
                if self._previous_a_thermal_Bx != self.a_thermal_Bx[0] \
                or self._previous_a_thermal_By != self.a_thermal_By[0] \
                or self._previous_a_thermal_Bz != self.a_thermal_Bz[0]:
                    print('ERROR: Thermal field not continuous for domain a.', self._previous_a_thermal_Bx, self.a_thermal_Bx[0])
        
            # Remember the previous value for the next iteration
            self._previous_a_thermal_Bx = self.a_thermal_Bx[-1]
            self._previous_a_thermal_By = self.a_thermal_By[-1]
            self._previous_a_thermal_Bz = self.a_thermal_Bz[-1]
            
        if self['b/mode']:
            
            if self.plot_test.h('n') > 0:
                
                if self._previous_b_thermal_Bx != self.b_thermal_Bx[0] \
                or self._previous_b_thermal_By != self.b_thermal_By[0] \
                or self._previous_b_thermal_Bz != self.b_thermal_Bz[0]:
                    print('ERROR: Thermal field not continuous for domain b.')
                
            # Remember the last value for the next check
            self._previous_b_thermal_Bx = self.b_thermal_Bx[-1]
            self._previous_b_thermal_By = self.b_thermal_By[-1]
            self._previous_b_thermal_Bz = self.b_thermal_Bz[-1]
    
        return Ta, Na          
    
    def test_field_sweep(self):
        """
        Runs a stepped sweep between the specified start and end conditions.
        """
        
        # Get the arrays of values
        Bs     = _n.linspace(self.settings_test['field_sweep/B_start'],
                             self.settings_test['field_sweep/B_stop'],
                             self.settings_test['field_sweep/steps'])
        
        thetas = _n.linspace(self.settings_test['field_sweep/theta_start'],
                             self.settings_test['field_sweep/theta_stop'],
                             self.settings_test['field_sweep/steps'])
        
        phis   = _n.linspace(self.settings_test['field_sweep/phi_start'],
                             self.settings_test['field_sweep/phi_stop'],
                             self.settings_test['field_sweep/steps'])
        
        # Loop over the number of steps
        n = 0
        while n < len(Bs) and self.button_run_test.is_checked():
        
            # Get the applied fields
            Bx = Bs[n]*_n.sin(thetas[n]*_n.pi/180)*_n.cos(phis[n]*_n.pi/180)
            By = Bs[n]*_n.sin(thetas[n]*_n.pi/180)*_n.sin(phis[n]*_n.pi/180)
            Bz = Bs[n]*_n.cos(thetas[n]*_n.pi/180)
            
            print(n, Bx, By, Bz)    
        
            # Set it for both domains
            self['a/Bx'] = Bx
            self['a/By'] = By
            self['a/Bz'] = Bz
            self['b/Bx'] = Bx
            self['b/By'] = By
            self['b/Bz'] = Bz
            
            # Update the header
            self.plot_test.h(
                Bx = Bx,
                By = By,
                Bz = Bz,
                field_sweep_step = n
            )
            
            # Set the number of solver iterations
            self['iterations'] = 1
            
            # Loop over the number of solver iterations, pushing go and plotting
            m = 0
            while m < self.settings_test['field_sweep/solver_iterations'] \
              and self.button_run_test.is_checked():
                
                # Push Go!
                self.button_run.click()
                
                # Update the header
                self.plot_test.h(field_sweep_solver_iteration = m)
                
                # Get the new data point
                new_data  = [n, m, Bs[n], thetas[n], phis[n], Bx, By, Bz]
                new_ckeys = ['step', 'iteration', 'B', 'theta', 'phi', 'Bx', 'By', 'Bz']
                
                # Domain data
                if self['a/mode']:
                    new_data  = new_data  + [_n.mean(self['ax']), _n.mean(self['ay']), _n.mean(self['az'])]
                    new_ckeys = new_ckeys + ['ax', 'ay', 'az']
                if self['b/mode']:
                    new_data  = new_data  + [_n.mean(self['bx']), _n.mean(self['by']), _n.mean(self['bz'])]
                    new_ckeys = new_ckeys + ['bx', 'by', 'bz']
                
                # Now record the average values as a new data point
                self.plot_test.append_data_point(new_data, new_ckeys)
                
                # Update the plot
                self.plot_test.plot()
                self.window.process_events()
 
                # Increment the solver iteration
                m += 1
            
 
            # Increment the step
            n += 1
        

                
    

if __name__ == '__main__':
    
    import macrospinmob
    runfile(macrospinmob.__path__[0] + '/_tests/test_everything.py')
    
    
    
