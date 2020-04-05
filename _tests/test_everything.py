# -*- coding: utf-8 -*-
"""
Module for testing _data.py
"""
import os                # For loading fixtures
import numpy        as np
import pylab
import spinmob      as sm
import macrospinmob as ms

import unittest as _ut

a = b = c = d = api = None

path_log = os.getcwd() + '/engine.log'

class Test_everything(_ut.TestCase):
    """
    Test class for databox.
    """

    def test_domain(self):
        """
        Plays with the _domain() object.
        """
        global a
        
        # Create a domain
        a = ms._domain()
        
        # Set something
        a.set('T', 300)
        self.assertEqual(a.T, 300.0)
        
        a['V'] = 2e-23
        self.assertEqual(a.V, 2e-23)
        
        # Set several things
        a.set_multiple(T=377, V=1e-27)
        self.assertEqual(a['T'], 377.0)
        self.assertEqual(a['V'], 1e-27)
        
        return

    def test_solver_api(self):
        """
        Basic playtime with solver api.
        """
        global api
        
        # Create an api instance
        api = ms.solver_api()
        
        # Various ways to set things
        api.set('a/T', 3)
        self.assertEqual(api['a/T'], 3)
        self.assertEqual(api['b/T'], 0)
        
        api['T'] = 327
        self.assertEqual(api['a/T'], 327)
        self.assertEqual(api['b/T'], 327)
        
        api.set_multiple(T=27, V=32)
        self.assertEqual(api['a/T'], 27)
        self.assertEqual(api['b/V'], 32)
        
        # Array
        api['V'] = np.linspace(1,10,10)
        self.assertTrue((api['a/V'] == np.linspace(1,10,10)).all())
        self.assertTrue((api['b/V'] == np.linspace(1,10,10)).all())
        api['V'] = 1000

        # Solver Parameters
        api['steps'] = 10
        self.assertEqual(api['steps'], 10)
        api['dt']    = 0.5e-12
        self.assertEqual(api['dt'], 0.5e-12)
        
        # Remove the log file if it exists
        if os.path.exists(path_log): os.remove(path_log)
        
        # Run it
        api['T']      = 300
        api['steps']  = 5
        api.log_level = 4
        api.run()
        api.log_level = 0

        # Print the log file
        if os.path.exists(path_log):        
            f = open(path_log)
            print(f.read())
            f.close()
        
        # Plot the trajectory
        sm.pylab.figure(1)
        sm.plot.xy.data(None, [api.a.x, api.b.x])
        
        # Now, without the log, do a LOT more steps and check the Langevin field
        api['steps'] = 1e4
        api.run()
        
        # Histogram the Langevin fields
        sm.pylab.figure(2)
        sm.pylab.hist(api.a.Lx, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Ly, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Lz, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Lx, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Ly, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Lz, bins=100, alpha=0.1)
        sm.pylab.title('Langevin RMS = ' + str(
            np.sqrt(4*api['a/damping']*1.4e-23*api['a/T']*1.25663706e-6/api['a/gyro']/api['a/M']/api['a/V']/api['dt'])
            ) + '\nstd(a.Lx) = ' + str(np.std(api.a.Lx)))
        
        
        # Remove the log file if it exists
        if os.path.exists(path_log): os.remove(path_log)
        
        
        
if __name__ == "__main__": _ut.main()
