test_domain = True
test_api    = True
test_solver = True

import os                
import shutil
import numpy        as np
import spinmob      as sm
import macrospinmob as ms; ms.debug_enabled = True
import unittest     as ut


a = api = s = None
path_log = os.getcwd() + '/engine.log'

class Test_everything(ut.TestCase):
    """
    Test class for solver and api.
    """
    def run(self, result=None):
        """ Stop after first error """
        if not result.errors:
            super(Test_everything, self).run(result)

    def test_domain(self):
        """
        Plays with the _domain() object.
        """
        if not test_domain: 
            print('\nNot running test_domain()')
            return
        
        global a
        
        # Create a domain
        a = ms._domain()
        
        # Set something
        a.set('T', 300)
        self.assertEqual(a.T, 300.0)
        
        a['V'] = 2e-23
        self.assertEqual(a.V, 2e-23)
        
        # Set several things
        a.set_multiple(T=377, V=1e-27, enable_D=True)
        self.assertEqual(a['T'], 377.0)
        self.assertEqual(a['V'], 1e-27)
        self.assertEqual(a['enable_D'], True)
        return

    def test_solver_api(self):
        """
        Basic playtime with solver api.
        """
        if not test_api:
            print('\nNot running test_api()')
            return
        
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
        api['V'] = 1000e-27

        # Solver Parameters
        api['steps'] = 10
        self.assertEqual(api['steps'], 10)
        api['dt']    = 0.5e-12
        self.assertEqual(api['dt'], 0.5e-12)
        
        
        
        # Now do a short run to generate and view a log
        # for inspection.
        
        # Remove the log file if it exists
        if os.path.exists(path_log): os.remove(path_log)
        
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
        sm.pylab.subplot(221)
        sm.pylab.plot([0,1,2,3,4], api.a.x)
        sm.pylab.plot([0,1,2,3,4], api.b.x)
        
        
        
        
        # Now test that continuous works
        api.reset()
        api.run()
        
        # Initial values should match x0, y0, etc
        self.assertEqual(api.a.x[0], api.a.x0)
        self.assertEqual(api.a.y[0], api.a.y0)
        self.assertEqual(api.a.z[0], api.a.z0)
        self.assertEqual(api.b.x[0], api.b.x0)
        self.assertEqual(api.b.y[0], api.b.y0)
        self.assertEqual(api.b.z[0], api.b.z0)
        
        # Remember the last values
        last_ax = api.a.x[-1]; last_aLx = api.a.Lx[-1] 
        last_ay = api.a.y[-1]; last_aLy = api.a.Ly[-1]
        last_az = api.a.z[-1]; last_aLz = api.a.Lz[-1]
        
        last_bx = api.b.x[-1]; last_bLx = api.b.Lx[-1]
        last_by = api.b.y[-1]; last_bLy = api.b.Ly[-1]
        last_bz = api.b.z[-1]; last_bLz = api.b.Lz[-1]
        
        # Run it again
        api.continuous = True
        api.run()
        
        # Almost equal because of the initial normalization step in engine
        self.assertAlmostEqual(last_ax,  api.a.x[0]) 
        self.assertAlmostEqual(last_aLx, api.a.Lx[0])
        self.assertAlmostEqual(last_ay,  api.a.y[0])
        self.assertAlmostEqual(last_aLy, api.a.Ly[0])
        self.assertAlmostEqual(last_az,  api.a.z[0])
        self.assertAlmostEqual(last_aLz, api.a.Lz[0])
        
        self.assertAlmostEqual(last_bx,  api.b.x[0])
        self.assertAlmostEqual(last_bLx, api.b.Lx[0])
        self.assertAlmostEqual(last_by,  api.b.y[0])
        self.assertAlmostEqual(last_bLy, api.b.Ly[0])
        self.assertAlmostEqual(last_bz,  api.b.z[0])
        self.assertAlmostEqual(last_bLz, api.b.Lz[0])
        
        # Quick check that it initializes when continuous = False
        last_ax = api.a.x[-1]; last_aLx = api.a.Lx[-1] 
        api.continuous = False
        api.run()
        self.assertEqual(api.a.x[0], api.a.x0)
        
        # Now, without the log, do a LOT more steps and check the Langevin field
        api['steps'] = 1e4
        api.run()
        
        # Histogram the Langevin fields
        sm.pylab.subplot(222)
        sm.pylab.hist(api.a.Lx, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Ly, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Lz, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Lx, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Ly, bins=100, alpha=0.1)
        sm.pylab.hist(api.a.Lz, bins=100, alpha=0.1)
        sm.pylab.title('Langevin RMS = ' + str(
            np.sqrt(4*api['a/damping']*1.4e-23*api['a/T']*1.25663706e-6/api['a/gyro']/api['a/M']/api['a/V']/api['dt'])
            ) + '\nstd(a.Lx) = ' + str(np.std(api.a.Lx)))
        
        # CLEANUP Remove the log file if it exists
        if os.path.exists(path_log): os.remove(path_log)
    
    def test_solver(self):
        """
        Test the main solver object.
        """
        if not test_solver:
            print('\nNot running test_solver()')
            return
        
        global s
        
        # Remove any egg_settings from previous runs
        if os.path.exists('egg_settings'): shutil.rmtree('egg_settings')
        
        # Create an instance
        s = ms.solver()
        
        # First set a few things
        s['T'] = 270
        self.assertEqual(s['a/T'], 270)
        self.assertEqual(s['b/T'], 270)
        
        s['y0'] = 0.1
        self.assertEqual(s['a/y0'], 0.1)
        self.assertEqual(s['b/y0'], 0.1)
        
        s['V'] = 1000
        self.assertEqual(s['a/V'], 1000)
        self.assertEqual(s['b/V'], 1000)
        
        s['a/X'] = 0.777
        self.assertEqual(s['a/X'], 0.777)
        self.assertNotEqual(s['b/X'], 0.777)
        
        s['dt'] = 170e-15
        s['steps'] = 1000
        s['continuous'] = True
        s['a/enable'] = False
        s['iterations'] = 1
        
        # Now make sure the _transfer() process works, especially for ones 
        # having different names or units.
        s.transfer_to_api()
        self.assertEqual(s.api.a['T'], 270)
        self.assertEqual(s.api.b['T'], 270)
        self.assertEqual(s.api['a/X'], 0.777)
        self.assertNotEqual(s.api['b/X'], 0.777)
        self.assertEqual(s.api['dt'], 170e-15)
        
        # Let's take a look at that thermal field
        for n in range(10):
            s.button_run.click()
            s.plot_inspect['b/Lx'] = s.b.Lx  
            s.plot_inspect.plot()
        
        # Now clear out the columns and let's test energy conservation
        s.plot_inspect.clear()
        s['a/enable'] = True
        s['b/enable'] = False
        
        s['a/X']       = -0.1
        s['a/Bx']      = 0.3
        s['a/damping'] = 0
        s['a/T']       = 0
        s['a/y0']      = 1
        s['a/Dyy']     = 1
        
        s['continuous'] = False
        s['get_energy'] = True
        
        s.run()
        
        self.assertAlmostEqual(s['Ua'][0], s['Ua'][-1], 4)
        
        # Same for other domain
        # Now clear out the columns and let's test energy conservation
        s.plot_inspect.clear()
        s['a/enable'] = False
        s['b/enable'] = True
        
        s['b/X']       = -0.1
        s['b/Bx']      = 0.3
        s['b/damping'] = 0
        s['b/T']       = 0
        s['b/y0']      = 1
        s['b/Dyy']     = 1
        
        s['continuous'] = False
        s['get_energy'] = True
        
        s.run()
        
        self.assertAlmostEqual(s['Ub'][0], s['Ub'][-1], 4)
        
        
        # End with reasonable config for playtime.
        s['a/enable'] = True
        s['b/enable'] = False
        s['damping']  = 0.01
        s['T']        = 300
        s['steps']    = 2e5
        
        # CLEANUP: Remove egg_settings for next time
        if os.path.exists('egg_settings'): shutil.rmtree('egg_settings')
        
        
        
if __name__ == "__main__": ut.main()
