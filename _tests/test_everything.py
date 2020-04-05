# -*- coding: utf-8 -*-
"""
Module for testing _data.py
"""
import os                # For loading fixtures
import numpy        as np
import spinmob      as sm
import macrospinmob as ms

import unittest as _ut

a = b = c = d = api = None

class Test_everything(_ut.TestCase):
    """
    Test class for databox.
    """

    def setUp(self):
        """
        """
        return 
    
    def tearDown(self):
        """
        """
        return

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

        # Solver Parameters
        api['steps'] = 400
        self.assertEqual(api['steps'], 400)
        api['dt']    = 100e-15
        self.assertEqual(api['dt'], 100e-15)
        
        # Solve something!
        api.run()
        f = open(ms.__path__[0] + '/engine.log')
        print(f.read())
        f.close()
        
        
if __name__ == "__main__": _ut.main()
