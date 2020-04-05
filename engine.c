/**
 * This file is part of the Macrospyn distribution 
 * (https://github.com/Spinmob/macrospyn).
 * Copyright (c) 2002-2020 Jack Childress (Sankey).
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.

 * Building on Linux requires gcc.
 * Building on Windows is easiest with the dev-C++ IDE.
**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define kB 1.380649e-23     // Boltzmann constant [J/K]
#define u0 1.25663706212e-6 // Vacuum permeability [H/m | N/A^2]



///////////////////////////////////
// LOG STUFF
///////////////////////////////////
int log_level=4;
FILE *log_file;



///////////////////////////////
// STANDALONE FUNCTIONS
///////////////////////////////
double random_gaussian()
{
  // This produces a random number from a gaussian distribution having standard deviation 1

  static int noExtra = 1;  // the algorithm is a cute trick that produces 2 random numbers, 
                           // returns one of them, and keeps track of the other
                           // toggling noExtra to decide if a new calculation is required.
  static double gset;
  double fac, r, v1, v2;

  if (noExtra)
  {
    do // get two random numbers between -1 and 1, and make sure they are within the unit circle
    {
      v1 = 2.0*rand()/RAND_MAX - 1;
      v2 = 2.0*rand()/RAND_MAX - 1;
      r = v1*v1 + v2*v2;           // need two random numbers in the unit circle
    } while (r >= 1.0 || r == 0);

    // now do the transformation
    fac = sqrt(-2.0*log(r)/r);
    gset = v1*fac;
    noExtra = 0; // we have an extra now
    return v2*fac;
  }
  else
  {
    noExtra = 1;
    return gset;
  }
}



///////////////////////////////
// DOMAIN struct
///////////////////////////////

// Data structure for each domain. 
// We use a struct because there is no cross-platform-compiler means of
// having Python interact with a C++ class. Blame the C++ compilers.
typedef struct domain {

    //////////////////////////////
    // Model Inputs 
    //////////////////////////////

    // Highest index of the valid Langevin field
    long n_langevin_valid;

    // Temperature
    double T, *Ts;

    // Volume of magnetic material (m^3), used only for Langevin field
    double V, *Vs;

    // Magnitude of the gyromagnetic ratio [radians / (sec T)]
    double gyro, *gyros;

    // Saturation magnetization u0*Ms [T]
    double M, *Ms;

    // Gilbert damping parameter [unitless]
    double damping, *dampings;         

    // Exchange-like field strength [T], applied in the direction of the other domain's unit vector
    double X, *Xs; 

    // Spin transfer torque (rate) parallel to other domain [rad / s]
    double STT, *STTs;

    // Other torque (rate) unrelated to either domain [rad / s]
    double Qx, *Qxs;
    double Qy, *Qys; 
    double Qz, *Qzs; 

    // Applied field components [T]
    double Bx, *Bxs;
    double By, *Bys; 
    double Bz, *Bzs; 

    // Anisotropy tensor elements [unitless], defined such that Nxx+Nyy+Nzz=1 for an aligned ellipsoid
    double Nxx, *Nxxs, Nxy, *Nxys, Nxz, *Nxzs; 
    double Nyx, *Nyxs, Nyy, *Nyys, Nyz, *Nyzs;
    double Nzx, *Nzxs, Nzy, *Nzys, Nzz, *Nzzs;

    // Dipole tensor [unitless], representing the fraction of the other layer's saturation magnetization
    double Dxx, *Dxxs, Dxy, *Dxys, Dxz, *Dxzs;       
    double Dyx, *Dyxs, Dyy, *Dyys, Dyz, *Dyzs; 
    double Dzx, *Dzxs, Dzy, *Dzys, Dzz, *Dzzs;

    //////////////////////////////
    // Settings
    //////////////////////////////

    // 0 = disabled, 1 = LLG
    int mode;    

    // Initial conditions
    double x0, y0, z0;

    //////////////////////////////////////////
    // SOLVER STUFF
    //////////////////////////////////////////

    // Solution arrays
    double *x, *y, *z;    // Magnetization unit vector
    double *Lx, *Ly, *Lz; // Langevin field arrays, filled on the fly.

} domain; 

// Function for logging a single step
void log_step(domain *a, domain *b, long n) {
    fprintf(log_file, "n=%li --------------------------%p\n", n, a->x);
    fprintf(log_file, "  a->gyro=%f,    pointer=%p\n",     a->gyro,    a->gyros);
    fprintf(log_file, "  a->M=%f,       pointer=%p\n",     a->M,       a->Ms);
    fprintf(log_file, "  a->damping=%f, pointer=%p\n",     a->damping, a->dampings);
    fprintf(log_file, "  a->X=%f,       pointer=%p\n",     a->X,       a->Xs);
    fprintf(log_file, "  a->STT=%f,     pointer=%p\n",     a->STT,     a->STTs);

    fprintf(log_file, "\n");
}

// Sets all the instantaneous model inputs for step n.
// We do this to ease the user's ability to input parameters
// vs arrays.
void get_input_parameters(domain *a, long n) {

    // Always check that the array exists first, then assume it's
    // of sufficient length.
    if(a->Ts       != NULL) a->T       = a->Ts[n];       // Temperature
    if(a->gyros    != NULL) a->gyro    = a->gyros[n];    // gyro
    if(a->Ms       != NULL) a->M       = a->Ms[n];       // magnetization
    if(a->Vs       != NULL) a->V       = a->Vs[n];       // Magnetic volume (m^3)
    if(a->dampings != NULL) a->damping = a->dampings[n]; // damping
    if(a->Xs       != NULL) a->X       = a->Xs[n];       // exchange
    if(a->STTs     != NULL) a->STT     = a->STTs[n];     // spin torque
    
    if(a->Bxs != NULL) a->Bx = a->Bxs[n]; // applied B-field
    if(a->Bys != NULL) a->By = a->Bys[n];
    if(a->Bzs != NULL) a->Bz = a->Bzs[n];

    if(a->Qxs != NULL) a->Qx = a->Qxs[n]; // other independent torQues
    if(a->Qys != NULL) a->Qy = a->Qys[n];
    if(a->Qzs != NULL) a->Qz = a->Qzs[n];

    if(a->Nxxs != NULL) a->Nxx = a->Nxxs[n]; // aNisotropy
    if(a->Nxys != NULL) a->Nxy = a->Nxys[n];
    if(a->Nxzs != NULL) a->Nxz = a->Nxzs[n];
    if(a->Nyxs != NULL) a->Nyx = a->Nyxs[n];
    if(a->Nyys != NULL) a->Nyy = a->Nyys[n];
    if(a->Nyzs != NULL) a->Nyz = a->Nyzs[n];
    if(a->Nzxs != NULL) a->Nzx = a->Nzxs[n];
    if(a->Nzys != NULL) a->Nzy = a->Nzys[n];
    if(a->Nzzs != NULL) a->Nzz = a->Nzzs[n];
    
    if(a->Dxxs != NULL) a->Dxx = a->Dxxs[n]; // Dipole
    if(a->Dxys != NULL) a->Dxy = a->Dxys[n];
    if(a->Dxzs != NULL) a->Dxz = a->Dxzs[n];
    if(a->Dyxs != NULL) a->Dyx = a->Dyxs[n];
    if(a->Dyys != NULL) a->Dyy = a->Dyys[n];
    if(a->Dyzs != NULL) a->Dyz = a->Dyzs[n];
    if(a->Dzxs != NULL) a->Dzx = a->Dzxs[n];
    if(a->Dzys != NULL) a->Dzy = a->Dzys[n];
    if(a->Dzzs != NULL) a->Dzz = a->Dzzs[n];
};

// Calculate a single step for this domain, if enabled.
// Parameters
//   domain *a    The domain whose step we wish to calculate.
//   domain *b    The "other" domain that exerts exchange fields, dipolar fields, and spin transfer.
//   long n        The step at which to calculate.
void D(domain *a, domain *b, long n, double dt, double *dx, double *dy, double *dz) {
    
    if(log_level >= 4) fprintf(log_file, "\nD() %li", n);

    // At each step (including intermediate steps), make sure to get 
    // the most current model input values from any supplied arrays.
    get_input_parameters(a, n);
    get_input_parameters(b, n);
    
    // If our domain's dynamics are not enabled, no step
    if(a->mode == 0) {
        *dx = *dy = *dz = 0;
        return;
    }

    // Intermediate values
    double Nx, Ny, Nz; // aNisotropy field [T] from this domain
    double Dx, Dy, Dz; // Dipolar field [T] from "other" domain
    double Xx, Xy, Xz; // Exchange field [T] from "other" domain
    double Fx, Fy, Fz; // Non-damping forcer [rad/sec]
    double vx, vy, vz; // Total non-damping torque [rad/sec]
    double Bx, By, Bz; // Total applied field, including Langevin
    
    // Prefactor (involves square root, don't recalculate unless we need to)
    static double langevin_prefactor; 

    // Previous values that go into the langevin_prefactor
    static double T, damping, gyro, M, V;

    // Initialization for the first step
    if(n==0) {
      
      if(log_level >= 4) fprintf(log_file, "  initializing D variables\n");
    
      // Set all the "previous values" to zero, to trigger a new Langevin calculation
      langevin_prefactor = T = damping = gyro = M = V = 0;
    }

    // Get the magnetic field
    Bx = a->Bx;
    By = a->By;
    Bz = a->Bz;

    // If we have non-zero temperature, include the Langevin field
    if(a->T > 0) {

      // Only calculate a new value if we haven't already done so for this index
      if(n > a->n_langevin_valid) {

        // Only recalculate the prefactor if the previous values don't match the current values
        if(T != a->T || damping != a->damping || gyro != a->gyro || M != a->M || V != a->V) {
          
          // Remember these values for next time
          T       = a->T;
          damping = a->damping;
          gyro    = a->gyro;
          M       = a->M;
          V       = a->V;

          // Calculate the prefactor
          langevin_prefactor = sqrt( 4*u0*damping*kB*T / (gyro*M*V*dt) );
        } 
        
        // Now calculate the langevin field for this step
        a->Lx[n] = langevin_prefactor*random_gaussian();
        a->Ly[n] = langevin_prefactor*random_gaussian();
        a->Lz[n] = langevin_prefactor*random_gaussian();
        
        // If we come back to this value of n, we will use the existing value.
        a->n_langevin_valid = n;
      }

      // Now add the Langevin field from this index to the magnetic field
      Bx += a->Lx[n];
      By += a->Ly[n];
      Bz += a->Lz[n];
    }

    // Calculate the aNisotropy field
    Nx = -a->M*(a->Nxx*a->x[n] + a->Nxy*a->y[n] + a->Nxz*a->z[n]);
    Ny = -a->M*(a->Nyy*a->y[n] + a->Nyz*a->z[n] + a->Nyx*a->x[n]);
    Nz = -a->M*(a->Nzz*a->z[n] + a->Nzx*a->x[n] + a->Nzy*a->y[n]);
    
    // Now the Dipolar field from b
    Dx = -b->M*(a->Dxx*b->x[n] + a->Dxy*b->y[n] + a->Dxz*b->z[n]);
    Dy = -b->M*(a->Dyy*b->y[n] + a->Dyz*b->z[n] + a->Dyx*b->x[n]);
    Dz = -b->M*(a->Dzz*b->z[n] + a->Dzx*b->x[n] + a->Dzy*b->y[n]);
    
    // Now the eXchange field from b
    Xx = a->X*b->x[n];
    Xy = a->X*b->y[n];
    Xz = a->X*b->z[n];

    // Now we can get the components of F. Bx, By, and Bz include the Langevin values already.
    // (We combine the two so that if T=0 the user needn't supply langevin arrays)
    Fx = -a->gyro*(Bx+Nx+Dx+Xx) + a->STT*(a->z[n]*b->y[n] - a->y[n]*b->z[n]) + a->z[n]*a->Qy - a->y[n]*a->Qz;
    Fy = -a->gyro*(By+Ny+Dy+Xy) + a->STT*(a->x[n]*b->z[n] - a->z[n]*b->x[n]) + a->x[n]*a->Qz - a->z[n]*a->Qx;
    Fz = -a->gyro*(Bz+Nz+Dz+Xz) + a->STT*(a->y[n]*b->x[n] - a->x[n]*b->y[n]) + a->y[n]*a->Qx - a->x[n]*a->Qy;
    
    // Now we can compute the total non-damping torque for this step.
    vx = a->y[n]*Fz - a->z[n]*Fy; 
    vy = a->z[n]*Fx - a->x[n]*Fz; 
    vz = a->x[n]*Fy - a->y[n]*Fx; 

    // We store the step magnitude to help with Heun method.
    double scale = dt/(1.0+a->damping*a->damping);
    *dx = ( vx + a->damping*(a->y[n]*vz-a->z[n]*vy) ) * scale;
    *dy = ( vy + a->damping*(a->z[n]*vx-a->x[n]*vz) ) * scale;
    *dz = ( vz + a->damping*(a->x[n]*vy-a->y[n]*vx) ) * scale;
}



///////////////////////////////////
// SOLVER
///////////////////////////////////

void solve_heun(domain *a, domain *b, double dt, long steps) {
  
  // Log file
  if(log_level > 0) {
    log_file = fopen("engine.log", "w");
    fprintf(log_file, "solve_heun() beings\n------------------------------------------------\n\n");
    log_step(a,b,0);
  }

  // For measuring the time of the simulation
  long t0 = time(0);

  // The initial condition of the magnetization is assumed to be the 
  // first element of the array, but we should make sure it's length is 1!

  // Scale factor
  double scale;
  
  // Normalize a
  scale = 1.0/sqrt(a->x[0]*a->x[0] + a->y[0]*a->y[0] + a->z[0]*a->z[0]);
  a->x[0] *= scale;
  a->y[0] *= scale;
  a->z[0] *= scale;

  // Normalize b
  scale = 1.0/sqrt(b->x[0]*b->x[0] + b->y[0]*b->y[0] + b->z[0]*b->z[0]);
  b->x[0] *= scale;
  b->y[0] *= scale;
  b->z[0] *= scale;
  
  // These will hold the step values calculated by D();
  double adx1, ady1, adz1, bdx1, bdy1, bdz1;
  double adx2, ady2, adz2, bdx2, bdy2, bdz2;
  
  if(log_level >=1) fprintf(log_file, "STARTING LOOP: steps=%li\n", steps);

  // Now do the Heun loop
  // We don't go to the end because we don't want to overwrite the first step.
  for(long n=0; n<=steps-2; n++) {
   
    //  Heun method: with our derivative step dy(y,n), we calculate intermediate value
    //
    //    yi[n+1] = y[n] + dy(y,n)
    //
    //  then get a better estimate
    //    
    //    y[n+1] = y[n] + 0.5*( dy(y,n) + dy(yi,n+1) )
    //
    //  Importantly, dy(y,n) involves the current magnetization, field, etc, 
    //  whereas dy(yi, n+1) involves the intnermediate magnetization, field, etc at the next step.
    
    // Calculate dy(y,n)
    D(a, b, n, dt, &adx1, &ady1, &adz1);
    D(b, a, n, dt, &bdx1, &bdy1, &bdz1);

    if(log_level >= 3 && (n % (long)(steps/5) == 0 || n<5 || steps-n<7)) log_step(a, b, n);
    
    // Store the intermediate value yi at n+1
    a->x[n+1] = a->x[n] + adx1; 
    a->y[n+1] = a->y[n] + ady1; 
    a->z[n+1] = a->z[n] + adz1;
    b->x[n+1] = b->x[n] + bdx1; 
    b->y[n+1] = b->y[n] + bdy1; 
    b->z[n+1] = b->z[n] + bdz1;
    
    // Calculate dy(yi,n+1)
    D(a, b, n+1, dt, &adx2, &ady2, &adz2);
    D(b, a, n+1, dt, &bdx2, &bdy2, &bdz2);

    // Get the Heun step
    a->x[n+1] = a->x[n] + 0.5*(adx1 + adx2);
    a->y[n+1] = a->y[n] + 0.5*(ady1 + ady2);
    a->z[n+1] = a->z[n] + 0.5*(adz1 + adz2);
    b->x[n+1] = b->x[n] + 0.5*(bdx1 + bdx2);
    b->y[n+1] = b->y[n] + 0.5*(bdy1 + bdy2);
    b->z[n+1] = b->z[n] + 0.5*(bdz1 + bdz2);

    // Normalize the new magnetization using the Taylor expansion of sqrt() near 1 to speed up the calculation.
    double norminator;
    
    // domain a
    norminator = 1.0/(1.0 + 0.5 * (a->x[n+1]*a->x[n+1] + a->y[n+1]*a->y[n+1] + a->z[n+1]*a->z[n+1] - 1.0) );
    a->x[n+1] *= norminator;
    a->y[n+1] *= norminator;
    a->z[n+1] *= norminator;

    // domain b
    norminator = 1.0/(1.0 + 0.5 * (b->x[n+1]*b->x[n+1] + b->y[n+1]*b->y[n+1] + b->z[n+1]*b->z[n+1] - 1.0) );
    b->x[n+1] *= norminator;
    b->y[n+1] *= norminator;
    b->z[n+1] *= norminator;

  } // End of for loop.

  // At this point, the whole solution arrays should be populated.
  if(log_level>0) {
    fprintf(log_file, "\n\n------------------------------------------------\nsolve_heun() done after %li", time(0)-t0);
    fclose(log_file);
  }
}

