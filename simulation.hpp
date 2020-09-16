//////////////////////////////////////////////////////////////////////////////
//  Quantum Monte Carlo Simulation for Kitaev Models
//  with Green's-function-based Kernel Polynomial Method
//  written by: Tim Eschmann, May 2017
//  Modified version: February 2019
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//
//  The code skeleton of this file was derived from "ising_skeleton.cpp" 
//  (Copyright (C) 2003 by Brigitte Surer and Jan Gukelberger)   
//  which is part of the ALPS libraries and was published under the 
//  ALPS Library License (http://alps.comp-phys.org/)
//
//  This software incorporates the Armadillo C++ Library
//  Armadillo C++ Linear Algebra Library
//  Copyright 2008-2020 Conrad Sanderson (http://conradsanderson.id.au)
//  Copyright 2008-2016 National ICT Australia (NICTA)
//  Copyright 2017-2020 Arroyo Consortium
//  Copyright 2017-2020 Data61, CSIRO

//  This product includes software developed by Conrad Sanderson (http://conradsanderson.id.au)
//  This product includes software developed at National ICT Australia (NICTA)
//  This product includes software developed at Arroyo Consortium
//  This product includes software developed at Data61, CSIRO

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <armadillo>
#include <mpi.h>

#include <alps/scheduler/montecarlo.h>
#include <alps/alea.h>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <boost/filesystem.hpp>

// Include package to calculate thermodynamic observables
#include "functions.hpp"

// Include the desired lattice (read in header file from .hpp folder)
#include "Lattices/honeycomb_8_sites.hpp"

#ifndef _temp_hpp_
#define _temp_hpp_
double calc_temp(double T_min, double T_max, int me, int np, std::string dist);
#endif

using namespace arma;

class Simulation
{
public:
    Simulation(int np, int me, double T_min, double T_max, double T, double lambda, int intsteps, int M, std::string dist, std::string output_file)
    // Define interaction matrix A, parameters and measurable observables:
    :   eng_(3*me) // Random generator engine (different seed for each replica)
    ,   rng_(eng_, dist_) // Random generator
    ,   np_(np) // # of processes
    ,   me_(me) // process number / parallelization rank
    ,   T_min(T_min) // minimal temperature
    ,   T_max(T_max) // maximal temperature
    ,   dist(dist) // Temperature distribution
    ,   temp_(T)  // Replica Temperature
    ,   beta_(1/T) // Inverse replica Temperature
    ///////////////////////////////////////////////////
    // System configuration stored as CRS blockvectors
    ,   val()     
    ,   col_idx()
    ,   row_ptr()
    //////////////////
    ,   N_() // # system sites (IMPORTANT: THIS HAS CHANGED W.R.T. FORMER VERSIONS !!!)
    ,   v_() // Vector with coordinates of nonzero matrix entries
    ,   length_() // number of non-zero matrix elements
    ////////////////////////////////////
    //  Parameters for KPM Calculation
    ,   s_()
    ,   ld()
    ,   bw_save()
    ,   lambda_(lambda)
    ,   intsteps_(intsteps)
    ,   M_(M)
    ,   int_min() // Limits for all integrations ...
    ,   int_max()
    ,   step_()
    ,   pi()
    ,   twopi()
    ,   Del_rho()
    ,   y_values()
    ,   tanh_values()
    ,   kernel_factors()
    ,   dr()
    ,   mom_i_plus_ij()
    ,   mom_i_plus_j()
    ,   mom_ii()
    ,   mom_jj()
    ,   mom()
    ,   i_()
    ,   j_2()
    ,   j_1()
    ,   j_()
    ////////////////////////////////////
    ,   plaquettes() // matrix with elementary plaquettes
    ,   plaquettes2() // -------------- " ---------------
    ,   energy_("E") // Measurement data: energy
    ,   e2_("E2") // Measurement data: squared energy
    //,   e4_("E4") // Measurement data: energy^4
    ,   dE_db("dE_db") // Measurement data: dE / d(beta)
    ,   p_("p") // disorder
    ,   flux_real("Flreal") // Measurement data: average plaquet flux (real part)
    //,   flux_imag("Flimag") // "" (imaginary part)
    ,   flux_real_squared("Flreal2") // Measurement data: average plaquet flux (real part)
    //,   flux_imag_squared("Flimag2") // "" (imaginary part)
    ,   flux2_real("Fl2real") // Measurement data: average plaquet flux (real part)
    //,   flux2_imag("Fl2imag") // "" (imaginary part)
    ,   flux2_real_squared("Fl2real2") // Measurement data: average plaquet flux (real part)
    //,   flux_imag_squared("Fl2imag2") // "" (imaginary part)
    ,   spin_corr("Spin_corr")
    ,   flux_corr("Flux_corr")
    ,   flip_rate() // Single flip acceptance rate
    ,   filename_(output_file) // Filename for data saving
        
    {  
    ////////////////////////////////////////////////////////////////////////////////////////////
    // Some necessary initializations: /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    
    // Locate nonzero entries of interaction matrix (.hpp file):
    v_ = non_zeros();
    length_ = v_.size();
    
    // Fill interaction matrix A with coefficients due to lattice symmetry (.hpp file)
    // -> CRS format:
    val = get_val(def_matrix());
    col_idx = get_col_idx(def_matrix());
    row_ptr = get_row_ptr(def_matrix());

    // Initialize plaquettes for flux measurements:
    plaquettes = create_plaquettes();
    //plaquettes2 = create_plaquettes_2();

    // Number of system sites:
    N_ = size(row_ptr)[0] - 1;

    // Initial eigenvalues and bandwidth:
    vec ev = get_evals(val, col_idx, row_ptr);    
    
    bw_save << "bw.saved";
    std::ifstream bw_loadfile(bw_save.str().c_str(), std::ifstream::in);
    if(bw_loadfile.good())
    {
        bw_loadfile >> s_;
        bw_loadfile.close();
    }
    else
    {
        s_ = -ev[0];

        if (me_ == 1)
            std::cout << "Calculating bandwidth ..." << std::endl;
    
        // Determine bandwidth by diagonalizing 1000 random configurations:
        for (int ii = 0; ii < 1000; ii++)
        {
            // Random configuration:
            val = get_val(randomize(def_matrix())); 
            ev = get_evals(val, col_idx, row_ptr);

            if (s_ < -ev[0])
                s_ = -ev[0];

        }

        if (me_ == 1)
        {
            std::ofstream bw_savefile(bw_save.str().c_str(), std::ofstream::trunc);
            bw_savefile << s_;
        }
    }

    // Initialize parameters for KPM/GF calculations from bandwidth:
    int_min = 0.;
    int_max = 0.9999999*s_;
    step_ = (int_max - int_min)/double(intsteps_ - 1);

    if (me_ == 1)
    std::cout << "Bandwidth check: E_0 = " << ev[0] << ", int_max = " << int_max << ", s_ = " << s_ << std::endl;

    // This is pi, a famous number:
    pi = M_PI; 
    twopi = 2*M_PI;

    // Repeatedly used values for the integration in main part:
    Del_rho = vec(intsteps_); // reserve memory
    
    y_values = cx_mat(M_, intsteps_, fill::zeros); // cf. warning above!!!
    tanh_values = vec(intsteps_, fill::zeros);

    double EE;

    for (int jj = 0; jj < intsteps_; jj++)
    {
        // Linear distribution of abscissas (for trapezoidal or Simpson integration):
        EE = int_min + jj*step_;
        
        // Tabulation of values that are repeatedly needed during calculation of Green functions:
        tanh_values[jj] = tanh(beta_*EE/2.);

        for (int mm = 0; mm < M_; mm++)
        {
            // round brackets!!!
            y_values(mm, jj) = 2.0 * exp(-std::complex<double>(0.,1.)*double(mm)*acos(EE/s_)) / (sqrt(s_*s_ - EE*EE));
        }

    }

    // Calculate kernel factors (needed during the calculation of Chebyshev moments)
    kernel_factors = vec(M_, fill::zeros);

    for (int mm = 0; mm < M_; mm++)
    {
        // Jackson kernel:
        kernel_factors[mm] = ((double(M_) - double(mm) + 1)*cos(pi*double(mm)/(double(M_) + 1)) + sin(pi*double(mm)/(double(M_) + 1))*cos(pi/(double(M_) + 1))/sin(pi/(double(M_) + 1))) / (double(M_) + 1); 
    }

    // Reserve memory for vectors that are repeatedly filled below: 
    dr = vec(intsteps_, fill::zeros);
    mom_i_plus_ij = cx_vec(M_, fill::zeros);
    mom_i_plus_j = cx_vec(M_, fill::zeros);
    mom_ii = cx_vec(M_, fill::zeros);
    mom_jj = cx_vec(M_, fill::zeros);

    mom = cx_vec(M_);
    i_ = cx_vec(N_);
    j_2 = cx_vec(N_);
    j_1 = cx_vec(N_);
    j_ = cx_vec(N_);

    sp_cx_mat ham_sp(N_, N_);

    // Initialize single flip acceptance rate to 0:
    flip_rate = 0;

    } 

    // Replica Monte Carlo iteration
    void run(int n, int ntherm, int sweeps_per_swap, int sweeps_per_save)
    {
        engine_type loaded, saved; // needed to load and save random engine status
        sweeps_ = n;
        thermalization_ = ntherm; // thermalization steps

        double fr; // fliprate
        int n_tot = n + ntherm;
        double tau_en, tau_fl; // autocorrelation times

        vec eval;

        std::stringstream matrix_output; // needed for saving configurations (only 'val' needed)
        matrix_output << "val_temp_" << 1/beta_ << ".saved";

        std::stringstream rng_save; // needed for saving random generator status
        rng_save << "rng_" << me_ << ".saved";
         
        //Load so-far-obtained measurement data (ALPS)
        if (boost::filesystem::exists(filename_))
        {
            load(filename_);
        }
        
        // Load random generator status:
        std::ifstream rngfile(rng_save.str().c_str(), std::ifstream::in);
        if(rngfile.good())
        {
            rngfile >> loaded;
            rngfile.close();
            eng_ = loaded;
        }

        // Load last Z2 configuration from file (-> skip thermalization)
        std::ifstream matfile(matrix_output.str().c_str(), std::ifstream::in);
        if(matfile.good())
            val.load(matrix_output.str().c_str());

        // Thermalize for ntherm steps
        if (me_ == 1)
            std::cout << "Thermalizing ..." << std::endl;
        
        while(ntherm--)
        {
            // MC Step:
            step();

            // Get Eigenvalues for swapping and observable calculations:
            eval = get_evals(val, col_idx, row_ptr);

            // Communication with master process:
            if (ntherm % sweeps_per_swap == 0)
            {
                // Replica exchange: 
                swap();  

                if (((ntherm + n) / sweeps_per_swap) % sweeps_per_save == 0)
                {    
                    // Calculate single flip acceptance rate and send it to Master process:
                    fr = double(flip_rate)/(double(n_tot - n - ntherm)*N_);
                    MPI_Send(&fr, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
                } 
            }
            
            // Saving: 
            if (ntherm % sweeps_per_save == 0)
            {
                // Save Z2 configuration:
                val.save(matrix_output.str().c_str());

                // Save random generator status:
                saved = eng_;
                std::ofstream file(rng_save.str().c_str(), std::ofstream::trunc);
                file << saved;
                
                // Tell that everything is saved ...
                if (me_ == 1)
                    std::cout << "SAVE " << ntherm << std::endl;        
            }          
        }

        // Output for orientation (only for process 1):
        if (me_ == 1)
        {
            std::cout << "###############################" << std::endl;
            std::cout << "Sweeping ..." << std::endl;
        }

        // Run n steps
        while(n--)
        {              
            // MC step:
            step();

            // Get Eigenvalues for swapping and observable calculations:
            eval = get_evals(val, col_idx, row_ptr);
            
            // Output eigenvalue and flux configurations:
            //output_eigenvalues(eval);
            //output_flux_confs();
            //output_P_n();

            // Measure observables:
            measure(eval);
            
            // Communication with Master process:
            if (n % sweeps_per_swap == 0)
            {
                // Replica exchange: 
                swap();  

                if (((n) / sweeps_per_swap) % sweeps_per_save == 0)
                {
                    // Calculate single flip acceptance rate and send it to Master process:
                    fr = double(flip_rate)/(double(n_tot - n - ntherm)*N_);
                    MPI_Send(&fr, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
                }
            }
                  
            // Save all simulation data:
            if (n % sweeps_per_save == 0)
            {   
                // Save results:
                save(filename_);
                
                // Save Z2 configuration:
                val.save(matrix_output.str().c_str());

                // Save random generator status:
                saved = eng_;
                std::ofstream file(rng_save.str().c_str(), std::ofstream::trunc);
                file << saved;
                
                // Tell that everything is saved ...
                if (me_ == 1)
                    std::cout << "SAVE " << n << std::endl; 
            }    
        }

        //Save the observables to file
        save(filename_);

        tau_en = energy_.tau();
        tau_fl = flux_real.tau();
        MPI_Send(&tau_en, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
        MPI_Send(&tau_fl, 1, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
     
        // Print observables      
        /*std::cout.precision(17);
        std::cout << energy_.name() << ":\t" << energy_.mean()
            << " +- " << energy_.error() << ";\ttau = " << energy_.tau() 
            << ";\tconverged: " << alps::convergence_to_text(energy_.converged_errors())     
            << std::endl;
        std::cout << flux_real.name() << ":\t" << flux_real.mean()
            << " +- " << flux_real.error() << ";\ttau = " << flux_real.tau() 
            << ";\tconverged: " << alps::convergence_to_text(flux_real.converged_errors())
            << std::endl;
        std::cout << flux_imag.name() << ":\t" << flux_imag.mean()
            << " +- " << flux_imag.error() << ";\ttau = " << flux_imag.tau() 
            << ";\tconverged: " << alps::convergence_to_text(flux_imag.converged_errors())
            << std::endl;*/
    }
    
    // Iteration step (= "Metropolis sweep"): 
    void step()
    {
        int kk, i, j; // Running indices 
        int idx; // Index in val vector
         
        double Delta_F; // Free energy change
        
        int coord1, coord2; // Random bond coordinate
        double alpha, gamma; // Monte Carlo variables
        
        int count = 0;

        // One sweep = N tries (# lattice sites)
        for (kk = 0; kk < N_; kk++)
        {               
            // Switch sign of random matrix entry:
            int die = roll_die(length_);
            coord1 = v_[die]/(N_);
            coord2 = v_[die]%(N_);

            // Calculate change in energy density via Green function / KPM method:
            Del_rho = Delta_rho_dE(val, col_idx, row_ptr, coord1, coord2); 

            // Calculate free energy by integration:
            Delta_F = integrate_free_energy(Del_rho);
            
            // Weight and random number:
            alpha = 1./ (1. + exp(beta_ * (Delta_F)));
            gamma = rng_();

            // Accepted?
            if (gamma <= alpha) 
            {                  
                // Update pair of matrix entries:
                // Check which index in val is changed
                idx = get_idx(col_idx, row_ptr, coord1, coord2); 
                if (idx < size(val)[0])
                    val(idx) *= -1;
                else
                    std::cout << "WARNING: Severe error during update trial" << std::endl;

                idx = get_idx(col_idx, row_ptr, coord2, coord1);
                if (idx < size(val)[0])
                    val(idx) *= -1;
                else
                    std::cout << "WARNING: Severe error during update trial" << std::endl;
                
                // Single flip acceptance rate + 1
                flip_rate += 1;
            }
            count += 1;
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////
    // KPM / Green function part: //////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////

    // Recursive calculation of moments for the Chebyshev expansion (all diagonal):
    cx_vec moments(cx_vec val, vec col_idx, vec row_ptr, int i, int j, std::string num)
    {   
        mom = cx_vec(M_, fill::zeros);
        i_ = cx_vec(N_, fill::zeros);
        j_2 = cx_vec(N_, fill::zeros);

        if (i == j)
        {
            i_[i] = 1.;
            j_2[j] = 1.;
        }
        else if (i != j && num == "real")
        {
            i_[i] = 1.;
            i_[j] = 1.;
            j_2[i] = 1.;
            j_2[j] = 1.;
        }
        else if (i != j && num == "imag")
        {
            i_[i] = 1.;
            i_[j] = std::complex<double>(0.0, -1.0);
            j_2[i] = 1.;
            j_2[j] = std::complex<double>(0.0, 1.0);
        }
        
        j_1 = compl_mat_vec_multiply(val/s_, col_idx, row_ptr, j_2);
        
        // First two moments:
        mom[0] = dot(i_ , j_2);
        mom[1] = dot(i_ , j_1);
          
        // Improved Chebyshev recursion: 
        // mom[2m] = 2 <j_m|j_m> - mom[0]
        // mom[2m+1] = 2 <j_m+1|j_m> - mom[1]
        if (num == "real") // every odd moment is zero then!
        {
            for (int m = 1; m < M_/2; m++)
            {
                j_ = 2.*compl_mat_vec_multiply(val/s_, col_idx, row_ptr, j_1) - j_2;  
                mom[2*m] = 2.*cdot(j_1, j_1) - mom[0];
                j_2 = j_1;
                j_1 = j_; 
            }
        }
        else if (num == "imag") // Here we have to iterate all the moments!
        {
            for (int m = 1; m < M_/2; m++)
            {
                j_ = 2.*compl_mat_vec_multiply(val/s_, col_idx, row_ptr, j_1) - j_2;  
                mom[2*m] = 2.*cdot(j_1, j_1) - mom[0];
                mom[2*m+1] = 2.*cdot(j_, j_1) - mom[1];
                j_2 = j_1;
                j_1 = j_; 
            }
        }
        
        // Multiply each moment by kernel factor:
        for (int mm = 1; mm < M_; mm++)
        {
            mom[mm] *= kernel_factors[mm];
        }
            
        return mom;
    }
    
    // Green Function
    std::complex<double> green(cx_vec mmts, int MM, int jj, std::string num)
    {
        const std::complex<double> im(0., 1.);
        double EE = int_min + jj*step_;
        std::complex<double> value = mmts[0] / (sqrt(s_*s_ - EE*EE));

        // Approximate GF with MM Chebyshev moments:
        if (num == "real") // every second moment is zero here ...
        {    
            for (int mm = 2; mm < MM; mm+=2)
            {
                value += mmts[mm]*y_values(mm,jj);
            }
        }
        else if (num == "imag")
        {
            for (int mm = 1; mm < MM; mm++)
            {
                value += mmts[mm]*y_values(mm,jj);
            }
        }

        return im*value;
    }

    //////////////////////////////////////////////////////////////////////
    // Calculate function Im(log(d(E))) via Green Function / KPM method
    //////////////////////////////////////////////////////////////////////
    
    vec Delta_rho_dE(cx_vec val, vec col_idx, vec row_ptr, int c1, int c2)
    {
        std::complex<double> im(0., 1.);

        std::complex<double> Delta_ij = val(get_idx(col_idx, row_ptr, c1, c2));
        Delta_ij *= -2.0;
        std::complex<double> Delta_sq = Delta_ij * Delta_ij;
        std::complex<double> g_ij, g_ji, g_ii, g_jj, g_i_plus_j, g_i_plus_ij;
        std::complex<double> d;

        // Vectors with Chebyshev moments:
        mom_ii = moments(val, col_idx, row_ptr, c1, c1, "real");
        mom_jj = moments(val, col_idx, row_ptr, c2, c2, "real");
        mom_i_plus_j = moments(val, col_idx, row_ptr, c1, c2, "real");
        mom_i_plus_ij = moments(val, col_idx, row_ptr, c1, c2, "imag");

        // Calculate Im(log(d(E))) from Green Functions:
        for (int j = 0; j < intsteps_; ++j)
        {            
            // Calculate Chebyshev expansion of Green functions:
            g_ii = green(mom_ii, M_, j, "real");
            g_jj = green(mom_jj, M_, j, "real");
            g_i_plus_j = green(mom_i_plus_j, M_, j, "real");
            g_i_plus_ij = green(mom_i_plus_ij, M_, j, "imag");
            
            // (Antisymmetry can be included: g_ij = -g_ji) 
            g_ij = 0.5*(g_i_plus_j - im*g_i_plus_ij - (1. - im)*(g_ii + g_jj));
            g_ji = 0.5*(g_i_plus_j + im*g_i_plus_ij - (1. + im)*(g_ii + g_jj));
             
            // d(E) = det (1 + G(E)*Delta(E)):
            d = (1. + Delta_ij * g_ji) * (1. - Delta_ij * g_ij) + Delta_sq*g_ii*g_jj; 

            // im lim_(eps -> 0) log(d(E + i*eps)) = (rho(E) - rho'(E)) dE:
            dr[j] = imag(log(d));             
        }

        return dr;
        
    }
     
    /////////////////////////////////////////////////////////
    // Calculate free energy change by integration:
    /////////////////////////////////////////////////////////  

    double integrate_free_energy(vec d_rho)
    {
        double df, m;
        double integral1 = 0.0;
        double E;
        double value;
                
        // Semi-open integration (cf. "Numerical recipes in C", Press et al.)
        for (int jj = 0; jj < intsteps_ - 1; ++jj)
        {
            //std::cout << tanh_values[jj] << " " << d_rho[jj] << std::endl;
            value = tanh_values[jj]*d_rho[jj];

            if (jj == 1)
                integral1 += value / 2.;
            else if (jj == intsteps_ - 2) 
                integral1 += 1.5 * value;
            else
                integral1 += value;
        }

        df = -integral1 * step_ / twopi;
        
        return df;
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    
    void measure(vec eval)
    {      
        std::complex <double> fl, fl2, op;
        double E_, dE_;
        double p;
        double fl_real;
        //double fl_imag;
        double fl2_real;
        //double fl2_imag;
        double corr;
        double flcorr;
        //sp_cx_mat ham_sp = get_sparse(val, col_idx, row_ptr);
        ham_sp = get_sparse(val, col_idx, row_ptr);

        E_ = en(eval, beta_);
        dE_ = diffE(eval, beta_);

        // Measure average flux per plaquet and disorder: 
        p = get_p(ham_sp, plaquettes);
        fl = flux(ham_sp, plaquettes);
        fl_real = std::real(fl);
        //fl2 = flux(ham_sp, plaquettes2);
        //fl2_real = std::real(fl2);
        //op = pseudo_ord_par(ham_sp, plaquettes);
        //fl_imag = std::real(op);

        // Measure spin-spin correlation:
        corr = correlation(ham_sp, v_, beta_);
        flcorr = flux_correlation(ham_sp, plaquettes);
        
        // Add sample to observables:
        energy_ << E_/double(N_); // Energy per site
        e2_ << E_/double(N_)*E_/double(N_); // Squared energy per site
        //e4_ << E_/double(N_)*E_/double(N_)*E_/double(N_)*E_/double(N_);
        dE_db << dE_ / double(N_); // dE/d(beta)
        p_ << p;
        flux_real << fl_real;
        //flux_imag << fl_imag;
        flux_real_squared << fl_real*fl_real;
        //flux_imag_squared << fl_imag*fl_imag;
        //flux2_real << fl2_real;
        //flux2_imag << fl2_imag;
        //flux2_real_squared << fl2_real*fl2_real;
        //flux2_imag_squared << fl2_imag*fl2_imag;
        spin_corr << corr;
        flux_corr << flcorr;
    }

    void output_flux_confs()
    {
        //sp_cx_mat ham = get_sparse(val, col_idx, row_ptr);
        ham_sp = get_sparse(val, col_idx, row_ptr);
        cx_vec fl_confs = flux_confs(ham_sp, plaquettes);
        
        std::stringstream flux_output; // needed for saving configurations
        flux_output << "flux_configuration_temp_" << 1/beta_ << ".saved";

        std::ofstream flux(flux_output.str().c_str(), std::ofstream::app);
        for(int iii = 0; iii < size(plaquettes)[0]; iii++)
        {
            // Switch between real and imaginary fluxes:
            flux << std::setprecision(17) << std::real(fl_confs[iii]) << "   ";
            //flux << std::setprecision(17) << std::imag(fl_confs[iii]) << std::endl;
        }
        flux << std::endl;
    }

    void output_eigenvalues(vec eval)
    {        
        std::stringstream output; // needed for saving energies
        output << "eigenvalues_temp_" << 1/beta_ << ".saved";

        std::ofstream eig(output.str().c_str(), std::ofstream::app);
        for(int iii = 0; iii < size(eval)[0]/2; iii++)
        {
            eig << std::setprecision(17) << eval[iii] << "   ";
        }
        eig << std::endl;
    }
    
    // Swap replica with left neighbour ...
    void swapleft()
    {
        MPI_Status status;
        int control = 0;
        int jj;
        double beta_alt = 1/calc_temp(T_min, T_max, me_ - 1, np_, dist);
        cx_vec H_a(size(val)[0], fill::zeros); // receive
        cx_vec H_b = val; // send
        vec eigval = get_evals(val, col_idx, row_ptr);

        double f2 = -beta_alt * free_en(eigval, beta_alt);
        double f3 = beta_ * free_en(eigval, beta_);

        MPI_Send(&f2, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&f3, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        MPI_Recv(&control, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        
        if (control == 1)
        {
            // Receive replica from left neighbour
            for (jj = 0; jj < size(val)[0]; jj++)
            {
                MPI_Recv(&H_a(jj), 1, MPI_DOUBLE_COMPLEX, me_- 1, 3, MPI_COMM_WORLD, &status);
            }
            
            // Send own replica to left neighbour
            for (jj = 0; jj < size(val)[0]; jj++)
            {
                MPI_Send(&H_b(jj), 1, MPI_DOUBLE_COMPLEX, me_- 1, 4, MPI_COMM_WORLD);
            }

            val = H_a;
        }
    }

    // Swap replica with right neighbour ...
    void swapright()
    {
        MPI_Status status;
        int control = 0;
        int jj;
        double beta_alt = 1/calc_temp(T_min, T_max, me_ + 1, np_, dist);
        cx_vec H_b(size(val)[0], fill::zeros); // receive (here it's the other way round!!!)
        cx_vec H_a = val; // send
        vec eigval = get_evals(val, col_idx, row_ptr);

        double f1 = -beta_alt * free_en(eigval, beta_alt);
        double f4 = beta_ * free_en(eigval, beta_);

        MPI_Send(&f1, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&f4, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        MPI_Recv(&control, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        if (control == 1)
        {
            // Send own replica to right neighbour
            for (jj = 0; jj < size(val)[0]; jj++)
            {
                MPI_Send(&H_a(jj), 1, MPI_DOUBLE_COMPLEX, me_+ 1, 3, MPI_COMM_WORLD);
            }

            // Receive replica from right neighbour
            for (jj = 0; jj < size(val)[0]; jj++)
            {
                MPI_Recv(&H_b(jj), 1, MPI_DOUBLE_COMPLEX, me_+ 1, 4, MPI_COMM_WORLD, &status);
            }

            val = H_b;
        }

    }

    // Parallel Tempering for each temperature point (= "Swap")
    void swap()
    {
        if (me_ != 1)   
            swapleft();
        if (me_ != np_ - 1)     
            swapright();
    }

    // Master process for managing swaps:
    void master(int therm, int sweeps, int sweeps_per_save, int sweeps_per_swap)
    {
        MPI_Status status;
        engine_type loaded, saved; // needed to load and save random engine status
        int counts = (therm + sweeps)/sweeps_per_swap; // How many swaps in total?
        int control = 0;  // Signal for accepting / rejecting swap
        int sign_[np_];   // needed for ensemble optimization 
        int nplus_[np_];  // n+ histogram
        int nminus_[np_]; // n- histogram
        double counter[np_]; // How many accepted swaps?
        double den = 0;
        int s_i, s_i_plus_1; // sign for each replica (was it at T_min or T_max latest?)
        double f1, f2, f3, f4; // free energy variables
        double alpha_pt, gamma_pt; // Monte Carlo variables

        double fr; // Single flip acceptance rate

        // Autocorrelation times:
        double tau_en;
        double tau_fl;
        
        std::stringstream rng_save; // needed for saving random generator status
        rng_save << "rng_" << me_ << ".saved";

        std::stringstream sfar; // needed for saving single flip acceptance rates
        sfar << "single_flip_rate.saved";

        std::stringstream nplus_ratio; // needed for saving ratio function f = n_plus / n_tot
        nplus_ratio << "n_plus_ratio.saved";

        std::stringstream swap_ratio; // needed for saving replica exchange ratio
        swap_ratio << "swap_ratio.saved";

        std::stringstream tau_energy; // needed for saving energy autocorrelation time
        tau_energy << "tau_energy.saved";

        std::stringstream tau_flux; // needed for saving flux autocorrelation time
        tau_flux << "tau_flux.saved";

        // Load random generator status:
        std::ifstream rngfile(rng_save.str().c_str(), std::ifstream::in);
        if(rngfile.good())
        {
            rngfile >> loaded;
            rngfile.close();
            eng_ = loaded;
        }

        // Initialize sign array and histograms
        for (int k = 0; k < np_; k ++)
        {
            sign_[k] = 0;
            nplus_[k] = 0;
            nminus_[k] = 0;
            counter[k] = 0;
        }

        sign_[1] = 1;
        sign_[np_ - 1] = -1;

        // PT iteration
        while (counts--)
        {
            den += 2;

            // Regard temperature points from T_min to T_max
            for (int i = 1; i < np_ - 1; i++)
            {
                
                // Receive free energies from replicas
                MPI_Recv(&f1, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&f4, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&f2, 1, MPI_DOUBLE, i+1, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(&f3, 1, MPI_DOUBLE, i+1, 2, MPI_COMM_WORLD, &status);

                // Decide if replicas are swapped
                alpha_pt = exp(f1 + f2 + f3 + f4);
		        gamma_pt = rng_();

                if (gamma_pt <= alpha_pt) // accept
                {    
                    control = 1;

                    MPI_Send(&control, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                    MPI_Send(&control, 1, MPI_INT, i+1, 2, MPI_COMM_WORLD);
                    
                    // Record histogram for ensemble optimization:
                    s_i = check_sign_i(i, np_,  sign_[i], sign_[i+1]);
                    s_i_plus_1 = check_sign_i_plus_1(i, np_,  sign_[i], sign_[i+1]);
                    sign_[i] = s_i;
                    sign_[i+1] = s_i_plus_1;
                    if (counts < sweeps + therm - 100) // Start recording after a couple of steps ...
                    {
                        if (s_i == 1)
                            nplus_[i] += 1;
                        else if (s_i = -1)  
                            nminus_[i] += 1;
                        if (s_i_plus_1 == 1)
                            nplus_[i+1] += 1;
                        else if (s_i_plus_1 == -1)
                            nminus_[i+1] += 1;
                    }

                    // Record replica exchange rate:
                    counter[i] += 1;
                    counter[i+1] += 1;
                
                }
                else // refuse
                {
                    //std::cout << "NO SWAP " << i << " " << i + 1 << std::endl;
                    control = 0;
                    MPI_Send(&control, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                    MPI_Send(&control, 1, MPI_INT, i+1, 2, MPI_COMM_WORLD);
                }  
            }   

            if (counts % sweeps_per_save == 0)
            {
                // Give single flip acceptance rates as output ...
                std::ofstream sf__(sfar.str().c_str(), std::ofstream::trunc);
                for (int jj = 1; jj < np_; jj++)
                {
                    MPI_Recv(&fr, 1, MPI_DOUBLE, jj, 5, MPI_COMM_WORLD, &status);
                    sf__ << std::setprecision(17) << calc_temp(T_min, T_max, jj, np_, dist) << " " << fr << std::endl;
                }


                // Give histogram as output ...
                std::ofstream nplus__(nplus_ratio.str().c_str(), std::ofstream::trunc);
                for (int kk = 1; kk < np_ ; kk++)
                {
                    nplus__ << std::setprecision(17) << calc_temp(T_min, T_max, kk, np_, dist) << " " << nplus_[kk] / double(nplus_[kk] + nminus_[kk]) << std::endl;
                }

                // Give swap ratio as output ...
                std::ofstream swap_ratio__(swap_ratio.str().c_str(), std::ofstream::trunc);
                for (int ll = 1; ll < np_ ; ll++)
                {
                    swap_ratio__ << std::setprecision(17) << calc_temp(T_min, T_max, ll, np_, dist) << " " << counter[ll] / den << std::endl;
                }

                // Save random generator status:
                saved = eng_;
                std::ofstream file(rng_save.str().c_str(), std::ofstream::trunc);
                file << saved;
            }    
        }

        // Give autocorrelation times as output ...
        std::ofstream tau1(tau_energy.str().c_str(), std::ofstream::trunc);
        std::ofstream tau2(tau_flux.str().c_str(), std::ofstream::trunc);
        for (int ll = 1; ll < np_; ll++)
        {
            MPI_Recv(&tau_en, 1, MPI_DOUBLE, ll, 6, MPI_COMM_WORLD, &status);
            tau1 << std::setprecision(17) << calc_temp(T_min, T_max, ll, np_, dist) << " " << tau_en << std::endl;

            MPI_Recv(&tau_fl, 1, MPI_DOUBLE, ll, 7, MPI_COMM_WORLD, &status);
            tau2 << std::setprecision(17) << calc_temp(T_min, T_max, ll, np_, dist) << " " << tau_fl << std::endl;
        }
    }


    void load(std::string const & filename)
    {
        alps::hdf5::archive ar(filename, "a");
        ar["/simulation/results/"+energy_.representation()] >> energy_;
        ar["/simulation/results/"+e2_.representation()] >> e2_;
        //ar["/simulation/results/"+e4_.representation()] >> e4_;
        ar["/simulation/results/"+dE_db.representation()] >> dE_db;
        ar["/simulation/results/"+p_.representation()] >> p_;
        ar["/simulation/results/"+flux_real.representation()] >> flux_real;
        ar["/simulation/results/"+flux_imag.representation()] >> flux_imag;
        ar["/simulation/results/"+flux_real_squared.representation()] >> flux_real_squared;
        //ar["/simulation/results/"+flux_imag_squared.representation()] >> flux_imag_squared;
        //ar["/simulation/results/"+flux2_real.representation()] >> flux2_real;
        //ar["/simulation/results/"+flux2_imag.representation()] >> flux2_imag;
        //ar["/simulation/results/"+flux2_real_squared.representation()] >> flux2_real_squared;
        //ar["/simulation/results/"+flux2_imag_squared.representation()] >> flux2_imag_squared;
        ar["/simulation/results/"+spin_corr.representation()] >> spin_corr;
        ar["/simulation/results/"+flux_corr.representation()] >> flux_corr;
        ar["/parameters/T"] >> temp_;
        ar["/parameters/BETA"] >> beta_;
        ar["/parameters/SWEEPS"] >> sweeps_;
        ar["/parameters/THERMALIZATION"] >> thermalization_;
    }
    
    void save(std::string const & filename)
    {        
        alps::hdf5::archive ar(filename, "a");
        ar["/simulation/results/"+energy_.representation()] << energy_;
        ar["/simulation/results/"+e2_.representation()] << e2_;
        //ar["/simulation/results/"+e4_.representation()] << e4_;
        ar["/simulation/results/"+dE_db.representation()] << dE_db;
        ar["/simulation/results/"+p_.representation()] << p_;
        ar["/simulation/results/"+flux_real.representation()] << flux_real;
        ar["/simulation/results/"+flux_imag.representation()] << flux_imag;
        ar["/simulation/results/"+flux_real_squared.representation()] << flux_real_squared;
        //ar["/simulation/results/"+flux_imag_squared.representation()] << flux_imag_squared;
        //ar["/simulation/results/"+flux2_real.representation()] << flux2_real;
        //ar["/simulation/results/"+flux2_imag.representation()] << flux2_imag;
        //ar["/simulation/results/"+flux2_real_squared.representation()] << flux2_real_squared;
        //ar["/simulation/results/"+flux2_imag_squared.representation()] << flux2_imag_squared;
        ar["/simulation/results/"+spin_corr.representation()] << spin_corr;
        ar["/simulation/results/"+flux_corr.representation()] << flux_corr;
        ar["/parameters/T"] << temp_;
        ar["/parameters/BETA"] << beta_;
        ar["/parameters/SWEEPS"] << sweeps_;
        ar["/parameters/THERMALIZATION"] << thermalization_;
    }

    /////////////////////////////////////////////////////////////////////
    // Randomize zz-entries of matrix (likewise: all nonzero entries): //
    /////////////////////////////////////////////////////////////////////
    
    sp_cx_mat randomize(sp_cx_mat ham) 
    {    
        int N_ = size(ham)[0];
        int die, coord1, coord2;

        for (int j = 0 ; j < 3*N_/2; j++)
        {
            die = roll_die(length_);
            coord1 = v_[die]/N_;
            coord2 = v_[die]%N_;
        
            // Flip a coin:
            if (rng_() < 0.5)
            {
                ham(coord1, coord2) *= -1;
                ham(coord2, coord1) *= -1;
            }
        }

        return ham;
    }
    
    ////////////////////////////////////////////////////////////////
    
    protected:
    
    // Random int from the interval [0,max)
    int roll_die(int max) const
    {
        return static_cast<int>(max * rng_());
    }

private:
    typedef boost::mt19937 engine_type; // Mersenne twister
    typedef boost::uniform_real<> distribution_type;
    typedef boost::variate_generator<engine_type&, distribution_type> rng_type;
    engine_type eng_;
    distribution_type dist_;
    mutable rng_type rng_;

    size_t sweeps_;
    size_t thermalization_;

    // Everything here is described above:
    int np_;
    int me_;
    double T_min;
    double T_max; 
    double temp_;
    double beta_;

    double s_;
    double ld;
    std::stringstream bw_save;
    double lambda_;
    int intsteps_;
    int M_;
    
    // Parameters for all integrations:
    double int_min;
    double int_max;
    double step_;
    double pi;
    double twopi;

    size_t N_;

    // Configuration in CRS Format:
    cx_vec val;
    vec col_idx;
    vec row_ptr;

    std::vector<int> v_;
    int length_;

    vec Del_rho;
    cx_mat y_values;
    vec tanh_values;
    vec d_values;

    vec kernel_factors;

    vec dr;
    cx_vec mom_i_plus_ij;
    cx_vec mom_i_plus_j;
    cx_vec mom_ii;
    cx_vec mom_jj;

    cx_vec mom;
    cx_vec i_;
    cx_vec j_2;
    cx_vec j_1;
    cx_vec j_;

    sp_cx_mat ham_sp;

    Mat<int> plaquettes;
    Mat<int> plaquettes2;

    // ALPS Observables:
    alps::RealObservable energy_;
    alps::RealObservable e2_;
    //alps::RealObservable e4_;
    alps::RealObservable dE_db;
    alps::RealObservable p_;
    alps::RealObservable flux_real;
    alps::RealObservable flux_imag;
    alps::RealObservable flux_real_squared;
    //alps::RealObservable flux_imag_squared;
    alps::RealObservable flux2_real;
    //alps::RealObservable flux2_imag;
    alps::RealObservable flux2_real_squared;
    //alps::RealObservable flux2_imag_squared;
    alps::RealObservable spin_corr;
    alps::RealObservable flux_corr;

    int flip_rate;

    signed int sign;

    std::string dist;
    std::string filename_;
};
