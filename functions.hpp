//////////////////////////////////////////////////////////////////////////////
//  Observable functions for QMC-KPM Simulation
//  written by: Tim Eschmann, May 2017
//  Modified version: February 2019
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
   
#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>

using namespace arma;


///////////////////////////////////////////////////////////////
// Calculate free energy for a given Z2 configuration:
///////////////////////////////////////////////////////////////

double free_en(vec ev, double b)
{
    int p; // running index
    double fe = 0; // free energy
    
    // Calculating:
    for (p = 0; p < size(ev)[0]/2; p++)
    {
        fe -= (1/b)*logl(2*coshl(b*ev[p]/2));
        //fe -= 0.5*b*ev[p] - logl(1 + exp(-b * ev[p]));      
    }

    return fe;
       
}
    
///////////////////////////////////////////////////////////////
// Calculate internal energy of the Majorana fermion system:
///////////////////////////////////////////////////////////////

double en(vec ev, double b)
{
    int q; // running index;
    double en = 0; // internal energy
    
    for (q = 0; q < size(ev)[0]/2; q++)
    {
        en += ev[q]/2 * tanhl(-b * ev[q]/2);
    }
    
    return en;
}

///////////////////////////////////////////////////////////////
// Calculate derivative of E w.r.t. beta:
///////////////////////////////////////////////////////////////

double diffE(vec ev, double b)
{
    int qq; // running index
    double drv = 0;
    for (qq = 0; qq < size(ev)[0]/2; qq++)
    {
        //drv += ev[qq]*ev[qq]/4. * (1 - tanh(-beta_ * ev[qq]/2)*tanh(-beta_ * ev[qq]/2));
        drv += pow(ev[qq], 2)/4. / pow((coshl(-b * ev[qq]/2)),2);
    }

    return drv;
}

///////////////////////////////////////////////////////////////
// Calculate average flux per elementary plaquette:
///////////////////////////////////////////////////////////////

std::complex <double> flux(sp_cx_mat ham, Mat<int> plaq)
{
    std::complex <double> flux;
    std::complex <double> av_flux = std::complex<double>(0.0, 0.0);
    std::complex <double> fl;
    std::complex <double> abs_fl;
    int N = size(ham)[0];
    int M1 = size(plaq)[0];
    int M2 = size(plaq)[1];
    int i,j;
    int coord1, coord2;
        
    for (i = 0; i < M1; i++)
    {
        flux = std::complex<double>(1.0, 0.0);
        for (j = 0; j < M2; j++)
        {
            coord1 = plaq(i,j) / N;
            coord2 = plaq(i,j) % N;
            
            // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
            fl = ham(coord1, coord2);
            abs_fl = std::abs(fl);
            flux *= -fl / abs_fl;
        }
        av_flux += flux;
    }

    av_flux /= double(M1);

    return av_flux;
        
}

///////////////////////////////////////////////////////////////
// Give flux configurations as output:
///////////////////////////////////////////////////////////////
cx_vec flux_confs(sp_cx_mat ham, Mat<int> plaq)
{
    int M1 = size(plaq)[0];
    int M2 = size(plaq)[1];
    int N = size(ham)[0];
    int coord1, coord2;
    cx_vec confs(M1);
    std::complex <double> flux, fl, abs_fl;

    for (int i = 0; i < M1; i++)
    {
        flux = std::complex<double>(1.0, 0.0);
        for (int j = 0; j < M2; j++)
        {
            coord1 = plaq(i,j) / N;
            coord2 = plaq(i,j) % N;
            
            // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
            fl = ham(coord1, coord2);
            abs_fl = std::abs(fl);
            flux *= -fl / abs_fl;
        }

        confs[i] = flux;
    }

    return confs;

} 

///////////////////////////////////////////////////////////////
// Calculate flux disorder ratio p:
///////////////////////////////////////////////////////////////

double get_p(sp_cx_mat ham, Mat<int> plaq)
{
    std::complex <double> flux;
    std::complex <double> fl;
    std::complex <double> abs_fl;
    double p;
    int N = size(ham)[0];
    int M1 = size(plaq)[0];
    int M2 = size(plaq)[1];
    int coord1, coord2;
        
    for (int i = 0; i < M1; i++)
    {
        flux = std::complex<double>(1.0, 0.0);
        for (int j = 0; j < M2; j++)
        {
            coord1 = plaq(i,j) / N;
            coord2 = plaq(i,j) % N;
            
            // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
            fl = ham(coord1, coord2);
            abs_fl = std::abs(fl);
            flux *= -fl / abs_fl;
        }
        if ((M2 % 4 == 0) && (flux == std::complex<double>(-1, 0)))
            p += 1;
        else if ((M2 % 4 == 2) && (flux == std::complex<double>(1, 0)))
            p += 1;
        else if ((M2 % 2 != 0) && (flux == std::complex<double>(0, 1)))
            p += 1;
    }

    p /= double(M1);

    return p;
        
}

///////////////////////////////////////////////////////////////
// Calculate spin-spin correlation:
// Make sure only to consider one subset of bonds!
///////////////////////////////////////////////////////////////

double correlation(sp_cx_mat ham_sp, std::vector<int> v_, double b)
{
    double value = 0.;
    std::complex<double> av;
    cx_mat ham(ham_sp);
    vec eigval;
    cx_mat eigvec;
    int N = size(ham)[0];
    int coord1, coord2;
      
    eig_sym(eigval, eigvec, ham);
    //eigvec = normalise(eigvec);

    // Iterate over zz-bonds:
    for (int j = 0; j < v_.size(); j++)
    {
        coord1 = v_[j]/N;
        coord2 = v_[j]%N;
        
        for (int i = size(eigval)[0]/2; i < size(eigval)[0]; i++)
        {
            av = conj(eigvec(coord1,i))*((-ham(coord1, coord2)) / std::abs(ham(coord1, coord2)))*eigvec(coord2, i);
            av += conj(eigvec(coord2,i))*((-ham(coord2, coord1)) / std::abs(ham(coord2, coord1)))*eigvec(coord1, i);
            value -= real(av)*tanhl(b * eigval[i]/2.);
        }
    }

    value *= 2/double(N);    
    return value;
        
}

///////////////////////////////////////////////////////////////
// Calculate flux-flux correlation:
///////////////////////////////////////////////////////////////

double flux_correlation(sp_cx_mat ham_sp, Mat<int> plaq)
{
    std::complex <double> flux_ref;
    std::complex <double> flux_check;
    cx_mat ham(ham_sp);
    double W_p_W_p_prime;
    double corr = 0;
    int N = size(ham)[0];
    int M1 = size(plaq)[0];
    int M2 = size(plaq)[1];
    int coord1, coord2;
        
    for (int i = 0; i < M1; i++)
    {
        flux_ref = std::complex<double>(1.0, 0.0);
        for (int j = 0; j < M2; j++)
        {
            coord1 = plaq(i,j) / N;
            coord2 = plaq(i,j) % N;
            
            // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
            flux_ref *= (-ham(coord1, coord2)) / std::abs(ham(coord1, coord2));
        }
            
        for (int k = 0; k < M1; k++)
        {
            flux_check = std::complex<double>(1.0, 0.0);

            for (int kk = 0; kk < M2; kk++)
            {
                coord1 = plaq(k,kk) / N;
                coord2 = plaq(k,kk) % N;
            
                // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
                flux_check *= (-ham(coord1, coord2)) / std::abs(ham(coord1, coord2));
            }

            W_p_W_p_prime = real(flux_ref)*real(flux_check);
            corr += W_p_W_p_prime;
        }        
    }

    corr /= double(M1*M1);

    return corr;
        
        
}

/////////////////////////////////////////////////////////
// Set up simulation temperatures (w. different options):
/////////////////////////////////////////////////////////  

double calc_temp(double T_min, double T_max, int me, int np, std::string dist)
{
    int i;
    double T;
        
    // a) Read temperature distribution from file:
    if (dist == "external")
    {
        double temperatures[np];
        std::ifstream tempfile("temp.saved", std::ifstream::in);
        for (i = 0; i < np; i++)
        {
            tempfile >> temperatures[i];
        }

        T = temperatures[me - 1];
    }

    // b) Linear temperature distribution:
    else if (dist == "lin")
    {
        T = T_min + (T_max - T_min)*(me - 1)/float(np);
    }
    // c) Logarithmic temperature distribution:*/    
    else if (dist == "log")
    {
        T = pow(10,log10(T_min)+((log10(T_max) - log10(T_min))*(me - 1)/double(np)));
    }
    // d) Double-logarithmic temperature distribution:
    else if (dist == "double_log")
    {
        T = pow(10,log10(T_min)+((log10(T_max) - log10(T_min))*pow(10, -(np - (me - 1))/double(np))));
    }

    return T;


}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//  Ensemble optimization functions 
//  -> check if MC replica has been at T_min or T_max latest and assign
// "+1" or "-1" accordingly 
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Record if replica has been at lowest or highest T latest ...      ////////
/////////////////////////////////////////////////////////////////////////////

int check_sign_i(int i, int np, int s_i, int s_i_plus_1)
{
    int sign;
    
    if (i != 1 && (i+1 != (np - 1)))
    {
        if (s_i == 1 && s_i_plus_1 == 0) 
            sign = 0;
        else if (s_i == 0 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == 1 && s_i_plus_1 == -1)
            sign = -1;
        else if (s_i == -1 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == 0 && s_i_plus_1 == -1)
            sign = -1;
        else if (s_i == -1 && s_i_plus_1 == 0)
            sign = 0;     
        else if (s_i == 0 && s_i_plus_1 == 0)
            sign = 0;
        else if (s_i == 1 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == -1 && s_i_plus_1 == -1)
            sign = -1;
    }
    else if (i == 1)
        sign = 1;
    else if (i+1 == np - 1)
        sign = -1;
    else
        sign = 0;

    return sign;

}

int check_sign_i_plus_1(int i, int np, int s_i, int s_i_plus_1)
{
    int sign;
    
    if (i != 1 && (i+1 != (np - 1)))
    {
        if (s_i == 1 && s_i_plus_1 == 0)   
            sign = 1;
        else if (s_i == 0 && s_i_plus_1 == 1)
            sign = 0;
        else if (s_i == 1 && s_i_plus_1 == -1)
            sign = 1;
        else if (s_i == -1 && s_i_plus_1 == 1)
            sign = -1;
        else if (s_i == 0 && s_i_plus_1 == -1)
            sign = 0;
        else if (s_i == -1 && s_i_plus_1 == 0)
            sign = -1;     
        else if (s_i == 0 && s_i_plus_1 == 0)
            sign = 0;
        else if (s_i == 1 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == -1 && s_i_plus_1 == -1)
            sign = -1;  
    }
    else if (i == 1)
        sign = 1;
    else if (i+1 == np - 1)
        sign = -1;
    else 
        sign = 0;

    return sign;
}

    
/////////////////////////////////////////////////////////
// Create block vectors for CRS matrix representation: //
/////////////////////////////////////////////////////////

cx_vec get_val(sp_cx_mat ham)
{
    cx_vec v;
    
    // Be able to read in ground state configuration:
    std::stringstream matrix_output; 
    matrix_output << "matrix_gs.saved"; 
    
    // If ground state configuration exists in this folder, read it in
    // Otherwise, construct it from hpp-file
    std::ifstream matfile(matrix_output.str().c_str(), std::ifstream::in);
    if(matfile.good())
    {
        cx_mat A(size(ham)[0], size(ham)[0]);
        A.load(matrix_output.str().c_str());
        v = nonzeros(trans(A));
        std::cout << "INFO: Read matrix ground state configuration from file!" << std::endl;
    }
    else
    {
        // Warning: matrix has to be transposed before getting 
        // nonzero-values vector!
        v = nonzeros(trans(ham));
    }
    
   
    return v;
}

vec get_col_idx(sp_cx_mat ham)
{
    int N = size(ham)[0];
    int k = 0;
    std::complex <double> value;

    cx_vec val = nonzeros(trans(ham)); // (!)
    vec col_idx(size(val)[0]);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            value = ham(i,j);
            if (std::imag(value) != 0)
            {
                col_idx(k) = j;
                k += 1;
            }
        }
    }

    return col_idx;
    
}

vec get_row_ptr(sp_cx_mat ham)
{
    int N = size(ham)[0];
    int k = 0;
    int kk = 1;
    std::complex <double> value;

    cx_vec val = nonzeros(trans(ham)); // (!)
    vec col_idx(size(val)[0]);
    vec row_ptr(N+1);
    row_ptr(0) = 0;
    row_ptr(N) = size(col_idx)[0];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            value = ham(i,j);
            if (std::imag(value) != 0)
            {
                col_idx(k) = j;
                k += 1;
            }
        }
        row_ptr(kk) = k;
        kk += 1;
    }

    return row_ptr;

}

/////////////////////////////////////////////////////////////
// Rebuild sparse matrix from CRS vectors and diagonalize: //
/////////////////////////////////////////////////////////////

sp_cx_mat get_sparse(cx_vec val, vec col_idx, vec row_ptr)
{
    sp_cx_mat temp(size(row_ptr)[0] - 1, size(row_ptr)[0] - 1);
    int idx;

    for (int i = 0; i < size(row_ptr)[0] - 1; i++)
    {
        for (int j = row_ptr(i); j < row_ptr(i+1); j++)
        {
            temp(i, col_idx(j)) = val(j);
        }
    }

    return temp;

}


//////////////////////////////////////////////////////
// Rebuild matrix from CRS vectors and diagonalize: //
//////////////////////////////////////////////////////

vec get_evals(cx_vec val, vec col_idx, vec row_ptr)
{
    cx_mat temp(size(row_ptr)[0] - 1, size(row_ptr)[0] - 1, fill::zeros);
    int idx;

    for (int i = 0; i < size(row_ptr)[0] - 1; i++)
    {
        for (int j = row_ptr(i); j < row_ptr(i+1); j++)
        {
            temp(i, col_idx(j)) = val(j);
        }
    }
    
    vec eval = eig_sym(temp);

    return eval;

}

///////////////////////////////////////////
// Access CRS element: ////////////////////
///////////////////////////////////////////

int get_idx(vec col_idx, vec row_ptr, int i, int j)
{
    // Be careful with zero elements here:
    // Initialize idx s.t. simulation.hpp recognizes
    // if matrix element is zero:

    int idx = size(col_idx)[0] + 1;
    
    // Check for element at coordinates (i,j):
    for (int k = row_ptr(i); k < row_ptr(i+1); k++)
    {
        if (col_idx(k) == j)
            idx = k;
    }

    return idx;
}

///////////////////////////////////////////
// Complex matrix-vector multiplication: //
///////////////////////////////////////////

cx_vec compl_mat_vec_multiply(cx_vec val, vec col_idx, vec row_ptr, cx_vec x)
{
    cx_vec y(size(x)[0], fill::zeros);

    // Usual procedure for CRS-format matrices: 
    for (int i = 0; i < size(x)[0]; i++)
    {
        for (int j = row_ptr(i); j < row_ptr(i+1); j++)
        {
            y(i) += val(j) * x(col_idx(j));
        }
    }
    
    return y;

}


