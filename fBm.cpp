/****************************************************************************************

    Spectral representation of the fractional Brownian
    motion kernel using Legendre polynommials.

    Gadzhirasul Budaichiev (student-programmer@mail.ru): Programming;
    Konstantin Rybakov (rkoffice@mail.ru): Methodology, Programming.

    This program is the supplement to articles:
    Rybakov K. "Spectral Representation and Simulation of Fractional Brownian Motion"
    (Computation 2025, 13(1), 19; https://doi.org/10.3390/computation13010019),
    Rybakov K.A. "On approximate representation of fractional Brownian motion"
    (Methodology and Computing in Applied Probability 2025, 27(4), 88;
    https://doi.org/10.1007/s11009-025-10219-w).

****************************************************************************************/

// Libraries for streams
#include <iostream>
#include <fstream>
#include <sstream>
// Boost.Multiprecision library
#include <boost/multiprecision/cpp_dec_float.hpp>

// Recomended precisions:
//    96 for L = 128
//   192 for L = 256
//   384 for L = 512
//   768 for L = 1024
const int mp_precision = 375;

namespace MA = boost::math;
namespace MP = boost::multiprecision;

// High precision types
typedef MP::number<MP::cpp_dec_float<mp_precision / 4>> real1;
typedef MP::number<MP::cpp_dec_float<mp_precision / 2>> real2;
typedef MP::number<MP::cpp_dec_float<mp_precision * 5 / 8>> real3;
typedef MP::number<MP::cpp_dec_float<mp_precision * 3 / 4>> real4;
typedef MP::number<MP::cpp_dec_float<mp_precision * 13 / 16>> real5;
typedef MP::number<MP::cpp_dec_float<mp_precision * 7 / 8>> real6;
typedef MP::number<MP::cpp_dec_float<mp_precision * 15 / 16>> real7;
typedef MP::number<MP::cpp_dec_float<mp_precision>> real8;

// Default high precision type
typedef real8 real; 

// Vectors and matrices
typedef std::vector<double> double1D;
typedef std::vector<double1D> double2D;
typedef std::vector<double2D> double3D;
typedef std::vector<double3D> double4D;

#define _HIGH_PRECISION_FOR_Aalpha
#define _HIGH_PRECISION_FOR_Pbeta

// Coefficient a_H
double compute_aH(double H)
{
    return sqrt(2 * H * MA::tgamma(H + 0.5) * MA::tgamma(1.5 - H) / MA::tgamma(2 - 2 * H));
}

// Spectral representation of the function t^alpha
double1D compute_Falpha(double T, double alpha, int L)
{
    std::vector<real> F(L);
    double1D R(L);
    F[0] = 1 / real(alpha + 1);
    R[0] = pow(T, alpha + 0.5) / (alpha + 1);
    for (int i = 0; i < L - 1; ++i)
    {
        F[i + 1] = F[i] * sqrt(real(2 * i + 3) / real(2 * i + 1)) * (real(alpha) - i) / (real(alpha) + (i + 2));
        R[i + 1] = pow(T, alpha + 0.5) * F[i + 1].convert_to<double>();
    }
    return R;
}

#ifdef HIGH_PRECISION_FOR_Aalpha

// Spectral representation of the multiplication operator with multiplier t^alpha
double2D compute_Aalpha(double T, double alpha, int L)
{
    double2D A(L, double1D(L));
    double1D F = compute_Falpha(T, alpha, L);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j <= i && j < L; ++j)
        {
            real Pi = 1;
            real Sum = 0;
            for (int k = 0; k <= j; ++k)
            {
                if (k > 0)
                {   /* ver.1: native
                    Pi *= pow((alpha + real(k)) / k, 2)
                        * real(j - k + 1) / (alpha - real(i - k))
                        * real(j + k) / (alpha + real(i + k + 1)); */
                    /* ver.2
                    Pi *= pow(alpha + real(k), 2) * real((j - k + 1) * (j + k))
                        / ((k * k) * (alpha - real(i - k)) * (alpha + real(i + k + 1))); */
                    /* ver.3 */
                    real ak = alpha + k;
                    real aks = pow(alpha + k, 2);
                    Pi *= aks * real((j - k + 1) * (j + k))
                        / ((k * k) * (aks + ak - real(i * (i + 1))));
                }
                if ((j - k) % 2)
                    Sum -= Pi;
                else
                    Sum += Pi;
            }
            A[i][j] = sqrt((2 * j + 1) / T) * F[i] * Sum.convert_to<double>();
            if (i != j)
                A[j][i] = A[i][j];
        }
    }

    return A;
}

#else

// Spectral representation of the multiplication operator with multiplier t^alpha
double2D compute_Aalpha(double T, double alpha, int L)
{
    double2D A(L, double1D(L));
    double1D F = compute_Falpha(T, alpha, L);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j <= i && j < L; ++j)
        {
            double Pi = 1;
            double Sum = 0;
            for (int k = 0; k <= j; ++k)
            {
                if (k > 0)
                {   /* ver.1: native
                    Pi *= pow((alpha + k) / k, 2)
                        * ((j - k + 1) / (alpha - i + k))
                        * ((j + k) / (alpha + i + k + 1)); */
                    /* ver.2 */
                    double ak = alpha + k;
                    double aks = pow(ak, 2);
                    Pi *= aks * ((j - k + 1) * (j + k)) / (k * k * (aks + ak - (i * (i + 1))));
                }
                if ((j - k) % 2)
                    Sum -= Pi;
                else
                    Sum += Pi;
            }
            A[i][j] = sqrt((2 * j + 1) / T) * F[i] * Sum;
            if (i != j)
                A[j][i] = A[i][j];
        }
    }

    return A;
}

#endif

#ifdef HIGH_PRECISION_FOR_Pbeta

// Spectral representation of the integration operator of fractional order beta
double2D compute_Pbeta(double T, double beta, int L)
{
    double2D P(L, std::vector<double>(L));

    if (beta == 1)
    {
        P[0][0] = T / 2;
        for (int i = 1; i < L; ++i)
        {
            double denom = 2 * sqrt(4 * i * i - 1);
            P[i][i - 1] = T / denom;
            P[i - 1][i] = -T / denom;
        }
    }
    else
    {
        double1D F = compute_Falpha(T, beta, L);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j <= i && j < L; ++j)
            {
                real Pi = 1;
                real sum = 0;
                for (int k = 0; k <= j; ++k)
                {
                    if (k > 0)
                    {   /* ver.1: native
                        Pi *= (beta + real(k)) / real(k)
                            * real(j - k + 1) / (beta - real(i - k))
                            * real(j + k) / (beta + real(i + k + 1)); */
                        /* ver.2
                        Pi *= (beta + real(k)) * real((j - k + 1) * (j + k))
                            / (k * (beta - real(i - k)) * (beta + real(i + k + 1))); */
                        /* ver.3 */
                        real bk = beta + k;
                        Pi *= bk * real((j - k + 1) * (j + k))
                            / (k * (bk - real(i)) * (bk + real(i + 1)));
                        /* ver.3+
                        real bk = beta + k;
                        real bks = pow(beta + k, 2);
                        Pi *= bk * real((j - k + 1) * (j + k))
                            / (k * (bks + bk - real(i * (i + 1)))); */
                    }
                    if ((j - k) % 2)
                        sum -= Pi;
                    else
                        sum += Pi;
                }
                P[i][j] = sqrt((2 * j + 1) / T) * F[i] * sum.convert_to<double>() / tgamma(beta + 1);
                if (i != j)
                    P[j][i] = ((i + j) % 2 ? -P[i][j] : P[i][j]);
            }
        }
    }

    return P;
}

#else

// Spectral representation of the integration operator of fractional order beta
double2D compute_Pbeta(double T, double beta, int L)
{
    double2D P(L, double1D(L));

    if (beta == 1)
    {
        P[0][0] = T / 2;
        for (int i = 1; i < L; ++i)
        {
            double denom = 2 * sqrt(4 * i * i - 1);
            P[i][i - 1] = T / denom;
            P[i - 1][i] = -T / denom;
        }
    }
    else
    {
        double1D F = compute_Falpha(T, beta, L);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j <= i && j < L; ++j)
            {
                double Pi = 1;
                double sum = 0;
                for (int k = 0; k <= j; ++k)
                {
                    if (k > 0)
                    {   /* ver.1: native
                        Pi *= ((beta + k) / k)
                            * ((j - k + 1) / (beta - i + k))
                            * ((j + k) / (beta + i + k + 1)); */
                        /* ver.2 */
                        double bk = beta + k;
                        Pi *= bk * ((j - k + 1) * (j + k)) / (k * (bk - i) * (bk + i + 1));
                    }
                    if ((j - k) % 2)
                        sum -= Pi;
                    else
                        sum += Pi;
                }
                P[i][j] = sqrt((2 * j + 1) / T) * F[i] * sum / MA::tgamma(beta + 1);
                if (i != j)
                    P[j][i] = ((i + j) % 2 ? -P[i][j] : P[i][j]);
            }
        }
    }

    return P;
}

#endif

// Auxilary function for spectral representation of the fractional Brownian motion kernel
template<typename Type> double compute_preKHij(double H, int i, int j)
{
    Type H1 = H + 0.5;
    Type Pi = 1;
    Type Sum = 0;
    for (int k = 0; k <= j; ++k)
    {
        Type H1k = H1 + k;
        if (k > 0)
        {   /* ver.1: native
            Pi *= pow(H1 + k, 2)
                * (k - H1) / pow(k, 3)
                * Type(j - k + 1) / (H1 - Type(i - k))
                * Type(j + k) / (H1 + Type(i + k + 1)); */
            /* ver.2
            Pi *= pow(H1 + k, 2) * (k - H1) * Type((j - k + 1) * (j + k))
                / ((k * k * k) * (H1 - Type(i - k)) * (H1 + Type(i + k + 1))); */
            /* ver.3 */
            Type H1ks = pow(H1k, 2);
            Pi *= H1ks * (k - H1) * Type((j - k + 1) * (j + k))
                / ((k * k * k) * (H1ks + H1k - Type(i * (i + 1))));
        }
        if ((j - k) % 2)
            Sum -= Pi * ((k + 1) - H1) / H1k;
        else
            Sum += Pi * ((k + 1) - H1) / H1k;
    }
    return Sum.convert_to<double>();
}

// Auxilary function for spectral representation of the fractional Brownian motion kernel
double compute_preKHij_fast(double H, int i, int j)
{
    double H1 = H + 0.5;
    double Pi = 1;
    double Sum = 0;
    for (int k = 0; k <= j; ++k)
    {
        double H1k = H1 + k;
        if (k > 0)
        {   /* ver.1: native
            Pi *= pow(H1 + k, 2) * ((k - H1) / pow(k, 3))
                * ((j - k + 1) / (H1 - (i - k)))
                * ((j + k) / (H1 + i + k + 1)); */
            /* ver.2 */
            double H1ks = pow(H1k, 2);
            Pi *= H1ks * (k - H1) * ((j - k + 1) * (j + k))
                / ((k * k * k) * (H1ks + H1k - (i * (i + 1))));
        }
        if ((j - k) % 2)
            Sum -= Pi * ((k + 1) - H1) / H1k;
        else
            Sum += Pi * ((k + 1) - H1) / H1k;
    }
    return Sum;
}

// Spectral representation of the fractional Brownian motion kernel (exact)
double2D compute_KH(double T, double H, int L)
{
    if (H == 0.5)
        return compute_Pbeta(T, 1, L);

    double2D K(L, double1D(L));
    double1D F = compute_Falpha(T, H + 0.5, L);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < L; ++i)
    {   /*
        if (i == L / 4 - 1) std::cout << "25%..";
        if (i == L / 2 - 1) std::cout << "50%..";
        if (i == 3 * L / 4 - 1) std::cout << "75%..";
        if (i == L - 1) std::cout << "100%\n"; */
        for (int j = 0; j < L; ++j)
        {
            double Sum;
            if (i - j >= 0) Sum = compute_preKHij_fast(H, i, j);
            else
                if (L <= 64) Sum = compute_preKHij<real>(H, i, j);
                else
                    if (j < L / 4) Sum = compute_preKHij<real1>(H, i, j);
                    else
                        if (j < L / 2) Sum = compute_preKHij<real2>(H, i, j);
                        else
                            if (j < 5 * L / 8) Sum = compute_preKHij<real3>(H, i, j);
                            else
                                if (j < 3 * L / 4) Sum = compute_preKHij<real4>(H, i, j);
                                else 
                                    if (j < 13 * L / 16) Sum = compute_preKHij<real5>(H, i, j);
                                    else
                                        if (j < 7 * L / 8) Sum = compute_preKHij<real6>(H, i, j);
                                        else
                                            if (j < 15 * L / 16) Sum = compute_preKHij<real7>(H, i, j);
                                            else 
                                                Sum = compute_preKHij<real8>(H, i, j);
                 /* if (j < L / 4)
                        Sum = compute_preKHij<real1>(H, i, j);
                    if (j >= L / 4 && j < L / 2)
                        Sum = compute_preKHij<real2>(H, i, j);
                    if (j >= L / 2 && j < 5 * L / 8)
                        Sum = compute_preKHij<real3>(H, i, j);
                    if (j >= 5 * L / 8 && j < 3 * L / 4)
                        Sum = compute_preKHij<real4>(H, i, j);
                    if (j >= 3 * L / 4 && j < 13 * L / 16)
                        Sum = compute_preKHij<real5>(H, i, j);
                    if (j >= 13 * L / 16 && j < 7 * L / 8)
                        Sum = compute_preKHij<real6>(H, i, j);
                    if (j >= 7 * L / 8 && j < 15 * L / 16)
                        Sum = compute_preKHij<real7>(H, i, j);
                    if (j >= 15 * L / 16)
                        Sum = compute_preKHij<real8>(H, i, j); */
            K[i][j] = compute_aH(H) * sqrt((2 * j + 1) / T) * MA::tgamma(0.5 - H) * F[i] * Sum;
        }
    }

    return K;
}

// Matrix multiplication
double2D matrix_multiplication(const double2D& A, const double2D& B)
{
    int L = (int)A.size();
    double2D C(L, double1D(L));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
        {
            double Sum = 0;
            for (int k = 0; k < L; ++k)
                Sum += A[i][k] * B[k][j];
            C[i][j] = Sum;
        }

    return C;
}

// Spectral representation of the fractional Brownian motion kernel (approximate)
double2D compute_tilde_KH(double T, double H, int L, bool flag)
{
    if (H == 0.5)
        return compute_Pbeta(T, 1, L);

    double aH = compute_aH(H);
    double2D P_1, A_1, P_2, A_2;

    if (!((H < 0.5) ^ (flag)))
    {
        P_1 = compute_Pbeta(T, 2 * H, L);
        A_1 = compute_Aalpha(T, 0.5 - H, L);
        P_2 = compute_Pbeta(T, 0.5 - H, L);
        A_2 = compute_Aalpha(T, H - 0.5, L);
    }
    else
    {
        P_1 = compute_Pbeta(T, 1, L);
        A_1 = compute_Aalpha(T, H - 0.5, L);
        P_2 = compute_Pbeta(T, H - 0.5, L);
        A_2 = compute_Aalpha(T, 0.5 - H, L);
    }

    double2D M = matrix_multiplication(
        matrix_multiplication(P_1, A_1), 
        matrix_multiplication(P_2, A_2)
    );

    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            M[i][j] *= aH;
    
    return M;
}

// Squared Euclidean norm for matrix
double squared_norm(const double2D& K)
{
    double Sum = 0;
    for (int i = 0; i < K.size(); ++i)
        for (int j = 0; j < K[0].size(); ++j)
            Sum += pow(K[i][j], 2);
    return Sum;
}

// Squared Euclidean norm for difference of matrices
double squared_norm(const double2D& K1, const double2D& K2)
{
    double Sum = 0;
    for (int i = 0; i < K1.size(); ++i)
        for (int j = 0; j < K1[0].size(); ++j)
            Sum += pow(K1[i][j] - K2[i][j], 2);
    return Sum;
}

// Squared norm of the fractional Brownian motion kernel
double theoretical_value(double T, double H)
{
    return pow(T, 2 * H + 1) / (2 * H + 1);
}

// Saving matrix to file
void save_matrix(const double2D& K, const std::string& filename, int precision)
{
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Cannot open file: " + filename);
    out << std::fixed << std::setprecision(precision);

    for (std::size_t i = 0; i < K.size(); ++i)
    {
        for (std::size_t j = 0; j < K[0].size(); ++j)
        {
            out << K[i][j];
            if (j + 1 < K[0].size())
                out << '\t';
        }
        out << '\n';
    }

    out.close();
    std::cout << "\n" << "The matrix has been successfully saved in file." << "\n";
}

// Spectral representations of the fractional Brownian motion kernel for given arrays of values ​​H and L
double4D compute_K_table(double T, const double1D& H_values, const std::vector<int>& L_values)
{
    double4D K_table(H_values.size(), double3D(L_values.size()));

    for (size_t i = 0; i < H_values.size(); ++i)
        for (size_t j = 0; j < L_values.size(); ++j)
            K_table[i][j] = compute_KH(T, H_values[i], L_values[j]);

    return K_table;
}
 
// Calculation of mean square approximation errors for given arrays of values ​​H and L
double2D compute_MSE_table(double T, const double1D& H_values, const std::vector<int>& L_values, const double4D& K_table)
{
    double2D MSE_table(H_values.size(), double1D(L_values.size()));

    for (size_t i = 0; i < H_values.size(); ++i)
        for (size_t j = 0; j < L_values.size(); ++j)
            MSE_table[i][j] = theoretical_value(T, H_values[i]) - squared_norm(K_table[i][j]);

    return MSE_table;
}

// Computing and displaying Table 1
void print_table_one(double T, const double1D& H_values, const std::vector<int>& L_values, const double4D& K_table,  int precision)
{
    std::cout << "\n" << "===| MSE 1 |===" << "\n";
    std::cout << "\033[1;32m" << "H \\ L";
    for (int L : L_values)
        std::cout << "\t\tL = " << L;
    std::cout << "\033[0m" << "\n";

    double2D MSE_table = compute_MSE_table(T, H_values, L_values, K_table);
    for (size_t i = 0; i < H_values.size(); ++i)
    {
        std::cout << "\033[1;33m" << H_values[i] << "\033[0m";
        for (size_t j = 0; j < L_values.size(); ++j)
            std::cout << "\t" << MSE_table[i][j];
        std::cout << "\n";
    }
}

// Computing and displaying Table 2
void print_table_two(double T, const double1D& H_values, const std::vector<int>& L_values, const double4D& K_table, int precision)
{
    std::cout << "\n" << "===| MSE 2 |===" << "\n";
    std::cout << "\033[1;32m" << "H \\ L";
    for (int L : L_values)
        std::cout << "\t\tL = " << L;
    std::cout << "\033[0m" << "\n";

    double2D MSE_table = compute_MSE_table(T, H_values, L_values, K_table);
    for (size_t i = 0; i < H_values.size(); ++i)
    {
        std::cout << "\033[1;33m" << H_values[i] << "\033[0m";
        for (size_t j = 0; j < L_values.size(); ++j)
        {
            double2D tilde_K = compute_tilde_KH(T, H_values[i], L_values[j], true);
            std::cout << "\t" << MSE_table[i][j] + squared_norm(K_table[i][j], tilde_K);
        }
        std::cout << "\n";
    }
}

// Computing and displaying Table 3
void print_table_three(double T, const double1D& H_values, const std::vector<int>& L_values, const double4D& K_table, int precision)
{
    std::cout << "\n" << "===| MSE 3 |===" << "\n";
    std::cout << "\033[1;32m" << "H \\ L";
    for (int L : L_values)
        std::cout << "\t\tL = " << L;
    std::cout << "\033[0m" << "\n";

    double2D MSE_table = compute_MSE_table(T, H_values, L_values, K_table);
    for (size_t i = 0; i < H_values.size(); ++i)
    {
        std::cout << "\033[1;33m" << H_values[i] << "\033[0m";
        for (size_t j = 0; j < L_values.size(); ++j)
        {
            double2D tilde_K = compute_tilde_KH(T, H_values[i], L_values[j], false);
            std::cout << "\t" << MSE_table[i][j] + squared_norm(K_table[i][j], tilde_K);
        }
        std::cout << "\n";
    }
}

// The main function
int main()
{
    time_t time1, time2;

    std::cout << "Spectral representation for fractional Brownian motion kernel." << "\n\n";
    std::cout << "[1] Calculate the single matrix and save it." << "\n";
    std::cout << "[2] Calculate MSE tables." << "\n\n" << "Choose mode: ";
    int mode;
    std::cin >> mode;

    if (!(mode == 1 || mode == 2))
    {
        std::cout << "Wrong mode! Next time pay attention.";
        exit(1);
    }

    std::cout << "\n" << "Enter output precision: ";
    int precision;
    std::cin >> precision;
    std::cout << std::fixed << std::setprecision(precision);

    std::cout << "\n" << "Enter T (for time interval [0,T]): ";
    double T;
    std::cin >> T;

    if (mode == 1)
    {
        std::cout << "Enter H, the Hurst index: ";
        double H;
        std::cin >> H;

        std::cout << "Enter L, the truncation order: ";
        int L;
        std::cin >> L;

        time(&time1);
        double2D K = compute_KH(T, H, L);
        time(&time2);

        std::cout << std::fixed << std::setprecision(0);
        std::cout << "\n" << "Spectral representation of the fractional Brownian motion kernel has been computed (" << difftime(time2, time1) << " sec)." << "\n";
        std::cout << std::fixed << std::setprecision(precision);

        double S_num = squared_norm(K);
        double S_th = theoretical_value(T, H);

        std::cout << "\n";
        std::cout << "Squared norm = " << S_num << "\n";
        std::cout << "Theory       = " << S_th << "\n";
        std::cout << "Difference   = " << S_th - S_num << "\n";

        save_matrix(K, "K_matrix.txt", precision);
    }
    else
    {
        std::cout << "Enter array of H (Hurst indices, e.g., 0.25 0.5 0.75): ";
        double1D H_values;
        std::string line;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::getline(std::cin, line);
        std::stringstream s1(line);

        double H;
        while (s1 >> H)
            H_values.push_back(H);
        
        std::cout << "Enter array of L (truncation orders, e.g., 32 64 128): ";
        std::vector<int> L_values;
        std::cin.clear();
        std::getline(std::cin, line);
        std::stringstream s2(line);

        int L;
        while (s2 >> L)
            L_values.push_back(L);

        time(&time1);
        double4D K_matrix = compute_K_table(T, H_values, L_values);
        print_table_one(T, H_values, L_values, K_matrix, precision);
        print_table_two(T, H_values, L_values, K_matrix, precision);
        print_table_three(T, H_values, L_values, K_matrix, precision);
        time(&time2);

        std::cout << std::fixed << std::setprecision(0);
        std::cout << "\n" << "The results have been tabulated (" << difftime(time2, time1) << " sec)." << "\n";
    }

    return 0;
}