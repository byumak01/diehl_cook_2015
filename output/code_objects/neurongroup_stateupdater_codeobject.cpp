#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace {
        
    static inline int64_t _timestep(double t, double dt)
    {
        return (int64_t)((t + 1e-3*dt)/dt);
    }
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int32_t,int32_t> { typedef int32_t type; };
    template < > struct _higher_type<int32_t,int64_t> { typedef int64_t type; };
    template < > struct _higher_type<int32_t,float> { typedef float type; };
    template < > struct _higher_type<int32_t,double> { typedef double type; };
    template < > struct _higher_type<int32_t,long double> { typedef long double type; };
    template < > struct _higher_type<int64_t,int32_t> { typedef int64_t type; };
    template < > struct _higher_type<int64_t,int64_t> { typedef int64_t type; };
    template < > struct _higher_type<int64_t,float> { typedef float type; };
    template < > struct _higher_type<int64_t,double> { typedef double type; };
    template < > struct _higher_type<int64_t,long double> { typedef long double type; };
    template < > struct _higher_type<float,int32_t> { typedef float type; };
    template < > struct _higher_type<float,int64_t> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<float,long double> { typedef long double type; };
    template < > struct _higher_type<double,int32_t> { typedef double type; };
    template < > struct _higher_type<double,int64_t> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < > struct _higher_type<double,long double> { typedef long double type; };
    template < > struct _higher_type<long double,int32_t> { typedef long double type; };
    template < > struct _higher_type<long double,int64_t> { typedef long double type; };
    template < > struct _higher_type<long double,float> { typedef long double type; };
    template < > struct _higher_type<long double,double> { typedef long double type; };
    template < > struct _higher_type<long double,long double> { typedef long double type; };
    // General template, used for floating point types
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {
        return x-y*floor(1.0*x/y);
    }
    // Specific implementations for integer types
    // (from Cython, see LICENSE file)
    template <>
    inline int32_t _brian_mod(int32_t x, int32_t y)
    {
        int32_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int32_t x, int64_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int64_t x, int32_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int64_t x, int64_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    // General implementation, used for floating point types
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    // Specific implementations for integer types
    // (from Cython, see LICENSE file)
    template <>
    inline int32_t _brian_floordiv<int32_t, int32_t>(int32_t a, int32_t b) {
        int32_t q = a / b;
        int32_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int32_t, int64_t>(int32_t a, int64_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int64_t, int>(int64_t a, int32_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int64_t, int64_t>(int64_t a, int64_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif

}

////// HASH DEFINES ///////



void _run_neurongroup_stateupdater_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const double E_exc_for_exc = 0.0;
const double E_inh_for_exc = - 0.1;
const double E_rest_exc = - 0.065;
const int64_t N = 400;
const size_t _numdt = 1;
const size_t _numg_e = 400;
const size_t _numg_i = 400;
const size_t _numlastspike = 400;
const size_t _numnot_refractory = 400;
const size_t _numt = 1;
const double tau_ge = 0.001;
const double tau_gi = 0.002;
const double tau_lif_exc = 0.1;
const double tau_theta = 10000.0;
const size_t _numtheta = 400;
const size_t _numv = 400;
    ///// POINTERS ////////////
        
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double* __restrict  _ptr_array_neurongroup_g_e = _array_neurongroup_g_e;
    double* __restrict  _ptr_array_neurongroup_g_i = _array_neurongroup_g_i;
    double* __restrict  _ptr_array_neurongroup_lastspike = _array_neurongroup_lastspike;
    char* __restrict  _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    double* __restrict  _ptr_array_neurongroup_theta = _array_neurongroup_theta;
    double* __restrict  _ptr_array_neurongroup_v = _array_neurongroup_v;


    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double t = _ptr_array_defaultclock_t[0];
    const int64_t _lio_1 = _timestep(0.005, dt);
    const double _lio_2 = 1.0f*(- dt)/tau_ge;
    const double _lio_3 = 1.0f*(- dt)/tau_gi;
    const double _lio_4 = 1.0f*(- dt)/tau_theta;
    const double _lio_5 = 1.0f*dt/tau_lif_exc;


    const int _N = N;
    
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
                
        double g_e = _ptr_array_neurongroup_g_e[_idx];
        double g_i = _ptr_array_neurongroup_g_i[_idx];
        const double lastspike = _ptr_array_neurongroup_lastspike[_idx];
        char not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
        double theta = _ptr_array_neurongroup_theta[_idx];
        double v = _ptr_array_neurongroup_v[_idx];
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        const double _g_e = (_lio_2 * g_e) + g_e;
        const double _g_i = (_lio_3 * g_i) + g_i;
        const double _theta = (_lio_4 * theta) + theta;
        double _v;
        if(!not_refractory)
            _v = v;
        else 
            _v = (_lio_5 * ((E_rest_exc + ((g_e * (E_exc_for_exc - v)) + (g_i * (E_inh_for_exc - v)))) - v)) + v;
        g_e = _g_e;
        g_i = _g_i;
        theta = _theta;
        if(not_refractory)
            v = _v;
        _ptr_array_neurongroup_g_e[_idx] = g_e;
        _ptr_array_neurongroup_g_i[_idx] = g_i;
        _ptr_array_neurongroup_not_refractory[_idx] = not_refractory;
        _ptr_array_neurongroup_theta[_idx] = theta;
        _ptr_array_neurongroup_v[_idx] = v;

    }

}


