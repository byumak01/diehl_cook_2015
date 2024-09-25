#include "code_objects/synapses_2_pre_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>
#include "brianlib/stdint_compat.h"
#include "synapses_classes.h"

////// SUPPORT CODE ///////
namespace {
        
    template <typename T>
    static inline T _clip(const T value, const double a_min, const double a_max)
    {
        if (value < a_min)
            return a_min;
        if (value > a_max)
            return a_max;
        return value;
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



void _run_synapses_2_pre_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    double* const _array_synapses_2_Apost1_ee = _dynamic_array_synapses_2_Apost1_ee.empty()? 0 : &_dynamic_array_synapses_2_Apost1_ee[0];
const size_t _numApost1_ee = _dynamic_array_synapses_2_Apost1_ee.size();
double* const _array_synapses_2_Apost2_ee = _dynamic_array_synapses_2_Apost2_ee.empty()? 0 : &_dynamic_array_synapses_2_Apost2_ee[0];
const size_t _numApost2_ee = _dynamic_array_synapses_2_Apost2_ee.size();
double* const _array_synapses_2_Apre_ee = _dynamic_array_synapses_2_Apre_ee.empty()? 0 : &_dynamic_array_synapses_2_Apre_ee[0];
const size_t _numApre_ee = _dynamic_array_synapses_2_Apre_ee.size();
int32_t* const _array_synapses_2__synaptic_pre = _dynamic_array_synapses_2__synaptic_pre.empty()? 0 : &_dynamic_array_synapses_2__synaptic_pre[0];
const size_t _num_synaptic_pre = _dynamic_array_synapses_2__synaptic_pre.size();
const double eta_pre_ee = 0.0001;
const size_t _numg_e_post = 400;
double* const _array_synapses_2_lastupdate = _dynamic_array_synapses_2_lastupdate.empty()? 0 : &_dynamic_array_synapses_2_lastupdate[0];
const size_t _numlastupdate = _dynamic_array_synapses_2_lastupdate.size();
const size_t _numt = 1;
const double tau_Apost1_ee = 0.02;
const double tau_Apost2_ee = 0.04;
const double tau_Apre_ee = 0.02;
double* const _array_synapses_2_w_ee = _dynamic_array_synapses_2_w_ee.empty()? 0 : &_dynamic_array_synapses_2_w_ee[0];
const size_t _numw_ee = _dynamic_array_synapses_2_w_ee.size();
const int64_t w_max_ee = 1;
const int64_t w_min_ee = 0;
int32_t* const _array_synapses_2__synaptic_post = _dynamic_array_synapses_2__synaptic_post.empty()? 0 : &_dynamic_array_synapses_2__synaptic_post[0];
const size_t _num_postsynaptic_idx = _dynamic_array_synapses_2__synaptic_post.size();
    ///// POINTERS ////////////
        
    double* __restrict  _ptr_array_synapses_2_Apost1_ee = _array_synapses_2_Apost1_ee;
    double* __restrict  _ptr_array_synapses_2_Apost2_ee = _array_synapses_2_Apost2_ee;
    double* __restrict  _ptr_array_synapses_2_Apre_ee = _array_synapses_2_Apre_ee;
    int32_t* __restrict  _ptr_array_synapses_2__synaptic_pre = _array_synapses_2__synaptic_pre;
    double* __restrict  _ptr_array_neurongroup_g_e = _array_neurongroup_g_e;
    double* __restrict  _ptr_array_synapses_2_lastupdate = _array_synapses_2_lastupdate;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    double* __restrict  _ptr_array_synapses_2_w_ee = _array_synapses_2_w_ee;
    int32_t* __restrict  _ptr_array_synapses_2__synaptic_post = _array_synapses_2__synaptic_post;



    // This is only needed for the _debugmsg function below

    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double t = _ptr_array_defaultclock_t[0];
    const double _lio_1 = 1.0f*1.0/tau_Apost1_ee;
    const double _lio_2 = 1.0f*1.0/tau_Apost2_ee;
    const double _lio_3 = 1.0f*1.0/tau_Apre_ee;
    const double _lio_4 = - eta_pre_ee;


    
    {
    std::vector<int> *_spiking_synapses = synapses_2_pre.peek();
    const int _num_spiking_synapses = _spiking_synapses->size();

    
    {
        for(int _spiking_synapse_idx=0;
            _spiking_synapse_idx<_num_spiking_synapses;
            _spiking_synapse_idx++)
        {
            const size_t _idx = (*_spiking_synapses)[_spiking_synapse_idx];
            const size_t _vectorisation_idx = _idx;
                        
            const int32_t _postsynaptic_idx = _ptr_array_synapses_2__synaptic_post[_idx];
            double Apost1_ee = _ptr_array_synapses_2_Apost1_ee[_idx];
            double Apost2_ee = _ptr_array_synapses_2_Apost2_ee[_idx];
            double Apre_ee = _ptr_array_synapses_2_Apre_ee[_idx];
            double g_e_post = _ptr_array_neurongroup_g_e[_postsynaptic_idx];
            double lastupdate = _ptr_array_synapses_2_lastupdate[_idx];
            double w_ee = _ptr_array_synapses_2_w_ee[_idx];
            const double _Apost1_ee = Apost1_ee * exp(_lio_1 * (- (t - lastupdate)));
            const double _Apost2_ee = Apost2_ee * exp(_lio_2 * (- (t - lastupdate)));
            const double _Apre_ee = Apre_ee * exp(_lio_3 * (- (t - lastupdate)));
            Apost1_ee = _Apost1_ee;
            Apost2_ee = _Apost2_ee;
            Apre_ee = _Apre_ee;
            Apre_ee = 1;
            w_ee = _clip(w_ee + (_lio_4 * Apost1_ee), w_min_ee, w_max_ee);
            g_e_post += w_ee;
            lastupdate = t;
            _ptr_array_synapses_2_Apost1_ee[_idx] = Apost1_ee;
            _ptr_array_synapses_2_Apost2_ee[_idx] = Apost2_ee;
            _ptr_array_synapses_2_Apre_ee[_idx] = Apre_ee;
            _ptr_array_neurongroup_g_e[_postsynaptic_idx] = g_e_post;
            _ptr_array_synapses_2_lastupdate[_idx] = lastupdate;
            _ptr_array_synapses_2_w_ee[_idx] = w_ee;

        }
    }
    }

}

void _debugmsg_synapses_2_pre_codeobject()
{
    using namespace brian;
    std::cout << "Number of synapses: " << _dynamic_array_synapses_2__synaptic_pre.size() << endl;
}

