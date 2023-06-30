#!/usr/bin/env python3

import math
import time
import numpy as np
from pprint import pprint
# import pandas as pd
import scipy
from tqdm import tqdm


###===== Functions =====###
exp = math.exp
ln = math.log
e = exp(1)
tolerance = 1e-8

def lambertw(x, tol=tolerance):
    return scipy.special.lambertw(x, k=0, tol=tol)

def logwright(x, N=None, tol=tolerance, seed=None):
    """Returns g(x)=log(W(exp(x))) where W(x) is the lambert W function.
    Reduce the value of 'tol' to get a better approximation of the root."""
    # Seed for Halley's method
    if seed is None:
        if x<=-e:
            y0 = x
        elif x>=e:
            y0 = ln(x)
        else:
            y0 = -e + (1+e)*(x+e)/(2*e)
    else:
        y0 = seed

    # Refine estimate
    #halley = lambda y0: y0 - 2*(y0+exp(y0)-x)*(1+exp(y0))/(2*(1+exp(y0))**2 - (y0+exp(y0)-x)*exp(y0))
    def halley(y0):
        ey0 = exp(y0)
        # c1 = (y0+ey0-x)*ey0
        # return y0 - (y0+ey0-x+c1)/((1+ey0)**2 - 0.5*c1)
        return y0 - (y0+ey0-x)/((1+ey0) - 0.5*(y0+ey0-x)*ey0/(1+ey0))
    # halley = lambda y0: y0 - (y0+exp(y0)-x)/(1+exp(y0))
    # iter_count = 0
    if N:
        for _ in range(N):
            y0 = halley(y0)
        y = y0
    else:
        y = halley(y0)
        while abs(y-y0)>tol:
            y0 = y
            y = halley(y0)
            # iter_count += 1
    # print(f"Iterations = {iter_count}")
    return y

def lambertw_fg(x, N=None, tol=tolerance):
    """Return w0=x*z0, where z0 is the root of the modified form of
    lambert w function: fg(z)=z-G^z, with G=e^(-x) and z=W0(x)/x
    where W0(x) returns the value of the principal branch of the
    lambert w function for the given value of x."""
    G = exp(-x)
    z0 = (1+ln(1+x))/(1+2*x) #seed LB_T
    fg = lambda z: z-G**z
    fgd = lambda z: 1+G**z*x
    if N:
        for _ in range(N):
            z0 = z0 - fg(z0)/fgd(z0) #Finding root using N-R method
    else:
        z_prev = z0
        z0 = z_prev - fg(z_prev)/fgd(z_prev)
        while abs(z0-z_prev)>tol:
            z_prev = z0
            z0 = z_prev - fg(z_prev)/fgd(z_prev)
    return x*z0 #w0=x*z0

def lambertw_hx(c, d, N=None, tol=tolerance):
    """Return w0, where w0 is the root of the modified form of
    lambert w function: hx(z)=ln(w)+w-ln(x), with w=W0(x),
    where W0(x) returns the value of the principal branch of the
    lambert w function for the given value of x."""
    l1 = ln(c)+d
    l2 = ln(l1)
    l1_inv = 1/l1
    w0 = l1 - l2 + l2*l1_inv*(1 + (-2+l2)*0.5*l1_inv) #seed BC_4
    hx = lambda w: ln(w)+w-l1
    hxd = lambda w: 1/w + 1
    #N = 3 #num iterations for Newton Raphson
    if N:
        for _ in range(N):
            w0 = w0 - hx(w0)/hxd(w0) #Finding root using N-R method
    else:
        w_prev = w0
        w0 = w_prev - hx(w_prev)/hxd(w_prev)
        while abs(w_prev-w0)>tol:
            w_prev = w0
            w0 = w_prev - hx(w_prev)/hxd(w_prev)
    return w0

def lambertw_hybrid(c, d):
    """Lambert w hybrid explicit calculation from Batzelis et al.
    Get implementation from here: https://github.com/ebatzelis/Lambert-W-function-in-PV-modeling/blob/master/LambertWs.c
    """
    l1 = ln(c)+d
    if l1>=ln(9):
        l2 = ln(l1)
        l1_inv = 1/l1
        l2_sq = l2**2
        return l1 - l2 + l2*l1_inv*\
            (1 + 0.5*l1_inv*
             (-2 + l2 + l1_inv/3.0*
              (6 - 9*l2 - 2*l2_sq + 0.25*l1_inv*
               (-12 + 36*l2 + 22*l2_sq + 3*l2_sq*l2 + 0.2*l1_inv*
                (60 - 300*l2 + 350*l2_sq - 125*l2_sq*l2 + 12*l2_sq*l2_sq)))))
    else:
        u = exp(l1-1)
        u_sq = u*u
        p = 1-u
        r = 1/(1+u)
        pr2 = p*r**2
        return u + u*r*p*\
            (1 + pr2*0.5*
             (1 + pr2/3*
              (-2*u + 1 + pr2*0.25*
               (6*u_sq - 8*u + 1 + pr2*0.2*
                (24*u_sq*u - 58*u_sq + 22*u - 1)))))

###===== Solvers =====###

def standard_i_approach(V, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell"""

    arg = R_sh/(a*(R_sh+R_s))
    x = I_sat*R_s*arg * exp(arg * (R_s*(I_ph+I_sat)+V) )
    return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw(x, tol=tol) #output current I

def standard_i_approach_witharg(V, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell"""

    arg = R_sh/(a*(R_sh+R_s))
    x = I_sat*R_s*arg * exp(arg * (R_s*(I_ph+I_sat)+V) )
    return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw(x, tol=tol), x #output current I

def standard_v_approach(I, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output voltage V of the solar cell calculated using the
    current and the parameters of the Single-Diode model of the solar cell"""

    arg = R_sh/a
    x = I_sat*arg * exp( arg * (I_ph+I_sat-I) )
    return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw(x, tol=tol) #output current I

def standard_v_approach_witharg(I, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output voltage V of the solar cell calculated using the
    current and the parameters of the Single-Diode model of the solar cell"""

    arg = R_sh/a
    x = I_sat*arg * exp( arg * (I_ph+I_sat-I) )
    return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw(x, tol=tol), x #output current I

def toledo_i_approach(V, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell"""
    arg = R_sh/(a*(R_sh+R_s))
    c = I_sat*R_s*arg
    d = arg * (R_s*(I_ph+I_sat)+V)

    if ln(c)+d <= 1:
        return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw_fg(c*exp(d), tol=tol) #output current I
    else:
        return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw_hx(c, d, tol=tol) #output current I


def toledo_i_approach_witharg(V, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell"""
    arg = R_sh/(a*(R_sh+R_s))
    c = I_sat*R_s*arg
    d = arg * (R_s*(I_ph+I_sat)+V)

    if ln(c)+d <= 1:
        return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw_fg(c*exp(d), tol=tol), (c, d)  #output current I
    else:
        return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw_hx(c, d, tol=tol), (c, d) #output current I

def toledo_v_approach(I, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell"""
    arg = R_sh/a
    c = I_sat*arg
    d = arg * (I_ph+I_sat-I)

    if ln(c)+d <= 1:
        return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw_fg(c*exp(d), tol=tol)
    else:
        return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw_hx(c, d, tol=tol)


def toledo_v_approach_witharg(I, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell"""
    arg = R_sh/a
    c = I_sat*arg
    d = arg * (I_ph+I_sat-I)

    if ln(c)+d <= 1:
        return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw_fg(c*exp(d), tol=tol), (c, d)
    else:
        return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw_hx(c, d, tol=tol), (c, d)

def logwright_i_approach(V, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    c1 = R_sh/(a*(R_sh+R_s))
    c2 = ln(I_sat*R_s*c1)
    u = c2 + (R_s*(I_ph+I_sat)+V)*c1
    return (a*(logwright(u, tol=tol) - c2) - V)/R_s

def logwright_i_approach_witharg(V, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    c1 = 1/(a*(1+R_s/R_sh))
    c2 = ln(I_sat*R_s*c1)
    u = c2 + (R_s*(I_ph+I_sat)+V)*c1
    return (a*(logwright(u, tol=tol) - c2) - V)/R_s, u

def logwright_v_approach(I, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    log_arg = ln(I_sat*R_sh/a)
    v = log_arg + (I_ph+I_sat-I)*R_sh/a
    return a*(logwright(v, tol=tol)-log_arg) - I*R_s

def logwright_v_approach_witharg(I, I_ph, I_sat, R_s, R_sh, a, tol=tolerance):
    log_arg = ln(I_sat*R_sh/a)
    v = log_arg + (I_ph+I_sat-I)*R_sh/a
    return a*(logwright(v, tol=tol)-log_arg) - I*R_s, v

def hybrid_i_approach(V, I_ph, I_sat, R_s, R_sh, a, tol=None):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell.
    Keyword argument TOL isn't used in the function."""

    arg = R_sh/(a*(R_sh+R_s))
    c = I_sat*R_s*arg
    d = arg * (R_s*(I_ph+I_sat)+V)
    return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw_hybrid(c, d) #output current I

def hybrid_i_approach_witharg(V, I_ph, I_sat, R_s, R_sh, a, tol=None):
    """Return the output current I of the solar cell calculated using the
    voltage and the parameters of the Single-Diode model of the solar cell.
    Keyword argument TOL isn't used in the function."""

    arg = R_sh/(a*(R_sh+R_s))
    c = I_sat*R_s*arg
    d = arg * (R_s*(I_ph+I_sat)+V)
    return (R_sh*(I_ph+I_sat)-V)/(R_sh+R_s) - (a/R_s)*lambertw_hybrid(c, d), (c, d) #output current I

def hybrid_v_approach(I, I_ph, I_sat, R_s, R_sh, a, tol=None):
    """Return the output voltage V of the solar cell calculated using the
    current and the parameters of the Single-Diode model of the solar cell.
    Keyword argument TOL isn't used in the function."""

    arg = R_sh/a
    c = I_sat*arg
    d = arg * (I_ph+I_sat-I)
    return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw_hybrid(c,d) #output current I

def hybrid_v_approach_witharg(I, I_ph, I_sat, R_s, R_sh, a, tol=None):
    """Return the output voltage V of the solar cell calculated using the
    current and the parameters of the Single-Diode model of the solar cell.
    Keyword argument TOL isn't used in the function."""

    arg = R_sh/a
    c = I_sat*arg
    d = arg * (I_ph+I_sat-I)
    return R_sh*(I_ph+I_sat) - (R_sh+R_s)*I - a*lambertw_hybrid(c,d), (c, d) #output current I


###===== Solver Class =====###

class Solver:
    def __init__(self, fn, fn_witharg, name, approach, tol=1e-8) -> None:
        """Initialize with a solver function"""
        self.fn = fn
        self.fn_witharg = fn_witharg
        # self.fn_ret_arg = fn_ret_arg
        self.name = name
        self.approach = approach
        self.proc_times = []
        self.results = {}
        self.results["I"] = []
        self.results["V"] = []
        self.results["fn_arg"] = []
        self.tol = tol

    def solve_for_iter_time(self, inp, *params):
        """Evaluates the solver function once and returns time. INP is current
        if V-approach is used, else INP is voltage. PARAMS is a 5-tuple of
        (I_ph, I_sat, R_s, R_sh, a)"""
        # Record time
        start_time = time.time()
        result = self.fn(inp, *params, tol=self.tol)
        proc_time = time.time()-start_time

        # Log data
        # self.proc_times.append(proc_time)
        return proc_time, result

    def solve_for_iter_time_witharg(self, inp, *params):
        """Evaluates the solver function once and returns time. INP is current
        if V-approach is used, else INP is voltage. PARAMS is a 5-tuple of
        (I_ph, I_sat, R_s, R_sh, a)"""
        # Record time
        start_time = time.time()
        result, fn_arg = self.fn_witharg(inp, *params, tol=self.tol)
        proc_time = time.time()-start_time

        # Log data
        # self.proc_times.append(proc_time)
        return proc_time, result

    def solve_for_iv_char(self, *params, num_steps=1):
        """Return I-V characteristics as tuple of two lists (lst_I, lst_V).
        params should not include current or voltage. PARAMS is a 7-tuple of
        (I_ph, I_sat, R_s, R_sh, a, I_sc, V_oc)"""
        I_ph, I_sat, R_s, R_sh, a, I_sc, V_oc = params
        upper_lim = I_sc if self.approach=="V" else V_oc
        step_size = upper_lim/num_steps
        inp_lst = []
        out_lst = []
        inp = 0.0
        out = self.fn(inp, I_ph, I_sat, R_s, R_sh, a, tol=self.tol)
        while inp<=upper_lim:
            inp_lst.append(inp)
            out_lst.append(out)
            out = self.fn(inp, I_ph, I_sat, R_s, R_sh, a, tol=self.tol)
            inp += step_size
        return (inp_lst, out_lst) if self.approach=="V" else (out_lst, inp_lst)

    def solve_for_iv_char_witharg(self, *params, num_steps=100):
        """Return I-V characteristics as tuple of two lists (lst_I, lst_V).
        params should not include current or voltage. PARAMS is a 7-tuple of
        (I_ph, I_sat, R_s, R_sh, a, I_sc, V_oc)"""
        I_ph, I_sat, R_s, R_sh, a, I_sc, V_oc = params
        upper_lim = I_sc if self.approach=="V" else V_oc
        step_size = upper_lim/num_steps
        inp_lst = []
        out_lst = []
        fn_arg_lst = []
        inp = 0.0
        out, fn_arg = self.fn_witharg(inp, I_ph, I_sat, R_s, R_sh, a, tol=self.tol)
        while inp<=upper_lim:
            inp_lst.append(inp)
            out_lst.append(out)
            fn_arg_lst.append(fn_arg)
            out, fn_arg = self.fn_witharg(inp, I_ph, I_sat, R_s, R_sh, a, tol=self.tol)
            inp += step_size
        return (inp_lst, out_lst, fn_arg_lst) if self.approach=="V" else (out_lst, inp_lst, fn_arg_lst)

    def get_iv_char_time(self, *params, num_steps=1, num_iters=1):
        """Return I-V characteristics as tuple of two lists (lst_I, lst_V). params
        should not include current or voltage. PARAMS is a 7-tuple of
        (I_ph, I_sat, R_s, R_sh, a, I_sc, V_oc)"""
        proc_times = []
        I, V = self.solve_for_iv_char(*params, num_steps=num_steps) #in case the loop doesnt run
        for _ in tqdm(range(num_iters)):
            proc_time = 0
            start_time = time.time()
            I, V = self.solve_for_iv_char(*params, num_steps=num_steps)
            proc_time = time.time() - start_time
            proc_times.append(proc_time)
        return I, V, proc_times

    def profile_iv_char_time(self, *params, num_steps=1, num_iters=1, time_metric="median"):
        """Calculates time taken to compute each point in the I-V
        characteristics. Increase step size to make computation faster. Return
        I-V characteristics and list of process times for each computation.
        params should not include current or voltage. PARAMS is a 7-tuple of
        (I_ph, I_sat, R_s, R_sh, a, I_sc, V_oc)"""
        I_ph, I_sat, R_s, R_sh, a, I_sc, V_oc = params
        upper_lim = I_sc if self.approach=="V" else V_oc
        step_size = upper_lim/num_steps
        all_proc_times = []
        for _ in tqdm(range(num_iters)):
            inp_lst = []
            out_lst = []
            proc_times = []
            inp = 0
            start_time = time.time()
            out = self.fn(inp, I_ph, I_sat, R_s, R_sh, a, tol=self.tol)
            proc_time = time.time()-start_time
            while inp<=upper_lim:
                proc_times.append(proc_time)
                inp_lst.append(inp)
                out_lst.append(out)
                inp += step_size
                start_time = time.time()
                out = self.fn(inp, I_ph, I_sat, R_s, R_sh, a, tol=self.tol)
                proc_time = time.time()-start_time
            all_proc_times.append(proc_times)
        if time_metric=="mean":
            out_proc_times = np.mean(np.array(all_proc_times), axis=0)
        else:
            out_proc_times = np.median(np.array(all_proc_times), axis=0)

        input_var = "I" if self.approach=="V" else "V"
        output_var = self.approach
        iv_char = {}
        iv_char[input_var] = inp_lst
        iv_char[output_var] = out_lst
        return iv_char["I"], iv_char["V"], out_proc_times

    def get_total_time(self):
        return np.sum(self.proc_times)

    def get_mean_time(self):
        return np.mean(self.proc_times)

    def get_median_time(self):
        return np.median(self.proc_times)

    def get_max_time(self):
        return np.max(self.proc_times)

    def get_min_time(self):
        return np.min(self.proc_times)

    def print_summary(self):
        print(f"{self.name} {self.approach}-approach - Program Summary")
        print("-----------------------------------------")
        print("")
        print("Results and Parameters for last solve:")
        pprint(self.results, sort_dicts=False)
        print("")
        print("Metrics:")
        print(f"Mean calculation time = {self.get_mean_time()}")
        print(f"Median calculation time = {self.get_median_time()}")
        print(f"Max calculation time = {self.get_max_time()}")
        print(f"Min calculation time = {self.get_min_time()}")
        print("=========================================")
        print("\n")


if __name__ == '__main__':
    I_ph = 15.88
    I_sat = 7.44e-10
    a = 14.67
    R_s = 2.04
    R_sh = 425.2
    num_iterations = 2000

    solvers = {"LogWright": {"I":logwright_i_approach, "V":logwright_v_approach,
                             "I_witharg":logwright_i_approach_witharg, "V_witharg":logwright_v_approach_witharg},
               "Toledo": {"I":toledo_i_approach, "V":toledo_v_approach,
                             "I_witharg":toledo_i_approach_witharg, "V_witharg":toledo_v_approach_witharg}}
    metrics = {}
    for algo in ["LogWright","Toledo"]:
        for approach in ["I","V"]:
            solver = Solver(solvers[algo][approach], solvers[algo][approach+"_witharg"], name=algo, approach=approach)
            proc_times = []
            for i in range(num_iterations):
                proc_time, _ = solver.solve_for_iter_time(0, I_ph, I_sat, R_s, R_sh, a)
                proc_times.append(proc_time)
            metrics[algo+"-"+approach] = {"mean time":np.mean(proc_times),
                                          "median time":np.median(proc_times),
                                          "max time":np.max(proc_times),
                                          "min time":np.min(proc_times)}


    # params_df = pd.DataFrame(params)
    # metrics_df = pd.DataFrame(metrics)
    # pd.options.display.float_format = "{:e}".format
    # print(params_df)
    # print(metrics_df)
