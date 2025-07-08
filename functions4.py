from pylab import *
import stim
import sinter
import os
from typing import List
from scipy.optimize import curve_fit
# # correct order of stabilizer measurement
# z_order = [2,1,3,0]
# z_order_leaf = [3,1,0,2]
# x_order = [2,3,1,0]
# x_order_leaf = [3,2,0,1]

# correct order of stabilizer measurement
z_order = [3,0,2,1]
z_order_leaf = [1,3,2,0]
x_order = [3,2,0,1]
x_order_leaf = [2,3,1,0]

ler = lambda a: a.errors/(a.shots-a.discards)
ler_err = lambda a: binom_error(a.errors, (a.shots-a.discards))
discs = lambda a: a.discards/a.shots
discs_err = lambda a: binom_error(a.discards, a.shots)
probs = lambda a: a.json_metadata['p']

def trans_cnot(rmax=3, shift_qubits=100):
    a = data_list(rmax)
    b = data_list(rmax)+shift_qubits
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

def init_remote_pair_transversal(rmax=3, repeats=1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in all_qubits_list(rmax))+' \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(n) for n in data_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)

    # initialize Bell
    injection = ' CX '+' '.join(str(n) for n in trans_cnot(rmax, shift_qubits))+' #   INJECTION   \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)
    
    # check all the deterministic stabs
    ###############################################################    
    x_stabs_det, z_stabs_det = stabs_all(rmax)
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in (x_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)

    # parity detectors
    detectors_pair = check_remote_pair(all_stabs_list(rmax), all_stabs_list(rmax), len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors_pair)

    ###############################################################  
    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in (x_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    detectors2 = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors2)
    
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    
    ###############################################################  
    
    return full_string[0]

def init_remote_pair_zero(rmax=3, repeats=1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax)[1:])+' \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in all_qubits_list(rmax))+' \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(0))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)

    # initialize Bell
    injection = ' CX '+f'0 {shift_qubits}'+' #  INJECTION  \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)
    
    # check all the deterministic stabs
    ###############################################################    
    x_stabs_det, z_stabs_det = stabs_all(rmax)
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in (x_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    # detectors on deterministic stabs without adjacent to pair
    detectors = check_det_single(all_stabs_list(rmax), np.setdiff1d(z_stabs_all(rmax), [1,3]))
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    # detectors on deterministic stabs
    detectors2 = check_det_single(all_stabs_list(rmax), np.setdiff1d(z_stabs_all(rmax), [1,3]))
    full_string = np.char.add(full_string, detectors2)
    # parity detectors around pair
    detectors_pair = check_remote_pair(all_stabs_list(rmax), [1,3], len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors_pair)

    ###############################################################  
    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in (x_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    detectors2 = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors2)
    
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    
    ###############################################################  
    
    return full_string[0]

def init_remote_pair_line(rmax=3, repeats=1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))

    central_col = array([0])
    for r in range(2,rmax)[::2]:
        central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax)[1:])+' \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in all_qubits_list(rmax))+' \n TICK \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(n) for n in central_col)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)

    # initialize Bell
    injection = ' CX '+f'0 {shift_qubits}'+' #  INJECTION  \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)
    
    # check all the deterministic stabs
    ###############################################################    
    x_stabs_det, z_stabs_det = stabs_all(rmax)
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in (x_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    # detectors on deterministic stabs without adjacent to pair
    detectors = check_det_single(all_stabs_list(rmax), np.setdiff1d(z_stabs_all(rmax), [1,3]))
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    # detectors on deterministic stabs
    detectors2 = check_det_single(all_stabs_list(rmax), np.setdiff1d(z_stabs_all(rmax), [1,3]))
    full_string = np.char.add(full_string, detectors2)
    # parity detectors around pair
    detectors_pair = check_remote_pair(all_stabs_list(rmax), [1,3], len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors_pair)

    ###############################################################  
    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in (x_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    detectors2 = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors2)
    
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    
    ###############################################################  
    
    return full_string[0]

def detector_data_corner_magic(rmax): # detect all deterministic stabs via data qubits and prev measurement
    full_string = np.array([""])
    N = len(data_list(rmax))

    # no leaves
    for r in arange(rmax)[1::2]: # measure all deterministic internal stabilizers via data qubits
        for a in range(4):
            for b in range(r):
                if n_polar(r, a, b) in np.append(det_stabs_lower(rmax), det_stabs_upper(rmax)):
                    stab = n_polar(r, a, b)
                    data_qubs = neighbour_data(r, a, b)
                    _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                    ind_data = len(data_list(rmax)) - ind_data
                    _,_,ind_stab = np.intersect1d(stab, all_stabs_list(rmax), return_indices=True)
                    ind_stab = len(all_stabs_list(rmax)) - ind_stab
                    full_string = np.char.add(full_string, 
                                              f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}] rec[-{N+ind_stab[0]}]  \n ')

    # leaves
    for b in arange(rmax)[1::2]:
        stab_z = n_polar(rmax, 1, b)
        data_qubs = neighbour_data(rmax, 1, b)
        shift = 0
        _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
        ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
        _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
        ind_stab = len(all_stabs_list(rmax)) - ind_stab
        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')

        stab_x = n_polar(rmax, 2, b)
        data_qubs = neighbour_data(rmax, 2, b)
        shift = 0
        _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
        ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
        _,_,ind_stab = np.intersect1d(stab_x, all_stabs_list(rmax), return_indices=True)
        ind_stab = len(all_stabs_list(rmax)) - ind_stab
        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')
    
    return full_string[0]

def corner_growing_remote_transversal(start_d=3, stop_d=5, repeats_small=1, repeats_full=1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))
                
    # reset zeros
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    # reset zeros
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'  \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(n) for n in translate_diag_array(data_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)

    # initialize Bell
    injection = ' CX '+' '.join(str(translate_diag(n, stop_d, stop_d-start_d, shift_qubits)) for n in (trans_cnot(start_d, shift_qubits)))+' #   INJECTION   \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)
    
    ########################## SMALL CODE #####################################
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'  \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        small_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+' \n  '
        full_string = np.char.add(full_string, small_check)
        small_check1 = ' CX '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, small_check1)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' \n  '
    full_string = np.char.add(full_string, MR) 
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(start_d), all_stabs_list(start_d), len(all_stabs_list(start_d)))
    full_string = np.char.add(full_string, detectors) 
    
    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_small} {{'+' \n ')

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam1)
    
    # measure checks of the full code
    for i in range(4):
        small_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+' \n  '
        full_string = np.char.add(full_string, small_check)
        small_check = ' CX '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, small_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits)
    MR = ' MR '+' '.join(str(n) for n in small_stabs)+' \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(start_d), all_stabs_list(start_d), len(all_stabs_list(start_d))*2)
    full_string = np.char.add(full_string, detectors)
    
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in small_stabs)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(start_d), all_stabs_list(start_d), len(all_stabs_list(start_d))*2)
    full_string = np.char.add(full_string, detectors)
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    

    ########################## FULL CODE ##################################### 
    # init datas in +
    patch_plus_small = translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d, shift_qubits)
    patch_plus_big = setdiff1d(lower_triangle_datas(stop_d), patch_plus_small)
    patch_zero_small = translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d, shift_qubits)
    patch_zero_big = setdiff1d(upper_triangle_datas(stop_d), patch_zero_small)
    reset = ' RX '+' '.join(str(n) for n in patch_plus_big)+'  \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' RX '+' '.join(str(n+shift_qubits) for n in patch_plus_big)+'  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in patch_zero_big)+' \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in patch_zero_big)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  '
        full_string = np.char.add(full_string, big_check)
        big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # patch det stabs
    small_det_stabs = translate_diag_array(np.append(det_stabs_lower(start_d), det_stabs_upper(start_d), shift_qubits), stop_d, stop_d-start_d, shift_qubits)
    big_det_stabs = np.append(det_stabs_lower(rmax), det_stabs_upper(rmax))
    patch_det_stabs = setdiff1d(big_det_stabs, small_det_stabs)
    detectors = check_det_single(all_stabs_list(rmax), patch_det_stabs)
    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  '
    full_string = np.char.add(full_string, MR)
    full_string = np.char.add(full_string, detectors)
    small_stabs_all = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits)
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), small_stabs_all, len(all_stabs_list(rmax))+len(all_stabs_list(start_d)))
    full_string = np.char.add(full_string, detectors)
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(rmax), patch_det_stabs)
    full_string = np.char.add(full_string, detectors)
    small_stabs_all = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits)
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), small_stabs_all, len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_full} {{'+' \n ')

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  '
        full_string = np.char.add(full_string, big_check)
        big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  '
    full_string = np.char.add(full_string, MR)
    detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)
    
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors2 = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors2)
    ############################################################### 
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    # # Hadamard on X data qubits
    # hadam = ' H '+' '.join(str(n) for n in patch_plus_big)+' \n  '
    # full_string = np.char.add(full_string, hadam)
    # hadam = ' H '+' '.join(str(n+shift_qubits) for n in patch_plus_big)+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)
        
    return full_string[0] 

def corner_growing_remote_transversal_noiseless(start_d=3, stop_d=5, repeats_small=1, repeats_full=1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))
                
    # reset zeros
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+'#   NOISELESS   \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in all_qubits_list(rmax))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    # reset zeros
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' #   NOISELESS  \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(n) for n in translate_diag_array(data_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)

    # initialize Bell
    injection = ' CX '+' '.join(str(translate_diag(n, stop_d, stop_d-start_d, shift_qubits)) for n in (trans_cnot(start_d, shift_qubits)))+' #   INJECTION  NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)
    
    ########################## SMALL CODE #####################################
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS   \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        small_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  '
        full_string = np.char.add(full_string, small_check)
        small_check1 = ' CX '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
        full_string = np.char.add(full_string, small_check1)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  '
    full_string = np.char.add(full_string, MR) 
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(start_d), all_stabs_list(start_d), len(all_stabs_list(start_d)))
    full_string = np.char.add(full_string, detectors) 
    
    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_small} {{'+' \n ')

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam1)
    
    # measure checks of the small code
    for i in range(4):
        small_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  '
        full_string = np.char.add(full_string, small_check)
        small_check = ' CX '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
        full_string = np.char.add(full_string, small_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits)
    MR = ' MR '+' '.join(str(n) for n in small_stabs)+'#   NOISELESS  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(start_d), all_stabs_list(start_d), len(all_stabs_list(start_d))*2)
    full_string = np.char.add(full_string, detectors)
    
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in small_stabs)+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(start_d), all_stabs_list(start_d), len(all_stabs_list(start_d))*2)
    full_string = np.char.add(full_string, detectors)
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    

    ########################## FULL CODE ##################################### 
    # init datas in +
    patch_plus_small = translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d, shift_qubits)
    patch_plus_big = setdiff1d(lower_triangle_datas(stop_d), patch_plus_small)
    patch_zero_small = translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d, shift_qubits)
    patch_zero_big = setdiff1d(upper_triangle_datas(stop_d), patch_zero_small)
    reset = ' RX '+' '.join(str(n) for n in patch_plus_big)+'  \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' RX '+' '.join(str(n+shift_qubits) for n in patch_plus_big)+'  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in patch_zero_big)+' \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in patch_zero_big)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  '
        full_string = np.char.add(full_string, big_check)
        big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # patch det stabs
    small_det_stabs = translate_diag_array(np.append(det_stabs_lower(start_d), det_stabs_upper(start_d), shift_qubits), stop_d, stop_d-start_d, shift_qubits)
    big_det_stabs = np.append(det_stabs_lower(rmax), det_stabs_upper(rmax))
    patch_det_stabs = setdiff1d(big_det_stabs, small_det_stabs)
    detectors = check_det_single(all_stabs_list(rmax), patch_det_stabs)
    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  '
    full_string = np.char.add(full_string, MR)
    full_string = np.char.add(full_string, detectors)
    small_stabs_all = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits)
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), small_stabs_all, len(all_stabs_list(rmax))+len(all_stabs_list(start_d)))
    full_string = np.char.add(full_string, detectors)
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(rmax), patch_det_stabs)
    full_string = np.char.add(full_string, detectors)
    small_stabs_all = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits)
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), small_stabs_all, len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_full} {{'+' \n ')

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  '
        full_string = np.char.add(full_string, big_check)
        big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  '
    full_string = np.char.add(full_string, MR)
    detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)
    
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors2 = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors2)
    ############################################################### 
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    # # Hadamard on X data qubits
    # hadam = ' H '+' '.join(str(n) for n in patch_plus_big)+' \n  '
    # full_string = np.char.add(full_string, hadam)
    # hadam = ' H '+' '.join(str(n+shift_qubits) for n in patch_plus_big)+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)
        
    return full_string[0] 

def corner_growing_remote_transversal_noiseless_norep(start_d=3, stop_d=5, repeats_small=1, repeats_full=1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))
                
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' #   NOISELESS  \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(n) for n in translate_diag_array(data_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)
    
    ########################## SMALL CODE #####################################
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS   \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        small_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  '
        full_string = np.char.add(full_string, small_check)
        small_check1 = ' CX '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
        full_string = np.char.add(full_string, small_check1)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  '
    full_string = np.char.add(full_string, MR) 
    detectors = check_det_single(all_stabs_list(rmax), x_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors) 
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors) 
     
    # initialize Bell
    injection = ' CX '+' '.join(str(translate_diag(n, stop_d, stop_d-start_d, shift_qubits)) for n in (trans_cnot(start_d, shift_qubits)))+' # \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)

    ########################## FULL CODE ##################################### 
    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'#      \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+'#      \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'#      \n  '
        full_string = np.char.add(full_string, big_check)
        big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+'#      \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'#      \n  '
    full_string = np.char.add(full_string, MR)
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+'#    \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(rmax), z_stabs_all(rmax), len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors)
        
    return full_string[0] 

def transversal_cnot_reps(start_d=3, stop_d=3, repeats = 1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))
                
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' #   NOISELESS  \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(n) for n in translate_diag_array(data_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)
    
    ########################## INIT THE CODE #####################################
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS   \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        small_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  '
        full_string = np.char.add(full_string, small_check)
        small_check1 = ' CX '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
        full_string = np.char.add(full_string, small_check1)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  '
    full_string = np.char.add(full_string, MR) 
    detectors = check_det_single(all_stabs_list(rmax), x_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors) 
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS  \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors) 
     
    ########################## CNOTS Bell pair ##################################### 
    
    # transversal CNOT
    injection = ' CX '+' '.join(str(translate_diag(n, stop_d, stop_d-start_d, shift_qubits)) for n in (trans_cnot(start_d, shift_qubits)))+' # \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'#      \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+'#      \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'#      \n  '
        full_string = np.char.add(full_string, big_check)
        big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+'#      \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'#      \n  '
    full_string = np.char.add(full_string, MR)
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+'#    \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_remote_pair(all_stabs_list(rmax), all_stabs_list(rmax), len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors)

    # # repeat 'repeats' times
    # full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # ########################## REPEATED CNOTS separable ##################################### 
    
    # # transversal CNOT
    # injection = ' CX '+' '.join(str(translate_diag(n, stop_d, stop_d-start_d, shift_qubits)) for n in (trans_cnot(start_d, shift_qubits)))+' # \n  TICK  \n  '
    # full_string = np.char.add(full_string, injection)

    # # Hadamard on X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'#      \n  '
    # full_string = np.char.add(full_string, hadam)    
    # hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+'#      \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam1)    
    
    # # measure checks of the full code
    # for i in range(4):
    #     big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'#      \n  '
    #     full_string = np.char.add(full_string, big_check)
    #     big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+'#      \n  TICK  \n  '
    #     full_string = np.char.add(full_string, big_check)

    # # Hadamard on X stabs
    # full_string = np.char.add(full_string, hadam)
    # full_string = np.char.add(full_string, hadam1)

    # # measure all stabs
    # MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'#      \n  '
    # full_string = np.char.add(full_strisng, MR)
    # detectors = check_det_single(all_stabs_list(rmax), x_stabs_all(rmax))
    # full_string = np.char.add(full_string, detectors) 
    # N = len(all_stabs_list(rmax))*2
    # detectors = check_det_1_round_patches(all_stabs_list(rmax), all_stabs_list(rmax), z_stabs_all(rmax), N)
    # full_string = np.char.add(full_string, detectors) 
    # MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+'#    \n  TICK  \n  '
    # full_string = np.char.add(full_string, MR)
    # detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    # full_string = np.char.add(full_string, detectors) 
    # N = len(all_stabs_list(rmax))*2
    # detectors = check_det_1_round_patches(all_stabs_list(rmax), all_stabs_list(rmax), x_stabs_all(rmax), N)
    # full_string = np.char.add(full_string, detectors) 

    # ########################## REPEATED CNOTS Bell pair ##################################### 
    
    # # transversal CNOT
    # injection = ' CX '+' '.join(str(translate_diag(n, stop_d, stop_d-start_d, shift_qubits)) for n in (trans_cnot(start_d, shift_qubits)))+' # \n  TICK  \n  '
    # full_string = np.char.add(full_string, injection)

    # # Hadamard on X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'#      \n  '
    # full_string = np.char.add(full_string, hadam)    
    # hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+'#      \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam1)    
    
    # # measure checks of the full code
    # for i in range(4):
    #     big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'#      \n  '
    #     full_string = np.char.add(full_string, big_check)
    #     big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+'#      \n  TICK  \n  '
    #     full_string = np.char.add(full_string, big_check)

    # # Hadamard on X stabs
    # full_string = np.char.add(full_string, hadam)
    # full_string = np.char.add(full_string, hadam1)

    # # measure all stabs
    # MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'#      \n  '
    # full_string = np.char.add(full_string, MR)
    # MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+'#    \n  TICK  \n  '
    # full_string = np.char.add(full_string, MR)
    # detectors = check_remote_pair(all_stabs_list(rmax), all_stabs_list(rmax), len(all_stabs_list(rmax)))
    # full_string = np.char.add(full_string, detectors)

    # # close repeats 
    # full_string = np.char.add(full_string, '}'+' \n ')
    
    return full_string[0] 

def transversal_cnot_reps_corr(start_d=3, stop_d=3, repeats = 1, shift_coord=20, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))
                
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+' #  NOISELESS   \n  '
    full_string = np.char.add(full_string, reset)
    reset = ' R '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#  NOISELESS   \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    reset0 = ' RX '+' '.join(str(n) for n in translate_diag_array(data_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS   \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)
    
    ########################## INIT THE CODE #####################################
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS    \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS   \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        small_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#  NOISELESS    \n  '
        full_string = np.char.add(full_string, small_check)
        small_check1 = ' CX '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS   \n  TICK  \n  '
        full_string = np.char.add(full_string, small_check1)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#   NOISELESS   \n  '
    full_string = np.char.add(full_string, MR) 
    detectors = check_det_single(all_stabs_list(rmax), x_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors) 
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d, shift_qubits))+'#  NOISELESS    \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors) 


    # repeat 'repeats' times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')


    ########################## CNOTS Bell pair ##################################### 
    
    # transversal CNOT
    injection = ' CX '+' '.join(str(translate_diag(n, stop_d, stop_d-start_d, shift_qubits)) for n in (trans_cnot(start_d, shift_qubits)))+' # \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'#      \n  '
    full_string = np.char.add(full_string, hadam)    
    hadam1 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+'#      \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam1)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'#      \n  '
        full_string = np.char.add(full_string, big_check)
        big_check = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+'#      \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam1)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'#      \n  '
    full_string = np.char.add(full_string, MR)
    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+'#    \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # X stabs on target
    detectors = check_corr_cnot_single(measurement_last=all_stabs_list(rmax), M=0, 
                                       measurement_prev=all_stabs_list(rmax), checks=x_stabs_all(rmax), N=len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    # Z stabs on control
    detectors = check_corr_cnot_single(measurement_last=all_stabs_list(rmax), M=len(all_stabs_list(rmax)), 
                                       measurement_prev=all_stabs_list(rmax), checks=z_stabs_all(rmax), N=len(all_stabs_list(rmax))*3)
    full_string = np.char.add(full_string, detectors)


    # X stabs on control - parity
    detectors = check_corr_cnot_parity(measurement_last=all_stabs_list(rmax), M=len(all_stabs_list(rmax)), 
                                       measurement_prev=all_stabs_list(rmax), checks=x_stabs_all(rmax), N=len(all_stabs_list(rmax))*2, P=len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors)

    # Z stabs on target - parity
    detectors = check_corr_cnot_parity(measurement_last=all_stabs_list(rmax), M=0, 
                                       measurement_prev=all_stabs_list(rmax), checks=z_stabs_all(rmax), N=len(all_stabs_list(rmax))*2, P=len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors)

    
    
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')
    
    return full_string[0] 

def corner_growing_magic(start_d=3, stop_d=5, repeats_small=1, repeats_full=1):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset zeros
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    ########################## SMALL CODE #####################################
    # reset small code
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    # init datas in +
    reset = ' RX '+' '.join(str(n) for n in translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d))+'  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(start_d), np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_small} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d)
    MR = ' MR '+' '.join(str(n) for n in small_stabs)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(start_d))
    full_string = np.char.add(full_string, detectors)
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    

    ########################## FULL CODE ##################################### 
    # init datas in +
    patch_plus_small = translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d)
    patch_plus_big = setdiff1d(lower_triangle_datas(stop_d), patch_plus_small)
    patch_zero_small = translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d)
    patch_zero_big = setdiff1d(upper_triangle_datas(stop_d), patch_zero_small)
    reset = ' RX '+' '.join(str(n) for n in patch_plus_big)+'  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in patch_zero_big)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # only_det_checks = setdiff1d(z_stabs_all(rmax), gauge_checks(start_d, stop_d))
    # small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d)
    small_det_stabs = translate_diag_array(np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)), stop_d, stop_d-start_d)
    big_det_stabs = np.append(det_stabs_lower(rmax), det_stabs_upper(rmax))
    patch_det_stabs = setdiff1d(big_det_stabs, small_det_stabs)
    detectors = check_det_single(all_stabs_list(rmax), patch_det_stabs)
    full_string = np.char.add(full_string, detectors)
    # check that didn't change
    detectors = check_det_1_round_RX(small_stabs, all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_full} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    # # Hadamard on X data qubits
    # hadam = ' H '+' '.join(str(n) for n in lower_triangle_datas(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)
        
    return full_string[0]     

def corner_growing_magic_upd(start_d=3, stop_d=5, repeats_small=1, repeats_full=1):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset zeros
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    ########################## SMALL CODE #####################################
    # reset small code
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    # init datas in +
    reset = ' RX '+' '.join(str(n) for n in translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d))+'#   NOISELESS  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(start_d), np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_small} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d)
    MR = ' MR '+' '.join(str(n) for n in small_stabs)+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(start_d))
    full_string = np.char.add(full_string, detectors)
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    

    ########################## FULL CODE ##################################### 
    # init datas in +
    patch_plus_small = translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d)
    patch_plus_big = setdiff1d(lower_triangle_datas(stop_d), patch_plus_small)
    patch_zero_small = translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d)
    patch_zero_big = setdiff1d(upper_triangle_datas(stop_d), patch_zero_small)
    reset = ' RX '+' '.join(str(n) for n in patch_plus_big)+'  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in patch_zero_big)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # only_det_checks = setdiff1d(z_stabs_all(rmax), gauge_checks(start_d, stop_d))
    # small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d)
    small_det_stabs = translate_diag_array(np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)), stop_d, stop_d-start_d)
    big_det_stabs = np.append(det_stabs_lower(rmax), det_stabs_upper(rmax))
    patch_det_stabs = setdiff1d(big_det_stabs, small_det_stabs)
    detectors = check_det_single(all_stabs_list(rmax), patch_det_stabs)
    full_string = np.char.add(full_string, detectors)
    # check that didn't change
    detectors = check_det_1_round_RX(small_stabs, all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_full} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    # # Hadamard on X data qubits
    # hadam = ' H '+' '.join(str(n) for n in lower_triangle_datas(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)
        
    return full_string[0]   

def corner_growing_magic_upd_norep(start_d=3, stop_d=5, repeats_small=1):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset zeros
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    ########################## SMALL CODE #####################################
    # reset small code
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    # init datas in +
    reset = ' RX '+' '.join(str(n) for n in translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d))+'#   NOISELESS  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(start_d), np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_small} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+'#   NOISELESS \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d)
    MR = ' MR '+' '.join(str(n) for n in small_stabs)+'#   NOISELESS \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(start_d))
    full_string = np.char.add(full_string, detectors)
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    

    ########################## FULL CODE ##################################### 
    # init datas in +
    patch_plus_small = translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d)
    patch_plus_big = setdiff1d(lower_triangle_datas(stop_d), patch_plus_small)
    patch_zero_small = translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d)
    patch_zero_big = setdiff1d(upper_triangle_datas(stop_d), patch_zero_small)
    reset = ' RX '+' '.join(str(n) for n in patch_plus_big)+'  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in patch_zero_big)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # only_det_checks = setdiff1d(z_stabs_all(rmax), gauge_checks(start_d, stop_d))
    # small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d)
    small_det_stabs = translate_diag_array(np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)), stop_d, stop_d-start_d)
    big_det_stabs = np.append(det_stabs_lower(rmax), det_stabs_upper(rmax))
    patch_det_stabs = setdiff1d(big_det_stabs, small_det_stabs)
    detectors = check_det_single(all_stabs_list(rmax), patch_det_stabs)
    full_string = np.char.add(full_string, detectors)
    # check that didn't change
    detectors = check_det_1_round_RX(small_stabs, all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)

    # # repeat d times
    # full_string = np.char.add(full_string, f'REPEAT {repeats_full} {{'+' \n ')

    # # Hadamard on X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)    
    
    # # measure checks of the full code
    # for i in range(4):
    #     big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
    #     full_string = np.char.add(full_string, big_check)

    # # Hadamard on X stabs
    # full_string = np.char.add(full_string, hadam)

    # # measure all stabs
    # MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, MR)

    # # check that all the stabs are the same
    # detectors = detectors_parity(all_stabs_list(rmax))
    # full_string = np.char.add(full_string, detectors)
    # ############################################################### 
    # # close repeats 
    # full_string = np.char.add(full_string, '}'+' \n ')

    # # Hadamard on X data qubits
    # hadam = ' H '+' '.join(str(n) for n in lower_triangle_datas(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)
        
    return full_string[0]   

def corner_init_magic(start_d, repeats_small=1):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    rmax = start_d
    stop_d = start_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset zeros
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    ########################## SMALL CODE #####################################
    # reset small code
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    # init datas in +
    reset = ' RX '+' '.join(str(n) for n in translate_diag_array(lower_triangle_datas(start_d), stop_d, stop_d-start_d))+'  \n  '
    full_string = np.char.add(full_string, reset)
    # init datas in 0
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(upper_triangle_datas(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(start_d), np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_small} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    small_stabs = translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d)
    MR = ' MR '+' '.join(str(n) for n in small_stabs)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(start_d))
    full_string = np.char.add(full_string, detectors)
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    

    # Hadamard on X data qubits
    hadam = ' H '+' '.join(str(n) for n in lower_triangle_datas(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)
        
    return full_string[0]

def corner_growing_zero(start_d, stop_d, repeats_small=1, repeats_full=1):
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    rmax = stop_d
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # for i in range(len(all_qubits_list(rmax))):
    #     full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    
    ########################## SMALL CODE #####################################
    # reset small code
    reset = ' R '+' '.join(str(n) for n in translate_diag_array(all_qubits_list(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_single(all_stabs_list(start_d), z_stabs_all(start_d))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_small} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in translate_diag_array(x_stabs_all(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in translate_diag_array(all_checks_upd(start_d)[i], stop_d, stop_d-start_d))+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in translate_diag_array(all_stabs_list(start_d), stop_d, stop_d-start_d))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(start_d))
    full_string = np.char.add(full_string, detectors)
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')    

    ########################## FULL CODE ##################################### 
    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    only_det_checks = setdiff1d(z_stabs_all(rmax), gauge_checks(start_d, stop_d))
    detectors = check_det_single(all_stabs_list(rmax), only_det_checks)
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats_full} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')
        
    return full_string[0]   

def gauge_checks(start_d, stop_d): # non-deterministic Z stabs during lattice surgery
    init_coord = all_in_r_a(start_d,2)[::2]
    new_coord = translate_diag_array(init_coord, stop_d, stop_d-start_d)
    return new_coord

def det_stabs_lower(rmax): # only internal X stabs
    datas = np.array([], dtype=int)
    for r in range(1, rmax-1)[0::2]:
        datas = np.append(datas, np.roll(all_in_r(r),-1)[2*r-1:][1::2])
    # append bottom leaves
    datas = np.append(datas, all_in_r_a(rmax, 2)[:-1][1::2])
    return datas

def det_stabs_upper(rmax): # only internal Z stabs
    datas = np.array([], dtype=int)
    for r in range(1, rmax-1)[0::2]:
        datas = np.append(datas, all_in_r(r)[2:2*r-1][0::2])
    # append right leaves
    datas = np.append(datas, all_in_r_a(rmax, 1)[:-1][1::2])
    return datas

def lower_triangle_datas(rmax): # init in + state
    datas = np.array([], dtype=int)
    datas = np.append(datas, 0)
    for r in range(2, rmax)[0::2]:
        datas = np.append(datas, np.roll(all_in_r(r),-1)[2*r-1:])
    magic = all_in_r(rmax-1)[0]
    magic_index = np.where(datas == magic)
    datas = np.delete(datas, magic_index)
    return datas      

def upper_triangle_datas(rmax): # init in 0 state
    datas = np.array([], dtype=int)
    for r in range(2, rmax)[0::2]:
        datas = np.append(datas, np.roll(all_in_r(r),-1)[0:2*r-1])
    return datas  

def diag_line(rmax, offset):
    datas = np.array([], dtype=int)
    datas1 = np.array([], dtype=int)
    for r in arange(abs(offset), rmax):
        if offset>=0:
            a = 0
            b = offset
            datas = np.append(datas, n_polar(r, a, b))
            a = 1
            b = offset+1
            datas1 = np.append(datas1, all_in_r_a(r,a)[-b])
        if offset<0:
            a = 3
            b = offset-1
            datas = np.append(datas, all_in_r_a(r,a)[b])
            a = 2
            b = -offset
            datas1 = np.append(datas1, n_polar(r, a, b))  
    return np.append(datas[::-1], datas1[1:])        

def diag_line_array(rmax):
    rmax = rmax+1
    datas = []
    for i in arange(-(rmax-1), rmax):
        datas.append(diag_line(rmax, i))
    return datas

def translate_diag(qubit, rmax, shift, shift_qubits=100):
    if (qubit<shift_qubits):
        list_of_arrays = diag_line_array(rmax)
        for i, arr in enumerate(list_of_arrays): 
            result = np.argwhere(arr == qubit)
            if result.size > 0:
                i1 = i; i2 = result[0][0]
        qubit_new = list_of_arrays[i1][i2-shift]
        return qubit_new
    elif (qubit>=shift_qubits):
        qubit = qubit-shift_qubits
        list_of_arrays = diag_line_array(rmax)
        for i, arr in enumerate(list_of_arrays): 
            result = np.argwhere(arr == qubit)
            if result.size > 0:
                i1 = i; i2 = result[0][0]
        qubit_new = list_of_arrays[i1][i2-shift]
        return qubit_new + shift_qubits

def translate_diag_array(qubit, rmax, shift, shift_qubits=100):
    if (np.array(qubit)<shift_qubits).all():
        qubit_new = np.zeros(len(qubit), dtype=int)
        list_of_arrays = diag_line_array(rmax)
        for i, arr in enumerate(list_of_arrays): 
            for j, q in enumerate(qubit):
                result = np.argwhere(arr == q)
                if result.size > 0:
                    qubit_new[j] = list_of_arrays[i][result[0][0]-shift]
        return qubit_new 
    elif (np.array(qubit)>=shift_qubits).all():
        qubit = qubit-shift_qubits
        qubit_new = np.zeros(len(qubit), dtype=int)
        list_of_arrays = diag_line_array(rmax)
        for i, arr in enumerate(list_of_arrays): 
            for j, q in enumerate(qubit):
                result = np.argwhere(arr == q)
                if result.size > 0:
                    qubit_new[j] = list_of_arrays[i][result[0][0]-shift]
        return qubit_new + shift_qubits

def binom_error(errors, shots):
    p = errors/shots
    return 3*sqrt(p*(1-p)/shots)

def bell_error(p):
    return 2*p-4/3*p**2

def bell_error_v0(p):
    return 2*p-p**2

def write_grow_code(init_d=3, target_d=21, start='zero'):
    full_string = np.array([""])
    if start=='zero':
        full_string = np.char.add(full_string, d_init_upd(init_d, repeats=init_d))
    elif start=='bell':
        if int(((target_d-init_d)/2)%2)==0:
            full_string = np.char.add(full_string, d_init_bell(init_d, repeats=init_d))
        elif int(((target_d-init_d)/2)%2)==1:
            print('Please, select other target_d')
            # full_string = np.char.add(full_string, d_init_bell(init_d, repeats=init_d))
    else:
        print('Wrong start key')

    counter = 0
    for d in arange(init_d+2, target_d+2)[::2]:
        full_string = np.char.add(full_string, grow_distance_upd_debug(d, repeats=d))
        counter+=1
    
    full_string = np.char.add(full_string, measure_logical(target_d))
    full_string = np.char.add(full_string, observe_logical_PF_v3(target_d, n_grow=(counter+1)//2))

    return full_string[0]

def write_inject_grow_code(init_d=3, target_d=21, start='bell'):
    full_string = np.array([""])
    if start=='zero':
        full_string = np.char.add(full_string, d_init_upd(init_d, repeats=1))
    elif start=='bell':
        if int(((target_d-init_d)/2)%2)==0:
            full_string = np.char.add(full_string, d_init_bell_det_stabs_v2(init_d, repeats=1))
        elif int(((target_d-init_d)/2)%2)==1:
            print('Please, select other target_d')
            full_string = np.char.add(full_string, d_init_bell_det_stabs_v2(init_d, repeats=1))
            # full_string = np.char.add(full_string, d_init_bell(init_d, repeats=init_d))
    else:
        print('Wrong start key')

    counter = 0
    for d in arange(init_d+2, target_d+2)[::2]:
        full_string = np.char.add(full_string, grow_distance_upd_debug(d, repeats=d))
        counter+=1
    
    full_string = np.char.add(full_string, measure_logical(target_d))
    full_string = np.char.add(full_string, observe_logical_PF_v3(target_d, n_grow=(counter+1)//2))

    return full_string[0]

def write_inject_grow_code_3d4(init_d=3, target_d=11, start='zero'):
    full_string = np.array([""])
    if start=='zero':
        full_string = np.char.add(full_string, d_init_upd(init_d, repeats=1))
    elif start=='bell':
        if int(((target_d-init_d)/2)%2)==0:
            full_string = np.char.add(full_string, d_init_bell_det_stabs_v2(init_d, repeats=1))
        elif int(((target_d-init_d)/2)%2)==1:
            print('Please, select other target_d')
            full_string = np.char.add(full_string, d_init_bell_det_stabs_v2(init_d, repeats=1))
            # full_string = np.char.add(full_string, d_init_bell(init_d, repeats=init_d))
    else:
        print('Wrong start key')

    if init_d==3:
        full_string = np.char.add(full_string, grow_distance_upd_debug(5, repeats=2))
        full_string = np.char.add(full_string, grow_distance_3d4(5, repeats=2))

    full_string = np.char.add(full_string, measure_logical_z(target_d))
    full_string = np.char.add(full_string, observe_logical_op(target_d))

    return full_string[0]

def write_inject_grow_code_debug(init_d=3, target_d=21, start='bell'):
    full_string = np.array([""])
    if start=='zero':
        full_string = np.char.add(full_string, d_init_upd(init_d, repeats=1))
    elif start=='bell':
        if int(((target_d-init_d)/2)%2)==0:
            full_string = np.char.add(full_string, d_init_bell_det_stabs_v2(init_d, repeats=1))
        elif int(((target_d-init_d)/2)%2)==1:
            print('Please, select other target_d')
            full_string = np.char.add(full_string, d_init_bell_det_stabs_v2(init_d, repeats=1))
            # full_string = np.char.add(full_string, d_init_bell(init_d, repeats=init_d))
    else:
        print('Wrong start key')

    counter = 0
    for d in arange(init_d+2, target_d+2)[::2]:
        full_string = np.char.add(full_string, grow_distance_upd_debug_debug(d, repeats=d))
        counter+=1
    
    full_string = np.char.add(full_string, measure_logical(target_d))
    full_string = np.char.add(full_string, observe_logical_PF_v3(target_d, n_grow=(counter+1)//2))

    return full_string[0]

def write_injection_code(init_d=3, repeats=1):
    full_string = np.array([""])
    full_string = np.char.add(full_string, d_init_bell_det_stabs_v2(init_d, repeats=repeats))
    
    full_string = np.char.add(full_string, measure_logical(init_d))
    full_string = np.char.add(full_string, observe_logical_op(init_d))

    return full_string[0]

def write_injection_bell_code(init_d=3, repeats=1, shift_coord=15, shift_qubits=100):
    full_string = np.array([""])
    full_string = np.char.add(full_string, d_init_bell_det_stabs_pair(rmax=init_d, shift_coord=shift_coord, shift_qubits=shift_qubits))
    
    full_string = np.char.add(full_string, measure_logical_bell(init_d, shift_qubits))
    full_string = np.char.add(full_string, observe_logical_bell(init_d))

    return full_string[0]

def write_injection_code_RX(init_d=3):
    full_string = np.array([""])
    full_string = np.char.add(full_string, d_init_bell_det_stabs_v3(init_d, repeats=1))
    
    full_string = np.char.add(full_string, measure_logical(init_d))
    full_string = np.char.add(full_string, observe_logical_op(init_d))

    return full_string[0]

def write_grow_code_no_repeats(init_d=3, target_d=21, start='zero'):
    full_string = np.array([""])
    if start=='zero':
        full_string = np.char.add(full_string, d_init_upd(init_d))
    elif start=='bell':
        if int(((target_d-init_d)/2)%2)==0:
            full_string = np.char.add(full_string, d_init_bell(init_d))
        elif int(((target_d-init_d)/2)%2)==1:
            print('Please, select other target_d')
            # full_string = np.char.add(full_string, d_init_bell(init_d, repeats=init_d))
    else:
        print('Wrong start key')

    counter = 0
    for d in arange(init_d+2, target_d+2)[::2]:
        full_string = np.char.add(full_string, grow_distance_upd_debug(d))
        counter+=1
    
    full_string = np.char.add(full_string, measure_logical(target_d))
    full_string = np.char.add(full_string, observe_logical_PF_v3(target_d, n_grow=(counter+1)//2))

    return full_string[0]

def num_to_coord(r, a, b):
    n = n_polar(r, a, b)
    if r==0:
        x=0; y=0
    elif r>0:
        if a==0:
            y = r
            x = -r+2*b
        elif a==1:
            x = r
            y = r-2*b
        elif a==2:
            y = -r
            x = r-2*b
        elif a==3:
            x = -r
            y = -r+2*b
    if r%2==0:
        qtype = 'data'
    elif r%2==1:
        if n%2==0:
            qtype = 'X stab'
        if n%2==1:
            qtype = 'Z stab'        
            
    return f'QUBIT_COORDS({int(x)}, {int(-y)}) {int(n)} # {qtype} \n  '

def num_to_coord_shifted(r, a, b, shift_coord=20, shift_qubits=100):
    n = n_polar(r, a, b)
    if r==0:
        x=0; y=0
    elif r>0:
        if a==0:
            y = r
            x = -r+2*b
        elif a==1:
            x = r
            y = r-2*b
        elif a==2:
            y = -r
            x = r-2*b
        elif a==3:
            x = -r
            y = -r+2*b
    if r%2==0:
        qtype = 'data'
    elif r%2==1:
        if n%2==0:
            qtype = 'X stab'
        if n%2==1:
            qtype = 'Z stab'        
            
    return f'QUBIT_COORDS({int(x)}, {int(-(y+shift_coord))}) {int(n+shift_qubits)} # {qtype} \n  '

def num_to_coord_plot(r, a, b):
    n = n_polar(r, a, b)
    if r==0:
        x=0; y=0
    elif r>0:
        if a==0:
            y = r
            x = -r+2*b
        elif a==1:
            x = r
            y = r-2*b
        elif a==2:
            y = -r
            x = r-2*b
        elif a==3:
            x = -r
            y = -r+2*b
    if r%2==0:
        qtype = 'data'
    elif r%2==1:
        if n%2==0:
            qtype = 'X stab'
        if n%2==1:
            qtype = 'Z stab'        
            
    return x, y, n

def n_polar(r, a, b):
    if r == 0:
        n = 0
    else:
        if b==r:
            a+=1
            b=0
            n = 2*(r-1)*r+1+r*arange(4)[a%4]+b
        else:
            n = 2*(r-1)*r+1+r*arange(4)[a%4]+b
    return n

def all_in_r_a(r,a):
    qubits_ns = np.array([], dtype=int)
    for b in range(r+1):
        qubits_ns = np.append(qubits_ns, n_polar(r, a, b))
    return qubits_ns   

def all_in_r(r):     
    qubits_ns = np.array([], dtype=int)
    if r==0:
        qubits_ns = np.append(qubits_ns, 0)
    else:
        for a in range(4):
            for b in range(r):
                qubits_ns = np.append(qubits_ns, n_polar(r, a, b))
    return qubits_ns

def stab_list(rmax): # rmax is equiv to code distance (here only inner stabs)
    stab_x = np.array([], dtype=int)
    stab_z = np.array([], dtype=int)
    for r in arange(rmax)[1::2]:
        stab_x = np.append(stab_x, all_in_r(r)[1::2])
        stab_z = np.append(stab_z, all_in_r(r)[::2])
    return stab_x, stab_z

def data_list(rmax): # rmax is equiv to code distance, here all the data qubits
    datas = np.array([], dtype=int)
    # datas = np.append(datas, 0)
    for r in arange(rmax)[::2]:
        datas = np.append(datas, all_in_r(r))
    return datas

def all_qubits_list(rmax): # all the qubits with leaves for d=rmax code
    datas = np.array([], dtype=int)
    datas = np.append(datas, data_list(rmax)) # append all data qubits
    stab_x, stab_z = stab_list(rmax)
    leaves_x, leaves_z = leaves(rmax)
    datas = np.append(datas, stab_x) # append all inner X stabs
    datas = np.append(datas, stab_z) # append all inner Z stabs
    datas = np.append(datas, leaves_x) # append all X leaves
    datas = np.append(datas, leaves_z) # append all Z leaves
    return datas
    
def all_stabs_list(rmax): # all the stabs with leaves for d=rmax code
    # reordered so that merge checks are the last
    datas = np.array([], dtype=int)
    stab_x, stab_z = stab_list(rmax-2) # inner checks
    leaves_x, leaves_z = leaves(rmax)
    datas = np.append(datas, stab_x) # append all inner X stabs of smaller code
    datas = np.append(datas, stab_z) # append all inner Z stabs of smaller code
    datas = np.append(datas, leaves_x) # append all X leaves
    datas = np.append(datas, leaves_z) # append all Z leaves
    datas = np.append(datas, all_in_r(rmax-2)) # append merge region
    return datas

def all_stabs_list_v1(rmax): # all the stabs without leaves for d=rmax code
    # reordered so that merge checks are the last
    datas = np.array([], dtype=int)
    stab_x, stab_z = stab_list(rmax-2) # inner checks
    leaves_x, leaves_z = leaves(rmax)
    datas = np.append(datas, stab_x) # append all inner X stabs of smaller code
    datas = np.append(datas, stab_z) # append all inner Z stabs of smaller code
    # datas = np.append(datas, leaves_x) # append all X leaves
    # datas = np.append(datas, leaves_z) # append all Z leaves
    datas = np.append(datas, all_in_r(rmax-2)) # append merge region
    return datas
    
def x_stabs_all(rmax): # all the X stabs with leaves for d=rmax code
    stabs = np.array([], dtype=int)
    stab_x, stab_z = stab_list(rmax)
    leaves_x, leaves_z = leaves(rmax)
    stabs = np.append(stabs, stab_x) # append all inner X stabs
    stabs = np.append(stabs, leaves_x) # append all X leaves
    return stabs

def z_stabs_all(rmax): # all the Z stabs with leaves for d=rmax code
    stabs = np.array([], dtype=int)
    stab_x, stab_z = stab_list(rmax)
    leaves_x, leaves_z = leaves(rmax)
    stabs = np.append(stabs, stab_z) # append all inner X stabs
    stabs = np.append(stabs, leaves_z) # append all X leaves
    return stabs

def stabs_all(rmax): # all the X stabs with leaves for d=rmax code
    stabs_x = np.array([], dtype=int)
    stabs_z = np.array([], dtype=int)
    stab_x, stab_z = stab_list(rmax)
    leaves_x, leaves_z = leaves(rmax)
    stabs_z = np.append(stabs_z, stab_z) # append all inner X stabs
    stabs_z = np.append(stabs_z, leaves_z) # append all X leaves
    stabs_x = np.append(stabs_x, stab_x) # append all inner Z stabs
    stabs_x = np.append(stabs_x, leaves_x) # append all Z leaves
    return stabs_x, stabs_z  

def deterministic_stabs(rmax):
    x_stabs_all, z_stabs_all = stabs_all(rmax)
    x_stabs = np.array([], dtype=int)
    z_stabs = np.array([], dtype=int)
    for r in range(3, rmax+1)[::2]:
        for a in range(4)[int(((rmax-1)/2+1)%2)::2]:
            for b in range(r//2+1):
                x_stabs = np.append(x_stabs, n_polar(r,a,b))
        for a in range(4)[int(((rmax-1)/2)%2)::2]:
            for b in range(r)[::-1][:r//2-1]:
                x_stabs = np.append(x_stabs, n_polar(r,a,b))
                
        for a in range(4)[int(((rmax-1)/2)%2)::2]:
            for b in range(r//2+1):
                z_stabs = np.append(z_stabs, n_polar(r,a,b))
        for a in range(4)[int(((rmax-1)/2+1)%2)::2]:
            for b in range(r)[::-1][:r//2-1]:
                z_stabs = np.append(z_stabs, n_polar(r,a,b))
                
    return np.intersect1d(x_stabs, x_stabs_all), np.intersect1d(z_stabs, z_stabs_all)  

def leaves(rmax): # rmax is equiv to code distance
    leaves_x = np.array([], dtype=int)
    leaves_z = np.array([], dtype=int)
    # if ((rmax-1)/2)%2==1: # d=3,7 etc, blue X on top
    for b in arange(rmax)[1::2]:
        leaves_x = np.append(leaves_x, n_polar(rmax, 0, b))
        leaves_z = np.append(leaves_z, n_polar(rmax, 1, b))
        leaves_x = np.append(leaves_x, n_polar(rmax, 2, b))
        leaves_z = np.append(leaves_z, n_polar(rmax, 3, b))

    return leaves_x, leaves_z
    
def leaves_alt(rmax): # rmax is equiv to code distance
    leaves_x = np.array([], dtype=int)
    leaves_z = np.array([], dtype=int)
    if ((rmax-1)/2)%2==1: # d=3,7 etc, blue X on top
        for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
            leaves_x = np.append(leaves_x, n_polar(rmax, 0, b))
            leaves_z = np.append(leaves_z, n_polar(rmax, 1, b))
        for b in arange(rmax)[1::2]:
            leaves_x = np.append(leaves_x, n_polar(rmax, 2, b))
            leaves_z = np.append(leaves_z, n_polar(rmax, 3, b))
    elif ((rmax-1)/2)%2==0: # d=5,9 etc, red Z on top
        for b in arange(rmax)[2::2]:
            leaves_z = np.append(leaves_z, n_polar(rmax, 0, b))
            leaves_x = np.append(leaves_x, n_polar(rmax, 1, b))
        for b in arange(rmax)[2::2]:
            leaves_z = np.append(leaves_z, n_polar(rmax, 2, b))
            leaves_x = np.append(leaves_x, n_polar(rmax, 3, b))
    return leaves_x, leaves_z

def neighbour_data(r, a, b): # for stabilizers
    data_ns = np.array([], dtype=int)
    if b%r == 0: # on corners
        if b==r:
            a+=1
            data_ns = np.append(data_ns, n_polar(r+1, a, 0))
            data_ns = np.append(data_ns, n_polar(r+1, a, 1))
            data_ns = np.append(data_ns, n_polar(r-1, a, 0))
            data_ns = np.append(data_ns, n_polar(r+1, a-1, r))
        elif b==0:
            data_ns = np.append(data_ns, n_polar(r+1, a, 0))
            data_ns = np.append(data_ns, n_polar(r+1, a, 1))
            data_ns = np.append(data_ns, n_polar(r-1, a, 0))
            data_ns = np.append(data_ns, n_polar(r+1, a-1, r))
        
    else:
        data_ns = np.append(data_ns, n_polar(r+1, a, b))
        data_ns = np.append(data_ns, n_polar(r+1, a, b+1))
        data_ns = np.append(data_ns, n_polar(r-1, a, b))
        data_ns = np.append(data_ns, n_polar(r-1, a, b-1))
        
    return np.roll(data_ns, a)

def all_checks(rmax): # measure all the stabilizers of d=rmax code (with leaves)
    CX_strings = [[],[],[],[]]
    
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append(target_data)
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append(target_data)

    # append leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[(a+2)%4].append(leaf_n)
            CX_strings[(a+2)%4].append(target_data[(a+2)%4])
            CX_strings[(a+3)%4].append(leaf_n)
            CX_strings[(a+3)%4].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[(a+2)%4].append(target_data[(a+2)%4])
            CX_strings[(a+2)%4].append(leaf_n)
            CX_strings[(a+3)%4].append(target_data[(a+3)%4])
            CX_strings[(a+3)%4].append(leaf_n)
    
    return CX_strings

def all_checks_upd(rmax): # measure all the stabilizers of d=rmax code (with leaves)
    CX_strings = [[],[],[],[]]
   
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    
                    if (a+b)%2==0: # Z stab
                        # CX_strings[i].append(target_data[i])
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order][i])
                        # CX_strings[i].append(target_data[i])

    # append leaves 
    shift = 3      
    for b in arange(rmax)[int(((shift-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((shift-1)/2+1)%2)::2]: # X stabs
            leaf_n = n_polar(rmax, a, b)
            # target_data_ord = [neighbour_data(rmax, a, b)[k] for k in x_order]
            target_data = neighbour_data(rmax, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((shift-1)/2+0)%2)::2]: # Z stabs
            leaf_n = n_polar(rmax, a, b)
            # target_data_ord = [neighbour_data(rmax, a, b)[k] for k in z_order]
            target_data = neighbour_data(rmax, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)

    # if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
    #     CX_strings = [CX_strings[i] for i in [0,2,1,3]]
    
    # CX_strings = [CX_strings[i] for i in [0,2,1,3]]

    return CX_strings[::-1]

def all_checks_upd_nox(rmax): # measure all the stabilizers of d=rmax code (with leaves)
    CX_strings = [[],[],[],[]]
   
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    
                    if (a+b)%2==0: # Z stab
                        # CX_strings[i].append(target_data[i])
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    # elif (a+b)%2==1: # X stab
                    #     CX_strings[i].append(stab_num)
                    #     CX_strings[i].append([target_data[k] for k in x_order][i])

    # append leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        # for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
        #     leaf_n = n_polar(rmax, a, b)
        #     # target_data_ord = [neighbour_data(rmax, a, b)[k] for k in x_order]
        #     target_data = neighbour_data(rmax, a, b)
        #     CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
        #     CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
        #     CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
        #     CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            leaf_n = n_polar(rmax, a, b)
            # target_data_ord = [neighbour_data(rmax, a, b)[k] for k in z_order]
            target_data = neighbour_data(rmax, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)

    # if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
    #     CX_strings = [CX_strings[i] for i in [0,2,1,3]]
    
    return CX_strings[::-1]

def all_det_checks_upd(rmax): # measure all the deterministic stabilizers of d=rmax code (with leaves)
    CX_strings = [[],[],[],[]]
    x_stabs_det, z_stabs_det = deterministic_stabs(rmax)

    if int(((rmax-1)/2+1)%2)==0: # d=3,7,etc, normal order
        z_order1 = z_order
        z_order_leaf1 = z_order_leaf
        x_order1 = x_order
        x_order_leaf1 = x_order_leaf
    elif int(((rmax-1)/2+1)%2)==1: # d=3,7,etc, rotated order
        z_order1 = x_order
        z_order_leaf1 = x_order_leaf
        x_order1 = z_order
        x_order_leaf1 = z_order_leaf
   
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    
                    if stab_num in z_stabs_det: # Z stab
                        # CX_strings[i].append(target_data[i])
                        CX_strings[i].append([target_data[k] for k in z_order1][i])
                        CX_strings[i].append(stab_num)
                    elif stab_num in x_stabs_det: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order1][i])
                        # CX_strings[i].append(target_data[i])

    # append leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            if leaf_n in x_stabs_det:
                CX_strings[x_order_leaf1[(a+2)%4]].append(leaf_n)
                CX_strings[x_order_leaf1[(a+2)%4]].append(target_data[(a+2)%4])
                CX_strings[x_order_leaf1[(a+3)%4]].append(leaf_n)
                CX_strings[x_order_leaf1[(a+3)%4]].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            if leaf_n in z_stabs_det:
                CX_strings[z_order_leaf1[(a+2)%4]].append(target_data[(a+2)%4])
                CX_strings[z_order_leaf1[(a+2)%4]].append(leaf_n)
                CX_strings[z_order_leaf1[(a+3)%4]].append(target_data[(a+3)%4])
                CX_strings[z_order_leaf1[(a+3)%4]].append(leaf_n)
    
    if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
        CX_strings = [CX_strings[i] for i in [0,2,1,3]]

    return CX_strings[::-1]

def intermediate_stabs(rmax):
    xstabs = []
    zstabs = []
    
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    zstabs.append(stab_num)
                if (a+b)%2==1: # X stab
                    xstabs.append(stab_num)
                    
    # append former leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            leaf_n = n_polar(rmax, a, b)
            xstabs.append(leaf_n)
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            leaf_n = n_polar(rmax, a, b)
            zstabs.append(leaf_n)

    # append new leaves: outer + side
    rmax2 = rmax + 2
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax2-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            xstabs.append(leaf_n)

            # append new leaves: side
            shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
            leaf_n = n_polar(rmax, a, b+shift)
            xstabs.append(leaf_n)
          
        for a in arange(4)[int(((rmax2-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            zstabs.append(leaf_n)

            # append new leaves: side
            shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
            leaf_n = n_polar(rmax, a, b+shift)
            zstabs.append(leaf_n)

    return xstabs, zstabs

def all_checks_intermediate(rmax): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    CX_strings = [[],[],[],[]]
    
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append(target_data)
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append(target_data)
    
    # append former leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)[i]
                CX_strings[i].append(leaf_n)
                CX_strings[i].append(target_data)
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)[i]
                CX_strings[i].append(target_data)
                CX_strings[i].append(leaf_n)

    # append new leaves: outer + side
    rmax2 = rmax + 2
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax2-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            target_data = neighbour_data(rmax2, a, b+1)
            CX_strings[(a+2)%4].append(leaf_n)
            CX_strings[(a+2)%4].append(target_data[(a+2)%4])
            CX_strings[(a+3)%4].append(leaf_n)
            CX_strings[(a+3)%4].append(target_data[(a+3)%4])

            # append new leaves: side
            shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
            leaf_n = n_polar(rmax, a, b+shift)
            target_data = neighbour_data(rmax, a, b+shift)
            CX_strings[(a+shift+2)%4].append(leaf_n)
            CX_strings[(a+shift+2)%4].append(target_data[(a+shift+2)%4])
            CX_strings[(a+shift+3)%4].append(leaf_n)
            CX_strings[(a+shift+3)%4].append(target_data[(a+shift+3)%4])
            
        for a in arange(4)[int(((rmax2-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            target_data = neighbour_data(rmax2, a, b+1)
            CX_strings[(a+2)%4].append(target_data[(a+2)%4])
            CX_strings[(a+2)%4].append(leaf_n)
            CX_strings[(a+3)%4].append(target_data[(a+3)%4])
            CX_strings[(a+3)%4].append(leaf_n)

            # append new leaves: side
            shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
            leaf_n = n_polar(rmax, a, b+shift)
            target_data = neighbour_data(rmax, a, b+shift)
            CX_strings[(a+shift+2)%4].append(target_data[(a+shift+2)%4])
            CX_strings[(a+shift+2)%4].append(leaf_n)
            CX_strings[(a+shift+3)%4].append(target_data[(a+shift+3)%4])
            CX_strings[(a+shift+3)%4].append(leaf_n)    
    if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
        CX_strings = [CX_strings[i] for i in [0,2,1,3]]            
    return CX_strings[::-1]

def all_checks_intermediate_upd(rmax): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    CX_strings = [[],[],[],[]]
    
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order][i])
    
    # append former leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)#[i]
                CX_strings[i].append(leaf_n)
                CX_strings[i].append([target_data[k] for k in x_order][i])
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)#[i]
                CX_strings[i].append([target_data[k] for k in z_order][i])
                CX_strings[i].append(leaf_n)

    # append new leaves: outer + side
    rmax2 = rmax + 2
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax2-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            target_data = neighbour_data(rmax2, a, b+1)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])

            # append new leaves: side
            shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
            leaf_n = n_polar(rmax, a, b+shift)
            target_data = neighbour_data(rmax, a, b+shift)
            CX_strings[x_order_leaf[(a+shift+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+shift+2)%4]].append(target_data[(a+shift+2)%4])
            CX_strings[x_order_leaf[(a+shift+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+shift+3)%4]].append(target_data[(a+shift+3)%4])
            
        for a in arange(4)[int(((rmax2-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            target_data = neighbour_data(rmax2, a, b+1)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)

            # append new leaves: side
            shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
            leaf_n = n_polar(rmax, a, b+shift)
            target_data = neighbour_data(rmax, a, b+shift)
            CX_strings[z_order_leaf[(a+shift+2)%4]].append(target_data[(a+shift+2)%4])
            CX_strings[z_order_leaf[(a+shift+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+shift+3)%4]].append(target_data[(a+shift+3)%4])
            CX_strings[z_order_leaf[(a+shift+3)%4]].append(leaf_n)    
    if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
        CX_strings = [CX_strings[i] for i in [0,2,1,3]]            
    return CX_strings[::-1]

def all_checks_intermediate_upd_fixed(rmax): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    CX_strings = [[],[],[],[]]
    
    # only inner stabs
    for r in arange(rmax-1)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order][i])
    
    # append former leaves     
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)#[i]
                CX_strings[i].append(leaf_n)
                CX_strings[i].append([target_data[k] for k in x_order][i])
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)#[i]
                CX_strings[i].append([target_data[k] for k in z_order][i])
                CX_strings[i].append(leaf_n)
                
    # append inbetween leaves      
    for b in arange(rmax-1)[int(((rmax-1)/2+1)%2+2)::2]:
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # X stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)#[i]
                CX_strings[i].append(leaf_n)
                CX_strings[i].append([target_data[k] for k in x_order][i])
            
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # Z stabs
            for i in arange(4):
                leaf_n = n_polar(rmax, a, b)
                target_data = neighbour_data(rmax, a, b)#[i]
                CX_strings[i].append([target_data[k] for k in z_order][i])
                CX_strings[i].append(leaf_n)             

    # append new leaves: outer
    rmax2 = rmax + 2
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax2-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            target_data = neighbour_data(rmax2, a, b+1)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((rmax2-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(rmax2, a, b+1)
            target_data = neighbour_data(rmax2, a, b+1)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)

    # append new leaves: side (always only 4)
    if (int(((rmax-1)/2+1)%2)==0): #d=3,7..
        b = 1
    elif (int(((rmax-1)/2+1)%2)==1): #d=5,9..
        b = rmax-1

    for a in arange(4)[int(((rmax2-1)/2+1)%2)::2]: # X stabs
        shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
        leaf_n = n_polar(rmax, a, b+shift)
        target_data = neighbour_data(rmax, a, b+shift)
        CX_strings[x_order_leaf[(a+shift+2)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift+2)%4]].append(target_data[(a+shift+2)%4])
        CX_strings[x_order_leaf[(a+shift+3)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift+3)%4]].append(target_data[(a+shift+3)%4])

    for a in arange(4)[int(((rmax2-1)/2+0)%2)::2]: # Z stabs
        shift = int(((rmax-1)/2+1)%2*2-1) # shift=-1 for d=3,7.. and +1 for d=5,9..
        leaf_n = n_polar(rmax, a, b+shift)
        target_data = neighbour_data(rmax, a, b+shift)
        CX_strings[z_order_leaf[(a+shift+2)%4]].append(target_data[(a+shift+2)%4])
        CX_strings[z_order_leaf[(a+shift+2)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+shift+3)%4]].append(target_data[(a+shift+3)%4])
        CX_strings[z_order_leaf[(a+shift+3)%4]].append(leaf_n)    

    if int(((rmax-1)/2+1)%2) == 0: # d=5,9..
        CX_strings = [CX_strings[i] for i in [0,2,1,3]]
    return CX_strings[::-1]

def all_checks_intermediate_3d4(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    CX_strings = [[],[],[],[]]
    rmax = 3*start_d-4

    # only inner stabs of the small code
    for r in arange(start_d)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order][i])
    
    # only inner stabs of patches
    for r in range(start_d-1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(r//2, r//2+(start_d-2)):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order][i])             

    # append new leaves: outer
    r = rmax
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)

    # append new leaves: side right up
    r = rmax-2
    b = r//2+(start_d-2)
    shift = -1
    for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
    for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)

    # append new leaves: side left down
    r = start_d+2
    b = r//2-1
    shift = 1
    for a in range(4)[int(((start_d-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
    for a in arange(4)[int(((start_d-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)   

    # append new additional leaves: side right down
    r = start_d
    b = r//2+(start_d-2)
    shift = -1
    for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
    for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)
    if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
        CX_strings = [CX_strings[i] for i in [0,2,1,3]]
    return CX_strings[::-1] 

def all_checks_intermediate_3d4_list(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    stabs_x = np.array([], dtype=int)
    stabs_z = np.array([], dtype=int)
    rmax = 3*start_d-4

    # only inner stabs of the small code
    for r in arange(start_d)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)
    
    # only inner stabs of patches
    for r in range(start_d-1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(r//2, r//2+(start_d-2)):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)            

    # append new leaves: outer
    r = rmax
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_x = np.append(stabs_x, leaf_n)
            
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_z = np.append(stabs_z, leaf_n)

    # append new leaves: side right up
    r = rmax-2
    b = r//2+(start_d-2)
    for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_x = np.append(stabs_x, leaf_n)
    for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_z = np.append(stabs_z, leaf_n)

    # append new leaves: side left down
    r = start_d+2
    b = r//2-1
    for a in range(4)[int(((start_d-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_x = np.append(stabs_x, leaf_n)
    for a in arange(4)[int(((start_d-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_z = np.append(stabs_z, leaf_n)  

    # append new additional leaves: side right down
    r = start_d
    b = r//2+(start_d-2)
    for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_x = np.append(stabs_x, leaf_n)
    for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_z = np.append(stabs_z, leaf_n)

    return stabs_x, stabs_z

def all_checks_intermediate_3d4_list_noleaf(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    stabs_x = np.array([], dtype=int)
    stabs_z = np.array([], dtype=int)
    rmax = 3*start_d-4

    # only inner stabs of the small code
    for r in arange(start_d)[1::2]: 
        for a in arange(4):
            for b in arange(r):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)
    
    # only inner stabs of patches
    for r in range(start_d-1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(r//2, r//2+(start_d-2)):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)            

    # append new leaves: outer
    r = rmax
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_x = np.append(stabs_x, leaf_n)
            
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_z = np.append(stabs_z, leaf_n)

    # # append new leaves: side right up
    # r = rmax-2
    # b = r//2+(start_d-2)
    # for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_x = np.append(stabs_x, leaf_n)
    # for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_z = np.append(stabs_z, leaf_n)

    # # append new leaves: side left down
    # r = start_d+2
    # b = r//2-1
    # for a in range(4)[int(((start_d-1)/2+1)%2)::2]: # X stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_x = np.append(stabs_x, leaf_n)
    # for a in arange(4)[int(((start_d-1)/2+0)%2)::2]: # Z stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_z = np.append(stabs_z, leaf_n)  

    # # append new additional leaves: side right down
    # r = start_d
    # b = r//2+(start_d-2)
    # for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_x = np.append(stabs_x, leaf_n)
    # for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_z = np.append(stabs_z, leaf_n)

    return np.append(stabs_x, stabs_z)

def all_checks_patches_3d4_1_list(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    # CX_strings = [[],[],[],[]]
    stabs_x = np.array([], dtype=int)
    stabs_z = np.array([], dtype=int)
    rmax = 3*start_d-4
    
    # only inner stabs
    for r in range(start_d+1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(r//2, r//2+(start_d-2)):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)            

    # append new leaves: outer
    r = rmax
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_x = np.append(stabs_x, leaf_n)            
     
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_z = np.append(stabs_z, leaf_n)

    # append new leaves: inner
    r = start_d
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_x = np.append(stabs_x, leaf_n)
            
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_z = np.append(stabs_z, leaf_n)

    # append new leaves: side right up
    r = rmax-2
    b = r//2+(start_d-2)
    for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_x = np.append(stabs_x, leaf_n)

    for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_z = np.append(stabs_z, leaf_n)

    # append new leaves: side left down
    r = start_d+2
    b = r//2-1
    for a in range(4)[int(((start_d-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_x = np.append(stabs_x, leaf_n)

    for a in arange(4)[int(((start_d-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        stabs_z = np.append(stabs_z, leaf_n)   

    return stabs_x, stabs_z

def all_checks_patches_3d4_1_list_noleaf(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    # CX_strings = [[],[],[],[]]
    stabs_x = np.array([], dtype=int)
    stabs_z = np.array([], dtype=int)
    rmax = 3*start_d-4
    
    # only inner stabs
    for r in range(start_d+1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(r//2, r//2+(start_d-2)):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)            

    # append new leaves: outer
    r = rmax
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_x = np.append(stabs_x, leaf_n)            
     
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            stabs_z = np.append(stabs_z, leaf_n)

    # # append new leaves: inner
    # r = start_d
    # for b in range(r//2, r//2+(start_d-2))[::2]:
    #     for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
    #         leaf_n = n_polar(r, a, b)
    #         target_data = neighbour_data(r, a, b)
    #         stabs_x = np.append(stabs_x, leaf_n)
            
    #     for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
    #         leaf_n = n_polar(r, a, b)
    #         target_data = neighbour_data(r, a, b)
    #         stabs_z = np.append(stabs_z, leaf_n)

    # # append new leaves: side right up
    # r = rmax-2
    # b = r//2+(start_d-2)
    # for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_x = np.append(stabs_x, leaf_n)

    # for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_z = np.append(stabs_z, leaf_n)

    # # append new leaves: side left down
    # r = start_d+2
    # b = r//2-1
    # for a in range(4)[int(((start_d-1)/2+1)%2)::2]: # X stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_x = np.append(stabs_x, leaf_n)

    # for a in arange(4)[int(((start_d-1)/2+0)%2)::2]: # Z stabs
    #     leaf_n = n_polar(r, a, b)
    #     target_data = neighbour_data(r, a, b)
    #     stabs_z = np.append(stabs_z, leaf_n)   

    return np.append(stabs_x, stabs_z)

def all_checks_patches_3d4_1(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    CX_strings = [[],[],[],[]]
    rmax = 3*start_d-4
    
    # only inner stabs
    for r in range(start_d+1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(r//2, r//2+(start_d-2)):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order][i])             

    # append new leaves: outer
    r = rmax
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)

    # append new leaves: inner
    r = start_d
    for b in range(r//2, r//2+(start_d-2))[::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[x_order_leaf[(a+0)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+0)%4]].append(target_data[(a+0)%4])
            CX_strings[x_order_leaf[(a+1)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+1)%4]].append(target_data[(a+1)%4])
            
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[z_order_leaf[(a+0)%4]].append(target_data[(a+0)%4])
            CX_strings[z_order_leaf[(a+0)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+1)%4]].append(target_data[(a+1)%4])
            CX_strings[z_order_leaf[(a+1)%4]].append(leaf_n)

    # append new leaves: side right up
    r = rmax-2
    b = r//2+(start_d-2)
    shift = -1
    for a in range(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
    for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)

    # append new leaves: side left down
    r = start_d+2
    b = r//2-1
    shift = 1
    for a in range(4)[int(((start_d-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
    for a in arange(4)[int(((start_d-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)   
    if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
        CX_strings = [CX_strings[i] for i in [0,2,1,3]]
    return CX_strings[::-1]

def all_checks_patches_3d4_2(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    CX_strings = [[],[],[],[]]
    rmax = 3*start_d-4
    
    # only inner stabs
    for r in range(start_d+1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(0, r//2-1):
                for i in arange(4):
                    target_data = neighbour_data(r, a, b)#[i]
                    stab_num = n_polar(r, a, b)
                    if (a+b)%2==0: # Z stab
                        CX_strings[i].append([target_data[k] for k in z_order][i])
                        CX_strings[i].append(stab_num)
                    elif (a+b)%2==1: # X stab
                        CX_strings[i].append(stab_num)
                        CX_strings[i].append([target_data[k] for k in x_order][i])             
    r=rmax-2
    b = r-1
    for a in range(4):
        target_data = neighbour_data(r, a, b)#[i]
        stab_num = n_polar(r, a, b)
        for i in arange(4):
            if (a+b)%2==0: # Z stab
                CX_strings[i].append([target_data[k] for k in z_order][i])
                CX_strings[i].append(stab_num)
            elif (a+b)%2==1: # X stab
                CX_strings[i].append(stab_num)
                CX_strings[i].append([target_data[k] for k in x_order][i]) 


    # append new leaves: outer
    r = rmax
    shift = 2
    for b in range(r//2)[1::2]:
        for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
            CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
            
        for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
            # append new leaves: outer
            leaf_n = n_polar(r, a, b)
            target_data = neighbour_data(r, a, b)
            CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
            CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
            CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)

    # append new leaves: inner 1
    r = start_d
    b = 0
    shift = 0
    for a in arange(4)[int(((r-1)/2+1)%2)::2]: # X stabs
        # append new leaves: outer
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        
    for a in arange(4)[int(((r-1)/2+0)%2)::2]: # Z stabs
        # append new leaves: outer
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)

    # append new leaves: inner 2
    r = rmax-2
    b = r-2
    shift = 1
    for a in arange(4)[int(((r-1)/2+0)%2)::2]: # X stabs
        # append new leaves: outer
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        
    for a in arange(4)[int(((r-1)/2+1)%2)::2]: # Z stabs
        # append new leaves: outer
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)

    # append new leaves: side right down
    r = start_d+2
    b = r//2-1
    shift = -1
    for a in range(4)[int(((start_d-1)/2+1)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
    for a in arange(4)[int(((start_d-1)/2+0)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)

    # append new leaves: side left down
    r = rmax
    b = rmax-2
    shift = 2
    for a in range(4)[int(((start_d-1)/2+0)%2)::2]: # X stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[x_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(leaf_n)
        CX_strings[x_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
    for a in arange(4)[int(((start_d-1)/2+1)%2)::2]: # Z stabs
        leaf_n = n_polar(r, a, b)
        target_data = neighbour_data(r, a, b)
        CX_strings[z_order_leaf[(a+shift)%4]].append(target_data[(a+shift)%4])
        CX_strings[z_order_leaf[(a+shift)%4]].append(leaf_n)
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(target_data[(a+1+shift)%4])
        CX_strings[z_order_leaf[(a+1+shift)%4]].append(leaf_n)   
    if int(((rmax-1)/2+1)%2) == 1: # d=5,9..
        CX_strings = [CX_strings[i] for i in [0,2,1,3]]
    return CX_strings[::-1]

def all_checks_patches_3d4_2_list(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    stabs_x = np.array([], dtype=int)
    stabs_z = np.array([], dtype=int)
    rmax = 3*start_d-4
    
    # only inner stabs
    for r in range(start_d+1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(0, r//2-1):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)             
    r=rmax-2
    b = r-1
    for a in range(4):
        target_data = neighbour_data(r, a, b)#[i]
        stab_num = n_polar(r, a, b)
        if (a+b)%2==0: # Z stab
            stabs_z = np.append(stabs_z, stab_num)
        elif (a+b)%2==1: # X stab
            stabs_x = np.append(stabs_x, stab_num)

    # append new leaves: outer
    r = rmax
    for a in range(4):
        for b in range(r//2)[1::2]:
            leaf_n = n_polar(r, a, b)
            if (a+b)%2==0: # Z stab
                stabs_z = np.append(stabs_z, leaf_n)
            elif (a+b)%2==1: # X stab
                stabs_x = np.append(stabs_x, leaf_n)

    # append new leaves: inner 1
    r = start_d
    b = 0
    for a in range(4):
        leaf_n = n_polar(r, a, b)
        if (a+b)%2==0: # Z stab
            stabs_z = np.append(stabs_z, leaf_n)
        elif (a+b)%2==1: # X stab
            stabs_x = np.append(stabs_x, leaf_n)

    # append new leaves: inner 2
    r = rmax-2
    b = r-2
    for a in range(4):
        leaf_n = n_polar(r, a, b)
        if (a+b)%2==0: # Z stab
            stabs_z = np.append(stabs_z, leaf_n)
        elif (a+b)%2==1: # X stab
            stabs_x = np.append(stabs_x, leaf_n)

    # append new leaves: side right down
    r = start_d+2
    b = r//2-1
    for a in range(4):
        leaf_n = n_polar(r, a, b)
        if (a+b)%2==0: # Z stab
            stabs_z = np.append(stabs_z, leaf_n)
        elif (a+b)%2==1: # X stab
            stabs_x = np.append(stabs_x, leaf_n)

    # append new leaves: side left down
    r = rmax
    b = rmax-2
    for a in range(4):
        leaf_n = n_polar(r, a, b)
        if (a+b)%2==0: # Z stab
            stabs_z = np.append(stabs_z, leaf_n)
        elif (a+b)%2==1: # X stab
            stabs_x = np.append(stabs_x, leaf_n)

    return stabs_x, stabs_z

def all_checks_patches_3d4_2_list_noleaf(start_d): # measure all the stabilizers of (intermediate) d=rmax (smaller) code
    stabs_x = np.array([], dtype=int)
    stabs_z = np.array([], dtype=int)
    rmax = 3*start_d-4
    
    # only inner stabs
    for r in range(start_d+1,rmax)[1::2]: 
        for a in arange(4):
            for b in range(0, r//2-1):
                target_data = neighbour_data(r, a, b)#[i]
                stab_num = n_polar(r, a, b)
                if (a+b)%2==0: # Z stab
                    stabs_z = np.append(stabs_z, stab_num)
                elif (a+b)%2==1: # X stab
                    stabs_x = np.append(stabs_x, stab_num)             
    r=rmax-2
    b = r-1
    for a in range(4):
        target_data = neighbour_data(r, a, b)#[i]
        stab_num = n_polar(r, a, b)
        if (a+b)%2==0: # Z stab
            stabs_z = np.append(stabs_z, stab_num)
        elif (a+b)%2==1: # X stab
            stabs_x = np.append(stabs_x, stab_num)

    # append new leaves: outer
    r = rmax
    for a in range(4):
        for b in range(r//2)[1::2]:
            leaf_n = n_polar(r, a, b)
            if (a+b)%2==0: # Z stab
                stabs_z = np.append(stabs_z, leaf_n)
            elif (a+b)%2==1: # X stab
                stabs_x = np.append(stabs_x, leaf_n)

    # # append new leaves: inner 1
    # r = start_d
    # b = 0
    # for a in range(4):
    #     leaf_n = n_polar(r, a, b)
    #     if (a+b)%2==0: # Z stab
    #         stabs_z = np.append(stabs_z, leaf_n)
    #     elif (a+b)%2==1: # X stab
    #         stabs_x = np.append(stabs_x, leaf_n)

    # # append new leaves: inner 2
    # r = rmax-2
    # b = r-2
    # for a in range(4):
    #     leaf_n = n_polar(r, a, b)
    #     if (a+b)%2==0: # Z stab
    #         stabs_z = np.append(stabs_z, leaf_n)
    #     elif (a+b)%2==1: # X stab
    #         stabs_x = np.append(stabs_x, leaf_n)

    # # append new leaves: side right down
    # r = start_d+2
    # b = r//2-1
    # for a in range(4):
    #     leaf_n = n_polar(r, a, b)
    #     if (a+b)%2==0: # Z stab
    #         stabs_z = np.append(stabs_z, leaf_n)
    #     elif (a+b)%2==1: # X stab
    #         stabs_x = np.append(stabs_x, leaf_n)

    # # append new leaves: side left down
    # r = rmax
    # b = rmax-2
    # for a in range(4):
    #     leaf_n = n_polar(r, a, b)
    #     if (a+b)%2==0: # Z stab
    #         stabs_z = np.append(stabs_z, leaf_n)
    #     elif (a+b)%2==1: # X stab
    #         stabs_x = np.append(stabs_x, leaf_n)

    return np.append(stabs_x, stabs_z)

def grow_distance(rmax): # grow from d=(rmax-2) to rmax
    # define new qubits around
    # entangle pairs around (where will be the new leaves)
    # measure checks of intermediate code
    # measure checks of the full code
    
    # define new qubits around
    full_string = np.array([""])
    new_list = np.array([], dtype=int)
    for r in [rmax-1,rmax]:
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                new_list = np.append(new_list, n_polar(r, a, b))

    # reset new qubits
    reset = ' R '+' '.join(str(n) for n in new_list)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)    
    
    # entangle data pairs around (where will be the new leaves)
    data_ns = np.roll(all_in_r(rmax-1),int(((rmax+1)/2)%2))
    epr = ' CX '+' '.join(str(n) for n in data_ns)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, epr)

    #########################################
    # measure checks of intermediate code
    x_stab_inter, z_stab_inter = intermediate_stabs(rmax-2)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(x_stab_inter+z_stab_inter)
    full_string = np.char.add(full_string, detectors)
    #########################################


    #########################################
    # measure checks of the full code
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)    

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    #########################################
        
    return full_string[0]

def add_bell_pairs(rmax):# to grow from d=(rmax-2) to rmax
    
    full_string = np.array([""])
    new_list = np.array([], dtype=int)   

    # init data qubits in X basis 
    shift = int(((rmax)/2)%2)
    a = all_in_r(rmax-1)
    data_ns = np.roll(a,shift-1).reshape(4,-1)[shift::2].flatten()
    had_bell = ' H '+' '.join(str(n) for n in data_ns)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, had_bell)
    leaves_x, leaves_z = leaves(rmax)

    # checks
    CX_strings = [[],[],[],[]]
    # append leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in leaves_x)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.append(leaves_x,leaves_z)
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # full_string = np.char.add(full_string, detector_single(all_leaves))
    
    return full_string

def pairs_inner(rmax):
    leaves_x = np.array([], dtype=int)
    leaves_z = np.array([], dtype=int)
    if ((rmax-1)/2)%2==0: # d=3,7 etc, blue X on top
        for b in arange(rmax)[1::2]:
            leaves_x = np.append(leaves_x, n_polar(rmax-2, 0, b))
            leaves_z = np.append(leaves_z, n_polar(rmax-2, 1, b))
        for b in arange(rmax)[1::2]:
            leaves_x = np.append(leaves_x, n_polar(rmax-2, 2, b))
            leaves_z = np.append(leaves_z, n_polar(rmax-2, 3, b))
    elif ((rmax-1)/2)%2==1: # d=5,9 etc, red Z on top
        for b in arange(rmax)[1::2]:
            leaves_z = np.append(leaves_z, n_polar(rmax-2, 0, b))
            leaves_x = np.append(leaves_x, n_polar(rmax-2, 1, b))
        for b in arange(rmax)[1::2]:
            leaves_z = np.append(leaves_z, n_polar(rmax-2, 2, b))
            leaves_x = np.append(leaves_x, n_polar(rmax-2, 3, b))    
    return leaves_x, leaves_z
    
def add_bell_pairs_zx(rmax):# to grow from d=(rmax-2) to rmax
    
    full_string = np.array([""])
    new_list = np.array([], dtype=int)   
    leaves_x, leaves_z = leaves(rmax)
    leaves_x_in, leaves_z_in = pairs_inner(rmax)

    # checks
    CX_strings = [[],[],[],[]]
    # append leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])

            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[z_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[z_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[z_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            CX_strings[z_order_leaf[(a+1)%4]].append(leaf_n_in)
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)
            
            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[x_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[x_order_leaf[(a+1)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            
    # round 1
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in np.append(leaves_x, leaves_x_in))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x, leaves_z, leaves_x_in, leaves_z_in))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # round 2
    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x, leaves_z, leaves_x_in, leaves_z_in))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    full_string = np.char.add(full_string, detectors_parity(all_leaves))
    
    return full_string

def add_bell_pairs_zx_step2(rmax):# to grow from d=(rmax-2) to rmax
    
    full_string = np.array([""])
    new_list = np.array([], dtype=int)   
    leaves_x, leaves_z = leaves(rmax)
    leaves_x_in, leaves_z_in = pairs_inner(rmax)

    # checks
    CX_strings = [[],[],[],[]]
    # append leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)+2::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])

            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[z_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[z_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[z_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            CX_strings[z_order_leaf[(a+1)%4]].append(leaf_n_in)
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)
            
            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[x_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[x_order_leaf[(a+1)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            
    # round 1
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in np.append(leaves_x[1::2], leaves_x_in[1::2]))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x[1::2], leaves_z[1::2], leaves_x_in[1::2], leaves_z_in[1::2]))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # round 2
    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x[1::2], leaves_z[1::2], leaves_x_in[1::2], leaves_z_in[1::2]))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    full_string = np.char.add(full_string, detectors_parity(all_leaves))
    
    return full_string

def add_bell_pairs_zx_step1(rmax):# to grow from d=(rmax-2) to rmax
    
    full_string = np.array([""])
    new_list = np.array([], dtype=int)   
    leaves_x, leaves_z = leaves(rmax)
    leaves_x_in, leaves_z_in = pairs_inner(rmax)

    # checks
    CX_strings = [[],[],[],[]]
    # append leaves       
    for b in arange(rmax-2)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])

            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[z_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[z_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[z_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            CX_strings[z_order_leaf[(a+1)%4]].append(leaf_n_in)
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)
            
            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[x_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[x_order_leaf[(a+1)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            
    # round 1
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in np.append(leaves_x[::2], leaves_x_in[::2]))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x[::2], leaves_z[::2], leaves_x_in[::2], leaves_z_in[::2]))
    # all_leaves = np.concatenate((leaves_x[::2], leaves_z[::2]))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # round 2
    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x[::2], leaves_z[::2], leaves_x_in[::2], leaves_z_in[::2]))
    # all_leaves = np.concatenate((leaves_x[::2], leaves_z[::2]))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    full_string = np.char.add(full_string, detectors_parity(all_leaves))
    
    return full_string
    
def grow_distance_upd(rmax): # grow from d=(rmax-2) to rmax
    # define new qubits around
    # entangle pairs around (where will be the new leaves)
    # measure checks of intermediate code
    # measure checks of the full code
    
    # define new qubits around
    full_string = np.array([""])
    new_list = np.array([], dtype=int)
    for r in np.arange(rmax-1,rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                new_list = np.append(new_list, n_polar(r, a, b))
    for a in range(4):
        for b in range(rmax-2):
            new_list = np.append(new_list, n_polar(rmax-2, a, b))
    # reset new qubits
    reset = ' R '+' '.join(str(n) for n in new_list)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)    

    # add bell pairs via stabilizers
    # full_string = np.char.add(full_string, add_bell_pairs_zx(rmax))
    
    # directly entangle data pairs around (where will be the new leaves)
    data_ns = np.roll(all_in_r(rmax-1),int(((rmax+1)/2)%2))
    had_epr = ' H '+' '.join(str(n) for n in data_ns[::2])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, had_epr)
    epr = ' CX '+' '.join(str(n) for n in data_ns)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, epr)

    #########################################
    # measure checks of intermediate code
    x_stab_inter, z_stab_inter = intermediate_stabs(rmax-2)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(x_stab_inter+z_stab_inter)
    full_string = np.char.add(full_string, detectors)
    #########################################


    #########################################
    # measure checks of the full code
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)    

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ########################################
        
    return full_string[0]

def grow_distance_upd_debug(rmax=5, repeats=1): # grow from d=(rmax-2) to rmax
    # define new qubits around
    # entangle pairs around (where will be the new leaves)
    # measure checks of intermediate code
    # measure checks of the full code
    
    # define new qubits around
    full_string = np.array([""])
    new_list = np.array([], dtype=int)
    for r in np.arange(rmax-1,rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                new_list = np.append(new_list, n_polar(r, a, b))
    for a in range(4):
        for b in range(rmax-2):
            new_list = np.append(new_list, n_polar(rmax-2, a, b))
    # reset new qubits
    reset = ' R '+' '.join(str(n) for n in new_list)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)    

    # add bell pairs
    # full_string = np.char.add(full_string, add_bell_pairs_zx(rmax))
    
    # directly entangle data pairs around (where will be the new leaves)
    data_ns = np.roll(all_in_r(rmax-1),int(((rmax+1)/2)%2))
    had_epr = ' H '+' '.join(str(n) for n in data_ns[::2])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, had_epr)
    epr = ' CX '+' '.join(str(n) for n in data_ns)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, epr)


    # repeat d times (intermediate)
    # full_string = np.char.add(full_string, f'REPEAT {max(repeats-2,1)} {{')
    full_string = np.char.add(full_string, f'REPEAT {1} {{'+' \n ')

    #########################################
    # measure checks of intermediate code
    x_stab_inter, z_stab_inter = intermediate_stabs(rmax-2)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = check_det_1_round_RX(all_stabs_list(rmax-2), (x_stab_inter+z_stab_inter))
    full_string = np.char.add(full_string, detectors)

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(x_stab_inter+z_stab_inter)
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    #########################################
    # measure checks of the full code
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR) 
    # check that all the stabs are the same
    detectors = check_det_1_round_RX((x_stab_inter+z_stab_inter),all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)   
    # full_string = np.char.add(full_string, observe_logical_op(len(all_in_r(rmax-2))))

    # repeat d times (full)
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    return full_string[0]

def add_patches1(start_d=3):
    full_string = np.array([""])
    new_list = np.array([], dtype=int)   
    leaves_x, leaves_z = leaves(rmax)
    leaves_x_in, leaves_z_in = pairs_inner(rmax)

    # checks
    CX_strings = [[],[],[],[]]
    # append leaves       
    for b in arange(rmax)[int(((rmax-1)/2+1)%2+1)::2]:
        for a in arange(4)[int(((rmax-1)/2+1)%2)::2]: # X stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[x_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[x_order_leaf[(a+3)%4]].append(leaf_n)
            CX_strings[x_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])

            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[z_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[z_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[z_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            CX_strings[z_order_leaf[(a+1)%4]].append(leaf_n_in)
            
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]: # Z stabs
            # append outer leaves       
            leaf_n = n_polar(rmax, a, b)
            target_data = neighbour_data(rmax, a, b)
            CX_strings[z_order_leaf[(a+2)%4]].append(target_data[(a+2)%4])
            CX_strings[z_order_leaf[(a+2)%4]].append(leaf_n)
            CX_strings[z_order_leaf[(a+3)%4]].append(target_data[(a+3)%4])
            CX_strings[z_order_leaf[(a+3)%4]].append(leaf_n)
            
            # append inner leaves       
            leaf_n_in = n_polar(rmax-2, a, b-1)
            target_data_in = neighbour_data(rmax-2, a, b-1)
            CX_strings[x_order_leaf[(a)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a)%4]].append(target_data_in[(a)%4])
            CX_strings[x_order_leaf[(a+1)%4]].append(leaf_n_in)
            CX_strings[x_order_leaf[(a+1)%4]].append(target_data_in[(a+1)%4])
            
    # round 1
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in np.append(leaves_x, leaves_x_in))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x, leaves_z, leaves_x_in, leaves_z_in))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # round 2
    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in CX_strings[::-1][i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all bell leaves checks
    all_leaves = np.concatenate((leaves_x, leaves_z, leaves_x_in, leaves_z_in))
    MR = ' MR '+' '.join(str(n) for n in all_leaves)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    full_string = np.char.add(full_string, detectors_parity(all_leaves))  

    return full_string

def grow_distance_3d4(start_d=3, repeats=1): # grow from d=(rmax-2) to rmax
    # define new qubits around
    # entangle pairs around (where will be the new leaves)
    # measure checks of intermediate code
    # measure checks of the full code
    
    rmax = 3*start_d-4
    # define new qubits around
    full_string = np.array([""])
    new_list = np.array([], dtype=int)
    for r in np.arange(start_d+1,rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    for r in np.arange(start_d,rmax+1):
        for a in range(4):
            for b in range(r):
                new_list = np.append(new_list, n_polar(r, a, b))

    reset = ' R '+' '.join(str(n) for n in new_list)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset) 

    #########################################
    # add patches 1

    x_stab_patch1, z_stab_patch1 = all_checks_patches_3d4_1_list(start_d)   
    stabs_patches = np.append(x_stab_patch1, z_stab_patch1)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_patch1)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam) 

    for i in range(4):
        patches_check = ' CX '+' '.join(str(n) for n in all_checks_patches_3d4_1(start_d=start_d)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, patches_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    # measure all patch. stabs
    MR = ' MR '+' '.join(str(n) for n in stabs_patches)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check all Z stabs
    detectors = detector_single(z_stab_patch1)
    full_string = np.char.add(full_string, detectors)

    full_string = np.char.add(full_string, f'REPEAT {3} {{'+' \n ')
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_patch1)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam) 

    for i in range(4):
        patches_check = ' CX '+' '.join(str(n) for n in all_checks_patches_3d4_1(start_d=start_d)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, patches_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    # measure all patch. stabs
    MR = ' MR '+' '.join(str(n) for n in stabs_patches)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check all the stabs
    detectors = detectors_parity(stabs_patches)
    full_string = np.char.add(full_string, detectors)
    full_string = np.char.add(full_string, '}'+' \n ')

    # repeat d times (intermediate)
    # full_string = np.char.add(full_string, f'REPEAT {max(repeats-2,1)} {{')

    #########################################
    # measure checks of intermediate code
    x_stab_inter, z_stab_inter = all_checks_intermediate_3d4_list(start_d)
    stabs_inter = np.append(x_stab_inter,z_stab_inter)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_3d4(start_d)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (stabs_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that the stabs are the same all_checks_patches_3d4_1_list_noleaf
    # detectors = check_det_1_round_pair_grow(z_stab_inter, z_stab_patch1, len(stabs_inter))
    detectors = check_det_1_round_patches(stabs_inter, stabs_patches, all_checks_patches_3d4_1_list_noleaf(start_d), len(stabs_inter))
    full_string = np.char.add(full_string, detectors)
    # compare to previous distance
    # detectors = check_det_1_round_pair_grow(stabs_inter, np.append(stab_list(start_d)[0],stab_list(start_d)[1]), len(stabs_inter)+len(stabs_patches)*2)
    detectors = check_det_1_round_patches(stabs_inter, all_stabs_list(start_d), all_stabs_list_v1(start_d), len(stabs_inter)+len(stabs_patches)*4)
    # detectors = check_det_1_round_pair_grow(stabs_inter, stab_list(start_d), len(stabs_inter)+len(stabs_patches)*2)
    full_string = np.char.add(full_string, detectors)

    full_string = np.char.add(full_string, f'REPEAT {5} {{'+' \n ')
    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_3d4(start_d)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in stabs_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(stabs_inter)
    full_string = np.char.add(full_string, detectors)
    #########################################
    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    #########################################
    # add patches 2

    x_stab_patch2, z_stab_patch2 = all_checks_patches_3d4_2_list(start_d)   
    stabs_patches = np.append(x_stab_patch2, z_stab_patch2)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_patch2)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam) 

    for i in range(4):
        patches_check = ' CX '+' '.join(str(n) for n in all_checks_patches_3d4_2(start_d=start_d)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, patches_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    # measure all patch. stabs
    MR = ' MR '+' '.join(str(n) for n in stabs_patches)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check all Z stabs
    detectors = detector_single(z_stab_patch2)
    full_string = np.char.add(full_string, detectors)

    full_string = np.char.add(full_string, f'REPEAT {3} {{'+' \n ')
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_patch2)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam) 

    for i in range(4):
        patches_check = ' CX '+' '.join(str(n) for n in all_checks_patches_3d4_2(start_d=start_d)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, patches_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    # measure all patch. stabs
    MR = ' MR '+' '.join(str(n) for n in stabs_patches)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check all the stabs
    detectors = detectors_parity(stabs_patches)
    full_string = np.char.add(full_string, detectors)
    full_string = np.char.add(full_string, '}'+' \n ')

    #########################################

    #########################################
    # measure checks of the full code all_checks_patches_3d4_2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR) 
    # check that all the stabs are the same
    detectors = check_det_1_round_patches(all_stabs_list(rmax), stabs_inter, all_checks_intermediate_3d4_list_noleaf(start_d), len(all_stabs_list(rmax))+len(stabs_patches)*4)
    full_string = np.char.add(full_string, detectors)   
    detectors = check_det_1_round_patches(all_stabs_list(rmax), stabs_patches, all_checks_patches_3d4_2_list_noleaf(start_d), len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors) 

    # repeat d times (full)
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    return full_string[0]

def grow_distance_upd_debug_debug(rmax=5, repeats=1): # grow from d=(rmax-2) to rmax
    # define new qubits around
    # entangle pairs around (where will be the new leaves)
    # measure checks of intermediate code
    # measure checks of the full code
    
    # define new qubits around
    full_string = np.array([""])
    new_list = np.array([], dtype=int)
    for r in np.arange(rmax-1,rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                new_list = np.append(new_list, n_polar(r, a, b))
    for a in range(4):
        for b in range(rmax-2):
            new_list = np.append(new_list, n_polar(rmax-2, a, b))
    # reset new qubits
    reset = ' R '+' '.join(str(n) for n in new_list)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)    

    # add bell pairs
    # full_string = np.char.add(full_string, add_bell_pairs_zx(rmax))
    
    # directly entangle data pairs around (where will be the new leaves)
    data_ns = np.roll(all_in_r(rmax-1),int(((rmax+1)/2)%2))
    had_epr = ' H '+' '.join(str(n) for n in data_ns[::2])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, had_epr)
    epr = ' CX '+' '.join(str(n) for n in data_ns)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, epr)


    # repeat d times (intermediate)
    # full_string = np.char.add(full_string, f'REPEAT {max(repeats-2,1)} {{')

    #########################################
    # measure checks of intermediate code
    x_stab_inter, z_stab_inter = intermediate_stabs(rmax-2)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = check_det_1_round_RX(all_stabs_list(rmax-2), (x_stab_inter+z_stab_inter))
    full_string = np.char.add(full_string, detectors)

    full_string = np.char.add(full_string, f'REPEAT {2} {{'+' \n ')
    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(x_stab_inter+z_stab_inter)
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    #########################################
    # measure checks of the full code
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR) 
    # check that all the stabs are the same
    detectors = check_det_1_round_RX((x_stab_inter+z_stab_inter),all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)   
    # full_string = np.char.add(full_string, observe_logical_op(len(all_in_r(rmax-2))))

    # repeat d times (full)
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    return full_string[0]

def grow_distance_upd_debug_pair(rmax=5, repeats=1, shift_coord=15, shift_qubits=100): # grow from d=(rmax-2) to rmax
    # define new qubits around
    # entangle pairs around (where will be the new leaves)
    # measure checks of intermediate code
    # measure checks of the full code
    
    # define new qubits around
    full_string = np.array([""])
    new_list = np.array([], dtype=int)
    new_list2 = np.array([], dtype=int)
    for r in np.arange(rmax-1,rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))
                new_list = np.append(new_list, n_polar(r, a, b))
                new_list2 = np.append(new_list2, n_polar(r, a, b)+shift_qubits)
    for a in range(4):
        for b in range(rmax-2):
            new_list = np.append(new_list, n_polar(rmax-2, a, b))
            new_list2 = np.append(new_list2, n_polar(rmax-2, a, b)+shift_qubits)
    # reset new qubits
    reset = ' R '+' '.join(str(n) for n in new_list)+'  \n  '
    reset2 = ' R '+' '.join(str(n+shift_qubits) for n in new_list)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)    
    full_string = np.char.add(full_string, reset2)    

    # add bell pairs
    # full_string = np.char.add(full_string, add_bell_pairs_zx(rmax))
    
    # directly entangle data pairs around (where will be the new leaves)
    data_ns = np.roll(all_in_r(rmax-1),int(((rmax+1)/2)%2))
    had_epr = ' H '+' '.join(str(n) for n in data_ns[::2])+'  \n  '
    had_epr2 = ' H '+' '.join(str(n+shift_qubits) for n in data_ns[::2])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, had_epr)
    full_string = np.char.add(full_string, had_epr2)
    epr = ' CX '+' '.join(str(n) for n in data_ns)+'  \n  '
    epr2 = ' CX '+' '.join(str(n+shift_qubits) for n in data_ns)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, epr)
    full_string = np.char.add(full_string, epr2)


    # repeat d times (intermediate)
    # full_string = np.char.add(full_string, f'REPEAT {max(repeats-2,1)} {{')
    full_string = np.char.add(full_string, f'REPEAT {1} {{'+' \n ')

    #########################################
    # measure checks of intermediate code
    x_stab_inter, z_stab_inter = intermediate_stabs(rmax-2)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    full_string = np.char.add(full_string, hadam2)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+'  \n  '
        inter_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)
        full_string = np.char.add(full_string, inter_check2)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # # measure all intermed. stabs
    # check that all the stabs are the same
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+'  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_1_round_pair_grow(np.array(x_stab_inter+z_stab_inter),all_stabs_list(rmax-2),len((x_stab_inter+z_stab_inter))+len(all_stabs_list(rmax-2)))
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    detectors = check_det_1_round_pair_grow(np.array(x_stab_inter+z_stab_inter),all_stabs_list(rmax-2),len((x_stab_inter+z_stab_inter))*2)
    full_string = np.char.add(full_string, detectors)

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    full_string = np.char.add(full_string, hadam2)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+'  \n  '
        inter_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)
        full_string = np.char.add(full_string, inter_check2)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # # measure all intermed. stabs
    # check that all the stabs are the same
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+'  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = detectors_parity_pair(x_stab_inter+z_stab_inter, len(x_stab_inter+z_stab_inter)*2)
    full_string = np.char.add(full_string, detectors)

    #detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    detectors = detectors_parity_pair(x_stab_inter+z_stab_inter, len(x_stab_inter+z_stab_inter)*2)
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    #########################################
    # measure checks of the full code
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      
    full_string = np.char.add(full_string, hadam2)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    # check that all the stabs are the same
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR) 
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), np.array(x_stab_inter+z_stab_inter),len(x_stab_inter+z_stab_inter)+len(all_stabs_list(rmax)))
    full_string = np.char.add(full_string, detectors)   
    # check_det_1_round_pair_grow(np.array(x_stab_inter+z_stab_inter),all_stabs_list(rmax-2),len((x_stab_inter+z_stab_inter)))

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2) 
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), np.array(x_stab_inter+z_stab_inter),len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)      
    # full_string = np.char.add(full_string, observe_logical_op(len(all_in_r(rmax-2))))

    # repeat d times (full)
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      
    full_string = np.char.add(full_string, hadam2)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    # check that all the stabs are the same
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    MR = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_1_round_pair_grow(all_stabs_list(rmax), all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    return full_string[0]

def grow_distance_det(rmax=5, repeats=1): # grow from d=(rmax-2) to rmax
    # define new qubits around
    # entangle pairs around (where will be the new leaves)
    # measure checks of intermediate code
    # measure checks of the full code
    
    # define new qubits around
    full_string = np.array([""])
    new_list = np.array([], dtype=int)
    for r in np.arange(rmax-1,rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                new_list = np.append(new_list, n_polar(r, a, b))
    for a in range(4):
        for b in range(rmax-2):
            new_list = np.append(new_list, n_polar(rmax-2, a, b))
    # reset new qubits
    reset = ' R '+' '.join(str(n) for n in new_list)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)    

    # add bell pairs
    # full_string = np.char.add(full_string, add_bell_pairs_zx(rmax))
    
    # directly entangle data pairs around (where will be the new leaves)
    data_ns = np.roll(all_in_r(rmax-1),int(((rmax+1)/2)%2))
    had_epr = ' H '+' '.join(str(n) for n in data_ns[::2])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, had_epr)
    epr = ' CX '+' '.join(str(n) for n in data_ns)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, epr)


    # repeat d times (intermediate)
    # full_string = np.char.add(full_string, f'REPEAT {max(repeats-2,1)} {{')
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    #########################################
    # measure checks of intermediate code
    x_stab_inter, z_stab_inter = intermediate_stabs(rmax-2)
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stab_inter)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)   
    
    # measure checks of intermediate code
    for i in range(4):
        inter_check = ' CX '+' '.join(str(n) for n in all_checks_intermediate_upd_fixed(rmax-2)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, inter_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # # measure all intermed. stabs
    MR = ' MR '+' '.join(str(n) for n in (x_stab_inter+z_stab_inter))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(x_stab_inter+z_stab_inter)
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    # repeat d times (intermediate)
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    #########################################
    # measure checks of the full code
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)    
    # full_string = np.char.add(full_string, observe_logical_op(len(all_in_r(rmax-2))))

    # round 2
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)      

    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    #########################################

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')

    return full_string[0]

def d_init(rmax=3): # with 2 rounds of checks to measure parity
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
        
    return full_string[0]

def d_init_upd_reserve(rmax=3, repeats=1): # with 2x'repeats' rounds of checks to measure parity
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    ###############################################################    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # check deterministic stabs (Z)
    # detectors = detector_single(z_stabs_all(rmax))
    detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')
        
    return full_string[0]

def d_init_upd(rmax=3, repeats=1): # with 2x'repeats' rounds of checks to measure parity
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    ###############################################################    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # check deterministic stabs (Z)
    # detectors = detector_single(z_stabs_all(rmax))
    detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')
        
    return full_string[0]

def d_init_upd_norep(rmax=3, repeats=1): # with 2x'repeats' rounds of checks to measure parity
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    ###############################################################    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # check deterministic stabs (Z)
    # detectors = detector_single(z_stabs_all(rmax))
    detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors)    
        
    return full_string[0]

def d_init_upd_norep1(rmax=3, repeats=1): # with 2x'repeats' rounds of checks to measure parity
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    ###############################################################    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # check deterministic stabs (Z)
    # detectors = detector_single(z_stabs_all(rmax))
    detectors = check_det_single(all_stabs_list(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')
        
    return full_string[0]

def d_init_upd_nox(rmax=3, repeats=1): # with 2x'repeats' rounds of checks to measure parity
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    ###############################################################    
    # Hadamard on X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd_nox(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # # Hadamard on X stabs
    # full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in z_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)
    # check deterministic stabs (Z)
    # detectors = detector_single(z_stabs_all(rmax))
    detectors = check_det_single(z_stabs_all(rmax), z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors)    

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # # Hadamard on X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd_nox(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # # Hadamard on X stabs
    # full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in z_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(z_stabs_all(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')
        
    return full_string[0]

def init_bell_plus(rmax):
    datas = np.array([], dtype=int)
    for r in range(2, rmax)[::2]:
        for a in range(4)[int(((rmax-1)/2+1)%2)::2]:
            for b in range(r//2+1):
                datas = np.append(datas, n_polar(r,a,b))
        for a in range(4)[int(((rmax-1)/2)%2)::2]:
            for b in range(r)[::-1][:r//2-1]:
                datas = np.append(datas, n_polar(r,a,b))
    return datas

def d_init_bell(rmax=3, repeats=1): # with 2x'repeats' rounds of checks to measure parity
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    # init in special form
    init_plus = ' H '+' '.join(str(n) for n in init_bell_plus(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, init_plus)   

    # repeat d times
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    ###############################################################    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 

    # close repeats 
    full_string = np.char.add(full_string, '}'+' \n ')
        
    return full_string[0]

def add_noise(original_string, p1 = 1e-3, p2 = 1e-3, pin = 1e-3, pm = 1e-3):
    # test = ' R 1 2 3 4 5 \n  H 1 2 3 4 5 \n  MR 1 2 3 4 5 \n  R 1 2 3 4 5  \n  TICK  \n CX 1 2 3 4  \n  R 1 2 3 4 5 \n M 1 2 3  \n TICK'
    list_of_strings = original_string.split("\n")
    
    i = 0
    while i < len(list_of_strings):
        line = list_of_strings[i]
        if ' R ' in line:
            err = line.replace('R', f'X_ERROR({pin})')
            list_of_strings.insert(i+1, err)
            i += 2  # Skip both original and inserted
        elif ' RX ' in line:
            err = line.replace('RX', f'X_ERROR({pin})')
            list_of_strings.insert(i+1, err)
            i += 2  # Skip both original and inserted
        elif ' H' in line:
            err = line.replace('H', f'DEPOLARIZE1({p1})')
            list_of_strings.insert(i+1, err)
            i += 2
        elif ' CX' in line:
            err = line.replace('CX', f'DEPOLARIZE2({p2})')
            list_of_strings.insert(i+1, err)
            i += 2
        elif ' M ' in line:
            err = line.replace('M', f'X_ERROR({pm})')
            list_of_strings.insert(i, err)
            i += 2  # Skip both inserted and original
        elif ' MR ' in line:
            err_r = line.replace('MR', f'X_ERROR({pin})')
            err_m = line.replace('MR', f'X_ERROR({pm})')
            list_of_strings.insert(i, err_m)
            list_of_strings.insert(i+2, err_r)
            i += 3
        elif ' MRX ' in line:
            err_r = line.replace('MRX', f'X_ERROR({pin})')
            err_m = line.replace('MRX', f'X_ERROR({pm})')
            list_of_strings.insert(i, err_m)
            list_of_strings.insert(i+2, err_r)
            i += 3
        else:
            i += 1

    return ' \n '.join(list_of_strings)

def add_noise_inj_reserve(original_string, p1 = 1e-3, p2 = 1e-3, pin = 1e-3, pm = 1e-3, pinj = 2e-3):
    # test = ' R 1 2 3 4 5 \n  H 1 2 3 4 5 \n  MR 1 2 3 4 5 \n  R 1 2 3 4 5  \n  TICK  \n CX 1 2 3 4  \n  R 1 2 3 4 5 \n M 1 2 3  \n TICK'
    list_of_strings = original_string.split("\n")
    
    i = 0
    while i < len(list_of_strings):
        line = list_of_strings[i]
        if (' R ' in line) and (' INJECTION ' not in line):
            err = line.replace('R', f'X_ERROR({pin})')
            list_of_strings.insert(i+1, err)
            i += 2  # Skip both original and inserted
        elif (' RX ' in line) and (' INJECTION ' not in line):
            err = line.replace('RX', f'DEPOLARIZE1({p1})')
            list_of_strings.insert(i+1, err)
            i += 2 
        elif ' H' in line:
            err = line.replace('H', f'DEPOLARIZE1({p1})')
            list_of_strings.insert(i+1, err)
            i += 2
        elif (' R ' in line) and (' INJECTION ' in line):
            err = line.replace('R', f'DEPOLARIZE1({pinj})')
            list_of_strings.insert(i+1, err)
            i += 2
        elif (' CX' in line) and (' INJECTION ' not in line):
            err = line.replace('CX', f'DEPOLARIZE2({p2})')
            list_of_strings.insert(i+1, err)
            i += 2
        elif (' CX' in line) and (' INJECTION ' in line):
            err = line.replace('CX', f'DEPOLARIZE2({pinj})')
            list_of_strings.insert(i+1, err)
            i += 2
        elif ' M ' in line:
            err = line.replace('M', f'X_ERROR({pm})')
            list_of_strings.insert(i, err)
            i += 2  # Skip both inserted and original
        elif ' MX ' in line:
            err = line.replace('MX', f'DEPOLARIZE1({p1})')
            list_of_strings.insert(i, err)
            i += 2
        elif ' MR ' in line:
            err_r = line.replace('MR', f'X_ERROR({pin})')
            err_m = line.replace('MR', f'X_ERROR({pm})')
            list_of_strings.insert(i, err_m)
            list_of_strings.insert(i+2, err_r)
            i += 3
        else:
            i += 1

    return ' \n '.join(list_of_strings)

def add_noise_inj(original_string, p1 = 1e-3, p2 = 1e-3, pin = 1e-3, pm = 1e-3, pinj = 2e-3):
    # test = ' R 1 2 3 4 5 \n  H 1 2 3 4 5 \n  MR 1 2 3 4 5 \n  R 1 2 3 4 5  \n  TICK  \n CX 1 2 3 4  \n  R 1 2 3 4 5 \n M 1 2 3  \n TICK'
    list_of_strings = original_string.split("\n")
    
    i = 0
    while i < len(list_of_strings):
        line = list_of_strings[i]
        if (' INJECTION ' in line) and (' NOISELESS ' not in line):
            if (' R ' in line):
                err = line.replace('R', f'DEPOLARIZE1({pinj})')
                list_of_strings.insert(i+1, err)
                i += 2
            elif (' CX' in line):
                err = line.replace('CX', f'DEPOLARIZE2({pinj})')
                list_of_strings.insert(i+1, err)
                i += 2
            if (' RX ' in line):
                err = line.replace('RX', f'DEPOLARIZE1({pinj})')
                list_of_strings.insert(i+1, err)
                i += 2
            else:
                i += 1
        
        if (' INJECTION ' not in line) and (' NOISELESS ' not in line):
            if (' R ' in line) :
                err = line.replace('R', f'X_ERROR({pin})')
                list_of_strings.insert(i+1, err)
                i += 2  # Skip both original and inserted
            elif (' RX ' in line):
                err = line.replace('RX', f'DEPOLARIZE1({p1})')
                list_of_strings.insert(i+1, err)
                i += 2 
            elif ' H' in line:
                err = line.replace('H', f'DEPOLARIZE1({p1})')
                list_of_strings.insert(i+1, err)
                i += 2
            elif (' CX' in line):
                err = line.replace('CX', f'DEPOLARIZE2({p2})')
                list_of_strings.insert(i+1, err)
                i += 2
            elif ' M ' in line:
                err = line.replace('M', f'X_ERROR({pm})')
                list_of_strings.insert(i, err)
                i += 2  # Skip both inserted and original
            elif ' MX ' in line:
                err = line.replace('MX', f'DEPOLARIZE1({p1})')
                list_of_strings.insert(i, err)
                i += 2
            elif ' MR ' in line:
                err_r = line.replace('MR', f'X_ERROR({pin})')
                err_m = line.replace('MR', f'X_ERROR({pm})')
                list_of_strings.insert(i, err_m)
                list_of_strings.insert(i+2, err_r)
                i += 3
            else:
                i += 1
        else:
            i += 1            

    return ' \n '.join(list_of_strings)

def count_errors(original_string):
    list_of_strings = original_string.split("\n")
    counter = 0
    for i in range(len(list_of_strings)):
        if ' R ' in list_of_strings[i]:
            counter+=1
        if ' H ' in list_of_strings[i]:
            counter+=1
        if ' CX ' in list_of_strings[i]:
            counter+=1
        if ' M ' in list_of_strings[i]:
            counter+=1
        if ' MR ' in list_of_strings[i]:
            counter+=2
            
    return counter

# check that all the stabs are the same
def detectors_parity(stabs):
    N = len(stabs)
    full_string = np.array([""])
    for i in range(N):
        full_string = np.char.add(full_string, f'DETECTOR rec[-{i+1}] rec[-{i+1+N}]  \n ')

    return full_string

# check that all the stabs are the same
def detectors_parity_pair(stabs, N):
    # N = len(stabs)
    full_string = np.array([""])
    for i in range(len(stabs)):
        full_string = np.char.add(full_string, f'DETECTOR rec[-{i+1}] rec[-{i+1+N}]  \n ')

    return full_string

# check that all the stabs are the same
def detector_single(stabs):
    N = len(stabs)
    full_string = np.array([""])
    for i in range(N):
        full_string = np.char.add(full_string, f'DETECTOR rec[-{i+1}] \n ')

    return full_string    

def measure_logical_z(rmax): # logical Z on middle row
    central_row = array([0])
    for r in range(2,rmax)[::2]:
        central_row = np.append(central_row, n_polar(r, 1, (r-1)//2+1))
        central_row = np.append(central_row, n_polar(r, 3, (r-1)//2+1))
        
    MZ = ' M '+' '.join(str(n) for n in central_row)+'  \n  '
    return MZ

def measure_logical_x_had(rmax): # logical X on middle col
    central_col = array([0])
    for r in range(2,rmax)[::2]:
        central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
        central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
        
    MH = ' H '+' '.join(str(n) for n in central_col)+' \n TICK \n  '
    MX = ' M '+' '.join(str(n) for n in central_col)+'  \n  '
    return MH + MX

def measure_logical_x(rmax): # logical X on middle col
    central_col = array([0])
    for r in range(2,rmax)[::2]:
        central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
        central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
        
    # MH = ' H '+' '.join(str(n) for n in central_col)+' \n TICK \n  '
    MX = ' M '+' '.join(str(n) for n in central_col)+'  \n  '
    return MX
    
def measure_logical(rmax): # logical X/Z on middle col/row
    if ((rmax-1)/2)%2==1: # d=3,7 etc, measure Z
        central_row = array([0])
        for r in range(2,rmax)[::2]:
            central_row = np.append(central_row, n_polar(r, 1, (r-1)//2+1))
            central_row = np.append(central_row, n_polar(r, 3, (r-1)//2+1))
            
        M = ' M '+' '.join(str(n) for n in central_row)+'  \n  '

    elif ((rmax-1)/2)%2==0: # d=5,9 etc, measure X
        central_col = array([0])
        for r in range(2,rmax)[::2]:
            central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
            central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
            
        M = ' M '+' '.join(str(n) for n in central_col)+'  \n  '

    return M

def measure_logical_MZZ(rmax): # logical X/Z on middle col/row
    if ((rmax-1)/2)%2==1: # d=3,7 etc, measure Z
        central_row = array([0])
        for r in range(2,rmax)[::2]:
            central_row = np.append(central_row, n_polar(r, 1, (r-1)//2+1))
            central_row = np.append(central_row, n_polar(r, 3, (r-1)//2+1))
            
        M = ' MZZ '+' '.join(str(n) for n in np.roll(np.repeat(central_row, 2), -1))+'  \n  '

    elif ((rmax-1)/2)%2==0: # d=5,9 etc, measure X
        central_col = array([0])
        for r in range(2,rmax)[::2]:
            central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
            central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
            
        M = ' MZZ '+' '.join(str(n) for n in np.roll(np.repeat(central_col, 2), -1))+'  \n  '

    return M

def measure_logical_bell(rmax, shift_qubits): # logical X/Z on middle col/row
    if ((rmax-1)/2)%2==1: # d=3,7 etc, measure Z
        central_row = array([0])
        for r in range(2,rmax)[::2]:
            central_row = np.append(central_row, n_polar(r, 1, (r-1)//2+1))
            central_row = np.append(central_row, n_polar(r, 3, (r-1)//2+1))
            
        M = ' M '+' '.join(str(n) for n in central_row)+' '+' '.join(str(n+shift_qubits) for n in central_row)+'  \n  '

    elif ((rmax-1)/2)%2==0: # d=5,9 etc, measure X
        central_col = array([0])
        for r in range(2,rmax)[::2]:
            central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
            central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
            
        M = ' M '+' '.join(str(n) for n in central_col)+' '+' '.join(str(n+shift_qubits) for n in central_col)+'  \n  '

    return M

def observe_logical_PF(rmax): # logical operator with Pauli frame
    rmax2 = rmax-2
    meas_num = np.array([], dtype=int)
    
    if int(((rmax-1)/2+0)%2)==0: # d=5,9..
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]:
            for b in arange(rmax2-1)[::2][:]:
                meas_num = np.append(meas_num, a*rmax2+b)
                        
    elif int(((rmax-1)/2+0)%2)==1: # d=3,7..
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]:
            for b in arange(rmax-1)[3::2]:
                meas_num = np.append(meas_num, a*rmax2+b)

    meas_list = arange(rmax)
    pauli_frame = len(all_in_r(rmax2)) - meas_num%len(all_in_r(rmax2)) + rmax - 1
    meas_list = np.append(meas_list, pauli_frame)
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+1)+']' for n in meas_list)+' \n '
    return LZ

def observe_logical_PF_v2(rmax): # logical operator with Pauli frame
    rmax2 = rmax-2
    meas_num = np.array([], dtype=int)
    indent = (rmax+1)//4
    
    if int(((rmax-1)/2+0)%2)==0: # d=5,9,13..
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]:
            for b in arange(rmax2-1)[::2][:indent]:
                meas_num = np.append(meas_num, a*rmax2+b)
                        
    elif int(((rmax-1)/2+0)%2)==1: # d=3,7,11,15..
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]:
            for b in arange(rmax-1)[::-1][::2][:indent][::-1]:
                meas_num = np.append(meas_num, a*rmax2+b)

    meas_list = arange(rmax)
    pauli_frame = len(all_in_r(rmax2)) - meas_num%len(all_in_r(rmax2)) + rmax - 1
    meas_list = np.append(meas_list, pauli_frame)
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+1)+']' for n in meas_list)+' \n '
    return LZ

def observe_logical_PF_v2_debug(rmax): # logical operator with Pauli frame
    rmax2 = rmax-2
    meas_num = np.array([], dtype=int)
    qub_num = np.array([], dtype=int)
    indent = (rmax+1)//4
    
    if int(((rmax-1)/2+0)%2)==0: # d=5,9,13..
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]:
            for b in arange(rmax2-1)[::2][:indent]:
                meas_num = np.append(meas_num, a*rmax2+b)
                qub_num = np.append(qub_num, n_polar(rmax2, a, b))
                        
    elif int(((rmax-1)/2+0)%2)==1: # d=3,7,11,15..
        for a in arange(4)[int(((rmax-1)/2+0)%2)::2]:
            for b in arange(rmax-1)[::-1][::2][:indent][::-1]:
                meas_num = np.append(meas_num, a*rmax2+b)
                qub_num = np.append(qub_num, n_polar(rmax2, a, b))

    meas_list = arange(rmax)
    pauli_frame = len(all_in_r(rmax2)) - meas_num%len(all_in_r(rmax2)) + rmax - 1
    meas_list = np.append(meas_list, pauli_frame)
    # LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+1)+']' for n in meas_list)+' \n '
    return qub_num

def find_PFs(source, targets):
    """
    Returns the indices of elements in 'targets' as found in 'source'.
    If an element is not found, returns -1 for that element.

    """
    source_index = {value: idx for idx, value in enumerate(source)}
    return len(source)-np.array([source_index.get(target, -1) for target in targets])

def observe_logical_PF_v3(rmax, n_grow=1):
    meas_list = arange(rmax)+1
    for i in range(n_grow):
        pauli_frame = find_PFs(all_stabs_list(rmax), observe_logical_PF_v2_debug(rmax-4*(i))) + rmax
        meas_list = np.append(meas_list, pauli_frame)
        
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n)+']' for n in meas_list)+' \n '

    return LZ

def observe_logical(rmax): # logical operator
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+1)+']' for n in arange(rmax))+' \n '
    return LZ

def observe_logical_op(rmax): # logical operator
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+1)+']' for n in arange(rmax))+' \n '
    return LZ

def observe_logical_op_3d4(rmax): # logical operator
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+1)+']' for n in arange(rmax))+' rec[-87] rec[-86] rec[-82] rec[-91]'+' \n '
    return LZ

def observe_logical_bell(rmax): # logical operator
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+1)+']' for n in arange(rmax*2))+' \n '
    return LZ

def plot_LER(circuit, errs=linspace(0.1,1,10)*1e-3):
    tasks = [
        sinter.Task(
            circuit=stim.Circuit(add_noise(circuit,noise/10,noise,noise/10,noise/10)),
            json_metadata={'d': 3, 'p': noise},
        )
        # for d in [3, 5, 7, 9]
        for noise in errs
    ]
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=1000,
        save_resume_filepath = 'stats.csv',
    )
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax.loglog()
    # ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate")
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.legend()
    fig.set_dpi(120)  # Show it bigger
    show()
    return

bit_packed_array_for_this_circ_dets = packbits(ones(32,dtype=np.uint8))
    
def cr_circuit_noise_str_upd_magic(p1 = 1e-3, p2 = 1e-3, pin = 1e-3, pm = 1e-3, p_magic = 1e-3): # injection from the corner
    cr_circuit_noise_str_upd_magic = f"""
        QUBIT_COORDS(1, 1) 1 # magic
        
        QUBIT_COORDS(2, 0) 2 # top leaf
        QUBIT_COORDS(3, 1) 3 # data
        QUBIT_COORDS(5, 1) 5 # data
        QUBIT_COORDS(1, 3) 8 # data
        QUBIT_COORDS(2, 2) 9
        QUBIT_COORDS(3, 3) 10 # data
        QUBIT_COORDS(4, 2) 11
        QUBIT_COORDS(5, 3) 12 # data
        QUBIT_COORDS(6, 2) 13 # right leaf
        QUBIT_COORDS(0, 4) 14 # left leaf
        QUBIT_COORDS(1, 5) 15 # data
        QUBIT_COORDS(2, 4) 16
        QUBIT_COORDS(3, 5) 17 # data
        QUBIT_COORDS(4, 4) 18
        QUBIT_COORDS(5, 5) 19 # data
        QUBIT_COORDS(4, 6) 25 # bottom leaf
    
        # qubits initialization
        R 1 3 5 8 10 12 15 17 19 2 9 11 13 14 16 18 25
        H 8 10 15 17 19
        # S 10 # central qubit preparation
        X_ERROR({p_magic}) 1
        X_ERROR({pin}) 3 5 8 10 12 15 17 19 2 9 11 13 14 16 18 25
        # DEPOLARIZE1({p1}) 1 3 17 19
        # S 10 # central qubit preparation
        # DEPOLARIZE1({p1}) 10
        
        TICK
        # check the hatched stabs
        H 16 25
        DEPOLARIZE1({p1}) 16 25
        CX 25 17 12 13 16 8 16 10  
        DEPOLARIZE2({p2}) 25 17 12 13 16 8 16 10 
        CX 25 19 5 13 16 17 16 15
        DEPOLARIZE2({p2}) 25 19 5 13 16 17 16 15
        H 16 25
        DEPOLARIZE1({p1}) 16 25
        
        X_ERROR({pm}) 13 16 25
        MR 13 16 25
        X_ERROR({pin}) 13 16 25
        
        DETECTOR(6, 2, 0) rec[-3]
        DETECTOR(2, 4, 0) rec[-2]
        DETECTOR(4, 6, 0) rec[-1]
    
        # stabilizer measurements round 1
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        CX 2 3 16 17 11 12 15 14 10 9 19 18
        DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        TICK
        CX 2 1 16 15 11 10 8 14 3 9 12 18
        DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        TICK
        CX 16 10 11 5 25 19 8 9 17 18 12 13
        DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        TICK
        CX 16 8 11 3 25 17 1 9 10 18 5 13
        DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        X_ERROR({pm}) 2 9 11 13 14 16 18 25
        MR 2 9 11 13 14 16 18 25
        X_ERROR({pin}) 2 9 11 13 14 16 18 25
        SHIFT_COORDS(0, 0, 1) # to change notation to round 1
        # check that hatched are unchanged
        DETECTOR(6, 2, 0) rec[-11] rec[-5]
        DETECTOR(2, 4, 0) rec[-10] rec[-3]
        DETECTOR(4, 6, 0) rec[-9] rec[-1]
    
        # stabilizer measurements round 2
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        CX 2 3 16 17 11 12 15 14 10 9 19 18
        DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        TICK
        CX 2 1 16 15 11 10 8 14 3 9 12 18
        DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        TICK
        CX 16 10 11 5 25 19 8 9 17 18 12 13
        DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        TICK
        CX 16 8 11 3 25 17 1 9 10 18 5 13
        DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        X_ERROR({pm}) 2 9 11 13 14 16 18 25
        MR 2 9 11 13 14 16 18 25
        X_ERROR({pin}) 2 9 11 13 14 16 18 25

        SHIFT_COORDS(0, 0, 1)
        # check that all the ancilla qubits had the same output in both rounds
        DETECTOR(2, 0, 0) rec[-8] rec[-16]
        DETECTOR(2, 2, 0) rec[-7] rec[-15]
        DETECTOR(4, 2, 0) rec[-6] rec[-14]
        DETECTOR(6, 2, 0) rec[-5] rec[-13]
        DETECTOR(0, 4, 0) rec[-4] rec[-12]
        DETECTOR(2, 4, 0) rec[-3] rec[-11]
        DETECTOR(4, 4, 0) rec[-2] rec[-10]
        DETECTOR(4, 6, 0) rec[-1] rec[-9]


        # # stabilizer measurements round 3-4
        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # CX 2 3 16 17 11 12 15 14 10 9 19 18
        # DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        # TICK
        # CX 2 1 16 15 11 10 8 14 3 9 12 18
        # DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        # TICK
        # CX 16 10 11 5 25 19 8 9 17 18 12 13
        # DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        # TICK
        # CX 16 8 11 3 25 17 1 9 10 18 5 13
        # DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # X_ERROR({pm}) 2 9 11 13 14 16 18 25
        # MR 2 9 11 13 14 16 18 25
        # X_ERROR({pin}) 2 9 11 13 14 16 18 25
        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # CX 2 3 16 17 11 12 15 14 10 9 19 18
        # DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        # TICK
        # CX 2 1 16 15 11 10 8 14 3 9 12 18
        # DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        # TICK
        # CX 16 10 11 5 25 19 8 9 17 18 12 13
        # DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        # TICK
        # CX 16 8 11 3 25 17 1 9 10 18 5 13
        # DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # X_ERROR({pm}) 2 9 11 13 14 16 18 25
        # MR 2 9 11 13 14 16 18 25
        # X_ERROR({pin}) 2 9 11 13 14 16 18 25

        # SHIFT_COORDS(0, 0, 1)
        # # check that all the ancilla qubits had the same output in both rounds
        # DETECTOR(2, 0, 0) rec[-8] rec[-16]
        # DETECTOR(2, 2, 0) rec[-7] rec[-15]
        # DETECTOR(4, 2, 0) rec[-6] rec[-14]
        # DETECTOR(6, 2, 0) rec[-5] rec[-13]
        # DETECTOR(0, 4, 0) rec[-4] rec[-12]
        # DETECTOR(2, 4, 0) rec[-3] rec[-11]
        # DETECTOR(4, 4, 0) rec[-2] rec[-10]
        # DETECTOR(4, 6, 0) rec[-1] rec[-9]
        
        SHIFT_COORDS(0, 0, 1) # to change notation to round 2
        TICK
        # # check the hatched stabs again
        # H 16 25
        # DEPOLARIZE1({p1}) 16 25
        # CX 25 17 12 13 16 8 16 10  
        # DEPOLARIZE2({p2}) 25 17 12 13 16 8 16 10 
        # CX 25 19 5 13 16 17 16 15
        # DEPOLARIZE2({p2}) 25 19 5 13 16 17 16 15
        # H 16 25
        # DEPOLARIZE1({p1}) 16 25
        
        X_ERROR({pm}) 13 16 25
        MR 13 16 25
        # X_ERROR({pin}) 13 16 25
        
        DETECTOR(6, 2, 0) rec[-3]
        DETECTOR(2, 4, 0) rec[-2]
        DETECTOR(4, 6, 0) rec[-1]

        # DETECTOR(6, 2, 0) rec[-5]
        # DETECTOR(2, 4, 0) rec[-3]
        # DETECTOR(4, 6, 0) rec[-1]
        
        # final measurements of data qubits
        X_ERROR({pm}) 1 3 5 8 10 12 15 17 19
        M 1 3 5 8 10 12 15 17 19

        # DO NOT POSTSELECT ON DATA QUBITS!!!!!!!!
    
        # DETECTOR(0, 4, 1) rec[-3] rec[-6] rec[-13] # bottom left triangle
        # DETECTOR(2, 2, 1) rec[-5] rec[-6] rec[-8] rec[-9] rec[-16] # top left square
        # DETECTOR(4, 4, 1) rec[-1] rec[-2] rec[-4] rec[-5] rec[-11] # bottom right square
        # DETECTOR(6, 2, 1) rec[-4] rec[-7] rec[-14] # top right triangle
        # DETECTOR(0, 4, 1) rec[-3] rec[-6] rec[-13] # bottom left triangle
        # DETECTOR(2, 2, 1) rec[-8] rec[-9] rec[-17] # top left triangle
        # DETECTOR(4, 4, 1) rec[-1] rec[-2] rec[-10] # bottom right triangle
        # DETECTOR(6, 2, 1) rec[-4] rec[-7] rec[-14] # top right triangle
        
        OBSERVABLE_INCLUDE(0) rec[-7] rec[-8] rec[-9] # logical Z on top row
        # OBSERVABLE_INCLUDE(0) rec[-4] rec[-5] rec[-6] # logical Z on middle row
        # OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] # logical Z on bottom row
        # OBSERVABLE_INCLUDE(0) rec[-2] rec[-5] rec[-8] # logical X on middle row
        # OBSERVABLE_INCLUDE(0) rec[-3] rec[-6] rec[-9] # logical X on left row
        # OBSERVABLE_INCLUDE(0) rec[-1] rec[-4] rec[-7] # logical X on right row
    """    
    return cr_circuit_noise_str_upd_magic

def mr_circuit_noise_str_upd_magic(p1 = 1e-3, p2 = 1e-3, pin = 1e-3, pm = 1e-3, p_magic = 1e-3):
    mr_circuit_noise_str_upd_magic = f"""
        QUBIT_COORDS(1, 1) 1 # data
        QUBIT_COORDS(2, 0) 2 # top leaf
        QUBIT_COORDS(3, 1) 3 # data
        QUBIT_COORDS(5, 1) 5 # data
        QUBIT_COORDS(1, 3) 8 # data
        QUBIT_COORDS(2, 2) 9
        
        QUBIT_COORDS(3, 3) 10 # magic qubit
        
        QUBIT_COORDS(4, 2) 11
        QUBIT_COORDS(5, 3) 12 # data
        QUBIT_COORDS(6, 2) 13 # right leaf
        QUBIT_COORDS(0, 4) 14 # left leaf
        QUBIT_COORDS(1, 5) 15 # data
        QUBIT_COORDS(2, 4) 16
        QUBIT_COORDS(3, 5) 17 # data
        QUBIT_COORDS(4, 4) 18
        QUBIT_COORDS(5, 5) 19 # data
        QUBIT_COORDS(4, 6) 25 # bottom leaf
    
        # qubits initialization
        R 1 3 5 8 10 12 15 17 19 2 9 11 13 14 16 18 25
        H 1 3 17 19
        # S 10 # central qubit preparation
        X_ERROR({p_magic}) 10
        X_ERROR({pin}) 1 3 5 8 12 15 17 19 2 9 11 13 14 16 18 25
        # DEPOLARIZE1({p1}) 1 3 17 19
        # S 10 # central qubit preparation
        # DEPOLARIZE1({p1}) 10
        
        # TICK
        # check the leaves
        H 2 25
        DEPOLARIZE1({p1}) 2 25
        CX 2 1 13 5 25 19 14 15
        DEPOLARIZE2({p2}) 2 1 13 5 25 19 14 15
        CX 2 3 13 12 25 17 14 8
        DEPOLARIZE2({p2}) 2 3 13 12 25 17 14 8 
        H 2 25
        DEPOLARIZE1({p1}) 2 25
        
        X_ERROR({pm}) 2 13 14 25
        MR 2 13 14 25
        X_ERROR({pin}) 2 13 14 25
        
        DETECTOR(2, 0, 0) rec[-4]
        DETECTOR(6, 2, 0) rec[-3]
        DETECTOR(0, 4, 0) rec[-2]
        DETECTOR(4, 6, 0) rec[-1]
    
        # stabilizer measurements round 1
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        CX 2 3 16 17 11 12 15 14 10 9 19 18
        DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        TICK
        CX 2 1 16 15 11 10 8 14 3 9 12 18
        DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        TICK
        CX 16 10 11 5 25 19 8 9 17 18 12 13
        DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        TICK
        CX 16 8 11 3 25 17 1 9 10 18 5 13
        DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        X_ERROR({pm}) 2 9 11 13 14 16 18 25
        MR 2 9 11 13 14 16 18 25
        X_ERROR({pin}) 2 9 11 13 14 16 18 25
        SHIFT_COORDS(0, 0, 1) # to change notation to round 1
        # check that leaves are unchanged
        DETECTOR(2, 0, 0) rec[-12] rec[-8]
        DETECTOR(6, 2, 0) rec[-11] rec[-5]
        DETECTOR(0, 4, 0) rec[-10] rec[-4]
        DETECTOR(4, 6, 0) rec[-9] rec[-1]
    
        # stabilizer measurements round 2
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        CX 2 3 16 17 11 12 15 14 10 9 19 18
        DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        TICK
        CX 2 1 16 15 11 10 8 14 3 9 12 18
        DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        TICK
        CX 16 10 11 5 25 19 8 9 17 18 12 13
        DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        TICK
        CX 16 8 11 3 25 17 1 9 10 18 5 13
        DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        TICK
        H 2 11 16 25
        DEPOLARIZE1({p1}) 2 11 16 25
        TICK
        X_ERROR({pm}) 2 9 11 13 14 16 18 25
        MR 2 9 11 13 14 16 18 25
        X_ERROR({pin}) 2 9 11 13 14 16 18 25

        SHIFT_COORDS(0, 0, 1)
        # check that all the ancilla qubits had the same output in both rounds
        DETECTOR(2, 0, 0) rec[-8] rec[-16]
        DETECTOR(2, 2, 0) rec[-7] rec[-15]
        DETECTOR(4, 2, 0) rec[-6] rec[-14]
        DETECTOR(6, 2, 0) rec[-5] rec[-13]
        DETECTOR(0, 4, 0) rec[-4] rec[-12]
        DETECTOR(2, 4, 0) rec[-3] rec[-11]
        DETECTOR(4, 4, 0) rec[-2] rec[-10]
        DETECTOR(4, 6, 0) rec[-1] rec[-9]

        # # stabilizer measurements round 3-4
        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # CX 2 3 16 17 11 12 15 14 10 9 19 18
        # DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        # TICK
        # CX 2 1 16 15 11 10 8 14 3 9 12 18
        # DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        # TICK
        # CX 16 10 11 5 25 19 8 9 17 18 12 13
        # DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        # TICK
        # CX 16 8 11 3 25 17 1 9 10 18 5 13
        # DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # X_ERROR({pm}) 2 9 11 13 14 16 18 25
        # MR 2 9 11 13 14 16 18 25
        # X_ERROR({pin}) 2 9 11 13 14 16 18 25

        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # CX 2 3 16 17 11 12 15 14 10 9 19 18
        # DEPOLARIZE2({p2}) 2 3 16 17 11 12 15 14 10 9 19 18
        # TICK
        # CX 2 1 16 15 11 10 8 14 3 9 12 18
        # DEPOLARIZE2({p2}) 2 1 16 15 11 10 8 14 3 9 12 18
        # TICK
        # CX 16 10 11 5 25 19 8 9 17 18 12 13
        # DEPOLARIZE2({p2}) 16 10 11 5 25 19 8 9 17 18 12 13
        # TICK
        # CX 16 8 11 3 25 17 1 9 10 18 5 13
        # DEPOLARIZE2({p2}) 16 8 11 3 25 17 1 9 10 18 5 13
        # TICK
        # H 2 11 16 25
        # DEPOLARIZE1({p1}) 2 11 16 25
        # TICK
        # X_ERROR({pm}) 2 9 11 13 14 16 18 25
        # MR 2 9 11 13 14 16 18 25
        # X_ERROR({pin}) 2 9 11 13 14 16 18 25

        # SHIFT_COORDS(0, 0, 1)
        # # check that all the ancilla qubits had the same output in both rounds
        # DETECTOR(2, 0, 0) rec[-8] rec[-16]
        # DETECTOR(2, 2, 0) rec[-7] rec[-15]
        # DETECTOR(4, 2, 0) rec[-6] rec[-14]
        # DETECTOR(6, 2, 0) rec[-5] rec[-13]
        # DETECTOR(0, 4, 0) rec[-4] rec[-12]
        # DETECTOR(2, 4, 0) rec[-3] rec[-11]
        # DETECTOR(4, 4, 0) rec[-2] rec[-10]
        # DETECTOR(4, 6, 0) rec[-1] rec[-9]
        
        # SHIFT_COORDS(0, 0, 1) # to change notation to round 2
        TICK
        # # check the leaves
        # H 2 25
        # DEPOLARIZE1({p1}) 2 25
        # CX 2 1 13 5 25 19 14 15
        # DEPOLARIZE2({p2}) 2 1 13 5 25 19 14 15
        # CX 2 3 13 12 25 17 14 8
        # DEPOLARIZE2({p2}) 2 3 13 12 25 17 14 8 
        # H 2 25
        # DEPOLARIZE1({p1}) 2 25
        
        X_ERROR({pm}) 2 13 14 25
        MR 2 13 14 25
        X_ERROR({pin}) 2 13 14 25

        # check the leaves again
        DETECTOR(2, 0, 0) rec[-4]
        DETECTOR(6, 2, 0) rec[-3]
        DETECTOR(0, 4, 0) rec[-2]
        DETECTOR(4, 6, 0) rec[-1]

        # DETECTOR(2, 0, 0) rec[-8]
        # DETECTOR(6, 2, 0) rec[-5]
        # DETECTOR(0, 4, 0) rec[-4]
        # DETECTOR(4, 6, 0) rec[-1]

        # DETECTOR(0, 4, 1) rec[-2] rec[-4] rec[-5] rec[-7]
        
        # final measurements of data qubits
        X_ERROR({pm}) 1 3 5 8 10 12 15 17 19
        M 1 3 5 8 10 12 15 17 19

        # DO NOT POSTSELECT ON DATA QUBITS!!!!!!!!
    
        # DETECTOR(0, 4, 1) rec[-3] rec[-6] rec[-13] # bottom left triangle
        # DETECTOR(2, 2, 1) rec[-5] rec[-6] rec[-8] rec[-9] rec[-16] # top left square
        # DETECTOR(4, 4, 1) rec[-1] rec[-2] rec[-4] rec[-5] rec[-11] # bottom right square
        # DETECTOR(6, 2, 1) rec[-4] rec[-7] rec[-14] # top right triangle
        # DETECTOR(0, 4, 1) rec[-3] rec[-6] rec[-13] # bottom left triangle
        # DETECTOR(2, 2, 1) rec[-8] rec[-9] rec[-17] # top left triangle
        # DETECTOR(4, 4, 1) rec[-1] rec[-2] rec[-10] # bottom right triangle
        # DETECTOR(6, 2, 1) rec[-4] rec[-7] rec[-14] # top right triangle
        
        # OBSERVABLE_INCLUDE(0) rec[-7] rec[-8] rec[-9] # logical Z on top row
        OBSERVABLE_INCLUDE(0) rec[-4] rec[-5] rec[-6] # logical Z on middle row
        # OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] # logical Z on bottom row
        # OBSERVABLE_INCLUDE(0) rec[-2] rec[-5] rec[-8] # logical X on middle row
        # OBSERVABLE_INCLUDE(0) rec[-3] rec[-6] rec[-9] # logical X on left row
        # OBSERVABLE_INCLUDE(0) rec[-1] rec[-4] rec[-7] # logical X on right row
    """    
    return mr_circuit_noise_str_upd_magic

def plot_LER_post_magic_fixed(circuit = mr_circuit_noise_str_upd_magic(), errs=linspace(0.1,1,10)*1e-3, filename='stats.csv', noise_type='unbiased', magic_err=1e-3, bit_packed_array_for_this_circ_dets=bit_packed_array_for_this_circ_dets):
    if noise_type=='unbiased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(circuit(noise/1,noise,noise/1,noise/1,noise/1)),
                json_metadata={'d': 3, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    elif noise_type=='biased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(circuit(noise/10,noise,noise/10,noise/10,noise/1)),
                json_metadata={'d': 3, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    else:
        print('Wrong noise type! Can be biased or unbiased only.')
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=100_000,
        # save_resume_filepath = filename,
    )
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.25)
    sinter.plot_error_rate(
        ax=ax[0],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    sinter.plot_discard_rate(
        ax=ax[1],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax.loglog()
    # ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    fig.suptitle('MR injection '+noise_type+' noise', fontsize=12)
    ax[0].set_xlabel("Physical Error Rate (p2)")
    ax[0].set_ylabel("Logical Error Rate")
    ax[0].grid(which='major')
    ax[0].grid(which='minor')
    ax[0].legend()
    ax[1].set_xlabel("Physical Error Rate (p2)")
    ax[1].set_ylabel("Discard Rate")
    ax[1].grid(which='major')
    ax[1].grid(which='minor')
    ax[1].legend()
    fig.set_dpi(200)  # Show it bigger
    show()
    return collected_stats

def plot_LER_post_v2(circuit, dist=3, errs=linspace(0.1,1,10)*1e-3, filename='stats.csv', noise_type='biased', magic_err=1e-3, bit_packed_array_for_this_circ_dets=bit_packed_array_for_this_circ_dets):
    if noise_type=='unbiased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/1,noise,noise/1,noise/1,noise/1)),
                json_metadata={'d': dist, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    elif noise_type=='biased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/10,noise/1,noise/10,noise/10,noise/1)),
                json_metadata={'d': dist, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            for noise in errs
        ]
    else:
        print('Wrong noise type! Can be biased or unbiased only.')
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=100_000,
        # save_resume_filepath = filename,
    )
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.25)
    sinter.plot_error_rate(
        ax=ax[0],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    sinter.plot_discard_rate(
        ax=ax[1],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax[0].loglog()
    # ax[1].loglog()
    # ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    fig.suptitle('MR injection '+noise_type+' noise', fontsize=12)
    ax[0].set_xlabel("Physical Error Rate (p2)")
    ax[0].set_ylabel("Logical Error Rate")
    ax[0].grid(which='major')
    ax[0].grid(which='minor')
    ax[0].legend()
    ax[1].set_xlabel("Physical Error Rate (p2)")
    ax[1].set_ylabel("Discard Rate")
    ax[1].grid(which='major')
    ax[1].grid(which='minor')
    ax[1].legend()
    fig.set_dpi(200)  # Show it bigger
    # show()
    return collected_stats

def plot_LER_post_magic_fixed(circuit = mr_circuit_noise_str_upd_magic(), errs=linspace(0.1,1,10)*1e-3, filename='stats.csv', noise_type='unbiased', magic_err=1e-3, bit_packed_array_for_this_circ_dets=bit_packed_array_for_this_circ_dets):
    if noise_type=='unbiased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(circuit(noise/1,noise,noise/1,noise/1,noise/1)),
                json_metadata={'d': 3, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    elif noise_type=='biased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(circuit(noise/10,noise,noise/10,noise/10,noise/1)),
                json_metadata={'d': 3, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    else:
        print('Wrong noise type! Can be biased or unbiased only.')
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=100_000,
        # save_resume_filepath = filename,
    )
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.25)
    sinter.plot_error_rate(
        ax=ax[0],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    sinter.plot_discard_rate(
        ax=ax[1],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax.loglog()
    # ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    fig.suptitle('MR injection '+noise_type+' noise', fontsize=12)
    ax[0].set_xlabel("Physical Error Rate (p2)")
    ax[0].set_ylabel("Logical Error Rate")
    ax[0].grid(which='major')
    ax[0].grid(which='minor')
    ax[0].legend()
    ax[1].set_xlabel("Physical Error Rate (p2)")
    ax[1].set_ylabel("Discard Rate")
    ax[1].grid(which='major')
    ax[1].grid(which='minor')
    ax[1].legend()
    fig.set_dpi(200)  # Show it bigger
    show()
    return collected_stats

def plot_LER_post_v2_nopost(circuit, dist=3, errs=linspace(0.1,1,10)*1e-3, filename='stats.csv', noise_type='biased', magic_err=1e-3, bit_packed_array_for_this_circ_dets=bit_packed_array_for_this_circ_dets):
    if noise_type=='unbiased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/1,noise,noise/1,noise/1,noise/1)),
                json_metadata={'d': dist, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    elif noise_type=='biased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/10,noise/1,noise/10,noise/10,noise/1)),
                json_metadata={'d': dist, 'p': noise},
                # postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            for noise in errs
        ]
    else:
        print('Wrong noise type! Can be biased or unbiased only.')
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=100_000,
        # save_resume_filepath = filename,
    )
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.25)
    sinter.plot_error_rate(
        ax=ax[0],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    sinter.plot_discard_rate(
        ax=ax[1],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax[0].loglog()
    # ax[1].loglog()
    # ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    fig.suptitle('MR injection '+noise_type+' noise', fontsize=12)
    ax[0].set_xlabel("Physical Error Rate (p2)")
    ax[0].set_ylabel("Logical Error Rate")
    ax[0].grid(which='major')
    ax[0].grid(which='minor')
    ax[0].legend()
    ax[1].set_xlabel("Physical Error Rate (p2)")
    ax[1].set_ylabel("Discard Rate")
    ax[1].grid(which='major')
    ax[1].grid(which='minor')
    ax[1].legend()
    fig.set_dpi(200)  # Show it bigger
    # show()
    return collected_stats

def plot_LER_post_v2_nopost_bell(circuit, dist=3, errs=linspace(0.1,1,10)*1e-3, bell_err_ratio=1, filename='stats.csv', noise_type='biased', magic_err=1e-3):
    if noise_type=='unbiased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/1,noise,noise/1,noise/1,noise/1)),
                json_metadata={'d': dist, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    elif noise_type=='biased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/10,noise/1,noise/10,noise/10,magic_err)),
                json_metadata={'d': dist, 'p': noise},
                # postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            for noise in errs
        ]
    else:
        print('Wrong noise type! Can be biased or unbiased only.')
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=100_000,
        # save_resume_filepath = filename,
    )
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.25)
    sinter.plot_error_rate(
        ax=ax[0],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    sinter.plot_discard_rate(
        ax=ax[1],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax[0].loglog()
    # ax[1].loglog()
    # ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    fig.suptitle('MR injection '+noise_type+' noise', fontsize=12)
    ax[0].set_xlabel("Physical Error Rate (p2)")
    ax[0].set_ylabel("Logical Error Rate")
    ax[0].grid(which='major')
    ax[0].grid(which='minor')
    ax[0].legend()
    ax[1].set_xlabel("Physical Error Rate (p2)")
    ax[1].set_ylabel("Discard Rate")
    ax[1].grid(which='major')
    ax[1].grid(which='minor')
    ax[1].legend()
    fig.set_dpi(200)  # Show it bigger
    # show()
    return collected_stats

def plot_LER_post_v2_nopost_bell_transversal(circuit, dist=3, errs=linspace(0.1,1,10)*1e-3, bell_err_ratio=1, filename='stats.csv', noise_type='no local', magic_err=1e-3):
    if noise_type=='local':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit,noise/10,noise,noise/10,noise/10,magic_err)),
                json_metadata={'d': dist, 'p': noise},
                # postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    elif noise_type=='no local':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, 0e-3,0e-3,0e-3,0e-3,noise)),
                json_metadata={'d': dist, 'p': noise},
                # postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            for noise in errs
        ]
    else:
        print('Wrong noise type! Can be biased or unbiased only.')
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=100_000,
        # save_resume_filepath = filename,
    )
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.25)
    sinter.plot_error_rate(
        ax=ax[0],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    sinter.plot_discard_rate(
        ax=ax[1],
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    # ax[0].loglog()
    # ax[1].loglog()
    # ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    # fig.suptitle('MR injection '+noise_type+' noise', fontsize=12)
    ax[0].set_xlabel("Physical Error Rate (p2)")
    ax[0].set_ylabel("Logical Error Rate")
    ax[0].grid(which='major')
    ax[0].grid(which='minor')
    ax[0].legend()
    ax[1].set_xlabel("Physical Error Rate (p2)")
    ax[1].set_ylabel("Discard Rate")
    ax[1].grid(which='major')
    ax[1].grid(which='minor')
    ax[1].legend()
    fig.set_dpi(200)  # Show it bigger
    # show()
    return collected_stats

def plot_LER_post_v2_nopost_noplot(circuit, dist=3, errs=linspace(0.1,1,10)*1e-3, filename='stats.csv', noise_type='biased', magic_err=1e-3, bit_packed_array_for_this_circ_dets=bit_packed_array_for_this_circ_dets):
    if noise_type=='unbiased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/1,noise,noise/1,noise/1,noise/1)),
                json_metadata={'d': dist, 'p': noise},
                postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            # for d in [3, 5, 7, 9]
            for noise in errs
        ]
    elif noise_type=='biased':
        tasks = [
            sinter.Task(
                circuit=stim.Circuit(add_noise_inj(circuit, noise/10,noise/1,noise/10,noise/10,noise/1)),
                json_metadata={'d': dist, 'p': noise},
                # postselection_mask = bit_packed_array_for_this_circ_dets,
            )
            for noise in errs
        ]
    else:
        print('Wrong noise type! Can be biased or unbiased only.')
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=1_000_000,
        max_errors=100_000,
        # save_resume_filepath = filename,
    )
    return collected_stats

def check_det_1_round_RX(first_meas, second_meas):
    full_string = np.array([""])
    all_checks = second_meas
    det_checks = first_meas

    L = len(first_meas)
    N = len(second_meas)
    indices = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices[i] = np.where(all_checks == det_checks[::-1][i])[0][0]
        indices[i] = len(all_checks)-indices[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices[i]}] rec[-{i+1+N}]  \n ')
        
    return full_string

def check_det_single(measurement, checks):
    full_string = np.array([""])
    all_checks = measurement
    det_checks = checks

    L = len(det_checks)
    N = len(all_checks)
    indices = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices[i] = np.where(all_checks == det_checks[::-1][i])[0][0]
        indices[i] = len(all_checks)-indices[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices[i]}]  \n ')
        
    return full_string

def check_det_1_round(rmax):
    full_string = np.array([""])
    all_checks = all_stabs_list(rmax)
    x_stabs_det, z_stabs_det = deterministic_stabs(rmax)
    det_checks = np.append(x_stabs_det, z_stabs_det)

    L = len(det_checks)
    N = len(all_checks)
    indices = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices[i] = np.where(all_checks == det_checks[::-1][i])[0][0]
        indices[i] = len(all_checks)-indices[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices[i]}] rec[-{i+1+N}]  \n ')
        
    return full_string

def check_det_1_round_pair(rmax, N):
    full_string = np.array([""])
    all_checks = all_stabs_list(rmax)
    x_stabs_det, z_stabs_det = deterministic_stabs(rmax)
    det_checks = np.append(x_stabs_det, z_stabs_det)

    L = len(det_checks)
    # N = len(all_checks)
    indices = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices[i] = np.where(all_checks == det_checks[::-1][i])[0][0]
        indices[i] = len(all_checks)-indices[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices[i]}] rec[-{i+1+N}]  \n ')
        
    return full_string

def check_det_1_round_pair_grow(measurement, checks, N):
    full_string = np.array([""])
    all_checks = measurement
    det_checks = checks

    L = len(det_checks)
    # N = len(all_checks)
    indices = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices[i] = np.where(all_checks == det_checks[::-1][i])[0][0]
        indices[i] = len(all_checks)-indices[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices[i]}] rec[-{i+1+N}]  \n ')
        
    return full_string

def check_remote_pair(measurement, checks, N):
    full_string = np.array([""])
    all_checks = measurement
    det_checks = checks

    L = len(det_checks)
    # N = len(all_checks)
    indices = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices[i] = np.where(all_checks == det_checks[::-1][i])[0][0]
        indices[i] = len(all_checks)-indices[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices[i]}] rec[-{indices[i]+N}]  \n ')
        
    return full_string

def check_det_1_round_patches(measurement_last, measurement_prev, checks, N):
    full_string = np.array([""])
    # all_checks = measurement
    # det_checks = checks

    L = len(checks)
    # N = len(all_checks)
    indices1 = np.zeros(L, dtype=int)
    indices2 = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices1[i] = np.where(measurement_last == checks[::-1][i])[0][0]
        indices1[i] = len(measurement_last)-indices1[i]
        indices2[i] = np.where(measurement_prev == checks[::-1][i])[0][0]
        indices2[i] = len(measurement_prev)-indices2[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices1[i]}] rec[-{indices2[i]+N}]  \n ')
        
    return full_string

def check_corr_cnot_single(measurement_last, M, measurement_prev, checks, N):
    ''' M is the shift for the first detector, N for the last
    '''
    full_string = np.array([""])
    # all_checks = measurement
    # det_checks = checks

    L = len(checks)
    # N = len(all_checks)
    indices1 = np.zeros(L, dtype=int)
    indices2 = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices1[i] = np.where(measurement_last == checks[::-1][i])[0][0]
        indices1[i] = len(measurement_last)-indices1[i]
        indices2[i] = np.where(measurement_prev == checks[::-1][i])[0][0]
        indices2[i] = len(measurement_prev)-indices2[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices1[i]+M}] rec[-{indices2[i]+N}]  \n ')
        
    return full_string

def check_corr_cnot_parity(measurement_last, M, measurement_prev, checks, N, P):
    ''' M is the shift for the first detector, N for the last; P is for the third one;
    '''
    full_string = np.array([""])
    # all_checks = measurement
    # det_checks = checks

    L = len(checks)
    # N = len(all_checks)
    indices1 = np.zeros(L, dtype=int)
    indices2 = np.zeros(L, dtype=int)
    
    for i in range(L):
        indices1[i] = np.where(measurement_last == checks[::-1][i])[0][0]
        indices1[i] = len(measurement_last)-indices1[i]
        indices2[i] = np.where(measurement_prev == checks[::-1][i])[0][0]
        indices2[i] = len(measurement_prev)-indices2[i]
        full_string = np.char.add(full_string, f'DETECTOR rec[-{indices1[i]+M}] rec[-{indices2[i]+N}] rec[-{indices2[i]+N+P}] \n ')
        
    return full_string

def d_init_bell_det_stabs_v2(rmax=3, repeats=1): # with 2 rounds of checks to measure parity + deterministic checks
    # + checks between first and second rounds, like in paper, NO REPEATS
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # reset all (except for 0)
    reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax)[1:])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset)

    # initialize central (magic/Bell injection)
    injection = ' R '+'0'+' #  INJECTION  \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)

    # init data qubits in special form
    init_plus = ' H '+' '.join(str(n) for n in init_bell_plus(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, init_plus)
    
    
    # check all the deterministic stabs
    ###############################################################    
    x_stabs_det, z_stabs_det = deterministic_stabs(rmax)
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_det_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all det stabs
    MR = ' MR '+' '.join(str(n) for n in np.append(x_stabs_det, z_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # detectors on deterministic stabs
    detectors = detector_single(np.append(x_stabs_det, z_stabs_det))
    full_string = np.char.add(full_string, detectors)
    ###############################################################    
    
    ###############################################################    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # !!!!!!!!!!!!!!!!
    # check that det stabs are the same
    detectors = check_det_1_round(rmax)
    full_string = np.char.add(full_string, detectors)

    # repeat the last step?
    full_string = np.char.add(full_string, f'REPEAT {repeats} {{'+' \n ')

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)

    full_string = np.char.add(full_string, '}'+' \n ')
    ############################################################### 
        
    return full_string[0]

def d_init_bell_det_stabs_pair(rmax=3, repeats=1, shift_coord=20, shift_qubits=100): # with 2 rounds of checks to measure parity + deterministic checks
    # + checks between firts and second rounds, like in paper, NO REPEATS
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    full_string = np.char.add(full_string, num_to_coord_shifted(0,0,0,shift_coord=shift_coord, shift_qubits=shift_qubits))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))
                full_string = np.char.add(full_string, num_to_coord_shifted(r, a, b, shift_coord=shift_coord, shift_qubits=shift_qubits))

    # reset all (with 0)
    # reset = ' R '+' '.join(str(n) for n in all_qubits_list(rmax)[1:])+'  \n  '
    # init data qubits in special form
    init_plus = ' RX '+' '.join(str(n) for n in init_bell_plus(rmax))+'  \n  '
    init_plus2 = ' RX '+' '.join(str(n+shift_qubits) for n in init_bell_plus(rmax))+'  \n  '
    full_string = np.char.add(full_string, init_plus)
    full_string = np.char.add(full_string, init_plus2)
    reset0 = ' RX '+' '.join(str(0))+'  \n  '
    reset1 = ' R '+' '.join(str(n) for n in np.setdiff1d(all_qubits_list(rmax)[1:], init_bell_plus(rmax)))+'  \n  '
    reset2 = ' R '+' '.join(str(n+shift_qubits) for n in np.setdiff1d(all_qubits_list(rmax)[0:], init_bell_plus(rmax)))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, reset0)
    full_string = np.char.add(full_string, reset1)
    full_string = np.char.add(full_string, reset2)


    # initialize Bell
    injection = ' CX '+f'0 {shift_qubits}'+' #  INJECTION  \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)
    
    # check all the deterministic stabs
    ###############################################################    
    x_stabs_det, z_stabs_det = deterministic_stabs(rmax)
    # Hadamard on det X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+'  \n  '
    hadam2 = ' H '+' '.join(str(n) for n in (x_stabs_det+shift_qubits))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_det_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_det_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on det X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all det stabs
    MR = ' MR '+' '.join(str(n) for n in np.append(x_stabs_det, z_stabs_det))+'  \n  '
    full_string = np.char.add(full_string, MR)
    # detectors on deterministic stabs
    detectors = detector_single(np.append(x_stabs_det, z_stabs_det))
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in np.append(x_stabs_det, z_stabs_det))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    # detectors on deterministic stabs
    detectors2 = detector_single(np.append(x_stabs_det, z_stabs_det))
    full_string = np.char.add(full_string, detectors2)

    ###############################################################    
    
    ###############################################################    
    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    detectors = check_det_1_round_pair(rmax, len(all_stabs_list(rmax))+len(np.append(x_stabs_det, z_stabs_det)))
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    # !!!!!!!!!!!!!!!!
    # check that det stabs are the same
    detectors = check_det_1_round_pair(rmax, len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    # Hadamard on X stabs
    hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+'  \n  '
    hadam2 = ' H '+' '.join(str(n+shift_qubits) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, hadam)    
    full_string = np.char.add(full_string, hadam2)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+'  \n  '
        big_check2 = ' CX '+' '.join(str(n+shift_qubits) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)
        full_string = np.char.add(full_string, big_check2)

    # Hadamard on X stabs
    full_string = np.char.add(full_string, hadam)
    full_string = np.char.add(full_string, hadam2)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in all_stabs_list(rmax))+'  \n  '
    full_string = np.char.add(full_string, MR)
    # check that all the stabs are the same
    detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)

    MR2 = ' MR '+' '.join(str(n+shift_qubits) for n in all_stabs_list(rmax))+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MR2)
    # check that all the stabs are the same
    detectors = detectors_parity_pair(all_stabs_list(rmax), len(all_stabs_list(rmax))*2)
    full_string = np.char.add(full_string, detectors)
    ############################################################### 
        
    return full_string[0]

def d_init_bell_det_stabs_v3(rmax=3, repeats=1): # with 2 rounds of checks to measure parity + deterministic checks
    # + checks between firts and second rounds, like in paper, NO REPEATS
    # rmax = 3
    full_string = np.array([""])
    full_string = np.char.add(full_string, num_to_coord(0,0,0))
    for r in range(rmax+1):
        for a in range(4):
            for b in range(r):
                full_string = np.char.add(full_string, num_to_coord(r, a, b))

    # init X data qubits in special form
    init_plus = ' RX '+' '.join(str(n) for n in init_bell_plus(rmax))+' \n  '
    full_string = np.char.add(full_string, init_plus)
    # init X stabs
    init_plus = ' RX '+' '.join(str(n) for n in stabs_all(rmax)[0])+' \n  '
    full_string = np.char.add(full_string, init_plus)
    # init Z data qubits in special form
    init_plus = ' R '+' '.join(str(n) for n in np.setdiff1d(data_list(rmax)[1:], init_bell_plus(rmax)))+' \n  '
    full_string = np.char.add(full_string, init_plus)
    # init Z stabs
    init_plus = ' R '+' '.join(str(n) for n in stabs_all(rmax)[1])+' \n  '
    full_string = np.char.add(full_string, init_plus)

    # initialize central (magic/Bell injection)
    injection = ' R '+'0'+' #  INJECTION  \n  TICK  \n  '
    full_string = np.char.add(full_string, injection)
    
    # check all the deterministic stabs
    ###############################################################    
    x_stabs_det, z_stabs_det = deterministic_stabs(rmax)
    # # Hadamard on det X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_det)+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)    
    
    # measure det checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_det_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # # Hadamard on det X stabs
    # full_string = np.char.add(full_string, hadam)

    # measure all det stabs
    MR = ' MR '+' '.join(str(n) for n in z_stabs_det)+' \n  '
    full_string = np.char.add(full_string, MR)
    MRX = ' MRX '+' '.join(str(n) for n in x_stabs_det)+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MRX)

    # detectors on deterministic stabs
    detectors = detector_single(np.append(x_stabs_det, z_stabs_det))
    full_string = np.char.add(full_string, detectors)
    ###############################################################    
    
    ###############################################################    
    # # Hadamard on X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # # Hadamard on X stabs
    # full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in stabs_all(rmax)[1])+'  \n  '
    full_string = np.char.add(full_string, MR)
    MRX = ' MRX '+' '.join(str(n) for n in stabs_all(rmax)[0])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MRX)

    # !!!!!!!!!!!!!!!!
    # check that det stabs are the same
    detectors = check_det_1_round_RX(np.append(z_stabs_det, x_stabs_det), np.append(stabs_all(rmax)[1], stabs_all(rmax)[0]))
    full_string = np.char.add(full_string, detectors)

    # # Hadamard on X stabs
    # hadam = ' H '+' '.join(str(n) for n in x_stabs_all(rmax))+' \n  TICK  \n  '
    # full_string = np.char.add(full_string, hadam)    
    
    # measure checks of the full code
    for i in range(4):
        big_check = ' CX '+' '.join(str(n) for n in all_checks_upd(rmax)[i])+' \n  TICK  \n  '
        full_string = np.char.add(full_string, big_check)

    # # Hadamard on X stabs
    # full_string = np.char.add(full_string, hadam)

    # measure all stabs
    MR = ' MR '+' '.join(str(n) for n in stabs_all(rmax)[1])+'  \n  '
    full_string = np.char.add(full_string, MR)
    MRX = ' MRX '+' '.join(str(n) for n in stabs_all(rmax)[0])+' \n  TICK  \n  '
    full_string = np.char.add(full_string, MRX)

    # check that all the stabs are the same
    detectors = detectors_parity(all_stabs_list(rmax))
    full_string = np.char.add(full_string, detectors)
    ############################################################### 
        
    return full_string[0]

def bitmask_for_postselection(full_string, init_d=3):
    inj_checks = d_init_bell_det_stabs_v2(rmax=init_d).count(" DETECTOR ")
    full_num_checks = full_string.count(" DETECTOR ")

    return packbits(np.append(ones(inj_checks,dtype=np.uint8), zeros(full_num_checks-inj_checks,dtype=np.uint8)))
    # return packbits(np.append(zeros(full_num_checks-inj_checks,dtype=np.uint8), ones(inj_checks,dtype=np.uint8)))

def bitmask_for_postselection_circ(circuit, init_d=3):
    inj_checks = d_init_bell_det_stabs_v2(rmax=init_d).count(" DETECTOR ")
    full_num_checks = circuit.num_detectors

    if full_num_checks >= inj_checks:
        return packbits(np.append(ones(inj_checks,dtype=np.uint8), zeros(full_num_checks-inj_checks,dtype=np.uint8)), bitorder='big')
    else:
        return packbits(ones(inj_checks,dtype=np.uint8), bitorder='big')    # return packbits(ones(full_num_checks, dtype=np.uint8))
    # return packbits(np.append(ones(full_num_checks-inj_checks,dtype=np.uint8), zeros(inj_checks,dtype=np.uint8)), bitorder='big')

def bitmask_for_postselection_corner(circuit, start_d=3):
    inj_checks = stim.Circuit(corner_init_magic(start_d)).num_detectors
    full_num_checks = stim.Circuit(circuit).num_detectors

    if full_num_checks >= inj_checks:
        return packbits(np.append(ones(inj_checks,dtype=np.uint8), zeros(full_num_checks-inj_checks,dtype=np.uint8)), bitorder='big')
    else:
        return packbits(ones(inj_checks,dtype=np.uint8), bitorder='big')

def bitmask_for_postselection_circ_zeros(circuit, init_d=3):
    inj_checks = d_init_bell_det_stabs_v2(rmax=init_d).count(" DETECTOR ")
    full_num_checks = circuit.num_detectors

    return packbits(np.append(zeros(inj_checks,dtype=np.uint8), zeros(full_num_checks-inj_checks,dtype=np.uint8)), bitorder='big')

def bitmask_for_postselection_bell(circuit_big, circuit_small):
    # inj_checks = d_init_bell_det_stabs_v2(rmax=init_d).count(" DETECTOR ")
    inj_checks = stim.Circuit(circuit_small).num_detectors
    full_num_checks = stim.Circuit(circuit_big).num_detectors

    return packbits(np.append(ones(inj_checks,dtype=np.uint8), zeros(full_num_checks-inj_checks,dtype=np.uint8)), bitorder='big')

def measure_all_data(rmax):
    datas = data_list(rmax)
    M = ' M '+' '.join(str(n) for n in datas)+'  \n  '
    return M

def observe_logical_from_all(rmax):
    datas = data_list(rmax)       
    indices = np.zeros(rmax, dtype=int)
    rshift = 3
    if ((rshift-1)/2)%2==1: # d=3,7 etc, measure Z
        central_row = array([0])
        for r in range(2,rmax)[::2]:
            central_row = np.append(central_row, n_polar(r, 1, (r-1)//2+1))
            central_row = np.append(central_row, n_polar(r, 3, (r-1)//2+1))
        
        for i in range(rmax):
            indices[i] = np.where(datas == central_row[::-1][i])[0][0]
            indices[i] = len(datas)-indices[i]

        LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n)+']' for n in indices)+' \n '

    # elif ((rshift-1)/2)%2==0: # d=5,9 etc, measure X
    #     central_col = array([0])
    #     for r in range(2,rmax)[::2]:
    #         central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
    #         central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
            
    #     for i in range(rmax):
    #         indices[i] = np.where(datas == central_col[::-1][i])[0][0]
    #         indices[i] = len(datas)-indices[i]    
            
    #     LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n)+']' for n in indices)+' \n '    

    return LZ

def observe_logical_top(rmax):
    datas = data_list(rmax)       
    indices = np.zeros(rmax, dtype=int)

    top_line = all_in_r_a(rmax-1,0)
    indices = np.intersect1d(data_list(rmax), top_line, return_indices=True)[1]
    indices = len(datas)-indices

    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n)+']' for n in indices)+' \n '

    return LZ

def detector_data(rmax): # detect all Z stabs via data qubits and prev measurement
    full_string = np.array([""])
    # stab_x, stab_z = stab_list(rmax)
    # leaves_x, leaves_z = leaves(rmax)
    N = len(data_list(rmax))

    # no leaves
    for r in arange(rmax)[1::2]:
        for a in range(4):
            for b in range(r):
                if (a+b)%2==0: # Z stab
                    stab_z = n_polar(r, a, b)
                    data_qubs = neighbour_data(r, a, b)
                    _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                    ind_data = len(data_list(rmax)) - ind_data
                    _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                    ind_stab = len(all_stabs_list(rmax)) - ind_stab
                    full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}] rec[-{N+ind_stab[0]}]  \n ')

    rshift = 3
    if ((rshift-1)/2)%2==1: # d=3,7 etc, blue X on top
        for b in arange(rmax)[int(((rshift-1)/2+1)%2+1)::2]:
            stab_z = n_polar(rmax, 1, b)
            data_qubs = neighbour_data(rmax, 1, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')
        for b in arange(rmax)[1::2]:
            stab_z = n_polar(rmax, 3, b)
            data_qubs = neighbour_data(rmax, 3, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')
            
    # elif ((rshift-1)/2)%2==0: # d=5,9 etc, red Z on top
    #     for b in arange(rmax)[2::2]:
    #         stab_z = n_polar(rmax, 0, b)
    #         data_qubs = neighbour_data(rmax, 0, b)
    #         shift = 0
    #         _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
    #         ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
    #         _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
    #         ind_stab = len(all_stabs_list(rmax)) - ind_stab
    #         full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')
    #     for b in arange(rmax)[2::2]:
    #         stab_z = n_polar(rmax, 2, b)
    #         data_qubs = neighbour_data(rmax, 2, b)
    #         shift = 0
    #         _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
    #         ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
    #         _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
    #         ind_stab = len(all_stabs_list(rmax)) - ind_stab
    #         full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')
    
    return full_string[0]

def detector_data_X(rmax): # detect all X stabs via data qubits and prev measurement
    full_string = np.array([""])
    N = len(data_list(rmax))

    # no leaves
    for r in arange(rmax)[1::2]:
        for a in range(4):
            for b in range(r):
                if (a+b)%2==1: # Z stab
                    stab_z = n_polar(r, a, b)
                    data_qubs = neighbour_data(r, a, b)
                    _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                    ind_data = len(data_list(rmax)) - ind_data
                    _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                    ind_stab = len(all_stabs_list(rmax)) - ind_stab
                    full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}] rec[-{N+ind_stab[0]}]  \n ')

    rshift = 3
    if ((rshift-1)/2)%2==1: # d=3,7 etc, blue X on top
        for b in arange(rmax)[int(((rshift-1)/2+1)%2+1)::2]:
            stab_z = n_polar(rmax, 0, b)
            data_qubs = neighbour_data(rmax, 0, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')
        for b in arange(rmax)[1::2]:
            stab_z = n_polar(rmax, 2, b)
            data_qubs = neighbour_data(rmax, 2, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N+ind_stab[0]}]  \n ')

    return full_string[0]

def detector_data_nox(rmax): # detect all Z stabs via data qubits and prev measurement
    full_string = np.array([""])
    # stab_x, stab_z = stab_list(rmax)
    # leaves_x, leaves_z = leaves(rmax)
    N = len(data_list(rmax))

    # no leaves
    for r in arange(rmax)[1::2]:
        for a in range(4):
            for b in range(r):
                if (a+b)%2==0: # Z stab
                    stab_z = n_polar(r, a, b)
                    data_qubs = neighbour_data(r, a, b)
                    _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                    ind_data = len(data_list(rmax)) - ind_data
                    _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                    ind_stab = len(all_stabs_list(rmax)) - ind_stab
                    full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}]  \n ')

    rshift = 3
    if ((rshift-1)/2)%2==1: # d=3,7 etc, blue X on top
        for b in arange(rmax)[int(((rshift-1)/2+1)%2+1)::2]:
            stab_z = n_polar(rmax, 1, b)
            data_qubs = neighbour_data(rmax, 1, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}]  \n ')
        for b in arange(rmax)[1::2]:
            stab_z = n_polar(rmax, 3, b)
            data_qubs = neighbour_data(rmax, 3, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}]  \n ')
    
    return full_string[0]

def observe_random_checks(rmax):
    if rmax == 5:
        checks = np.array([13, 19])
    elif rmax == 11:
        checks = np.array([49, 51, 59, 41])
    datas = data_list(rmax) 
    stabs = all_stabs_list(rmax)
    N = len(datas)
    indices = np.zeros(len(checks), dtype=int)
    for i in range(len(checks)):
        indices[i] = np.where(stabs == checks[::-1][i])[0][0]
        indices[i] = len(stabs)-indices[i]        
    LZ = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+N)+']' for n in indices)+' \n ' 
    return LZ

def func(x, r, b):
    return r*log(x*b)

def detector_data_pair(rmax=3): # detect all Z stabs via data qubits and prev measurement
    full_string = np.array([""])
    N1 = 2*len(data_list(rmax))#+len(all_stabs_list(rmax))
    N2 = len(data_list(rmax))+len(all_stabs_list(rmax))
    dN = len(data_list(rmax))

    # no leaves
    for r in arange(rmax)[1::2]:
        for a in range(4):
            for b in range(r):
                if (a+b)%2==0: # Z stab
                    stab_z = n_polar(r, a, b)
                    data_qubs = neighbour_data(r, a, b)
                    _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                    ind_data = len(data_list(rmax)) - ind_data
                    _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                    ind_stab = len(all_stabs_list(rmax)) - ind_stab
                    full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}] rec[-{N1+ind_stab[0]}]  \n ')
                    full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{ind_data[2]+dN}] rec[-{ind_data[3]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')

    for b in arange(rmax)[1::2]:
        stab_z = n_polar(rmax, 1, b)
        data_qubs = neighbour_data(rmax, 1, b)
        shift = 0
        _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
        ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
        _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
        ind_stab = len(all_stabs_list(rmax)) - ind_stab
        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N1+ind_stab[0]}]  \n ')
        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')

        stab_z = n_polar(rmax, 3, b)
        data_qubs = neighbour_data(rmax, 3, b)
        shift = 0
        _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
        ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
        _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
        ind_stab = len(all_stabs_list(rmax)) - ind_stab
        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N1+ind_stab[0]}]  \n ')
        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')

    return full_string[0]

def measure_all_data_pair(rmax=3, shift_qubits=100, meas_X=False):
    datas = data_list(rmax)
    M = ' M '+' '.join(str(n) for n in datas)+'  \n  '
    M = M + ' M '+' '.join(str(n+shift_qubits) for n in datas)+'  \n  '
    if meas_X:
        M = ' MX '+' '.join(str(n) for n in datas)+'  \n  '
        M = M + ' MX '+' '.join(str(n+shift_qubits) for n in datas)+'  \n  '        
    return M

def observe_logical_from_all_pair(rmax=3, both=True, meas_X=False):
    datas = data_list(rmax)       
    indices = np.zeros(rmax, dtype=int)
    central_row = array([0])
    for r in range(2,rmax)[::2]:
        central_row = np.append(central_row, n_polar(r, 1, (r-1)//2+1))
        central_row = np.append(central_row, n_polar(r, 3, (r-1)//2+1))
    
    for i in range(rmax):
        indices[i] = np.where(datas == central_row[::-1][i])[0][0]
        indices[i] = len(datas)-indices[i]

    # if meas_X:
    #     central_col = array([0])
    #     # for r in range(2,rmax)[::2]:
    #     #     central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
    #     #     central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
        
    #     # for i in range(rmax):
    #     #     indices[i] = np.where(datas == central_col[::-1][i])[0][0]
    #     #     indices[i] = len(datas)-indices[i]
    #     indices = np.where(datas == central_col[::-1])[0][0]
    #     indices = int(len(datas)-indices)
    #     L = 'OBSERVABLE_INCLUDE(0) '+'rec[-'+str(indices)+']'+' \n '
    #     if both:
    #         L = L + 'OBSERVABLE_INCLUDE(0) '+'rec[-'+str(indices+len(data_list(rmax)))+']'+' \n '
    if meas_X:
        central_col = array([0])
        for r in range(2,rmax)[::2]:
            central_col = np.append(central_col, n_polar(r, 0, (r-1)//2+1))
            central_col = np.append(central_col, n_polar(r, 2, (r-1)//2+1))
        
        for i in range(rmax):
            indices[i] = np.where(datas == central_col[::-1][i])[0][0]
            indices[i] = len(datas)-indices[i]

    L = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n)+']' for n in indices)+' \n '
    if both:
        L = L + 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+len(data_list(rmax)))+']' for n in indices)+' \n '

    return L

def observe_logical_from_all_pair_top(rmax=3, both=True, meas_X=False):
    datas = data_list(rmax)       
    indices = np.zeros(rmax, dtype=int)
    top_row = array([])
    top_row = np.append(top_row, all_in_r_a(rmax-1, 0))
    
    for i in range(rmax):
        indices[i] = np.where(datas == top_row[::-1][i])[0][0]
        indices[i] = len(datas)-indices[i]

    if meas_X:
        left_col = array([])
        left_col = np.append(left_col, all_in_r_a(rmax-1, 3))

        for i in range(rmax):
            indices[i] = np.where(datas == left_col[::-1][i])[0][0]
            indices[i] = len(datas)-indices[i]

    L = 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n)+']' for n in indices)+' \n '
    if both:
        L = L + 'OBSERVABLE_INCLUDE(0) '+' '.join('rec[-'+str(n+len(data_list(rmax)))+']' for n in indices)+' \n '

    return L

def plot_fit_scaling(arr, names, colors, ls, save=False, name='corr_scale'):
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    plt.subplots_adjust(wspace=0.25)
    
    for i, a in enumerate(arr):
        xdata1 = array(list(map(probs, a)))
        ydata1 = array(list(map(ler, a)))
        sorted_indices = np.argsort(xdata1)
        xdata1 = xdata1[sorted_indices]
        ydata1 = ydata1[sorted_indices]
        popt1, pcov1 = curve_fit(func, (xdata1), log(ydata1+1e-12), p0=[2,10]) 

        ax.scatter(1e0*array(list(map(probs, a))), 1e0*array(list(map(ler, a))), label=names[i], zorder=10, color=colors[i])
        ax.errorbar(1e0*array(list(map(probs, a))), 1e0*array(list(map(ler, a))), yerr=1e0*array(list(map(ler_err, a))), zorder=10, color=colors[i], fmt='none', capsize=5)
        ax.plot(1e0*xdata1, 1e0*exp(func((xdata1), *popt1)), color=colors[i], linestyle=ls[i], label='fit $a \\cdot x^r$: r=%5.2f, a=%5.1f' % tuple(popt1))
        
    ax.set_xlabel("Local Two-Qubit Physical Error Rate $p_2$")
    # ax.set_xlabel("Distributed CNOT Physical Error Rate $p_2$")
    ax.set_ylabel("Logical Error Rate")
    ax.grid(which='major', alpha=0.3)
    ax.grid(which='minor', alpha=0.3)
    ax.set_ylim(1e-7,2e-1)
    ax.legend(title='$p_{Bell}$=0%')
    
    ax.loglog()
    fig.set_dpi(120)  # Show it bigger
    show()
    if save:
        fig.savefig(name+'.png', transparent=True, dpi=600, bbox_inches='tight')
        fig.savefig(name+'.pdf', transparent=True, dpi=600, bbox_inches='tight')

def detector_data_pair_grow(start_d=3, stop_d=5): # detect all Z stabs via data qubits and prev measurement

    if start_d == stop_d:
        return detector_data_pair(rmax=start_d)
    elif start_d > stop_d:
        return 'Wrong sizes!'
    elif start_d < stop_d:
        rmax = stop_d
        full_string = np.array([""])
        N1 = 2*len(data_list(rmax))
        N2 = len(data_list(rmax))+len(all_stabs_list(rmax))
        dN = len(data_list(rmax))
        small_det_stabs = translate_diag_array(np.append(det_stabs_lower(start_d), det_stabs_upper(start_d)), stop_d, stop_d-start_d)
        big_det_stabs = np.append(det_stabs_lower(rmax), det_stabs_upper(rmax))
        patch_det_stabs = setdiff1d(big_det_stabs, small_det_stabs)

        # patch det stabs, no leaves, no correlations
        for r in arange(rmax)[1::2]:
            for a in range(4):
                for b in range(r):
                    if n_polar(r, a, b) in patch_det_stabs: # Z det stab
                        stab_z = n_polar(r, a, b)
                        data_qubs = neighbour_data(r, a, b)
                        _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                        ind_data = len(data_list(rmax)) - ind_data
                        _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                        ind_stab = len(all_stabs_list(rmax)) - ind_stab
                        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}] rec[-{N1+ind_stab[0]}]  \n ')
                        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{ind_data[2]+dN}] rec[-{ind_data[3]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')

        # patch det stabs, leaves, no correlations
        for b in arange(rmax)[1::2]:
            stab_z = n_polar(rmax, 1, b)
            data_qubs = neighbour_data(rmax, 1, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N1+ind_stab[0]}]  \n ')
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')
        
            stab_z = n_polar(rmax, 2, b)
            data_qubs = neighbour_data(rmax, 2, b)
            shift = 0
            _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
            ind_data = np.roll(len(data_list(rmax)) - ind_data, shift)
            _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
            ind_stab = len(all_stabs_list(rmax)) - ind_stab
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N1+ind_stab[0]}]  \n ')
            full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')

        # (not)correlated stabs of the smaller code
        for r in arange(start_d)[1::2]:
            for a in range(4):
                for b in range(r):
                    if n_polar(r, a, b) in z_stabs_all(start_d): # Z det stab
                        stab_z = translate_diag(n_polar(r, a, b), stop_d, stop_d-start_d) 
                        data_qubs = translate_diag_array(neighbour_data(r, a, b), stop_d, stop_d-start_d)
                        _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                        ind_data = len(data_list(rmax)) - ind_data
                        _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                        ind_stab = len(all_stabs_list(rmax)) - ind_stab
                        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}] rec[-{N1+ind_stab[0]}]  \n ')
                        full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{ind_data[2]+dN}] rec[-{ind_data[3]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')    
        
        r=start_d; a=1
        for b in range(r):
            if n_polar(r, a, b) in z_stabs_all(start_d): # Z det stab
                stab_z = translate_diag(n_polar(r, a, b), stop_d, stop_d-start_d) 
                data_qubs = translate_diag_array(neighbour_data(r, a, b), stop_d, stop_d-start_d)
                _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                ind_data = len(data_list(rmax)) - ind_data
                _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                ind_stab = len(all_stabs_list(rmax)) - ind_stab
                full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{ind_data[2]}] rec[-{ind_data[3]}] rec[-{N1+ind_stab[0]}]  \n ')
                full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{ind_data[2]+dN}] rec[-{ind_data[3]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ') 
        
        r=start_d; a=3
        for b in range(r):
            if n_polar(r, a, b) in z_stabs_all(start_d): # Z det stab
                stab_z = translate_diag(n_polar(r, a, b), stop_d, stop_d-start_d) 
                data_qubs = translate_diag_array(neighbour_data(r, a, b), stop_d, stop_d-start_d)[1:3]
                _,_,ind_data = np.intersect1d(data_qubs, data_list(rmax), return_indices=True)
                ind_data = len(data_list(rmax)) - ind_data
                _,_,ind_stab = np.intersect1d(stab_z, all_stabs_list(rmax), return_indices=True)
                ind_stab = len(all_stabs_list(rmax)) - ind_stab
                full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]}] rec[-{ind_data[1]}] rec[-{N1+ind_stab[0]}]  \n ')
                full_string = np.char.add(full_string, f'DETECTOR rec[-{ind_data[0]+dN}] rec[-{ind_data[1]+dN}] rec[-{N2+ind_stab[0]+dN}]  \n ')     
        
        return full_string[0]

def plot_qubit_map(d=5, d_lil=0):
    color_qubits = translate_diag_array(all_qubits_list(d_lil), d, d-d_lil)
    color_data = data_list(d)
    color_stabs = all_stabs_list(d)
    fig, ax = subplots(figsize=(10,10))
    for r in range(d+1):
        if r==0:
            b_span = [0]
            a_span = [0]
        else: 
            b_span = range(r)
            a_span = range(4)
        for a in a_span:
            for b in b_span:
                x, y, n = num_to_coord_plot(r, a, b)
                if (n) in color_qubits:
                    ax.text(x/20+0.5, y/20+0.5, n, color='C1')
                elif n in color_data:
                    ax.text(x/20+0.5, y/20+0.5, n, color='C2')
                elif n in color_stabs:
                    ax.text(x/20+0.5, y/20+0.5, n, color='C3')
                else:
                    ax.text(x/20+0.5, y/20+0.5, n)
    
    ax.axis('off')
    show()







