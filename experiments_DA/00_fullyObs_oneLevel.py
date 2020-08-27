datadir: /gpfs/work/nonnenma/data/emulators/L96/
res_dir: /gpfs/work/nonnenma/results/emulators/L96/
exp_id: spinup_oneLevel_learnedEmulator_fullyObs_T40
K : 36
J : 0
T : 605
dt : 0.05
N_trials : 1
back_solve_dt_fac : 1000
spin_up_time : 5.
l96_F : 10.
l96_h : 1.
l96_b : 10.
l96_c : 10.
obs_operator : ObsOp_identity 
obs_operator_r : 0.5
obs_operator_sig2 : 1.0
T_rollout : 40
T_pred : 50
n_chunks : 40
n_chunks_recursive : 40
n_starts : 4
model_exp_id : 24
model_forwarder : rk4_default
optimizer : LBFGS
n_steps : 50
lr : 1e0
max_iter : 1000
max_eval : -1
tolerance_grad : 1e-12
tolerance_change : 1e-12
history_size: 10
if_LBFGS_chunks : False
if_LBFGS_full_chunks : False
if_backsolve : True 
if_LBFGS_full_backsolve : True
if_LBFGS_full_persistence : True 
if_LBFGS_recurse_chunks : False