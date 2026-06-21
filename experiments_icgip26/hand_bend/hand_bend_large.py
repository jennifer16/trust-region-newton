#  python3 ../myexperiments/hand_bend/hand_bend_large.py

import os
import sys

# 1.mesh_names
mesh_names = ['hand_closed']

# 2.methods
diff_mode_strs = ['clamp_abs_blending_j','clamp_abs_blending_q','abs_nondiff','clamp_nondiff'] 

# 3.pr
# pr_list = ['0.3','0.495']
deform_scale_list = ['0.1'] # '3','5.5' # useless

pr_list = ['0.48']  #['0.48']
#旋转
rotate_ratio_list = [ '0.5']

# 4.ym
ym_list = ['1e8']

# 5.adaptive
adaptive = '1'

# 6.smooth_mode
smooth_mode_strs = ['none']

# 7.gamma_mode
gamma_strs = ['-2']

# 8.para_pos_mode
pos_mode_strs = ['1'] 

# 9.para_neg_mode
neg_mode_strs = ['1']

# 10.para_j_mode
j_mode_strs = ['0']

# 11.udpate_gamma
update_gamma_mode_strs = ['0']

# 12.eta_mode_strs
eta_mode_strs = ['5']

# 13.kappa_mode_strs
kappa_mode_strs = ['0']

# 14.grad_mode_strs
grad_mode_strs = ['31']

diff = True
proj_eps = '1e-7'

# 
deform_styles = ['bend_stretch'] #['twist_right_rotate_ratio',‘twist_top_rotate_ratio’]
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]
# =======================================

# smooth_mode_strs = ["face","none"] # ["none","face", "default"]
# diff = True
# diff_mode_strs = ['abs_nondiff'] # 'clamp_abs_nondiff' for trust region method
# # diff_mode_strs = ['clamp_abs_nondiff','abs_nondiff', 'clamp_nondiff','soft_clamp','soft_abs',
# # 'clamp_abs', 'auto','hybrid','clamp','abs '] # 'clamp_abs_nondiff' for trust region method
# proj_eps = '1e-7'

# # proj_eps_list = ['-1', '0', '-0.5']
# pr_list = ['0.48']
# pr_list.reverse()

# mesh_name = 'hand_closed'
# experiment_name = 'figure_' + os.path.basename(__file__)[:-3]

# for smooth_mode_str in smooth_mode_strs:
#   for diff_mode_str in diff_mode_strs:
#   # for proj_eps in proj_eps_list:
#     for pr in pr_list:
#       try:
#         command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
#           + ' -l bend_stretch -t 0.1' \
#           + ' --ym 1e8 --pr ' + pr \
#           + ' --experiment_name ' + experiment_name \
#           + ' --tr 0.01' \
#           + ' -c 4e-7' \
#           + ' --diff' \
#           + ' --diff_mode ' + diff_mode_str \
#           + ' --smooth_mode ' + smooth_mode_str \
#           + ' --rotate_ratio 0.5' 
#         print(command)
#         os.system(command)
#       except:
#         print('Error: ' + mesh_name)

for pos_mode_str in pos_mode_strs:
    for neg_mode_str in neg_mode_strs:
        for gamma_str in gamma_strs:
            for smooth_mode_str in smooth_mode_strs:
                for diff_mode_str in diff_mode_strs:
                # for proj_eps in proj_eps_list:
                    for pr in pr_list:
                        for ym in ym_list:
                            for deform_scale in deform_scale_list:
                                for mesh_name in mesh_names:
                                    for deform_style in deform_styles:
                                        for rotate_ratio in rotate_ratio_list:
                                            for j_mode_str in j_mode_strs:
                                                for update_gamma_mode_str in update_gamma_mode_strs:
                                                    for eta_mode_str in eta_mode_strs:
                                                        for kappa_mode_str in kappa_mode_strs:
                                                            for grad_mode_str in grad_mode_strs:
                                                                try:
                                                                    command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
                                                                        + ' -l '+ deform_style + ' -t ' + deform_scale \
                                                                        + ' --rotate_ratio ' + rotate_ratio_list \
                                                                        + ' --ym '+ ym + ' --pr ' + pr \
                                                                        + ' --experiment_name ' + experiment_name \
                                                                        + ' --tr 0.01 -c 4e-7' \
                                                                        + ' --diff' \
                                                                        + ' --diff_mode ' + diff_mode_str \
                                                                        + ' --smooth_mode ' + smooth_mode_str \
                                                                        + ' --adaptive ' + adaptive \
                                                                        + ' --para_gamma ' + gamma_str \
                                                                        + ' --para_pos_mode ' + pos_mode_str \
                                                                        + ' --para_neg_mode ' + neg_mode_str \
                                                                        + ' --para_j_mode ' + j_mode_str \
                                                                        + ' --update_gamma_mode ' + update_gamma_mode_str \
                                                                        + ' --eta_mode ' + eta_mode_str \
                                                                        + ' --kappa_mode ' + kappa_mode_str \
                                                                        + ' --grad_mode ' + grad_mode_str 
                                                                        

                                                                    print(command)
                                                                    os.system(command)
                                                                except:
                                                                    print('Error: ' + mesh_name)

