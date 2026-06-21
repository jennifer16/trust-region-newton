import os
import sys

# 1.mesh_names
mesh_names = ['cylinder']

# 2.methods
diff_mode_strs = ['clamp_abs_blending_j','clamp_abs_blending_q','abs_nondiff','clamp_nondiff'] 

# 3.pr
# pr_list = ['0.3','0.495']
# deform_scale_list = ['0.8'] # '3','5.5'
pr_list = ['0.1', '0.2', '0.3', '0.4', '0.45', '0.46', '0.47', '0.48', '0.49', '0.495', '0.499', '0.4999']
deform_scale_list = ['2.0']

#旋转
rotate_ratio_list = [ '0.25',  '0.6']
rotate_ratio_list = []

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
deform_styles = ['shear_top_percentage'] #['twist_right_rotate_ratio',‘twist_top_rotate_ratio’]
experiment_name = 'figure_' + os.path.basename(__file__)[:-3]
# =======================================



# for smooth_mode_str in smooth_mode_strs:
#   for diff_mode_str in diff_mode_strs:
#   # for proj_eps in proj_eps_list:
#     for pr in pr_list:
#       for deform_scale in deform_scale_list:
#         try:
#           command = './example -p ' + proj_eps +  ' -n ' + mesh_name \
#             + ' -l stretch_longest_axis -t ' + deform_scale \
#             + ' --ym 1e8 --pr ' + pr \
#             + ' --experiment_name ' + experiment_name \
#             + ' --diff' \
#             + ' --diff_mode ' + diff_mode_str \
#             + ' --smooth_mode ' + smooth_mode_str \
#             + ' --tr 0.01' 
#           print(command)
#           os.system(command)
#         except:
#           print('Error: ' + mesh_name)

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
                                                                        + ' --ym '+ ym + ' --pr ' + pr \
                                                                        + ' --experiment_name ' + experiment_name \
                                                                        + ' --tr 0.01' \
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
