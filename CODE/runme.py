from Stabilization import *
from Background_Subtraction import *
from Matting import *
from Tracking import *
import os
import time
from tqdm import tqdm

log_file = open("time_log_file.txt", "w")
log_file.write('blocks run time:\n')


CODE = "{0}".format(os.getcwd())
RootFile = os.path.split(CODE)[0]
OutputFile = os.path.join(RootFile, 'Outputs')
InputFile = os.path.join(RootFile, 'Input')
TempFile = os.path.join(RootFile, 'Temp')

t0 = time.time()


                            ########## stabilization block ##########

INPUT_vid, OUT_vid1 = os.path.join(InputFile, 'INPUT.avi'), os.path.join(TempFile, 'init_stabilized_1.avi')
Pbar = inital_stab(INPUT_vid, OUT_vid1, False, None)

OUT_vid2 = os.path.join(TempFile, 'init_stabilized_2.avi')
Pbar = inital_stab(OUT_vid1, OUT_vid2, True, Pbar)

Stab_vid = os.path.join(OutputFile, 'stabilized.avi')
center_vid(OUT_vid2, Stab_vid, Pbar)

t1 = time.time()
delta_t_stab = (t1-t0)/60
log_file.write('stabilization takes '+str(delta_t_stab)+' minutes\n')



                        ########## background substraction block ##########
# get 3 videos- S based substract with high and low thres, and V based substract with high thres
back_sub_S_20 = os.path.join(TempFile, 'back_sub_S_thres=20.avi')
back_sub_S_55 = os.path.join(TempFile, 'back_sub_S_thres=55.avi')
back_sub(Stab_vid, back_sub_S_20, back_sub_S_55, 71, 1)

back_sub_V_20 = os.path.join(TempFile, 'back_sub_V_thres=20.avi')
back_sub_V_55 = os.path.join(TempFile, 'back_sub_V_thres=55.avi')
back_sub(Stab_vid, back_sub_V_20, back_sub_V_55, 91, 2)

# sum 3 videos
S_V_bin_mask = os.path.join(TempFile, 'S&V_bin_mask.avi')
sum_vid(back_sub_S_20, back_sub_S_55, back_sub_V_55, S_V_bin_mask)

# keep only region of interest
centerd_S_V_bin_mask = os.path.join(TempFile, 'centerd_S&V_bin_mask.avi')
region_of_interest(S_V_bin_mask, centerd_S_V_bin_mask)

# get 3 best hues (+-8 spectrum) in time
centerd_S_V_bin_mask_3hues = os.path.join(TempFile, 'centerd_S&V_bin_mask_top 3 hues.avi')
dominant_hues(Stab_vid, centerd_S_V_bin_mask, centerd_S_V_bin_mask_3hues)

# sum filtered S&V with previews high threshold S and high threshold V (explained in document)
S_V_bin_mask_3hues_inhanced = os.path.join(TempFile, 'S&V_bin_mask_top 3 hues_with_high_thres_S&V.avi')
sum_vid(centerd_S_V_bin_mask_3hues, back_sub_S_55, back_sub_V_55, S_V_bin_mask_3hues_inhanced)

# keep only region of interest
centerd_S_V_bin_mask_3hues_inhanced = os.path.join(TempFile, 'centerd_S&V_bin_mask_top 3 hues_with_high_thres_S&V.avi')
region_of_interest(S_V_bin_mask_3hues_inhanced, centerd_S_V_bin_mask_3hues_inhanced)

# lose noise, fill holes and get final background substruction
final_bin_mask = os.path.join(OutputFile, 'binary.avi')
color_mask = os.path.join(OutputFile, 'extracted.avi')
get_object(centerd_S_V_bin_mask_3hues_inhanced, Stab_vid, final_bin_mask, color_mask)

t2 = time.time()
delta_t_Back_sub = (t2-t1)/60
log_file.write('background substraction takes '+str(delta_t_Back_sub)+' minutes\n')



                        ########## Mating block ##########
# get alpha
alpha_mask = os.path.join(OutputFile, 'alpha.avi')
get_alpha(Stab_vid, final_bin_mask, alpha_mask)

# unstable alpha
alpha_shaky = os.path.join(OutputFile, 'unstabilized_alpha.avi')
unstable(INPUT_vid, Stab_vid, alpha_mask, alpha_shaky)

# put in new background
new_back = os.path.join(OutputFile, 'matted.avi')
background_im = os.path.join(InputFile, 'background.jpg')
put_in_background(Stab_vid, alpha_mask, background_im,new_back)


t3 = time.time()
delta_t_Mat = (t3-t2)/60
log_file.write('Matting takes '+str(delta_t_Mat)+' minutes\n')



                            ##################### Tracking block #####################
vid_with_rect = os.path.join(OutputFile, 'OUTPUT.avi')
track_object(new_back, vid_with_rect)

t4 = time.time()
delta_t_Track = (t4-t3)/60

log_file.write('Tracking takes '+str(delta_t_Track)+' minutes\n\n')
delta_t_tot = delta_t_stab + delta_t_Back_sub + delta_t_Mat + delta_t_Track
log_file.write('total algorithm time is '+str(delta_t_tot)+' minutes\n')
log_file.close()