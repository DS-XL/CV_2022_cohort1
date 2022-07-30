import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import torch
from pycpd import RigidRegistration

import sys
from sklearn import preprocessing

from scipy import stats

def ang_comp(reference, student, round_tensor=False):
    # Get all joint pair angles, frames x number of joint pairs

    adjacent_limb_map = [
                          [[0, 1],  [1, 2], [2, 3]],     # Right leg
                          [[0, 4],  [4, 5], [5, 6]],     # Left leg
                          [[0, 7],  [7, 8]],             # Spine
                          [[8, 14], [14, 15], [15, 16]], # Right arm
                          [[8, 11], [11, 12], [12, 13]], # Left arm
                          [[8, 9],  [9, 10]]             # Neck
                        ]
    
    adjacent_limbs_ref = []
    adjacent_limbs_stu = []
    num_frames = len(reference)

    def update_adjacent_limbs(person, adj, limb_id):
        for adj_limb_id in range(len(adjacent_limb_map[limb_id]) - 1):
            joint1a, joint1b = adjacent_limb_map[limb_id][adj_limb_id]
            joint2a, joint2b = adjacent_limb_map[limb_id][adj_limb_id + 1]
            
            limb1_vector = person[joint1a] - person[joint1b]  # Difference vector between two joints
            limb2_vector = person[joint2a] - person[joint2b]
            
            # Normalize the vectors
            limb1_vector = torch.div(limb1_vector, torch.norm(limb1_vector)).unsqueeze(0)
            limb2_vector = torch.div(limb2_vector, torch.norm(limb2_vector)).unsqueeze(0)
            
            adj.append(torch.Tensor(torch.cat([limb1_vector, limb2_vector], dim=0)).unsqueeze(0))

    for idx in range(num_frames):
        frame_reference = reference[idx] # frame_reference contains the coordinates of 17 joints of the reference video
        frame_student   = student[idx]
        for limb_id in range(len(adjacent_limb_map)):
            update_adjacent_limbs(frame_reference, adjacent_limbs_ref, limb_id)
            update_adjacent_limbs(frame_student, adjacent_limbs_stu, limb_id)


    # editing - Cals speed sequence, only use instructor's speed
    speed_ls = []
    for idx in range(num_frames):
        frame_pre = reference[idx]
        if idx+1<num_frames:
            frame_next = reference[idx+1]
        else:
            frame_pre = reference[idx-1]
            frame_next = reference[idx]
        speed = abs(torch.sum(frame_next - frame_pre)) # frame_next is a tensor
        speed_ls.append(speed)

        
    adjacent_limbs_ref = torch.cat(adjacent_limbs_ref, dim=0)
    adjacent_limbs_stu = torch.cat(adjacent_limbs_stu, dim=0)

    # Get angles between adjacent limbs, each of the below tensors are of shape (num_frames x 10), aka scalars
    adjacent_limbs_ref = torch.bmm(adjacent_limbs_ref[:, 0:1, :], adjacent_limbs_ref[:, 1, :].unsqueeze(-1))
    adjacent_limbs_stu = torch.bmm(adjacent_limbs_stu[:, 0:1, :], adjacent_limbs_stu[:, 1, :].unsqueeze(-1))
    
    # Get absolute difference between instructor and student angles in degrees 
    # 57.296 * radians converts units to degrees
    absolute_diffs = torch.abs((57.296*(adjacent_limbs_ref - adjacent_limbs_stu))).reshape(num_frames, 10)
    return absolute_diffs.sum(dim=1), speed_ls

def overlap_animation(reference, student, error, speed_ls, importance_rolling):

    # Point set registration of reference and student
    transformed_student = []
    for idx in range(len(reference)):
        rt = RigidRegistration(X=reference[idx], Y=student[idx])
        rt.register()
        rt.transform_point_cloud()
        transformed_student.append(np.expand_dims(rt.TY, axis=0))

    student = np.concatenate(transformed_student, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(1,-1)
    error_text = ax.text2D(0.75, 0.95, 'Score: 0', transform=ax.transAxes)
    importance_text = ax.text2D(0.75, 0.85, 'Importance: 0', transform=ax.transAxes)
    speed_text = ax.text2D(0.05, 0.95, 'Speed: 0', transform=ax.transAxes)
    legend_1 = ax.text2D(0.05, 0.85, 'Teacher: Blue', transform=ax.transAxes)
    legend_2 = ax.text2D(0.05, 0.75, 'Student: Red', transform=ax.transAxes)
    
    # There are 17 joints, therefore 16 limbs
    ref_limbs = [ax.plot3D([], [], [], c='b') for _ in range(16)]
    stu_limbs = [ax.plot3D([], [], [], c='r') for _ in range(16)]
        
    limb_map = [
                [0, 1],  [1, 2], [2, 3],     # Right leg
                [0, 4],  [4, 5], [5, 6],     # Left leg
                [0, 7],  [7, 8],             # Spine
                [8, 14], [14, 15], [15, 16], # Right arm
                [8, 11], [11, 12], [12, 13], # Left arm
                [8, 9],  [9, 10]             # Neck
               ]
        
    def update_animation(idx):
        ref_frame = reference[idx]
        stu_frame = student[idx]
        
        for i in range(len(limb_map)):
            # ref_limbs[i][0].set_data(ref_frame[limb_map[i], :2].T)
            ref_limbs[i][0].set_data(ref_frame[limb_map[i], [[0,0], [2,2]]])
            ref_limbs[i][0].set_3d_properties(ref_frame[limb_map[i], 1])
            
            stu_limbs[i][0].set_data(stu_frame[limb_map[i], [[0,0], [2,2]]])
            stu_limbs[i][0].set_3d_properties(stu_frame[limb_map[i], 1])

        if idx < len(error):
            # print(type(error[idx]))
            # print(type(speed_ls[idx]))
            error_text.set_text('Score: {:0.4f}'.format(error[idx]))
            speed_text.set_text('Speed: {:0.4f}'.format(speed_ls[idx]))
            # print(importance_rolling.shape)
            importance_text.set_text('Importance: {:0.4f}'.format(importance_rolling[idx]))
        
    iterations = len(reference)
    ani = animation.FuncAnimation(fig, update_animation, iterations,
                                  interval=50, blit=False, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    return ani, writer


def error(angle_tensor, speed_ls, window_sz=50):

    # print("angle_tensor.shape", angle_tensor.shape)
    # print("len(speed_ls)", len(speed_ls))

    rolling_average = np.convolve(angle_tensor, np.ones(window_sz,), 'same') / window_sz # need to replace np.ones(window_sz,) with the speed sequence
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    errs_norm = scaler.fit_transform(np.array(rolling_average).reshape(-1,1))


    speed_ls, fitted_lambda = stats.boxcox(speed_ls)
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 100))
    speed_ls = scaler.fit_transform(np.array(speed_ls).reshape(-1,1))


    importance_ls = [1/i for i in speed_ls]
    # importance_ls, fitted_lambda = stats.boxcox(importance_ls)
    importance_rolling = np.convolve(np.squeeze(importance_ls), np.ones(window_sz,), 'same') / window_sz
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    importance_rolling = scaler.fit_transform(np.array(importance_rolling).reshape(-1,1))

    score = np.squeeze(100 - errs_norm)

    # error score = angle_tensor[i] * 1/speed_ls[i]
    # errs = []
    # for i in range(angle_tensor.shape[0]):
    #     err = angle_tensor[i] * 1/speed_ls[i]
    #     errs.append(err)

    # errs = np.array(errs).reshape(-1,1)

    # errs_norm, fitted_lambda = stats.boxcox(np.squeeze(errs))
    # print("fitted_lambda", fitted_lambda)

    
    

    # transformer = preprocessing.StandardScaler().fit(errs)
    # errs_norm = transformer.transform(errs)
    # normalize_scaler = preprocessing.normalize()

    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    # errs_norm = scaler.fit_transform(errs)

    
    # score_moving_avg = np.convolve(score, np.ones(window_sz), 'same') / window_sz


    #plt.plot(speed_ls)
    #plt.savefig('../outputs/speed_ls.png')
    
    # plt.plot(errs_norm)
    # plt.savefig('../outputs/errs_norm.png')

    #plt.plot(score)
    #plt.savefig('../outputs/score.png')

    #plt.plot(importance_rolling)
    #plt.savefig('../outputs/importance_rolling.png')



    # sys.exit()

    # print(score.shape, len(speed_ls), importance_rolling.shape)


    return score, np.squeeze(speed_ls), np.squeeze(importance_rolling)









