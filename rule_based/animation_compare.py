import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
#from common.visualization import *
import visualization

#import torch
from pycpd import RigidRegistration

def animation_compare(reference,student,error,
                      ref_video_path,stu_video_path,
                      ref_keypoints,stu_keypoints,frame_list, angle_list, targetJoint):
    """
    ref_video_path: input video path for teacher
    stu_video_path: input video path for student
    ref_keypoints: teacher key points (frames,17,2)
    stu_keypoints: student key points (frames,17,2)

    """

    # Point set registration of reference and student
    #transformed_student = []
    #for idx in range(len(reference)):
    #    rt = RigidRegistration(X=reference[idx], Y=student[idx])
    #    rt.register()
    #    rt.transform_point_cloud()
    #    transformed_student.append(np.expand_dims(rt.TY, axis=0))

    #student = np.concatenate(transformed_student, axis=0)

    plt.ioff()
    fig = plt.figure(figsize=(6*3,6*1))
    #one row 3 columns and last one is the skel
    ax_3d = fig.add_subplot(1,3,2, projection='3d')
    ax_3d.set_xlim(-1,1)
    ax_3d.set_ylim(-1,1)
    ax_3d.set_zlim(1,-1)
    #error_text = ax_3d.text2D(1, 1, 'Error: 0', transform = ax_3d.transAxes)
    
    #teacher plot
    ax_ref = fig.add_subplot(1,3,1)
    ax_ref.get_xaxis().set_visible(False)
    ax_ref.get_yaxis().set_visible(False)
    ax_ref.set_axis_off()
    ax_ref.set_title('Teacher')

    #student plot
    ax_stu = fig.add_subplot(1,3,3)
    ax_stu.get_xaxis().set_visible(False)
    ax_stu.get_yaxis().set_visible(False)
    ax_stu.set_axis_off()
    ax_stu.set_title('Single Mode')

    limb_map = [[0, 1],  [1, 2], [2, 3],     # Right leg
                [0, 4],  [4, 5], [5, 6],     # Left leg
                [0, 7],  [7, 8],             # Spine
                [8, 14], [14, 15], [15, 16], # Right arm
                [8, 11], [11, 12], [12, 13], # Left arm
                [8, 9],  [9, 10]             # Neck
               ]

    # correction_map = [[8,14],[14,15]]
    #targetJoint.append([1,2])
    #targetJoint.append([2,3])
    correction_map = targetJoint
    print(correction_map)
    l_corr = len(correction_map)

    for i in range(l_corr):
        limb_map.remove(correction_map[i])

    l_limb = len(limb_map)
    

    # There are 17 joints, therefore 16 limbs
    ref_limbs = [ax_3d.plot3D([], [], [], c='b') for _ in range(16)]
    stu_limbs = [ax_3d.plot3D([], [], [], c='b') for _ in range(l_limb)]
    print(stu_limbs)
    corr_limbs = [ax_3d.plot3D([], [], [], c='r') for _ in range(l_corr)]
    corr_limbsBlue = [ax_3d.plot3D([], [], [], c='b') for _ in range(l_corr)]
    corr_limbsBlue2 = [ax_3d.plot3D([], [], [], c='b') for _ in range(l_corr)]
        

    


    limit = len(reference)
    #teacher video frames
    ref_video_frames = []
    for f in visualization.read_video(ref_video_path, skip=0, limit=limit):
       ref_video_frames.append(f)
    
    #student video frames
    stu_video_frames = []
    for f in visualization.read_video(stu_video_path, skip=0, limit=limit):
        stu_video_frames.append(f)


    ref_image = None
    stu_image = None
    ref_points = None
    stu_points = None

    print("ref_keypoints shape {}".format(ref_keypoints.shape))

    def update_animation(idx):
        print (idx)
        nonlocal ref_image,stu_image,ref_points,stu_points 
        #teacher image
        if not ref_image:
           print("init ref image")
           ref_image = ax_ref.imshow(ref_video_frames[idx],aspect='equal')
        else:
           ref_image.set_data(ref_video_frames[idx]) 
        
        #student image
        if not stu_image:
            stu_image = ax_stu.imshow(stu_video_frames[idx],aspect='equal')
        else:
            stu_image.set_data(stu_video_frames[idx]) 

        #colors for points
        colors_2d = np.full(ref_keypoints.shape[1], 'black')
        joints_right_2d = [2, 4, 6, 8, 10, 12, 14, 16]
        colors_2d[joints_right_2d] = 'red'

        #teacher points
        if not ref_points:
           print("init ref points")
           ref_points = ax_ref.scatter(*ref_keypoints[idx].T, 10, color=colors_2d, edgecolors='white', zorder=10)
        else:
           ref_points.set_offsets(ref_keypoints[idx])

        #student points
        if not stu_points:
            stu_points = ax_stu.scatter(*stu_keypoints[idx].T, 10, color=colors_2d, edgecolors='white', zorder=10)
        else:
            stu_points.set_offsets(stu_keypoints[idx])


        #ref_frame = reference[idx]
        stu_frame = student[idx]
        
        for i in range(len(limb_map)):
            #ref_limbs[i][0].set_data(ref_frame[limb_map[i], [[0,0],[2,2]]])
            #ref_limbs[i][0].set_3d_properties(ref_frame[limb_map[i], 1])
            
            stu_limbs[i][0].set_data(stu_frame[limb_map[i], [[0,0],[2,2]]])
            stu_limbs[i][0].set_3d_properties(stu_frame[limb_map[i], 1])
        for i in range(len(correction_map)):
            corr_limbsBlue[i][0].set_data(stu_frame[correction_map[i], [[0,0],[2,2]]])
            corr_limbsBlue[i][0].set_3d_properties(stu_frame[correction_map[i], 1])
                   
        startIdx = frame_list[0]
        endIdx = frame_list[len(frame_list) - 1]    
        if idx == startIdx:
            for i in range(len(correction_map)):
                corr_limbsBlue[i][0].remove()
        if startIdx <= idx <= endIdx:
            print (idx)
            for i in range(len(correction_map)):
                corr_limbs[i][0].set_data(stu_frame[correction_map[i], [[0,0],[2,2]]])
                corr_limbs[i][0].set_3d_properties(stu_frame[correction_map[i], 1])
        if idx == endIdx + 1:
            for i in range(len(correction_map)):
                corr_limbs[i][0].remove()
        if idx > endIdx:
            for i in range(len(correction_map)):
                corr_limbsBlue2[i][0].set_data(stu_frame[correction_map[i], [[0,0],[2,2]]])
                corr_limbsBlue2[i][0].set_3d_properties(stu_frame[correction_map[i], 1])


        #if idx < len(error):
            #error_text.set_text('Error: {}'.format(int(error[idx])))
        
    iterations = len(reference)
    ani = animation.FuncAnimation(fig, update_animation, iterations,
                                  interval=50, blit=False, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=3000)

    return ani, writer

def error(angle_tensor, window_sz=15):

    rolling_average = np.convolve(angle_tensor, np.ones(window_sz,)) / window_sz
    return rolling_average
