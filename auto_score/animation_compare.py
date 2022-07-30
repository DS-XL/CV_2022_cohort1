import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from common.visualization import *

#import torch
from pycpd import RigidRegistration

def animation_compare(reference,student,error,speed_ls,importance_rolling,
                      ref_video_path,stu_video_path,
                      ref_keypoints,stu_keypoints):
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

    nrows=1
    ncols=3
    plt.ioff()
    fig = plt.figure(figsize=(6*ncols,6*nrows))
    #one row 3 columns and last one is the skeleton
    ax_3d = fig.add_subplot(nrows,ncols,3, projection='3d')
    ax_3d.set_xlim(-1,1)
    ax_3d.set_ylim(-1,1)
    ax_3d.set_zlim(1,-1)
    #error_text = ax_3d.text2D(1, 1, 'Error: 0', transform=ax_3d.transAxes)
    
    error_text = ax_3d.text2D(0.75, 0.95, 'Score: 0', transform=ax_3d.transAxes)
    importance_text = ax_3d.text2D(0.75, 0.85, 'Importance: 0', transform=ax_3d.transAxes)
    speed_text = ax_3d.text2D(0.05, 0.95, 'Speed: 0', transform=ax_3d.transAxes)
    legend_1 = ax_3d.text2D(0.05, 0.85, 'Teacher: Blue', transform=ax_3d.transAxes)
    legend_2 = ax_3d.text2D(0.05, 0.75, 'Student: Red', transform=ax_3d.transAxes)
    encourage = ax_3d.text2D(0.5,0.95,"",transform=ax_3d.transAxes)
    
    #teacher plot
    ax_ref = fig.add_subplot(nrows,ncols,1)
    ax_ref.get_xaxis().set_visible(False)
    ax_ref.get_yaxis().set_visible(False)
    ax_ref.set_axis_off()
    ax_ref.set_title('Teacher')

    #student plot
    ax_stu = fig.add_subplot(nrows,ncols,2)
    ax_stu.get_xaxis().set_visible(False)
    ax_stu.get_yaxis().set_visible(False)
    ax_stu.set_axis_off()
    ax_stu.set_title('Student')


    # There are 17 joints, therefore 16 limbs
    ref_limbs = [ax_3d.plot3D([], [], [], c='b') for _ in range(16)]
    stu_limbs = [ax_3d.plot3D([], [], [], c='r') for _ in range(16)]
        
    limb_map = [
                [0, 1],  [1, 2], [2, 3],     # Right leg
                [0, 4],  [4, 5], [5, 6],     # Left leg
                [0, 7],  [7, 8],             # Spine
                [8, 14], [14, 15], [15, 16], # Right arm
                [8, 11], [11, 12], [12, 13], # Left arm
                [8, 9],  [9, 10]             # Neck
               ]
    
    limit = len(reference)
    #teacher video frames
    ref_video_frames = []
    for f in read_video(ref_video_path, skip=0, limit=limit):
        ref_video_frames.append(f)
    
    #student video frames
    stu_video_frames = []
    for f in read_video(stu_video_path, skip=0, limit=limit):
        stu_video_frames.append(f)


    ref_image = None
    stu_image = None
    ref_points = None
    stu_points = None

    print("ref_keypoints shape {}".format(ref_keypoints.shape))

    def update_animation(idx):
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


        ref_frame = reference[idx]
        stu_frame = student[idx]
        
        for i in range(len(limb_map)):
            ref_limbs[i][0].set_data(ref_frame[limb_map[i], [[0,0],[2,2]]])
            ref_limbs[i][0].set_3d_properties(ref_frame[limb_map[i], 1])
            
            stu_limbs[i][0].set_data(stu_frame[limb_map[i], [[0,0],[2,2]]])
            stu_limbs[i][0].set_3d_properties(stu_frame[limb_map[i], 1])
            
            #if idx==20:
            #    pass
            #else:
            #    stu_limbs[i][0].set_data([],[])
            #    stu_limbs[i][0].set_3d_properties([])
            
        
        
        if idx < len(error):
            #error_text.set_text('Error: {}'.format(int(error[idx])))
            # print(type(error[idx]))
            # print(type(speed_ls[idx]))
            error_text.set_text('Score: {:0.4f}'.format(error[idx]))
            speed_text.set_text('Speed: {:0.4f}'.format(speed_ls[idx]))
            #print(importance_rolling.shape)
            importance_text.set_text('Importance: {:0.4f}'.format(importance_rolling[idx]))
            if error[idx]>70:
                encourage.set_text("Great !")
                encourage.set_backgroundcolor("lime")
            elif error[idx]<30:
                encourage.set_text("Come on !")
                encourage.set_backgroundcolor("aqua")
            else:
                encourage.set_text("")
            
            if error[idx]<20:
                stu_points.set_color('darkorange')
            else:
                stu_points.set_color(colors_2d)
        
    iterations = len(reference)
    ani = animation.FuncAnimation(fig, update_animation, iterations,
                                  interval=50, blit=False, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=3000)

    return ani, writer


def animation_compete(reference,
               student1,error1,speed_ls1,importance_rolling1,
               student2,error2,speed_ls2,importance_rolling2,
               ref_video_path,stu_video_path1,stu_video_path2,
               ref_keypoints,stu_keypoints1,stu_keypoints2):
    """
    ref_video_path: input video path for teacher
    stu_video_path1: input video path for student1
    stu_video_path2: input video path for student2
    ref_keypoints: teacher key points (frames,17,2)
    stu_keypoints1: student1 key points (frames,17,2)
    stu_keypoints2: student2 key points (frames,17,2)

    """

    # Point set registration of reference and student
    #transformed_student = []
    #for idx in range(len(reference)):
    #    rt = RigidRegistration(X=reference[idx], Y=student[idx])
    #    rt.register()
    #    rt.transform_point_cloud()
    #    transformed_student.append(np.expand_dims(rt.TY, axis=0))

    #student = np.concatenate(transformed_student, axis=0)

    nrows=1
    ncols=3
    plt.ioff()
    fig = plt.figure(figsize=(6*ncols,6*nrows))
    #one row 3 columns and last one is the skeleton
    ax_3d = fig.add_subplot(nrows,ncols,3, projection='3d')
    ax_3d.set_xlim(-1,1)#-1,0.5
    ax_3d.set_ylim(-1,1)#0,0.6
    ax_3d.set_zlim(1,-1)
    #error_text = ax_3d.text2D(1, 1, 'Error: 0', transform=ax_3d.transAxes)
    
    #error_text = ax_3d.text2D(0.75, 0.95, 'Score: 0', transform=ax_3d.transAxes)
    #importance_text = ax_3d.text2D(0.75, 0.85, 'Importance: 0', transform=ax_3d.transAxes)
    #speed_text = ax_3d.text2D(0.05, 0.95, 'Speed: 0', transform=ax_3d.transAxes)
    legend_1 = ax_3d.text2D(0.05, 0.85, 'Teacher:  Blue', transform=ax_3d.transAxes,color='b')
    legend_2 = ax_3d.text2D(0.05, 0.75, 'Student1: Red', transform=ax_3d.transAxes, color='r')
    legend_3 = ax_3d.text2D(0.05, 0.65, 'Student2: Orange',transform=ax_3d.transAxes, color='orange')
    #encourage = ax_3d.text2D(0.5,0.95,"",transform=ax_3d.transAxes)
    
    #teacher plot
    ax_stu1 = fig.add_subplot(nrows,ncols,1)
    ax_stu1.get_xaxis().set_visible(False)
    ax_stu1.get_yaxis().set_visible(False)
    ax_stu1.set_axis_off()
    ax_stu1.set_title('Student1')
    score_text1 = ax_stu1.text(0.95, 0.95, 'Score: 0', transform=ax_stu1.transAxes)

    #student plot
    ax_stu2 = fig.add_subplot(nrows,ncols,2)
    ax_stu2.get_xaxis().set_visible(False)
    ax_stu2.get_yaxis().set_visible(False)
    ax_stu2.set_axis_off()
    ax_stu2.set_title('Student2')
    score_text2 = ax_stu1.text(0.95, 0.95, 'Score: 0', transform=ax_stu2.transAxes)


    # There are 17 joints, therefore 16 limbs
    ref_limbs = [ax_3d.plot3D([], [], [], c='b') for _ in range(16)]
    stu_limbs1 = [ax_3d.plot3D([], [], [], c='r') for _ in range(16)]
    stu_limbs2 = [ax_3d.plot3D([], [], [], c='orange') for _ in range(16)]
        
    limb_map = [
                [0, 1],  [1, 2], [2, 3],     # Right leg
                [0, 4],  [4, 5], [5, 6],     # Left leg
                [0, 7],  [7, 8],             # Spine
                [8, 14], [14, 15], [15, 16], # Right arm
                [8, 11], [11, 12], [12, 13], # Left arm
                [8, 9],  [9, 10]             # Neck
               ]
    
    limit = len(reference)
    #teacher video frames
    stu_video_frames1 = []
    for f in read_video(stu_video_path1, skip=0, limit=limit):
        stu_video_frames1.append(f)
    
    #student video frames
    stu_video_frames2 = []
    for f in read_video(stu_video_path2, skip=0, limit=limit):
        stu_video_frames2.append(f)


    stu_image1 = None
    stu_image2 = None
    stu_points1 = None
    stu_points2 = None

    #print("ref_keypoints shape {}".format(ref_keypoints.shape))

    def update_animation(idx):
        nonlocal stu_image1,stu_image2,stu_points1,stu_points2 
        #teacher image
        if not stu_image1:
            print("init stu1 image")
            stu_image1 = ax_stu1.imshow(stu_video_frames1[idx],aspect='equal')
        else:
            stu_image1.set_data(stu_video_frames1[idx]) 
        
        #student image
        if not stu_image2:
            print("init stu2 image")
            stu_image2 = ax_stu2.imshow(stu_video_frames2[idx],aspect='equal')
        else:
            stu_image2.set_data(stu_video_frames2[idx]) 

        #colors for points
        colors_2d = np.full(ref_keypoints.shape[1], 'black')
        joints_right_2d = [2, 4, 6, 8, 10, 12, 14, 16]
        colors_2d[joints_right_2d] = 'red'

        #teacher points
        if not stu_points1:
            print("init stu1 points")
            stu_points1 = ax_stu1.scatter(*stu_keypoints1[idx].T, 10, color=colors_2d, edgecolors='white', zorder=10)
        else:
            stu_points1.set_offsets(stu_keypoints1[idx])

        #student points
        if not stu_points2:
            print("init stu2 points")
            stu_points2 = ax_stu2.scatter(*stu_keypoints2[idx].T, 10, color=colors_2d, edgecolors='white', zorder=10)
        else:
            stu_points2.set_offsets(stu_keypoints2[idx])


        ref_frame = reference[idx]
        stu_frame1 = student1[idx]
        stu_frame2 = student2[idx]
        
        for i in range(len(limb_map)):
            ref_limbs[i][0].set_data(ref_frame[limb_map[i], [[0,0],[2,2]]])
            ref_limbs[i][0].set_3d_properties(ref_frame[limb_map[i], 1])
            
            stu_limbs1[i][0].set_data(stu_frame1[limb_map[i], [[0,0],[2,2]]])
            stu_limbs1[i][0].set_3d_properties(stu_frame1[limb_map[i], 1])
            
            stu_limbs2[i][0].set_data(stu_frame2[limb_map[i], [[0,0],[2,2]]])
            stu_limbs2[i][0].set_3d_properties(stu_frame2[limb_map[i], 1])
            
        
        if idx < len(error1):
            score_text1.set_text('Score: {:0.4f}'.format(error1[idx]))
            score_text2.set_text('Score: {:0.4f}'.format(error2[idx]))
            if error1[idx]>=90:
                score_text1.set_backgroundcolor('aqua')
            elif error1[idx]<90 and error1[idx]>=75:
                score_text1.set_backgroundcolor('lime')
            elif error1[idx]<75 and error1[idx]>=60:
                score_text1.set_backgroundcolor('yellow')
            elif error1[idx]<60 and error1[idx]>=45:
                score_text1.set_backgroundcolor('darkorange')
            else:
                score_text1.set_backgroundcolor('red')
            
            if error2[idx]>=90:
                score_text2.set_backgroundcolor('aqua')
            elif error2[idx]<90 and error2[idx]>=75:
                score_text2.set_backgroundcolor('lime')
            elif error2[idx]<75 and error2[idx]>=60:
                score_text2.set_backgroundcolor('yellow')
            elif error2[idx]<60 and error2[idx]>=45:
                score_text2.set_backgroundcolor('darkorange')
            else:
                score_text2.set_backgroundcolor('red')
            
    
    iterations = len(reference)
    ani = animation.FuncAnimation(fig, update_animation, iterations,
                                  interval=50, blit=False, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=3000)

    return ani, writer


def error(angle_tensor, window_sz=15):

    rolling_average = np.convolve(angle_tensor, np.ones(window_sz,)) / window_sz
    return rolling_average
