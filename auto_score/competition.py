import os
import sys

import numpy as np

import torch
#from loguru import logger

import angles
import angles_ol
import animation_compare
from common.camera import image_coordinates


# instructor_pose = np.load('../joints/joints.npy'.format(instructor.split('.')[0]))
instructor_pose = np.load('./output_joints/bl_teacher_2.npy')
student_pose1   = np.load('./output_joints/bl_student_2.npy')
student_pose2   = np.load('./output_joints/bad_student_2.npy')

instructor_keypoints = np.load("output_data_2d/data_2d_custom_bl_teacher.npz", allow_pickle=True)
instructor_keypoints_metadata = instructor_keypoints['metadata'].item()
instructor_keypoints = instructor_keypoints['positions_2d'].item()
instructor_video_name="bl_teacher_2.mp4"
instructor_keypoints = instructor_keypoints[instructor_video_name]['custom'][0].copy() 
instructor_video_path = "./input_video/"+instructor_video_name

student_keypoints1 = np.load("output_data_2d/data_2d_custom_bl_student.npz", allow_pickle=True)
student_keypoints_metadata1 = student_keypoints1['metadata'].item()
student_keypoints1 = student_keypoints1['positions_2d'].item()
student_video_name1="bl_student_2.mp4"
student_keypoints1 = student_keypoints1[student_video_name1]['custom'][0].copy() 
student_video_path1 = "./input_video/"+student_video_name1 

student_keypoints2 = np.load("output_data_2d/data_2d_custom_bad_student.npz", allow_pickle=True)
student_keypoints_metadata2 = student_keypoints2['metadata'].item()
student_keypoints2 = student_keypoints2['positions_2d'].item()
student_video_name2="bad_student_24fps_flip_2.mp4"
student_keypoints2 = student_keypoints2[student_video_name2]['custom'][0].copy() 
student_video_path2 = "./input_video/"+student_video_name2 


instructor_pose = instructor_pose[:600]
student_pose1 = student_pose1[:600]
student_pose2 = student_pose2[:600]

print(instructor_pose.shape)
print(student_pose1.shape)
print(student_pose2.shape)

min_frames = min(len(instructor_pose), len(student_pose1),len(student_pose2))
if len(instructor_pose) > min_frames:
    instructor_pose = instructor_pose[:min_frames]
if len(student_pose1) > min_frames:
    student_pose1 = student_pose1[:min_frames]
if len(student_pose2) > min_frames:
    student_pose2 = student_pose2[:min_frames]


instructor_pose_tensor = torch.from_numpy(instructor_pose)
student_pose_tensor1  = torch.from_numpy(student_pose1)
student_pose_tensor2  = torch.from_numpy(student_pose2)


#angles_between = angles.ang_comp(instructor_pose_tensor, student_pose_tensor, round_tensor=True)
#error = angles.error(angles_between)
#print('Error {}'.format(error))

angle_tensor1, speed_ls1 = angles_ol.ang_comp(instructor_pose_tensor, student_pose_tensor1, round_tensor=True)
error1, speed_ls1, importance_rolling1 = angles_ol.error(angle_tensor1,speed_ls1)

angle_tensor2, speed_ls2 = angles_ol.ang_comp(instructor_pose_tensor, student_pose_tensor2, round_tensor=True)
error2, speed_ls2, importance_rolling2 = angles_ol.error(angle_tensor2,speed_ls2)


#print("error shape: {}".format(error1.shape))
#print("speed shape: ".format(speed_ls.shape))
#print("importance rolling shape: ".format(importance_rolling.shape))

print("student 1 score first half: {}".format(np.mean(error1[0:300])))
print("student 1 score last half: {}".format(np.mean(error1[300:-1])))

print("student 2 score first half: {}".format(np.mean(error2[0:300])))
print("student 2 score last half: {}".format(np.mean(error2[300:-1])))

#print(instructor_keypoints.shape)
#print(student_keypoints1.shape)
#print(student_keypoints2.shape)

ani, writer = animation_compare.animation_compete(instructor_pose,
                                  student_pose1, error1,speed_ls1,importance_rolling1,
                                  student_pose2, error2,speed_ls2,importance_rolling2,
                                  ref_video_path=instructor_video_path, 
                                  stu_video_path1 = student_video_path1,
                                  stu_video_path2 = student_video_path2,
                                  ref_keypoints = instructor_keypoints,
                                  stu_keypoints1 = student_keypoints1,
                                  stu_keypoints2 = student_keypoints2)
#ani, writer = angles.overlap_animation(instructor_pose, student_pose, error)
# instructor = instructor.split('.mp4')[0]
# student    = student.split('.mp4')[0]
ani.save('./output/output_compete_2.gif', dpi=80, writer='imagemagick')
