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
student_pose   = np.load('./output_joints/bad_student_2.npy')


instructor_keypoints = np.load("output_data_2d/data_2d_custom_bl_teacher.npz", allow_pickle=True)
instructor_keypoints_metadata = instructor_keypoints['metadata'].item()
instructor_keypoints = instructor_keypoints['positions_2d'].item()
#instructor_video_name = list(instructor_keypoints.keys())[0]
instructor_video_name="bl_teacher_2.mp4"
instructor_keypoints = instructor_keypoints[instructor_video_name]['custom'][0].copy() 
instructor_video_path = "./input_video/"+instructor_video_name
#instrctor_video_w = instructor_keypoints_metadata['video_metadata'][instructor_video_name]['w']
#instrctor_video_h = instructor_keypoints_metadata['video_metadata'][instructor_video_name]['h']
#instructor_keypoints = image_coordinates(instructor_keypoints[..., :2], 
#                                         w=instrctor_video_w,
#                                         h=instrctor_video_h)


student_keypoints = np.load("output_data_2d/data_2d_custom_bad_student.npz", allow_pickle=True)
student_keypoints_metadata = student_keypoints['metadata'].item()
student_keypoints = student_keypoints['positions_2d'].item()
#student_video_name = list(student_keypoints.keys())[0]
student_video_name="bad_student_24fps_flip_2.mp4"
student_keypoints = student_keypoints[student_video_name]['custom'][0].copy() 
student_video_path = "./input_video/"+student_video_name 
#student_video_w = student_keypoints_metadata['video_metadata'][student_video_name]['w']
#student_video_h = student_keypoints_metadata['video_metadata'][student_video_name]['h']
#student_keypoints = image_coordinates(student_keypoints[..., :2], 
#                                         w=student_video_w,
#                                         h=student_video_h)


instructor_pose = instructor_pose[:600]
student_pose = student_pose[:600]

print(instructor_pose.shape)
print(student_pose.shape)

min_frames = min(len(instructor_pose), len(student_pose))
if len(instructor_pose) > min_frames:
    instructor_pose = instructor_pose[:min_frames]
if len(student_pose) > min_frames:
    student_pose = student_pose[:min_frames]

instructor_pose_tensor = torch.from_numpy(instructor_pose)
student_pose_tensor    = torch.from_numpy(student_pose)


#angles_between = angles.ang_comp(instructor_pose_tensor, student_pose_tensor, round_tensor=True)
#error = angles.error(angles_between)
#print('Error {}'.format(error))

angle_tensor, speed_ls = angles_ol.ang_comp(instructor_pose_tensor, student_pose_tensor, round_tensor=True)
error, speed_ls, importance_rolling = angles_ol.error(angle_tensor,speed_ls)

print("error shape: ".format(importance_rolling.shape))
print("speed shape: ".format(speed_ls.shape))
print("importance rolling shape: ".format(importance_rolling.shape))

print(instructor_keypoints.shape)
print(student_keypoints.shape)

ani, writer = animation_compare.animation_compare(instructor_pose, student_pose, error,speed_ls,importance_rolling,
                                                  ref_video_path=instructor_video_path, 
                                                  stu_video_path = student_video_path,
                                                  ref_keypoints = instructor_keypoints,
                                                  stu_keypoints = student_keypoints)
#ani, writer = angles.overlap_animation(instructor_pose, student_pose, error)
# instructor = instructor.split('.mp4')[0]
# student    = student.split('.mp4')[0]
ani.save('./output/output_bl_teacher_bad_student_2_compare.gif', dpi=80, writer='imagemagick')
