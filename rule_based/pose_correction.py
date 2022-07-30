import os
import sys

import numpy as np
import torch
import wrong_angle
import animation_compare

#https://www.youtube.com/watch?v=ZGhvvMt7WMA
#https://www.youtube.com/watch?v=OxfXp10JRKw
#https://builtwithscience.com/perfect-push-up-form/

#instructor_pose = np.load('./joints/bl_teacher.npy')
instructor_pose = np.load('./joints/badSquat.npy') #actually good
#instructor_pose = np.load('./joints/joints_teacher_30fps.npy') 
#student_pose = np.load('./joints/bl_student.npy')
student_pose = np.load('./joints/Bad.npy',allow_pickle=True)
#student_pose = np.load('./joints/bad2.npy',allow_pickle=True)
#student_pose = np.load('./joints/joints_bad_student.npy',allow_pickle=True)

#instructor_keypoints = np.load("data/data_2d_custom_bl_teacher.npz", allow_pickle=True)
instructor_keypoints = np.load("data/data_2d_custom_squat.npz", allow_pickle=True)
#instructor_keypoints = np.load("data/data_2d_custom_teacher_30fps.npz", allow_pickle=True)

instructor_keypoints_metadata = instructor_keypoints['metadata'].item()
instructor_keypoints = instructor_keypoints['positions_2d'].item()
instructor_video_name = list(instructor_keypoints.keys())[0]
instructor_video_name = 'badSquat.mov'
#instructor_keypoints = instructor_keypoints[instructor_video_name]['custom'][0].copy() 
instructor_video_path = "./video/"+instructor_video_name

#student_keypoints = np.load("data/data_2d_custom_bl_student.npz", allow_pickle=True)
student_keypoints = np.load("data/data_2d_custom_squat.npz", allow_pickle=True)
#student_keypoints = np.load("data/data_2d_custom_myvideos_bad.npz", allow_pickle=True)
#student_keypoints = np.load("data/data_2d_custom_bad_student.npz", allow_pickle=True)

student_keypoints_metadata = student_keypoints['metadata'].item()
student_keypoints = student_keypoints['positions_2d'].item()
student_video_name = list(student_keypoints.keys())[0]
student_video_name = 'Bad.mov'
#student_keypoints = student_keypoints[student_video_name]['custom'][0].copy() 
student_video_path = "./video/"+student_video_name 


print (len(student_pose))
frame = 100
#frame = 898
startFrame = 55
#startFrame = 340
endFrame = 80
#endFrame = 898
targetJoint = 3
#targetJoint = 1
desiredAngle = 20
#desiredAngle = 90
tolerance = 5
instructor_pose = instructor_pose[:frame]
student_pose = student_pose[:frame]

min_frames = min(len(instructor_pose), len(student_pose))
if len(instructor_pose) > min_frames:
    instructor_pose = instructor_pose[:min_frames]
if len(student_pose) > min_frames:
    student_pose = student_pose[:min_frames]

instructor_pose_tensor = torch.from_numpy(instructor_pose)
student_pose_tensor = torch.from_numpy(student_pose)

angle_between = wrong_angle.ang_comp(student_pose_tensor, round_tensor = True).reshape([frame,10])
print(len(angle_between))
for ind,vec in enumerate(angle_between):
    print (ind,'--',vec)


jointTable = {0:[[0,1],[1,2]],
           1:[[1,2],[2,3]],
           2:[[0,4],[4,5]],
           3:[[4,5],[5,6]],
           4:[[0,7],[7,8]],
           5:[[8,14],[14,15]],
           6:[[14,15],[15,16]],
           7:[[8,11],[11,12]],
           8:[[11,12],[12,13]],
           9:[[8,9],[9,10]]}

def correction(startFrame,endFrame,targetJoint,desiredAngle,tolerance):
    frameList = []
    angleList = []
    for frame, angleVector in enumerate(angle_between):
        if startFrame <= frame <= endFrame and abs(angleVector[targetJoint] - desiredAngle) > tolerance:
            frameList.append(frame)
            angleList.append(angleVector[targetJoint])
    return frameList, angleList
frameList, angleList = correction(startFrame,endFrame,targetJoint,desiredAngle,tolerance)
print('frameList,angleList')
print(frameList)
print(angleList)
ani, writer = animation_compare.animation_compare(instructor_pose, student_pose, error=None, 
                                                  ref_video_path=instructor_video_path, 
                                                  stu_video_path = student_video_path,
                                                  ref_keypoints = instructor_keypoints,
                                                  stu_keypoints = student_keypoints,frame_list = frameList,
                                                  angle_list = angleList, targetJoint = jointTable[targetJoint])
ani.save('./output/test3.gif', dpi=80, writer='imagemagick')

