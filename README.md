# CV_2022_cohort1
The working repo for the 2022 advance AI program - CV
# Instruction 
This directory contains the codes generating the demo in the report. There are three inputs:
1. video
2. 2D pose estimation 
3. 3D pose estimation
2D pose and 3D pose can be got from the pre-trained model VideoPose3D (https://github.com/facebookresearch/VideoPose3D)

# directory and files
1. auto_score: codes for generate demo of autoScore or competition. 
    + commom: codes for preprocess (most comes from VideoPose3D)
    + input_video: the videos for input
    + output_data_2d: 2d pose estimation 
    + output_joints: 3d pose estimation
    + output: the output of the demo
    + angles.py, angles_ol.py, angles_org.py: algorithm calculating the difference between two 3D pose 
    + animation_compare.py: code for generating animation
    + pose_match.py: code generating demo of autoScore. To run just type: `python pose_match.py`
    + competition.py: code generating demo of competition. To run just type: `python competition.py`
2. rule_based: codes for generate demo of rule based. The code structure is similar to auto_score
3. Group2-Project2-Final Report.pdf: the final report for this project

# License
This project is licensed under the terms of the MIT license
