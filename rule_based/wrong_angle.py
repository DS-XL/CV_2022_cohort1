import numpy as np

import torch
def ang_comp(reference, round_tensor = False):
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
        frame_reference = reference[idx]
        for limb_id in range(len(adjacent_limb_map)):
            update_adjacent_limbs(frame_reference, adjacent_limbs_ref, limb_id)
    
    adjacent_limbs_ref = torch.cat(adjacent_limbs_ref, dim=0)
    
    # Get angles between adjacent limbs, each of the below tensors are of shape (num_frames x 10), aka scalars
    adjacent_limbs_ref = torch.bmm(adjacent_limbs_ref[:, 0:1, :], adjacent_limbs_ref[:, 1, :].unsqueeze(-1))
    adjacent_limbs_ref = adjacent_limbs_ref * 57.296
    return adjacent_limbs_ref


           
            
