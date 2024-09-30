import numpy as np
import torch
import random

import joblib
import sys
sys.path.append('.')
from configs import get_config

from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin

from utils.utils import *
from utils.plot_script import *
# from utils.preprocess import *


class InterXDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE

        ignore_list = []
        try:
            ignore_list = open(os.path.join(opt.DATA_ROOT, "splits", "ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)
        data_list = []
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "splits",  "train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        if self.opt.MODE == "train_small":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "splits",  "train_small.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "splits", "val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "splits", "test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        random.shuffle(data_list)
        # data_list = data_list[:70]

        self.interaction_order = joblib.load(os.path.join(opt.DATA_ROOT, "annots","interaction_order.pkl"))
        
        index = 0
        files = os.listdir(pjoin(opt.DATA_ROOT, "smpl_joints_30fps"))
        for file in tqdm(files):
            if (file+"\n" not in ignore_list) and (file+"\n" in data_list): # or int(motion_name)>1000

                if self.interaction_order[file] == 1: # ensure p1 is the first person
                    file_path_person1 = pjoin(opt.DATA_ROOT, "smpl_joints_30fps", file, "P1.npy")
                    file_path_person2 = pjoin(opt.DATA_ROOT, "smpl_joints_30fps", file, "P2.npy")
                    body_6d_path_person1 = pjoin(opt.DATA_ROOT, "body_6d_30fps", file, "P1.npy")
                    body_6d_path_person2 = pjoin(opt.DATA_ROOT, "body_6d_30fps", file, "P2.npy")
                else:
                    file_path_person2 = pjoin(opt.DATA_ROOT, "smpl_joints_30fps", file, "P1.npy")
                    file_path_person1 = pjoin(opt.DATA_ROOT, "smpl_joints_30fps", file, "P2.npy")
                    body_6d_path_person2 = pjoin(opt.DATA_ROOT, "body_6d_30fps", file, "P1.npy")
                    body_6d_path_person1 = pjoin(opt.DATA_ROOT, "body_6d_30fps", file, "P2.npy")

                text_path = pjoin(opt.DATA_ROOT, "texts", file + ".txt")
                emotion_text_path = pjoin(opt.DATA_ROOT, "emotion_texts", file +".txt")
                emotion_text = open(emotion_text_path, "r").readlines()
                texts = [item.replace("\n", " ") + emotion_text[0] for item in open(text_path, "r").readlines()]
                texts_swap = [item.replace("\n", " ").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") + emotion_text[0] for item in texts]


                motion1, motion1_swap = self.load_motion(file_path_person1, body_6d_path_person1, self.min_length, swap=True)
                motion2, motion2_swap = self.load_motion(file_path_person2, body_6d_path_person2, self.min_length, swap=True)
                if motion1 is None:
                    continue

                if self.cache:
                    self.motion_dict[index] = [motion1, motion2]
                    self.motion_dict[index+1] = [motion1_swap, motion2_swap]
                else:
                    self.motion_dict[index] = [file_path_person1, file_path_person2]
                    self.motion_dict[index + 1] = [file_path_person1, file_path_person2]



                self.data_list.append({
                    # "idx": idx,
                    "name": file,
                    "motion_id": index,
                    "swap":False,
                    "texts":texts
                })
                if opt.MODE == "train":
                    self.data_list.append({
                        # "idx": idx,
                        "name": file+"_swap",
                        "motion_id": index+1,
                        "swap": True,
                        "texts": texts_swap
                    })

                index += 2
            else: # some seq are not in txt
                print("skip: ", file)
        print("total dataset: ", len(self.data_list))

    def load_motion(self, file_path, smplx_path, min_length, swap=False):
        try:
            motion = np.load(file_path).astype(np.float32)
            poses_body = np.load(smplx_path).astype(np.float32)
        except:
            print("error when load motion: ", file_path)
            return None, None
        fn0 = motion.shape[0]
        fn1 = poses_body.shape[0]
        if fn0 != fn1:
            print("error shape0 between motion12: ", file_path)
            return None, None
        motion1 = motion.reshape((fn0, -1))
        motion2 = poses_body.reshape(fn1, -1)
        motion = np.concatenate([motion1, motion2], axis=1)
        if motion.shape[0] < min_length:
            return None, None
        if swap:
            motion_swap = self.swap_left_right(motion, 22)
        else:
            motion_swap = None
        return motion, motion_swap
     
    def swap_left_right(self, data, n_joints):
        T = data.shape[0]
        new_data = data.copy()
        positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
        rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

        positions = swap_left_right_position(positions)
        rotations = swap_left_right_rot(rotations)

        new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
        return new_data
    
    def downsample(self, data, rate):
        # data: np.array(frames, ...)
        # assert isinstance(rate, int), 'downsample rate should be int.'
        frames_to_keep = np.arange(0, data.shape[0], rate)
        downsampled_data = data[frames_to_keep, ...]
        return downsampled_data

    def process_motion_np(self, motion, feet_thre, prev_frames, n_joints):
        # inter-x motion: already y_up

        '''Uniform Skeleton'''
        # positions = uniform_skeleton(positions, tgt_offsets)

        positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
        rotations = motion[:, n_joints*3:]

        # positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

        '''Put on Floor'''
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height


        '''XZ at origin'''
        root_pos_init = positions[prev_frames]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz

        '''All initially face Z+'''
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = root_pos_init[r_hip] - root_pos_init[l_hip]
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init


        positions = qrot_np(root_quat_init_for_all, positions)

        """ Get Foot Contacts """

        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1,fid_l,1]
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1,fid_r,1]
            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
            return feet_l, feet_r
        #
        feet_l, feet_r = foot_detect(positions, feet_thre)


        '''Get Joint Rotation Representation'''
        rot_data = rotations

        '''Get Joint Rotation Invariant Position Represention'''
        joint_positions = positions.reshape(len(positions), -1)
        joint_vels = positions[1:] - positions[:-1]
        joint_vels = joint_vels.reshape(len(joint_vels), -1)

        data = joint_positions[:-1]
        data = np.concatenate([data, joint_vels], axis=-1)
        data = np.concatenate([data, rot_data[:-1]], axis=-1)
        data = np.concatenate([data, feet_l, feet_r], axis=-1)

        return data, root_quat_init, root_pose_init_xz[None]


    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len()*1

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        # swap = data["swap"]
        text = random.choice(data["texts"]).strip()

        if self.cache:
            full_motion1, full_motion2 = self.motion_dict[motion_id]
        else:
            raise NotImplementedError("only cache now")
            # file_path1, file_path2 = self.motion_dict[motion_id]
            # motion1, motion1_swap = self.load_motion(file_path1, self.min_length, swap=swap)
            # motion2, motion2_swap = self.load_motion(file_path2, self.min_length, swap=swap)
            # if swap:
            #     full_motion1 = motion1_swap
            #     full_motion2 = motion2_swap
            # else:
            #     full_motion1 = motion1
            #     full_motion2 = motion2

        length = full_motion1.shape[0]
        if length > self.max_length:
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        if np.random.rand() > 0.5:
            motion1, motion2 = motion2, motion1
        motion1, root_quat_init1, root_pos_init1 = self.process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = self.process_motion_np(motion2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)

        gt_motion1 = motion1
        gt_motion2 = motion2


        gt_length = len(gt_motion1)
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
            gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)


        assert len(gt_motion1) == self.max_gt_length
        assert len(gt_motion2) == self.max_gt_length

        if np.random.rand() > 0.5:
            gt_motion1, gt_motion2 = gt_motion2, gt_motion1

        return name, text, gt_motion1, gt_motion2, gt_length

if __name__ == "__main__":
    data_cfg = get_config("configs/datasets_interx.yaml").interx_small
    train_dataset = InterXDataset(data_cfg)
    data0 = train_dataset.__getitem__(0)
    joblib.dump(data0, "interx_2.joblib")