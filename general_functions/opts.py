class optss(object):
    def __init__(self):
        self.task = 'tracking'
        self.dataset = 'mot'
        self.dataset_version = '17trainval'
        self.data_dir = 'E:/Workspace/MOT17/'
        self.tracking = True
        self.not_max_crop = True
        self.flip = 0  # 0.2
        self.input_w = 1920
        self.input_h = 1080
        self.output_w = 480
        self.output_h = 270
        self.same_aug_pre = True
        self.max_frame_dist = 1  # 3
        self.pre_hm = True
        self.num_classes = 2
        self.debug = 0  # T/F
        self.down_ratio = 4
        self.hm_disturb = 0  # 0.3
        self.lost_disturb = 0  # 0.3
        self.fp_disturb = 0  # 0.3
        self.not_rand_crop = True
        self.scale = 0   # 0.4
        self.shift = 0   # 0.1
        self.aug_rot = 0  # prob
        self.rotate = 0  # degree
        self.velocity = False
        self.no_color_aug = True
        self.dense_reg = 1
        self.simple_radius = 0  # ???

        self.test_dataset = 'mot'
        self.test = False
        self.head_conv = -1
        self.num_head_conv = 1
        self.batch_size = 4

        self.pre_thresh = -1  # ?
        self.track_thresh = 0.3  # ?
        self.new_thresh = 0.3  # ?
        self.ltrb = False
        self.ltrb_amodal = False
        self.ltrb_amodal_weight = 0.1
        self.public_det = True
        self.no_pre_img = False
        self.zero_tracking = True
        self.hungarian = True
        self.max_age = -1

        self.tracking_weight = 1
        self.reg_loss = 'l1'
        self.hm_weight = 1
        self.off_weight = 1
        self.wh_weight = 1
        self.hp_weight = 0
        self.hm_hp_weight = 0
        self.amodel_offset_weight = 0
        self.dep_weight = 0
        self.dim_weight = 0
        self.rot_weight = 0
        self.nuscenes_att = False
        self.nuscenes_att_weight = 0
        self.velocity_weight = 0
        self.ltrb_weight = 0
        self.ltrb_amodal_weight = 0
        self.custom_dataset_img_path = None
        self.custom_dataset_ann_path = None  # ''


    def update_dataset_info_and_set_heads(self, dataset):
        self.num_classes = dataset.num_categories \
            if self.num_classes < 0 else self.num_classes
        input_h, input_w = dataset.default_resolution
        self.output_h = self.input_h // self.down_ratio  # !!!!
        self.output_w = self.input_w // self.down_ratio  # !!!!
        self.input_res = max(self.input_h, self.input_w)
        self.output_res = max(self.output_h, self.output_w)

        self.heads = {'hm': self.num_classes, 'reg': 2, 'wh': 2}  # heatmap一类一张，reg是local location？，wh宽高  输出通道

        if 'tracking' in self.task:
            self.heads.update({'tracking': 2})

        if 'ddd' in self.task:  # 3D obj det/trk
            self.heads.update({'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2})

        if 'multi_pose' in self.task:
            self.heads.update({
                'hps': dataset.num_joints * 2, 'hm_hp': dataset.num_joints,
                'hp_offset': 2})

        if self.ltrb:
            self.heads.update({'ltrb': 4})
        if self.ltrb_amodal:
            self.heads.update({'ltrb_amodal': 4})
        if self.nuscenes_att:
            self.heads.update({'nuscenes_att': 8})
        if self.velocity:
            self.heads.update({'velocity': 3})

        weight_dict = {'hm': self.hm_weight, 'wh': self.wh_weight,
                       'reg': self.off_weight, 'hps': self.hp_weight,
                       'hm_hp': self.hm_hp_weight, 'hp_offset': self.off_weight,
                       'dep': self.dep_weight, 'rot': self.rot_weight,
                       'dim': self.dim_weight,
                       'amodel_offset': self.amodel_offset_weight,
                       'ltrb': self.ltrb_weight,
                       'tracking': self.tracking_weight,
                       'ltrb_amodal': self.ltrb_amodal_weight,
                       'nuscenes_att': self.nuscenes_att_weight,
                       'velocity': self.velocity_weight}
        self.weights = {head: weight_dict[head] for head in self.heads}  # the weight of terms of loss  final loss = weighted sum
        for head in self.weights:
            if self.weights[head] == 0:
                del self.heads[head]
        self.head_conv = {head: [self.head_conv \
                                for i in range(self.num_head_conv if head != 'reg' else 1)] for head in self.heads}

        print('input h w:', self.input_h, self.input_w)
        print('heads', self.heads)
        print('weights', self.weights)
        print('head conv', self.head_conv)