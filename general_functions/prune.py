# OPEN WITH GBK CODING!!!!!!!!!! reload - GBK
from supernet_functions.lookup_table_builder import *
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Upsample, Identity, Zero
import copy
import torch
import torch.nn as nn

state_dict = torch.load('saves/supernet.pth')  # 读取保存的supernet权重
# state_dict = {'name': 'weights'}
# sampled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
sampled = ['skip', 'ir_k3_re', 'xxxxx', '...', 'none']  # 采样出来子网络每层的operator名字，暂时没有实装和主程序的联动
sampled_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # backbone
               11,                                # preprocess
               12, 13, 14, 15, 16, 17,            # fpn
               18,                                # convert
               19, 20, 21, 22, 23]                # head
# sampled_idx里保存每层operator对应下标，即比如[2,1,3,3,7,3,...]每个数字对应lookup_table.builder.CANDIDATES里的下标
# 此处没有实装里面具体的值，现在自己写的0-23代表每一层的下标/序号

residuals = [5, 8, 12, 15, 17, 19, 20, 21, 22, 23]  # 固定值，哪些层是residual block
pure_skip = []  # 哪些层是纯“短路”的skip/一条直线直接连接
cnt_backbone = 11  # backbone层数
cnt_fpn = 6
cnt_head = 5
no_preprocess = 11  # no. of preprocess 这层的序号
no_convert = 18  # no. (num) of convert

for i in residuals:
    if sampled[i] == 'skip':
        pure_skip.append(i)  # residual + skip = 纯短路skip

bn_weights = []  # 保存所有bn的scaling factor，格式[[{bn层1：参数}{bn层2：参数}{...}][{}{}{}][...]...] ← [[子网络层1][层2][][]...]
bn_weights_avg = []  # 保存经过同步后的scaling factor

# read bn weights 提取scaling factor
for i in range(len(sampled)):
    bn_paras = {}
    if i < cnt_backbone: # backbone部分
        if sampled[i] != 'skip':  # 普通block有三层 pw→dw→pwl→子网络下一层
            bn_paras['pw_w'] = state_dict['stages_to_search.{}.ops.{}.pw.bn.weight'.format(i, sampled_idx[i])]  # 提取scaling factor
            bn_paras['dw_w'] = state_dict['stages_to_search.{}.ops.{}.dw.bn.weight'.format(i, sampled_idx[i])]
            bn_paras['pwl_w'] = state_dict['stages_to_search.{}.ops.{}.pwl.bn.weight'.format(i, sampled_idx[i])]
        else:
            if i not in pure_skip:
                bn_paras['pwl_w'] = state_dict['stages_to_search.{}.ops.10.conv.bn.weight'.format(i)] # "regarded" as pwl_w so that same name
                # 对于非纯短路skip层，字典中key的名字也使用‘pwl_w’，因为对于普通的block来说'pwl_w'层是输出层，对于skip层来说只有一层即输出层，
                # 使它们的key的名字一样，便于代码编辑

    elif i == no_preprocess:
        bn_paras['pwl_w'] = state_dict['preprocess.bn.weight'] # 这里也是同理 总之'pwl_w'代表了子网络每层的输出channel数

    elif i > cnt_backbone and i < cnt_backbone + cnt_fpn + 1:
        if sampled[i] != 'skip' and sampled[i] != 'none': # 对于none和纯短路skip：bn_weights[i] = []空列表
            bn_paras['pw_w'] = state_dict['neck.{}.ops.{}.pw.bn.weight'.format(i, sampled_idx[i])]
            bn_paras['dw_w'] = state_dict['neck.{}.ops.{}.dw.bn.weight'.format(i, sampled_idx[i])]
            bn_paras['pwl_w'] = state_dict['neck.{}.ops.{}.pwl.bn.weight'.format(i, sampled_idx[i])]
        elif sampled[i] == 'skip':
            if i not in pure_skip:
                bn_paras['pwl_w'] = state_dict['neck.{}.ops.7.conv.bn.weight'.format(i)]

    elif i == no_convert:
        bn_paras['pwl_w'] = state_dict['convert.bn.weight']

    else:  # head
        if sampled[i] != 'skip':
            bn_paras['pw_w'] = state_dict['head.{}.ops.{}.pw.bn.weight'.format(i, sampled_idx[i])]
            bn_paras['dw_w'] = state_dict['head.{}.ops.{}.dw.bn.weight'.format(i, sampled_idx[i])]
            bn_paras['pwl_w'] = state_dict['head.{}.ops.{}.pwl.bn.weight'.format(i, sampled_idx[i])]

    bn_weights.append(bn_paras)

bn_weights_avg = copy.deepcopy(bn_weights)

# prune
total = 0  # 整个子网络共多少channel
for layer in bn_weights:
    for m in layer.values():
        total += m.shape[0] # num of channels

# synchrone  同步分组的代码
group_tot = 0  # 一共分了几组
group_ids = [0] * len(sampled) # 每层有自己的组号

for i in range(cnt_backbone - 1): # backbone
    if sampled[i] == skip:
        if i in pure_skip:
            continue
    k = i + 1
    while 1:
        if sampled[k] == skip:
            if k in pure_skip:
                continue
        if k in res:
            if group_ids[i] == 0:
                group_tot += 1
                group_ids[i] = group_tot
                group_ids[k] = group_tot
            else:
                group_ids[k] = group_ids[i]
        break

def synfpn(i, j):
    if sampled[i] != 'none' and i not in pure_skip and sampled[j] != 'none' and j not in pure_skip:
        if group_ids[i] == 0:
            group_tot += 1
            group_ids[i] = group_tot
            group_ids[j] = group_tot
        else:
            group_ids[j] = group_ids[i]

f0 = 12  # fpn0这层的序号，fpn1的序号就是f0 + 1
b0 = 6  # backbone下采样率8的出来的层的序号（backbone[6]）

synfpn(b0, f0)
synfpn(f0, f0 + 2)
if sampled[f0] == 'skip':
    synfpn(b0, f0 + 2)
synfpn(f0 - 1, f0 + 3)  # f0 - 1是preprocess层序号，从下采样率16出来的backbone层要经过preprocess
synfpn(f0 + 3, f0 + 1)
synfpn(f0 + 3, f0 + 5)
synfpn(f0 + 1, f0 + 5)
synfpn(f0 + 5, f0 + 4)
if sampled[f0 + 3] == 'skip':
    synfpn(f0 - 1, f0 + 1)
    synfpn(f0 - 1, f0 + 5)
    synfpn(f0 + 5, f0 + 4)
if sampled[f0 + 5] == 'skip':
    synfpn(f0 - 1, f0 + 4)
    synfpn(f0 + 1, f0 + 4)

group_tot += 1
for i in range(no_convert, len(sampled)): # convert层和head所有层都要同步
    if i not in pure_skip:
        group_ids[i] = group_tot

aver = []
# group finished, syn:
for i in range(group_tot):  # aver里存放每一组的组员 [[第一组的所有序号][二][三][]...]
    g = []
    for j, k in enumerate(group_ids):
        if k == i + 1:
            g.append(j)
    aver.append(g)

# averaging
for g in aver:
    gsum = torch.zeros(bn_weights[g[0]]['pwl_w'].shape) # 同步的层channel数相同，scaling factor的维度一样，都是channel数
    for i in g:
        gsum += bn_paras[i]['pwl_w']
    gsum /= len(g)  # 求均值
    for i in g:
        bn_weights_avg[i]['pwl_w'] = gsum  # 覆盖到bn_weights_avg对应这些层中

# finish syn
# prune 开始剪枝 这段出自代码pytorch-slimming-master
bn = torch.zeros(total)
index = 0
for layer in bn_weights_avg:
    for m in layer.values():
        size = m.shape[0]
        bn[index:(index+size)] = m.abs().clone()
        index += size

percent = 0.5

y, i = torch.sort(bn)
thre_index = int(total * percent)  # threshold index
thre = y[thre_index]

pruned = 0
result_channels = []  # dicts: {convlayer: preserved channel idx}  保存每层剪枝后存下来的channel的下标
for layer in bn_weights_avg:
    res = {}
    for k, m in layer.items():
        mask = m.abs().gt(thre) #.cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        preserved = torch.nonzero(mask).squeeze() # 所有非0值对应的下标，就是没剪掉的channel下标
        res[k] = preserved
    result_channels.append(res)

pruned_ratio = pruned/total  # 没用（？）

# 给每一层找出输入通道数
inputs = [0] * len(sampled)
inputs[0] = 16  # 网络第一层的输出通道数
for i in range(1, cnt_backbone + 1):
    if i - 1 in pure_skip: # （每一层都找一个input数/输入通道数，只不过不一定使用，便于代码编辑）
        inputs[i] = inputs[i - 1]  # no skip-skip exists
    else:
        inputs[i] = len(result_channels[i - 1]['pwl_w'])  # 上一层的输出channel数
# fpn部分，穷举法……
inputs[f0] = len(result_channels[b0]['pwl_w'])
inputs[f0 + 1] = len(result_channels[b0]['pwl_w'])
inputs[f0 + 2] = len(result_channels[f0 - 1]['pwl_w'])
inputs[f0 + 3] = len(result_channels[f0 - 1]['pwl_w'])
if sample[f0] == 'none' or f0 in pure_skip:
    if sample[f0 + 2] == 'none':
        inputs[f0 + 4] = inputs[f0]
    else:
        inputs[f0 + 4] = len(result_channels[f0 + 2]['pwl_w'])
else:
    inputs[f0 + 4] = len(result_channels[f0]['pwl_w'])
if sample[f0 + 3] = 'none' or f0 + 3 in pure_skip:
    if sample[f0 + 1] == 'none':
        inputs[f0 + 5] = inputs[f0 + 3]
    else:
        inputs[f0 + 5] = len(result_channels[f0 + 1]['pwl_w'])
else:
    inputs[f0 + 5] = len(result_channels[f0 + 3]['pwl_w'])
if sample[f0 + 5] = 'none' or f0 + 5 in pure_skip:
    if sample[f0 + 4] == 'none':
        inputs[no_convert] = inputs[f0 + 5]
    else:
        inputs[no_convert] = len(result_channels[f0 + 4]['pwl_w'])
else:
    inputs[no_convert] = len(result_channels[f0 + 5]['pwl_w'])

strides = [] # 每层的stride合起来
strides = strides + SEARCH_SPACE['strides']
strides.append(1)
strides = strides + NECK_SEARCH_SPACE['strides']
strides.append(1)
strides = strides + HEAD_SEARCH_SPACE['strides']

pruned_layers_parameters = [] # 存放子网络每一层的网络参数（元组） 格式[(层1的input，output，中间层1，中间层2，stride)()()...]
for i in range(len(sampled)):
    if sampled[i] != 'skip' and sample[i] != 'none':
        pruned_layers_parameters.append((inputs[i],
                                         len(result_channels[i]['pwl_w']),
                                         len(result_channels[i]['pw_w']),
                                         len(result_channels[i]['dw_w']),
                                         strides[i]))
    elif sampled[i] == 'skip':
        if i not in pure_skip:
            pruned_layers_parameters.append((inputs[i],
                                             len(result_channels[i]['pwl_w']),
                                             strides[i]))    # num of chns: fpn: min of two plussers
        else: # 是短路skip
            pruned_layers_parameters.append((inputs[i],
                                             inputs[i],
                                             strides[i]))
    else: # none
        pruned_layers_parameters.append((strides[i]))


class Pruned_IRFBlock(nn.Module): # 给子网络定义的新的Block，区别是多了两个参数mid_depth，两中间层通道数
    def __init__(
            self,
            input_depth,
            output_depth,
            stride,
            mid_depth_0,
            mid_depth_1,
            bn_type="bn",
            kernel=3,
            nl="relu",
            dil=1,
            width_divisor=1,
            shuffle_type=None,
            pw_group=1,
            se=False,
            dw_skip_bn=False,
            dw_skip_relu=False,
    ):
        super(Pruned_IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        # mid_depth = int(input_depth * expansion)
        # mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth_0,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=nl,
            bn_type=bn_type,
            group=pw_group,
        )

        # dw
        self.dw = ConvBNRelu(
            mid_depth_0,
            mid_depth_1,
            kernel=kernel,
            stride=stride,
            pad=(kernel // 2) * dil,
            dil=dil,
            group=mid_depth,
            no_bias=1,
            use_relu=nl if not dw_skip_relu else None,
            bn_type=bn_type if not dw_skip_bn else None,
        )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth_1,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=False,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)

        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y

# 为子网络新定义的PRIMITIVES
PRIMITIVES = {
    "none": lambda stride, **kwargs: Zero(
        stride
    ),
    "skip": lambda C_in, C_out, stride, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ir_k3_re": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=3, nl="relu", **kwargs
    ),
    "ir_k3_hs": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=3, nl="hswish", **kwargs
    ),
    "ir_k3_r2_re": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=3, nl="relu", dil=2, **kwargs
    ),
    "ir_k3_r2_hs": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=3, nl="hswish", dil=2, **kwargs
    ),
    "ir_k3_r3_re": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=3, nl="relu", dil=3, **kwargs
    ),
    "ir_k5_re": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=5, nl="relu", **kwargs
    ),
    "ir_k5_hs": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=5, nl="hswish", **kwargs
    ),
    "ir_k5_r2_re": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=5, nl="relu", dil=2, **kwargs
    ),
    "ir_k5_r2_hs": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=5, nl="hswish", dil=2, **kwargs
    ),
    "ir_k5_r3_re": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=5, nl="relu", dil=3, **kwargs
    ),
    "ir_k7_re": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=7, nl="relu", **kwargs
    ),
    "ir_k7_hs": lambda C_in, C_out, mid_depth_0, mid_depth_1, stride, **kwargs: Pruned_IRFBlock(
        C_in, C_out, stride, mid_depth_0, mid_depth_1, kernel=7, nl="hswish", **kwargs
    ),
}


class Sampled_pruned_net(nn.Module):  # 新的子网络class
    def __init__(self, paras):
        super(Sampled_pruned_net, self).__init__()

        # self.first identical to 'add_first' in the fbnet_building_blocks/fbnet_builder.py
        self.first = ConvBNRelu(input_depth=7, output_depth=16, kernel=3, stride=2,
                                pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")
        self.paras = paras
        # didn't add names to sequential
        self.backbone = nn.ModuleList()
        for i in range(cnt_backbone):
            self.backbone.append(PRIMITIVES[sampled[i]](*self.paras[i]))

        self.neck = nn.ModuleList()
        for i in range(f0, f0 + cnt_fpn):
            self.neck.append(PRIMITIVES[sampled[i]](*self.paras[i]))

        self.head = nn.ModuleList()
        for i in range(no_convert + 1, len(sampled)):
            self.head.append(PRIMITIVES[sampled[i]](*self.paras[i]))

        self.preprocess = ConvBNRelu(*self.paras[no_preprocess])

        self.upsample = Upsample(scale_factor=2, mode="nearest")

        self.convert = ConvBNRelu(*self.paras[no_convert])

    def forward(self, x):
        y = self.first(x)

        for idx, layer in enumerate(self.backbone):
            y = layer(y)
            if idx == 6
                y_8 = y

        y = self.preprocess(y)  # downratio16

        y_fpn0 = self.neck[0](y_8)
        y_fpn1 = self.neck[1](y_8)
        y_fpn2 = self.neck[2](y)
        y_fpn3 = self.neck[3](y)
        y_fpnhid0 = y_fpn0 + self.upsample(y_fpn2)  # upsample此处省略操作，因为如果按之前说的TransposeConv2d，还需要对应的量化处理，后续可以改成Transpose2d
        # 现在用的方法：这一层的stride设置成-1，传入class时识别“如果stride<0，那么取stride=1，同时因为传入stride不等于1，不触发residual条件
        y_fpnhid1 = y_fpn1 + y_fpn3
        y_fpn4 = self.neck[4](y_fpnhid0)
        y_fpn5 = self.neck[5](y_fpnhid1)
        y = y_fpn4 + y_fpn5

        y = self.convert(y)

        for head_layer in self.head:
            y = head_layer(y)
        # y = self.last_stages(y)
        out = []
        z = {}
        for out_head, out_chns in CONFIG_SUPERNET['output_heads'].items():
            out_layer = nn.Conv2d(len(result_channels[no_convert]['pwl_w']), out_chns,
                                  kernel_size=1, stride=1, padding=0, bias=True)
            z[head] = out_layer(y)
        out.append(z)

# 创建子网络对象函数，在主程序中调用，此函数的调用还没有实装
def create_sampled_net():

    model = Sampled_pruned_net(pruned_layers_parameters) # 新建网络对象
    new_state_dict = model.state_dict() # 拉出它的权重dict

    # 给新对象的dict对应位置赋值权重，开始这一块冗余代码了，但是也不舍得删 - -
    k_else_1 = ['.pw.bn.weight', '.pw.bn.bias', '.pw.bn.running_mean', '.pw.bn.running_var',
                '.dw.bn.weight', '.dw.bn.bias', '.dw.bn.running_mean', '.dw.bn.running_var',
                '.pwl.bn.weight', '.pwl.bn.bias', '.pwl.bn.running_mean', '.pwl.bn.running_var',]
    k_else_4 = ['.pw.conv.weight', '.pw.conv.quan_w_fn.s',
                '.dw.conv.weight', '.dw.conv.quan_w_fn.s'
                '.pwl.conv.weight', '.pwl.conv.quan_w_fn.s']
    # k_else_0 = [] activation???
    k_skip_1 = ['.conv.bn.weight', '.conv.bn.bias', '.conv.bn.running_mean', '.conv.bn.running_var']
    k_skip_4 = ['.conv.conv.weight', '.conv.conv.quan_w_fn.s']

    for i in range(cnt_backbone):
        k0 = 'backbone.{}'.format(i)
        k1 = 'stages_to_search.{}.ops.{}.'.format(i, sampled_idx[i])
        if sampled[i] != 'skip':
            for key in k_else_4:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i]['pwl_w'], :, :, :]
            for key in k_else_1:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i]['pwl_w']]
        else:
            for key in k_skip_4:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i]['pwl_w'], :, :, :]
            for key in k_skip_1:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i]['pwl_w']]

    for i in range(cnt_fpn):
        k0 = 'neck.{}'.format(i)
        k1 = 'neck.{}.ops.{}.'.format(i, sampled_idx[i + f0])
        if sampled[i + f0] != 'skip' und sampled[i + f0] != 'none':
            for key in k_else_4:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i + f0]['pwl_w'], :, :, :]
            for key in k_else_1:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i + f0]['pwl_w']]
        elif sampled[i + f0] == 'skip':
            for key in k_skip_4:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i + f0]['pwl_w'], :, :, :]
            for key in k_skip_1:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i + f0]['pwl_w']]

    for i in range(cnt_head):
        k0 = 'head.{}'.format(i)
        k1 = 'head.{}.ops.{}.'.format(i, sampled_idx[i + no_convert + 1])
        if sampled[i + no_convert + 1] != 'skip':
            for key in k_else_4:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['pwl_w'], :, :, :]
            for key in k_else_1:
                k = k0 + key
                p = k1 + key
                new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['pwl_w']]
        # else:  # head的skip都是纯短路
        #     for key in k_skip_4:
        #         k = k0 + key
        #         p = k1 + key
        #         new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['pwl_w'], :, :, :]
        #     for key in k_skip_1:
        #         k = k0 + key
        #         p = k1 + key
        #         new_state_dict[k] = state_dict[p][result_channels[i + no_convert + 1]['pwl_w']]

    # 像这样赋值比较简单
    for k, v in state_dict.items()
        if 'preprocess.conv' in k:
            new_state_dict[k] = state_dict[k][result_channels[no_preprocess], :, :, :]
        if 'preprocess.bn' in k:
            if 'num' not in k: # state_dict中有一个参数叫num_batches_tracked，一个空的参数，不知道是啥，跳过
                new_state_dict[k] = state_dict[k][result_channels[no_preprocess]]
        if 'convert.conv' in k:
            new_state_dict[k] = state_dict[k][result_channels[no_convert], :, :, :]
        if 'convert.bn' in k:
            if 'num' not in k:
                new_state_dict[k] = state_dict[k][result_channels[no_convert]]
        if 'first' in k:
            new_state_dict[k] = state_dict[k]

    # dict赋值完，扔进新的网络对象
    model.load_state_dict(new_state_dict)

    return model