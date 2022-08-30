import timeit
import torch
from collections import OrderedDict
import gc
from fbnet_building_blocks.fbnet_builder import PRIMITIVES
from general_functions.utils import add_text_to_file, clear_files_in_the_list
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

# the settings from the page 4 of https://arxiv.org/pdf/1812.03443.pdf
#### table 2
CANDIDATE_BLOCKS = ["ir_k3_r2_re", "ir_k3_r2_hs", "ir_k3_re",
                    "ir_k3_hs", "ir_k5_r2_re", "ir_k5_r2_hs",
                    "ir_k5_re", "ir_k5_hs", "ir_k7_re", "ir_k7_hs", "skip"]

NECK_CANDIBLOCKS = ["ir_k3_re", "ir_k3_r2_re", "ir_k3_r3_re",
                    "ir_k5_re", "ir_k5_r2_re", "ir_k5_r3_re",
                    "none", "skip"]

HEAD_CANDIBLOCKS = ["ir_k3_re", "ir_k3_r2_re", "ir_k5_re", "ir_k5_r2_re", "skip"]

# SEARCH_SPACE = OrderedDict([
#     #### table 1. input shapes of 22 searched layers (considering with strides)
#     # Note: the second and third dimentions are recommended (will not be used in training) and written just for debagging
#     ("input_shape", [(16, 208, 208),
#                      (32, 104, 104),  (64, 52, 52),   (128, 52, 52), (128, 26, 26), (256, 26, 26),
#                      (256, 26, 26),  (256, 26, 26),   (512, 26, 26),   (512, 13, 13),
#                      (1024, 13, 13)]),
#     # table 1. filter numbers over the 22 layers
#     ("channel_size", [32,  64,
#                       128, 128, 256, 256,
#                       256, 512, 512, 1024,
#                       1024]),
#     # table 1. strides over the 22 layers
#     ("strides", [2, 2, 1, 2,
#                  1, 1, 1, 1,
#                  2, 1, 1])
# ])

SEARCH_SPACE = OrderedDict([
    #### table 1. input shapes of 22 searched layers (considering with strides)
    # Note: the second and third dimentions are recommended (will not be used in training) and written just for debagging
    ("input_shape", [(16, 208, 208),
                     (32, 104, 104), (64, 104, 104), (64, 52, 52), (128, 52, 52), (256, 52, 52),
                     (256, 52, 52), (512, 52, 52), (512, 26, 26), (512, 26, 26),
                     (1024, 26, 26)]),
    # table 1. filter numbers over the 22 layers
    ("channel_size", [32, 64,
                      64, 128, 256, 256,
                      512, 512, 512, 1024,
                      1024]),
    # table 1. strides over the 22 layers
    ("strides", [2, 1, 2, 1,
                 1, 1, 1, 2,
                 1, 1, 1])
])

NECK_SEARCH_SPACE = OrderedDict([
    #### table 1. input shapes of 22 searched layers (considering with strides)
    # Note: the second and third dimentions are recommended (will not be used in training) and written just for debagging
    ("input_shape", [(512, 52, 52), (512, 52, 52), (512, 26, 26), (512, 26, 26), (512, 52, 52), (512, 26, 26)]),
    # table 1. filter numbers over the 22 layers
    ("channel_size", [512, 512, 512, 512, 512, 512]),
    # table 1. strides over the 22 layers
    ("strides", [1, 2, 1, 1,
                 2, 1])
])

HEAD_SEARCH_SPACE = OrderedDict([
    ("input_shape",
     [(256, 26, 26), (256, 26, 26),
      (256, 26, 26), (256, 26, 26),
      (256, 26, 26)]),
    ("channel_size",
     [256, 256,
      256, 256,
      256]),
    ("strides",
     [1, 1,
      1, 1,
      1])

])


# **** to recalculate latency use command:
# l_table = LookUpTable(calulate_latency=True, path_to_file='lookup_table.txt', cnt_of_runs=50)
# results will be written to './supernet_functions/lookup_table.txt''
# **** to read latency from the another file use command:
# l_table = LookUpTable(calulate_latency=False, path_to_file='lookup_table.txt')
class LookUpTable:
    def __init__(self, candidate_blocks=CANDIDATE_BLOCKS, search_space=SEARCH_SPACE, candidate_neck=NECK_CANDIBLOCKS,
                 search_space_neck=NECK_SEARCH_SPACE, candidate_head=HEAD_CANDIBLOCKS,
                 search_space_head=HEAD_SEARCH_SPACE, calulate_latency=False):
        self.cnt_layers = len(search_space["input_shape"])
        self.cnt_layers_neck = len(search_space_neck["input_shape"])
        self.cnt_layers_head = len(search_space_head["input_shape"])
        # constructors for each operation
        self.lookup_table_operations = {op_name: PRIMITIVES[op_name] for op_name in candidate_blocks}
        self.lookup_neck_operations = {op_name: PRIMITIVES[op_name] for op_name in candidate_neck}
        self.lookup_head_operations = {op_name: PRIMITIVES[op_name] for op_name in candidate_head}
        # arguments for the ops constructors. one set of arguments for all 9 constructors at each layer
        # input_shapes just for convinience
        self.layers_parameters, self.layers_input_shapes = self._generate_layers_parameters(search_space)
        self.layers_parameters_neck, self.layers_input_shapes_neck = self._generate_layers_parameters(search_space_neck)
        self.layers_parameters_head, self.layers_input_shapes_head = self._generate_layers_parameters(search_space_head)

        # lookup_table
        self.lookup_table_latency = None
        if calulate_latency:
            self._create_from_operations(cnt_of_runs=CONFIG_SUPERNET['lookup_table']['number_of_runs'],
                                         write_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'],
                                         write_neck=CONFIG_SUPERNET['lookup_table']['path_to_lookup_neck'],
                                         write_head=CONFIG_SUPERNET['lookup_table']['path_to_lookup_head'])
        else:
            self._create_from_file(path_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'])

    def _generate_layers_parameters(self, search_space):
        # layers_parameters are : C_in, C_out, expansion, stride
        layers_parameters = [(search_space["input_shape"][layer_id][0],
                              search_space["channel_size"][layer_id],
                              # expansion (set to -999) embedded into operation and will not be considered
                              # (look fbnet_building_blocks/fbnet_builder.py - this is facebookresearch code
                              # and I don't want to modify it)
                              -999,
                              search_space["strides"][layer_id]
                              ) for layer_id in range(self.cnt_layers)]

        # layers_input_shapes are (C_in, input_w, input_h)
        layers_input_shapes = search_space["input_shape"]

        return layers_parameters, layers_input_shapes

    # CNT_OP_RUNS us number of times to check latency (we will take average)
    def _create_from_operations(self, cnt_of_runs, write_to_file=None, write_neck=None, write_head=None):
        self.lookup_table_latency = self._calculate_latency(self.lookup_table_operations,
                                                            self.cnt_layers,
                                                            self.layers_parameters,
                                                            self.layers_input_shapes,
                                                            cnt_of_runs)

        self.lookup_neck_latency = self._calculate_latency(self.lookup_neck_operations,
                                                           self.cnt_layers_neck,
                                                           self.layers_parameters_neck,
                                                           self.layers_input_shapes_neck,
                                                           cnt_of_runs)

        self.lookup_head_latency = self._calculate_latency(self.lookup_table_operations,
                                                           self.cnt_layers_head,
                                                           self.layers_parameters_head,
                                                           self.layers_input_shapes_head,
                                                           cnt_of_runs)

        if write_to_file is not None:
            self._write_lookup_table_to_file(write_to_file)

        if write_neck is not None:
            self._write_lookup_table_to_file(write_neck)

        if write_head is not None:
            self._write_lookup_table_to_file(write_head)

    def _calculate_latency(self, operations, cnt_layers, layers_parameters, layers_input_shapes, cnt_of_runs):
        LATENCY_BATCH_SIZE = 1
        latency_table_layer_by_ops = [{} for i in range(cnt_layers)]

        for layer_id in range(cnt_layers):
            for op_name in operations:
                op = operations[op_name](*layers_parameters[layer_id])
                input_sample = torch.randn((LATENCY_BATCH_SIZE, *layers_input_shapes[layer_id]))
                globals()['op'], globals()['input_sample'] = op, input_sample
                total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()", \
                                           globals=globals(), number=cnt_of_runs)
                # measured in micro-second
                latency_table_layer_by_ops[layer_id][op_name] = total_time / cnt_of_runs / LATENCY_BATCH_SIZE * 1e6

        return latency_table_layer_by_ops

    def _write_lookup_table_to_file(self, path_to_file):
        clear_files_in_the_list([path_to_file])
        ops = [op_name for op_name in self.lookup_table_operations]
        text = [op_name + " " for op_name in ops[:-1]]
        text.append(ops[-1] + "\n")

        for layer_id in range(self.cnt_layers):
            for op_name in ops:
                text.append(str(self.lookup_table_latency[layer_id][op_name]))
                text.append(" ")
            text[-1] = "\n"
        text = text[:-1]

        text = ''.join(text)
        add_text_to_file(text, path_to_file)

    def _create_from_file(self, path_to_file):
        self.lookup_table_latency = self._read_lookup_table_from_file(path_to_file)
        self.lookup_neck_latency = self._read_lookup_table_from_file(path_to_neck)
        self.lookup_head_latency = self._read_lookup_table_from_file(path_to_head)

    def _read_lookup_table_from_file(self, path_to_file):
        latences = [line.strip('\n') for line in open(path_to_file)]
        ops_names = latences[0].split(" ")
        latences = [list(map(float, layer.split(" "))) for layer in latences[1:]]

        lookup_table_latency = [{op_name: latences[i][op_id]
                                 for op_id, op_name in enumerate(ops_names)
                                 } for i in range(self.cnt_layers)]
        return lookup_table_latency
