import random
import struct
import sys

import numpy as np
import torch


def main(args):
    np.random.seed(2147483647)
    np_mersenne_state = [int(s) for s in list(np.random.get_state()[1])]
    np_state = np_mersenne_state + [624]
    np_state = (3, tuple(np_state), None)
    random.setstate(np_state)

    np_rand = ["{:0.4f}".format(np.random.rand()) for _ in range(5)]
    print(f"Numpy:\t         {', '.join(np_rand)}")

    py_rand = ["{:0.4f}".format(random.random()) for _ in range(5)]
    print(f"Python:\t         {', '.join(py_rand)}")

    generator = torch.Generator().manual_seed(2147483647)
    generator_state = generator.get_state()
    byte_array = generator_state.numpy().tobytes()

    # Format string for CPUGeneratorImplState from PyTorch. Note that the last field is actually a bool; I think the
    # size of the struct was padded out for alignment
    format_string = "QiiQ624QdddifQ"
    fields = struct.unpack(format_string, byte_array)
    # initial_seed = fields[0]
    # left = fields[1]
    # seeded = fields[2]
    # next = fields[3]
    # mersenne_state = fields[4:4 + 624]
    # normal_x = fields[624 + 4]
    # normal_y = fields[624 + 5]
    # normal_rho = fields[624 + 6]
    # normal_is_valid = fields[624 + 7]
    # next_float_normal_sample = fields[624 + 8]
    # is_next_float_normal_sample_valid = fields[624 + 9]
    modified_fields = fields[0:4] + tuple(np_mersenne_state) + fields[628:]
    modified_byte_array = struct.pack("QiiQ624QdddifQ", *modified_fields)
    modified_generator_state = torch.ByteTensor(list(modified_byte_array))
    generator.set_state(modified_generator_state)
    torch_rand = torch.rand(5, generator=generator)
    print(f"PyTorch: {torch_rand}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))