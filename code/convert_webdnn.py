import os
os.environ["OPTIMIZE"] = "0"
import numpy as np
import chainer
from chainer import cuda, serializers, Variable
import chainer.functions as F
import argparse
import sys
sys.setrecursionlimit(10000)
from webdnn.frontend.chainer import ChainerConverter
from webdnn.backend.interface.generator import generate_descriptor
from chainer_dcgan import DCGANGenerator64, ResNetGenerator128, ResNetGenerator256

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='discriminator testing script')
    parser.add_argument("--chainer_model_path", '-l', default='', help='load generator model')
    # parser.add_argument("--arch", default='', help='generator architecture')
    parser.add_argument('--out', '-o', default='gan-test', help='output path')
    parser.add_argument("--latent_len", type=int, default=128, help='latent vector length')

    args = parser.parse_args()

    #if args.arch == 'dcgan64':
    #    generator_class = DCGANGenerator64
    #elif args.arch == 'resnet128':
    #    generator_class = ResNetGenerator128
    #elif args.arch == 'resnet256':
    #else:
    #    raise ValueError('Unknown -arch %s' % FLAGS.arch)
    generator_class = ResNetGenerator256
    gen = generator_class()
    serializers.load_npz(args.chainer_model_path, gen)
    print("Generator model loaded")

    x =  chainer.Variable(np.empty((1, args.latent_len), dtype=np.float32))
    with chainer.using_config('train', False):
        y = gen(x)
    print("Start Convert")
    graph = ChainerConverter().convert([x], [y])
    #exec_info = generate_descriptor("webgpu", graph)
    #exec_info.save(args.out)
    #exec_info = generate_descriptor("webgl", graph)
    #exec_info.save(args.out)
    exec_info = generate_descriptor("webgl", graph)
    exec_info.save(args.out)
    exec_info = generate_descriptor("webgl", graph, constant_encoder_name="eightbit")
    exec_info.save(args.out+"_8bit")
    #exec_info = generate_descriptor("webassembly", graph, constant_encoder_name="eightbit")
    #exec_info.save(args.out+"_8bit")
    #exec_info = generate_descriptor("webgpu", graph, constant_encoder_name="eightbit")
    #exec_info.save(args.out+"_8bit")
