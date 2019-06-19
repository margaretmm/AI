from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import tensorflow as tf

import tensorflowTrain.sytleTransfer.vgg_model as vgg_model
import  tensorflowTrain.sytleTransfer.utils as utils

# parameters to manage experiments
STYLE = '1'
CONTENT = '1'
STYLE_IMAGE = 'style/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 333
NOISE_RATIO = 0.6  # percentage of weight of the noise for intermixing with the content image

CONTENT_WEIGHT = 0.01
STYLE_WEIGHT = 1

# 在VGG的Layers中选择部分作为风格特征提取层
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
#风格特征的权重
STYLE_LAYERS_W = [0.5, 1.0, 1.5, 3.0, 4.0]  # give more weights to deeper layers.

# 在VGG的Layers中选择部分作为内容特征提取层
CONTENT_LAYER = 'conv4_2'

ITERS = 300
LR = 2.0

#为了对输入图片预处理，归一化处理，定义的均值像素,并转为一个4维数组
MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783


# 计算所有的损失，包括风格损失和内容损失
def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            # 把content image赋值给变量input_image
            sess.run(input_image.assign(content_image))
            # 计算变量model[CONTENT_LAYER]的值
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])
        style_loss = _create_style_loss(A, model)

        ## TO DO: create total loss.
        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

    return content_loss, style_loss, total_loss


# 定义内容损失函数（生成图片和 内容图片直接的损失）
# 内容损失：内容图片在指定层上提取出的特征矩阵，与噪声图片在对应层上的特征矩阵的差值的L2范数。即求两两之间的像素差值的平方
def _create_content_loss(p, f):
    """
    Inputs:
        p为噪音图像特征的特征矩阵 ，f为内容图片特征的特征矩阵
    Output:
        内容损失值
    """
    return tf.reduce_sum((f - p) ** 2) / (4.0 * p.size)


# 定义tensor F的 gram矩阵
def _gram_matrix(F, N, M):
    F = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(F), F)


# 每一层的风格损失函数
def _single_style_loss(a, g):
    """
    Inputs:
        a 是风格图片的矩阵
        g  是生成图片的矩阵
    Output:
        该layer的style loss
    """
    N = a.shape[3]  # 特征矩阵的信道数
    M = a.shape[1] * a.shape[2]  # 特征矩阵的 长*宽
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)
    return tf.reduce_sum((G - A) ** 2 / ((2 * N * M) ** 2))


# 计算所有层的风格损失函数
def _create_style_loss(A, model):
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]

    ## 每层的W(style)*Loss(sytle)之和
    return sum([STYLE_LAYERS_W[i] * E[i] for i in range(n_layers)])


def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    with tf.name_scope('summaries'):
        tf.summary.scalar('content loss', model['content_loss'])
        tf.summary.scalar('style loss', model['style_loss'])
        tf.summary.scalar('total loss', model['total_loss'])
        tf.summary.histogram('histogram content loss', model['content_loss'])
        tf.summary.histogram('histogram style loss', model['style_loss'])
        tf.summary.histogram('histogram total loss', model['total_loss'])
        return tf.summary.merge_all()


def train(model, generated_image, initial_image):
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()

        ## 1. 初始化所有的变量
        ## 2. 定义记录graph运行状态的writer
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('graphs', sess.graph)

        # 变量generated_image被赋值为initial_image
        sess.run(generated_image.assign(initial_image))

        # 如果特征可以从文件夹中加载，则加载相关特征到sess中
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 计算当前变量global_step的值
        initial_step = model['global_step'].eval()

        # 开始多次迭代训练模型
        start_time = time.time()
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20

            # 开始计算变量model['optimizer']的值，也就是梯度下降优化过程
            sess.run(model['optimizer'])

            # skip_step用于控制打印和保存训练结果数据的粒度
            # 在特定的训练迭代时，计算generated_image,model['total_loss'],  model['summary_op']这几个变量的值，generated_image图片保存到文件目录下，summary 并写入日志文件中，total_loss变量主要用于打印
            if (index + 1) % skip_step == 0:
                ##同时计算3个变量：gen_image,total_loss和summary
                gen_image, total_loss, summary = sess.run([generated_image, model['total_loss'], model['summary_op']])

                # 图片矩阵增加均值像素
                gen_image = gen_image + MEAN_PIXELS

                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                # 保存变量gen_image的计算结果
                filename = 'outputs/%d.png' % (index)
                utils.save_image(filename, gen_image)

                # 定期保存sess
                if (index + 1) % 20 == 0:
                    saver.save(sess, 'checkpoints/style_transfer', index)


def main():
    with tf.variable_scope('input') as scope:
        # use variable instead of placeholder because we're training the intial image to make it
        # look like both the content image and the style image
        input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)

    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    utils.make_dir('checkpoints')
    utils.make_dir('outputs')
    model = vgg_model._init(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model,
                                                                                     input_image, content_image,
                                                                                     style_image)
    ## TO DO: create optimizer
    model['optimizer'] = tf.train.AdamOptimizer(LR).minimize(model['total_loss'],
                                                             global_step=model['global_step'])

    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
    train(model, input_image, initial_image)


if __name__ == '__main__':
    main()
