# 验证码测试
import os
import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt

# 不同字符数量
CHAR_SET_LEN = 10
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 1
# tfrecord文件存放路径
TFRECORD_FILE = 'E:/SVN/Gavin/Learn/Python/pygame/captcha/test.tfrecords'

# placeholder
x = tf.placeholder(tf.float32,[None,224,224])

# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本,返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label0': tf.FixedLenFeature([], tf.int64),
            'label1': tf.FixedLenFeature([], tf.int64),
            'label2': tf.FixedLenFeature([], tf.int64),
            'label3': tf.FixedLenFeature([], tf.int64),
        }
    )
    img = features['image']
    # 获取图片数据
    image = tf.decode_raw(img, tf.uint8)
    # 没有经过预处理的灰度图
    image_raw = tf.reshape(image, [224,224])
    # 图片预处理
    image = tf.cast(image, tf.float32) /255.0
    image = tf.subtract(image,0.5)
    image = tf.multiply(image,2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    return image, image_raw,label0,label1,label2,label3


# 获取图片数据和标签
image, image_raw,label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)
print(image,image_raw,label0,label1,label2, label3)
# 使用shuffle_batch可以随机打乱输入 next_batch挨着往下取
# shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
# 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
# shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
# Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
img_batch,img_raw_batch, label_batch0,label_batch1,label_batch2,label_batch3 = tf.train.shuffle_batch(
                                         [image,image_raw, label0,label1,label2,label3],
                                        batch_size=BATCH_SIZE, capacity=5000,
                                        min_after_dequeue=1000,num_threads=1)
# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False
)

with tf.Session() as sess:
    X = tf.reshape(x,[BATCH_SIZE,224,224,1])
    # 数据输入网络得到输出值
    logits0,logits1,logits2,logits3,end_points = train_network_fn(X)

    # 预测值
    predict0 = tf.reshape(logits0,[-1,CHAR_SET_LEN])
    predict0 = tf.argmax(predict0,1)

    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
    predict1 = tf.argmax(predict1, 1)

    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
    predict2 = tf.argmax(predict2, 1)

    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
    predict3 = tf.argmax(predict3, 1)


    # 初始化
    sess.run(tf.global_variables_initializer())
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动队列
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(10):
        # 获取一个批次的数据和标签
        b_image,b_image_raw, b_label0,b_label1,b_label2,b_label3 = sess.run([img_batch,img_raw_batch,
                                                                 label_batch0, label_batch1, label_batch2, label_batch3])
        # 显示图片
        img = Image.fromarray(b_image_raw[0],'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # 打印标签
        print('label:',b_label0,b_label1,b_label2,b_label3)
        # 预测
        label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3],
                                               feed_dict={x:b_image})
        # print
        print('predict:',label0,label1,label2,label3)

        # 通知其他线程关闭
        coord.request_stop()
        # 其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)
