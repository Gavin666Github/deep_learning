# 验证码生成器,批量生成本地文件，用于训练
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import sys

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = [] # 验证码列表
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    # 获取随机生成的验证码
    captcha_text = random_captcha_text()
    # 把验证码列表转为字符串
    captcha_text = ''.join(captcha_text)
    # 生成验证码
    captcha = image.generate(captcha_text)
    image.write(captcha_text, 'captcha/images/'+captcha_text + '.jpg')  # 写到文件

# 批量生成num 个图像
num = 3000

if __name__ == '__main__':
    # 测试
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d ' % (i + 1, num) )
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print("生成完毕")


