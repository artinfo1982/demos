import numpy as np


def yuv_420sp_padding(ori_file, ori_h, ori_w, new_file, new_h, new_w):
    '''
    对一个yuv420sp文件进行height或者width的padding，用0填补
    输入：
        ori_file：原始的yuv文件
        ori_h：原始yuv的height
        ori_w：原始yuv的width
        new_file：padding之后生成的新文件
        new_h：新yuv的height
        new_w：新yuv的width
    '''
    # padding can not scale down
    if new_h < ori_h:
        print('Error: new height can not less than original height')
        return
    if new_w < ori_w:
        print('Error: new width can not less than original width')
        return
    # read from binary yuv file, convert to ndarray
    s = ori_h * ori_w
    uv_h = ori_h // 2
    data = np.fromfile(file=ori_file, dtype='uint8')
    Y = data[0:s].reshape(ori_h, ori_w)
    UV = data[s:].reshape(uv_h, ori_w)

    # do padding
    diff_h = new_h - ori_h
    diff_w = new_w - ori_w
    Y = np.pad(Y, ((0, diff_h), (0, diff_w)), 'constant', constant_values=0)
    UV = np.pad(UV, ((0, diff_h), (0, diff_w)), 'constant', constant_values=0)

    Y = Y.flatten()
    UV = UV.flatten()

    try:
        with open(new_file, 'wb') as f:
            f.write(Y)
            f.write(UV)
    except IOError as e:
        print(str(e))


if __name__ == '__main__':
    yuv_420sp_padding('d:\\a.yuv', 12, 12, 'd:\\b.yuv', 12, 16)
