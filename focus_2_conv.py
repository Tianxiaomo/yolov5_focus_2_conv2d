import cv2
import numpy as np
import torch
import torch.nn as nn

image_path = "test.jpg"
img = cv2.imread(image_path)
if img != None:
    img = cv2.resize(img, (416, 416))
    img = np.transpose(img, [2, 0, 1])
    img = torch.from_numpy(img)
    img = img.float()
    img = img / 255.0
    img = img[None]
    x = img
else:
    x = np.array(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ],
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                [130, 140, 150, 160]
            ],
            [
                [19, 29, 39, 49],
                [59, 69, 79, 89],
                [99, 109, 119, 129],
                [139, 149, 159, 169]
            ]
        ]
    )

    x = x[None]
    x = torch.tensor(x).type(torch.FloatTensor)

# yolov5 focus result
focus_port_res = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


weight = np.array(
    [
        [[1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0]],

        [[0, 0, 1, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 1, 0]],

        [[0, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 0, 0]],

        [[0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1]],
    ]
)

weight = weight.reshape(12, 3, 2, 2)
weight = torch.from_numpy(weight).type(torch.FloatTensor)

conv = nn.Conv2d(3, 12, (2, 2), (2, 2), bias=False)
conv.weight.data = weight

# conv2d result
conv_res = conv(x)

# result compare
print((focus_port_res == conv_res).sum())
