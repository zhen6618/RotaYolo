import copy

import numpy as np
from copy import deepcopy
from math import cos, sin, pi, atan2
import torch
# x y w h theda → EigenVector、EigenValue → Matrix: 2 × 3
# x y w h: 0~1, theda: 0~180
def xywhTheda2Eigen(input):  # [:, xywhtheda] [a1, a2, cx; a3, a4, cy]

    input[..., 4] = input[..., 4] / 180 * pi
    # 2D
    if len(input.shape) == 2:
        output = np.zeros((input.shape[0], 2, 3))
        output_vector = np.zeros((input.shape[0], 6))  # [cx, cy, a1, a2, a3, a4]
        output_vector_5 = np.zeros((input.shape[0], 5))  # [cx, cy, z1, z3, z2]

        EigenMatrix = np.zeros((input.shape[0], 2, 3))
        EigenMatrix[:, 0, 2] = input[:, 0]  # cx
        EigenMatrix[:, 1, 2] = input[:, 1]  # cy

        EigenValueMatrix = np.zeros((input.shape[0], 2, 2))
        EigenValueMatrix[:, 0, 0] = input[:, 2]  # w
        EigenValueMatrix[:, 1, 1] = input[:, 3]  # h

        theda = input[:, 4]
        EigenVectorMatrix = np.zeros((input.shape[0], 2, 2))
        EigenVectorMatrix[:, 0, 0] = np.cos(theda)
        EigenVectorMatrix[:, 1, 0] = -np.sin(theda)
        EigenVectorMatrix[:, 0, 1] = -np.sin(theda)
        EigenVectorMatrix[:, 1, 1] = -np.cos(theda)

        # EigenMatrix[:2, :2] = np.dot(np.dot(EigenVectorMatrix, EigenValueMatrix), EigenVectorMatrix.T)
        E_sin_temp = np.einsum('ijk,ikr->ijr', EigenVectorMatrix, EigenValueMatrix)
        EigenMatrix[:, :2, :2] = np.einsum('ijk,ikr->ijr', E_sin_temp, EigenVectorMatrix.transpose(0, 2, 1))
        output[:, :, :] = EigenMatrix[:, :, :]

        output_vector[:, 0] = output[:, 0, 2]  # cx
        output_vector[:, 1] = output[:, 1, 2]  # cy
        output_vector[:, 2] = output[:, 0, 0]  # a1
        output_vector[:, 3] = output[:, 0, 1]  # a2
        output_vector[:, 4] = output[:, 1, 0]  # a3
        output_vector[:, 5] = output[:, 1, 1]  # a4

        output_vector_5[:, 0] = output[:, 0, 2]  # cx
        output_vector_5[:, 1] = output[:, 1, 2]  # cy
        output_vector_5[:, 2] = output[:, 0, 0]  # z1  0~1
        output_vector_5[:, 3] = output[:, 1, 1]  # z3  0~1
        output_vector_5[:, 4] = output[:, 0, 1]  # z2  -1~1

    elif len(input.shape) == 3:
        output_vector_5_3 = np.zeros((input.shape[0], input.shape[1], 5))
        for index in range(input.shape[0]):
            output = np.zeros((input.shape[1], 2, 3))
            output_vector = np.zeros((input.shape[1], 6))  # [cx, cy, a1, a2, a3, a4]
            output_vector_5 = np.zeros((input.shape[1], 5))  # [cx, cy, z1, z3, z2]

            EigenMatrix = np.zeros((input.shape[1], 2, 3))
            EigenMatrix[:, 0, 2] = input[index, :, 0]  # cx
            EigenMatrix[:, 1, 2] = input[index, :, 1]  # cy

            EigenValueMatrix = np.zeros((input.shape[1], 2, 2))
            EigenValueMatrix[:, 0, 0] = input[index, :, 2]  # w
            EigenValueMatrix[:, 1, 1] = input[index, :, 3]  # h

            theda = input[index, :, 4]
            EigenVectorMatrix = np.zeros((input.shape[1], 2, 2))
            EigenVectorMatrix[:, 0, 0] = np.cos(theda)
            EigenVectorMatrix[:, 1, 0] = -np.sin(theda)
            EigenVectorMatrix[:, 0, 1] = -np.sin(theda)
            EigenVectorMatrix[:, 1, 1] = -np.cos(theda)

            # EigenMatrix[:2, :2] = np.dot(np.dot(EigenVectorMatrix, EigenValueMatrix), EigenVectorMatrix.T)
            E_sin_temp = np.einsum('ijk,ikr->ijr', EigenVectorMatrix, EigenValueMatrix)
            EigenMatrix[:, :2, :2] = np.einsum('ijk,ikr->ijr', E_sin_temp, EigenVectorMatrix.transpose(0, 2, 1))
            output[:, :, :] = EigenMatrix[:, :, :]

            output_vector[:, 0] = output[:, 0, 2]  # cx
            output_vector[:, 1] = output[:, 1, 2]  # cy
            output_vector[:, 2] = output[:, 0, 0]  # a1
            output_vector[:, 3] = output[:, 0, 1]  # a2
            output_vector[:, 4] = output[:, 1, 0]  # a3
            output_vector[:, 5] = output[:, 1, 1]  # a4

            output_vector_5[:, 0] = output[:, 0, 2]  # cx
            output_vector_5[:, 1] = output[:, 1, 2]  # cy
            output_vector_5[:, 2] = output[:, 0, 0]  # z1  0~1
            output_vector_5[:, 3] = output[:, 1, 1]  # z3  0~1
            output_vector_5[:, 4] = output[:, 0, 1]  # z2  -1~1

            output_vector_5_3[index, :, :] = output_vector_5[:, :]
        output_vector_5 = output_vector_5_3

    return output_vector_5

def xywhTheda2Eigen_numpy(input):  # x y w h: 0~1, theda: 0~180  to x, y, z1, z3, z2
    cx = copy.deepcopy(input[..., 0])
    cy = copy.deepcopy(input[..., 1])
    w = copy.deepcopy(input[..., 2])
    h = copy.deepcopy(input[..., 3])
    angle = copy.deepcopy(input[..., 4])

    angle = angle / 180 * pi
    sin_ = np.sin(angle)
    cos_ = np.cos(angle)

    z1 = w * (cos_**2) + h * (sin_**2)
    z3 = w * (sin_**2) + h * (cos_**2)
    z2 = (h - w) * sin_ * cos_

    output = np.concatenate([cx.reshape(-1, 1), cy.reshape(-1, 1), z1.reshape(-1, 1), z3.reshape(-1, 1), z2.reshape(-1, 1)], axis=1)

    return output

# EigenVector、EigenValue → Matrix: 2 × 3 →x y w h theda
# x y w h theda: 0~180
def Eigen2xywhTheda(input_vector):  # [:, xywhtheda] [a1, a2, cx; a3, a4, cy]

    # 2D
    if len(input_vector.shape) == 2:
        output = np.zeros((input_vector.shape[0], 5))
        input = np.zeros((input_vector.shape[0], 2, 3))   # Vector to Matrix
        input[:, 0, 2] = input_vector[:, 0]  # cx
        input[:, 1, 2] = input_vector[:, 1]  # cy
        input[:, 0, 0] = input_vector[:, 2]  # a1
        input[:, 0, 1] = input_vector[:, 4]  # a2
        input[:, 1, 0] = input_vector[:, 4]  # a3
        input[:, 1, 1] = input_vector[:, 3]  # a4

        output[:, 0] = input[:, 0, 2]  # cx
        output[:, 1] = input[:, 1, 2]  # cy

        values, vectors = np.linalg.eig(input[:, :, :2])
        a = values.argmax(axis=1)
        output[:, 2] = values[np.arange(a.size), a]

        b = 1 - a
        output[:, 3] = values[np.arange(b.size), b]
        theda_vector = vectors[np.arange(a.size), :, a]
        theda = np.arctan2(theda_vector[:, 1], theda_vector[:, 0]) + pi  # 0 ~ 2*pi
        theda[theda >= pi] -= pi
        theda[theda == pi] -= pi  # [0, 2*pi] to [0, pi)

        output_theda = pi - theda
        output_theda[output_theda == pi] -= pi  # (0, pi] to [0, pi)
        output[:, 4] = output_theda

    elif len(input_vector.shape) == 3:
        output_3 = np.zeros((input_vector.shape[0], input_vector.shape[1], 5))
        for index in range(input_vector.shape[0]):
            output = np.zeros((input_vector.shape[1], 5))
            input = np.zeros((input_vector.shape[1], 2, 3))   # Vector to Matrix
            input[:, 0, 2] = input_vector[index, :, 0]  # cx
            input[:, 1, 2] = input_vector[index, :, 1]  # cy
            input[:, 0, 0] = input_vector[index, :, 2]  # a1
            input[:, 0, 1] = input_vector[index, :, 4]  # a2
            input[:, 1, 0] = input_vector[index, :, 4]  # a3
            input[:, 1, 1] = input_vector[index, :, 3]  # a4

            output[:, 0] = input[:, 0, 2]  # cx
            output[:, 1] = input[:, 1, 2]  # cy

            values, vectors = np.linalg.eig(input[:, :, :2])  # np.where
            a = values.argmax(axis=1)
            output[:, 2] = values[np.arange(a.size), a]

            b = 1 - a
            output[:, 3] = values[np.arange(b.size), b]

            theda_vector = vectors[np.arange(a.size), :, a]
            theda = np.arctan2(theda_vector[:, 1], theda_vector[:, 0]) + pi  # 0 ~ 2*pi
            theda[theda >= pi] -= pi
            theda[theda == pi] -= pi  # [0, 2*pi] to [0, pi)

            output_theda = pi - theda
            output_theda[output_theda == pi] -= pi  # (0, pi] to [0, pi)
            output[:, 4] = output_theda

            output_3[index, :, :] = output
        output = output_3

    output[..., 4] = output[..., 4] / pi * 180
    return output

# x y w h theda: 0~180
def Eigen2xywhTheda_numpy(input_vector, h_thred=0, angle_thred=0.95):  # input_vector: (n, 2)  (cx, cy, z1, z3, z2) to (cx, cy, w, h, theda)
    cx = copy.deepcopy(input_vector[..., 0])
    cy = copy.deepcopy(input_vector[..., 1])
    z1 = copy.deepcopy(input_vector[..., 2])
    z2 = copy.deepcopy(input_vector[..., 4])
    z3 = copy.deepcopy(input_vector[..., 3])

    w = 0.5 * (z1 + z3 + ((z1 - z3) ** 2 + 4 * (z2 ** 2)) ** 0.5)
    h = 0.5 * (z1 + z3 - ((z1 - z3) ** 2 + 4 * (z2 ** 2)) ** 0.5)

    'h >= 0 means positive definiteness'
    non_positive_define = (h <= h_thred)

    sin_2theta = -2 * z2 / (w - h)
    cos_2theta = (z1 - z3) / (w - h)

    sin_2theta = np.clip(sin_2theta, -1, 1)
    cos_2theta = np.clip(cos_2theta, -1, 1)

    arcsin_2_theta = np.arcsin(sin_2theta)  # [-pi/2, pi/2]
    arccos_2_theta = np.arccos(cos_2theta)  # [0, pi]

    _2_theta = np.zeros_like(arcsin_2_theta)

    arcsin_2_theta_1 = (arcsin_2_theta >= 0) * (arcsin_2_theta <= pi / 2) * (arccos_2_theta >= 0) * (arccos_2_theta <= pi / 2)
    _2_theta[arcsin_2_theta_1] = arcsin_2_theta[arcsin_2_theta_1]

    arcsin_2_theta_2 = (arcsin_2_theta >= 0) * (arcsin_2_theta <= pi / 2) * (arccos_2_theta > pi / 2) * (arccos_2_theta <= pi)
    _2_theta[arcsin_2_theta_2] = pi - arcsin_2_theta[arcsin_2_theta_2]

    arcsin_2_theta_3 = (arcsin_2_theta >= -pi / 2) * (arcsin_2_theta < 0) * (arccos_2_theta > pi / 2) * (arccos_2_theta <= pi)
    _2_theta[arcsin_2_theta_3] = pi - arcsin_2_theta[arcsin_2_theta_3]

    arcsin_2_theta_4 = (arcsin_2_theta >= -pi / 2) * (arcsin_2_theta < 0) * (arccos_2_theta >= 0) * (arccos_2_theta <= pi / 2)
    _2_theta[arcsin_2_theta_4] = 2 * pi + arcsin_2_theta[arcsin_2_theta_4]

    theta = _2_theta / 2  # [0, pi)
    theta = theta / pi * 180  # [0, 180)
    theta = np.clip(theta, 1, 179)

    cx, cy, w, h, theta = cx.reshape(-1, 1), cy.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1), theta.reshape(-1, 1)
    output = np.concatenate([cx, cy, w, h, theta], axis=1)

    return output, non_positive_define

def arcsin_taylor(xi, x0):
    x = torch.arcsin(x0) + ((1 - x0**2)**(-0.5)) * (xi - x0) + x0 * ((1 - x0**2)**(-1.5)) * (xi - x0)**2

    return x

def arccos_taylor(xi, x0):
    x = torch.arccos(x0) - ((1 - x0**2)**(-0.5)) * (xi - x0) - x0 * ((1 - x0**2)**(-1.5)) * (xi - x0)**2

    return x

# x y w h theda: 0~180
def Eigen2xywhTheda_numpy_tensor(input_vector, h_thred=0, angle_thred=1.0):  # input_vector: (n, 2)  (cx, cy, z1, z3, z2) to (cx, cy, w, h, theda)
    cx = input_vector[..., 0]
    cy = input_vector[..., 1]
    z1 = input_vector[..., 2]
    z2 = input_vector[..., 4]
    z3 = input_vector[..., 3]

    w = 0.5 * (z1 + z3 + ((z1 - z3) ** 2 + 4 * (z2 ** 2)) ** 0.5)
    h = 0.5 * (z1 + z3 - ((z1 - z3) ** 2 + 4 * (z2 ** 2)) ** 0.5)

    'h >= 0 means positive definiteness'
    non_positive_define = (h <= h_thred)
    w[(w / h < 1.05) & (h > 0)] *= 1.05

    sin_2theta = -2 * z2 / (w - h)
    cos_2theta = (z1 - z3) / (w - h)

    non_positive_define |= (sin_2theta > angle_thred) | (sin_2theta < -angle_thred) | (cos_2theta > angle_thred) | (cos_2theta < -angle_thred)
    sin_2theta = torch.clip(sin_2theta, -angle_thred, angle_thred)
    cos_2theta = torch.clip(cos_2theta, -angle_thred, angle_thred)

    arcsin_2_theta = torch.arcsin(sin_2theta)  # [-pi/2, pi/2]
    arccos_2_theta = torch.arccos(cos_2theta)  # [0, pi]

    _2_theta = torch.zeros_like(arcsin_2_theta)

    arcsin_2_theta_1 = (arcsin_2_theta >= 0) * (arcsin_2_theta <= pi / 2) * (arccos_2_theta >= 0) * (arccos_2_theta <= pi / 2)
    _2_theta[arcsin_2_theta_1] = arcsin_2_theta[arcsin_2_theta_1]

    arcsin_2_theta_2 = (arcsin_2_theta >= 0) * (arcsin_2_theta <= pi / 2) * (arccos_2_theta > pi / 2) * (arccos_2_theta <= pi)
    _2_theta[arcsin_2_theta_2] = pi - arcsin_2_theta[arcsin_2_theta_2]

    arcsin_2_theta_3 = (arcsin_2_theta >= -pi / 2) * (arcsin_2_theta < 0) * (arccos_2_theta > pi / 2) * (arccos_2_theta <= pi)
    _2_theta[arcsin_2_theta_3] = pi - arcsin_2_theta[arcsin_2_theta_3]

    arcsin_2_theta_4 = (arcsin_2_theta >= -pi / 2) * (arcsin_2_theta < 0) * (arccos_2_theta >= 0) * (arccos_2_theta <= pi / 2)
    _2_theta[arcsin_2_theta_4] = 2 * pi + arcsin_2_theta[arcsin_2_theta_4]

    theta = _2_theta / 2  # [0, pi)
    theta = theta / pi * 180  # [0, 180)
    theta = torch.clip(theta, 1, 179)

    # cx, cy, w, h, theta = cx.view(-1, 1), cy.view(-1, 1), w.view(-1, 1), h.view(-1, 1), theta.view(-1, 1)
    cx, cy, w, h, theta = cx.unsqueeze(-1), cy.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1), theta.unsqueeze(-1)
    output = torch.cat([cx, cy, w, h, theta], dim=-1)

    # output = output.to(dtype=original_type)

    return output, non_positive_define

def Eigen2xywhTheda_gpu(input_vector):  # [:, xywhtheda] [a1, a2, cx; a3, a4, cy]

    # 2D
    if len(input_vector.shape) == 2:
        output = torch.zeros((input_vector.shape[0], 5))
        input = torch.zeros((input_vector.shape[0], 2, 3))   # Vector to Matrix
        input[:, 0, 2] = input_vector[:, 0]  # cx
        input[:, 1, 2] = input_vector[:, 1]  # cy
        input[:, 0, 0] = input_vector[:, 2]  # a1
        input[:, 0, 1] = input_vector[:, 4]  # a2
        input[:, 1, 0] = input_vector[:, 4]  # a3
        input[:, 1, 1] = input_vector[:, 3]  # a4

        output[:, 0] = input[:, 0, 2]  # cx
        output[:, 1] = input[:, 1, 2]  # cy

        values, vectors = torch.linalg.eigh(input[:, :, :2])  # np.where
        a = values.argmax(axis=1)
        output[:, 2] = values[torch.arange(len(a)), a]
        b = 1 - a
        output[:, 3] = values[torch.arange(len(b)), b]

        theda_vector = vectors[torch.arange(len(a)), :, a]
        theda = torch.arctan2(theda_vector[:, 1], theda_vector[:, 0]) + pi  # 0 ~ 2*pi
        theda[theda >= pi] -= pi
        theda[theda == pi] -= pi  # [0, 2*pi] to [0, pi)

        output_theda = pi - theda
        output_theda[output_theda == pi] -= pi  # (0, pi] to [0, pi)
        output[:, 4] = output_theda

    elif len(input_vector.shape) == 3:
        output_3 = torch.zeros((input_vector.shape[0], input_vector.shape[1], 5))
        for index in range(input_vector.shape[0]):
            output = torch.zeros((input_vector.shape[1], 5))
            input = torch.zeros((input_vector.shape[1], 2, 3))   # Vector to Matrix
            input[:, 0, 2] = input_vector[index, :, 0]  # cx
            input[:, 1, 2] = input_vector[index, :, 1]  # cy
            input[:, 0, 0] = input_vector[index, :, 2]  # a1
            input[:, 0, 1] = input_vector[index, :, 4]  # a2
            input[:, 1, 0] = input_vector[index, :, 4]  # a3
            input[:, 1, 1] = input_vector[index, :, 3]  # a4

            output[:, 0] = input[:, 0, 2]  # cx
            output[:, 1] = input[:, 1, 2]  # cy

            values, vectors = torch.linalg.eigh(input[:, :, :2])  # np.where
            a = values.argmax(axis=1)
            output[:, 2] = values[torch.arange(a.size), a]

            b = 1 - a
            output[:, 3] = values[torch.arange(b.size), b]

            theda_vector = vectors[torch.arange(a.size), :, a]
            theda = torch.arctan2(theda_vector[:, 1], theda_vector[:, 0]) + pi  # 0 ~ 2*pi
            theda[theda >= pi] -= pi
            theda[theda == pi] -= pi  # [0, 2*pi] to [0, pi)

            output_theda = pi - theda
            output_theda[output_theda == pi] -= pi  # (0, pi] to [0, pi)
            output[:, 4] = output_theda

            output_3[index, :, :] = output
        output = output_3

    output[..., 4] = output[..., 4] / pi * 180
    return output
    
def Eigen2x(input):  # clsid, cx, cy, z1, z3, z2  to  clsid, cx, cy, x1, x2, x3
    output = deepcopy(input)
    output[..., -3] = (input[..., -3] + input[..., -2]) / 2
    output[..., -2] = input[..., -3]
    output[..., -1] = (input[..., -3] + input[..., -2] + input[..., -1] * 2) / 2

    return output

def x2Eigen(input):  # clsid, cx, cy, x1, x2, x3  to clsid, cx, cy, z1, z3, z2
    if torch.is_tensor(input):
        output = torch.zeros_like(input).to(input.device)
    else:
        output = np.zeros_like(input)
    output[..., :-3] = input[..., :-3]

    output[..., -3] = input[..., -2]
    output[..., -2] = input[..., -3] * 2 - input[..., -2]
    output[..., -1] = input[..., -1] - input[..., -3]

    return output

if __name__ == '__main__':
    input = np.array([[[0.3, 0.4, 0.8, 0.5, 179 ],
                      [0.1, 0.2, 0.41, 0.4, 0 ],
                      [0.3, 0.4, 0.7, 0.6, 90]],
                      [[0.3, 0.4, 0.8, 0.5, 134],
                       [0.1, 0.2, 0.41, 0.4, 50],
                       [0.3, 0.4, 0.7, 0.6, 70]]]
                     )

    output = xywhTheda2Eigen(input)
    print("input: ", input.shape, input)
    print("output: ", output.shape, output)

    output2 = Eigen2xywhTheda(output)
    print("output: ", output.shape, output)
    print("output2: ", output2.shape, output2)

