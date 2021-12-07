import torch
import numpy
import torch.nn as nn


class custom_tensor:
    def __init__(self, input_tensor, target_size) -> None:
        self.size = target_size[2]
        self.input_tensor = input_tensor
        self.out_tensor = torch.zeros((1,1,self.size,self.size))
        self.weight = torch.zeros((self.size**2, input_tensor.shape[2]**2)).reshape(self.size**2, input_tensor.shape[2]**2)
        self.scale = self.size / input_tensor.shape[2]
        self.point = {}
        self.init_output()
        self.convert_tensor()
        self.save = {}
        self.origin_size = input_tensor.shape[2]

    def init_output(self):
        origin_size = self.input_tensor.shape[2]
        self.first_index = (self.scale - 1) / 2

        for i in range(origin_size):
            for j in range(origin_size):
                value = self.input_tensor[0][0][i][j]
                new_row = self.first_index + self.scale * i
                new_col = self.first_index + self.scale *j
                self.set_value(value, new_row, new_col)
                if new_row.is_integer() and new_col.is_integer():
                    self.set_weight(1, new_row*self.size+new_col, i*origin_size+j)
                self.point[(new_row, new_col)]= float(value)

    def convert_tensor(self):
        out_shape = self.out_tensor.shape[2:]
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                self.convert_value(i, j)

    def convert_value(self, row, col):
        min_coord = min(self.point.keys())[0]
        max_coord = max(self.point.keys())[0]
        min_index = (min_coord - self.first_index)/self.scale
        max_index = (max_coord - self.first_index)/self.scale
        origin_size = self.input_tensor.shape[2]
        if row <= min_coord:
            if col < min_coord:
                self.set_value(self.point[(min_coord, min_coord)], row, col)
                self.set_weight(1, row*self.size+col, min_index*origin_size+min_index)
            elif col in set([key[1] for key in self.point.keys()]):
                self.set_value(self.point[(min_coord, col)], row, col)
                self.set_weight(1, row*self.size+col, (col-self.first_index)/self.scale)
            elif col > max_coord:
                self.set_value(self.point[(min_coord, max_coord)], row, col)
                self.set_weight(1, row*self.size+col, min_index*origin_size+max_index)
            else: 
                self.interpolate(row, col)
        elif col <= min_coord:
            if row < min_coord:
                self.set_value(self.point[(min_coord, min_coord)], row, col)
                self.set_weight(1, row*self.size+col, 0)
            elif row in set([key[0] for key in self.point.keys()]):
                self.set_value(self.point[(row, min_coord)], row, col)
                self.set_weight(1, row*self.size+col, 
                        (row-self.first_index)/self.scale*origin_size+min_index)
            elif row > max_coord:
                self.set_value(self.point[(max_coord, min_coord)], row, col)
                self.set_weight(1, row*self.size+col, max_index*origin_size)
            else:
                self.interpolate(row, col)
        elif row >= max_coord:
            if col > max_coord:
                self.set_value(self.point[(max_coord, max_coord)], row, col)
                self.set_weight(1, row*self.size+col, max_index*origin_size+max_index)
            elif col in set([key[1] for key in self.point.keys()]):
                self.set_value(self.point[(max_coord, col)], row, col)
                self.set_weight(1, row*self.size+col, 
                    max_index*origin_size+(col-self.first_index)/self.scale)
            else:
                self.interpolate(row, col)
        elif col >= max_coord:
            if row > max_coord:
                self.set_value(self.point[(max_coord, max_coord)], row, col)
                self.set_weight(1, row*self.size+col, max_index*origin_size+max_index)
            elif row in set([key[0] for key in self.point.keys()]):
                self.set_value(self.point[(row, max_coord)], row, col)
                self.set_weight(1, row*self.size+col, 
                        (row-self.first_index)/self.scale*origin_size+max_index)
            else:
                self.interpolate(row, col)

        elif row in set([key[0] for key in self.point.keys()]) or \
                        col in set([key[1] for key in self.point.keys()]):
            self.interpolate(row, col)

        else:
            self.bilinear_interpolate(row, col)

    def find_nearest_points(self, row, col):
        xs = list(set([key[0] for key in self.point.keys()]))
        ys = list(set([key[1] for key in self.point.keys()]))

        for i in range(len(xs)):
            if xs[i] < col < xs[i+1]:
                min_col = xs[i]
                max_col = xs[i+1]
                break
        for j in range(len(ys)):
            if ys[j] < row < ys[j+1]:
                min_row = ys[j]
                max_row = ys[j+1]
                break

        return min_row, min_col, max_row, max_col

    def bilinear_interpolate(self, row, col):
        min_r, min_c, max_r, max_c = self.find_nearest_points(row, col)
        h1 = max_r - row
        h2 = row - min_r
        w1 = col - min_c
        w2 = max_c - col
        alpha = h1 / (h1+h2)
        beta = h2 / (h1+h2)
        p = w1 / (w1+w2)
        q = w2 / (w1+w2)

        ori_min_r = (min_r - self.first_index) / self.scale
        ori_min_c = (min_c - self.first_index) / self.scale
        ori_max_r = (max_r - self.first_index) / self.scale
        ori_max_c = (max_c - self.first_index) / self.scale
        origin_size = self.input_tensor.shape[2]

        value = q*beta*self.point[(max_r, min_c)] + q*alpha*self.point[(min_r, min_c)] + \
            p*beta*self.point[(max_r, max_c)] + p*alpha*self.point[(min_r, max_c)]
        self.set_weight(q*beta, row*self.size+col, ori_max_r*origin_size+ori_min_c)
        self.set_weight(q*alpha, row*self.size+col, ori_min_r*origin_size+ori_min_c)
        self.set_weight(p*beta, row*self.size+col, ori_max_r*origin_size+ori_max_c)
        self.set_weight(p*alpha, row*self.size+col, ori_min_r*origin_size+ori_max_c)
        self.set_value(value, row, col)

    def interpolate(self, row, col):
        min_coord = min(self.point.keys())[0]
        max_coord = max(self.point.keys())[0]
        xs = list(set([key[0] for key in self.point.keys()]))
        ys = list(set([key[1] for key in self.point.keys()]))
        origin_size = self.input_tensor.shape[2]

        def linear_interpolate(first_coord, second_coord):
            f_value= self.point[first_coord]
            s_value = self.point[second_coord]
            f_r = (first_coord[0]-self.first_index)/self.scale
            f_c = (first_coord[1]-self.first_index)/self.scale            
            s_r = (second_coord[0]-self.first_index)/self.scale
            s_c = (second_coord[1]-self.first_index)/self.scale
            if row <= min_coord or row >= max_coord or \
                    row in set([key[0] for key in self.point.keys()]):
                f_delta = abs(col - first_coord[1])
                s_delta = abs(col - second_coord[1])

            elif col <= min_coord or col >= max_coord or \
                    col in set([key[1] for key in self.point.keys()]):
                f_delta = abs(row - first_coord[0])
                s_delta = abs(row - second_coord[0])

            self.set_weight(s_delta/(f_delta+s_delta), row*self.size+col,
                    f_r*origin_size+f_c)
            self.set_weight(f_delta / (f_delta + s_delta), row*self.size+col,
                    s_r*origin_size+s_c)
            return f_value * (s_delta / (f_delta + s_delta)) + \
                        s_value * (f_delta / (f_delta + s_delta))
            
        if row <= min_coord or row >= max_coord:
            for i in range(len(xs)):
                if xs[i] < col < xs[i+1]:
                    if row <= min_coord:
                        value = linear_interpolate((min_coord, xs[i]), (min_coord, xs[i+1]))
                    elif row >= max_coord:
                        value = linear_interpolate((max_coord, xs[i]), (max_coord, xs[i+1]))

                    self.set_value(value, row, col)
                    break

        elif col <= min_coord or col >= max_coord:
            for i in range(len(ys)):
                if ys[i] < row < ys[i+1]:
                    if col <= min_coord:
                        value = linear_interpolate((ys[i], min_coord), (ys[i+1], min_coord))
                    elif col >= max_coord:
                        value = linear_interpolate((ys[i], max_coord), (ys[i+1], max_coord))
                    self.set_value(value, row, col)
                    break
        elif row in set([key[0] for key in self.point.keys()]):
            for i in range(len(xs)):
                if xs[i] < col < xs[i+1]:
                    value = linear_interpolate((row, xs[i]), (row, xs[i+1]))
                    self.set_value(value, row, col)
                    break
        elif col in set([key[1] for key in self.point.keys()]):
            for i in range(len(ys)):
                if ys[i] < row < ys[i+1]:
                    value = linear_interpolate((ys[i], col), (ys[i+1], col))
                    self.set_value(value, row, col)
                    break

    def set_weight(self, value, out_ch, in_ch):
        if type(out_ch) == float and not out_ch.is_integer():
            raise ValueError("out_ch value is float but not integer")
        if type(in_ch) == float and not in_ch.is_integer():
            raise ValueError("in_ch value is float but not integer")

        self.weight[int(out_ch)][int(in_ch)] = value
        
    def set_value(self, value, row, col):
        if (type(row) == float and row.is_integer()) or type(row) == int:
            self.out_tensor[0][0][int(row)][int(col)] = value

    def get_value(self, row, col):
        return float(self.out_tensor[0][0][int(row)][int(col)])
    

    def show(self):
        for i in range(self.size):
            for j in range(self.size):
                print('{:.2f}'.format(float(self.out_tensor[0][0][i][j])), end=' ')
            print()

if __name__ == "__main__":
    input_size = (1,1,3,3)
    output_size = (1,1,9,9)
    scale_factor = output_size[2] / input_size[2]

    size = input_size[0] * input_size[1] * input_size[2] * input_size[3]
    ex_tensor = torch.Tensor(range(1, size+1)).reshape(input_size)
    # print(ex_tensor.numpy())
    upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    output_tensor = upsample(ex_tensor)
    tensor = custom_tensor(ex_tensor, output_size)
    # tensor.show()
    # print(output_tensor.numpy())
    # print(tensor.weight)
    out = nn.functional.linear(ex_tensor.flatten(), tensor.weight)
    print(tensor.out_tensor)
    print(out.reshape(*output_size))
    print(tensor.out_tensor - out.reshape(*output_size))

