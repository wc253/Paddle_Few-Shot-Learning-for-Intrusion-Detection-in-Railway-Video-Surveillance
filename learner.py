import paddle
from paddle import nn
from paddle.nn import functional as F
import numpy as np
class Learner(paddle.nn.Layer):
    def __init__(self,config):
        super(Learner, self).__init__()
        self.config = config
        # self.vars = paddle.nn.ParameterList()  #设置容器存储权重参数
        self.vars = []
        self.vars_bn = []
        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                weight = paddle.static.create_parameter(shape=[param[0],param[1],param[2],param[3]], dtype='float32',
                                            default_initializer=nn.initializer.KaimingNormal(),
                                            is_bias = False) # 对conv2d构建32x4x3x3的size
                bias = paddle.static.create_parameter(shape = [param[0]], dtype='float32',
                                            is_bias = True)  #创建可学习偏置b
                self.vars.extend([weight, bias])

            elif name == 'linear':
                # [ch_out, ch_in]
                weight = paddle.static.create_parameter(shape=[param[1], param[0]], dtype='float32',
                                            default_initializer=nn.initializer.KaimingNormal(),
                                            is_bias = False)  # 对conv2d构建32x4x3x3的size
                bias = paddle.static.create_parameter(shape=[param[0]], dtype='float32',
                                            is_bias=True)  # 创建可学习偏置b
                self.vars.extend([weight, bias])

            elif name == 'bn':
                # [ch_out]
                weight = paddle.static.create_parameter(shape=[param[0]], dtype='float32',
                                            default_initializer=nn.initializer.Constant(value=1),
                                            is_bias=False)  # 对conv2d构建32x4x3x3的size
                bias = paddle.static.create_parameter(shape=[param[0]], dtype='float32',
                                            is_bias=True)  # 创建可学习偏置b
                self.vars.extend([weight, bias])
                running_mean = paddle.to_tensor(np.zeros([param[0]], np.float32), stop_gradient=True)
                running_var = paddle.to_tensor(np.zeros([param[0]], np.float32), stop_gradient=True)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'dropout2d', 'dropout']:
                continue
            else:
                raise NotImplementedError
        # print(self.vars)

    def forward(self, x, vars=None, bn_training=True, dp_training=False):

        if vars is None:
            vars = self.vars

        idx = 0
        idx_bn = 0
        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[idx_bn],self.vars_bn[idx_bn + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training,
                                 momentum=1)  # , momentum=1)，在论文的中设置为0.1
                idx += 2
                idx_bn += 2

            elif name == 'flatten':
                # print(x.shape)
                x = paddle.reshape(x,[x.shape[0],-1])
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x) # inplace=True会节省时间，可能报错，但不影响结果
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'dropout2d':
                x = F.dropout2d(x, p=param[0], training=dp_training)
            elif name == 'dropout':
                if dp_training:
                    x = F.dropout(x, p=param[0], training=dp_training)
                else:
                    x = x
            else:
                raise NotImplementedError
        return x


    def parameters(self,include_sublayers=True):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

if __name__ == '__main__':
    config = [
        ('conv2d', [32, 4, 3, 3, 2, 0]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 2, 0]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 2, 0]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('dropout', [0.3]),
        ('linear', [2, 32 * 4 * 6])
    ]
    learn = Learner(config)
    tmp = filter(lambda x: not (x.stop_gradient), learn.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print("the num of parameters is{}".format(num))
