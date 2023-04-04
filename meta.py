import paddle
from paddle.nn import functional as F
from learner import Learner
import numpy as np
from copy import deepcopy
import argparse
class Meta(paddle.nn.Layer):
    def __init__(self,args,config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = 2   #有无入侵的二分类
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(config)
        tmp = filter(lambda x: not (x.stop_gradient), self.net.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print("the num of parameters is{}".format(num))
        # print("-----------------------------------------")
        self.meta_optim = paddle.optimizer.Adam(parameters=self.net.parameters(), learning_rate=self.meta_lr)  #其中parameters函数相当于根据网络自动生成了参数输入到优化器中
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [batchsz, setsz, channel, h, w]
        :param y_spt:   [batchsz, setsz]
        :param x_qry:   [batchsz, querysz, channel, h, w]
        :param y_qry:   [batchsz, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.shape

        # # print(x_spt.size())
        querysz = x_qry.shape[1]

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True, dp_training=True)  # 前向传播
            y_spt_long = paddle.cast(y_spt[i],'int64')
            y_qry_long = paddle.cast(y_qry[i], 'int64')
            loss = F.cross_entropy(logits, y_spt_long)
            # print(self.net.parameters())
            grad = paddle.grad(outputs = loss, inputs = self.net.parameters())  # 计算网络各层梯度    求loss/net.parameters，即梯度
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  # 单一任务的梯度下降

            # this is the loss and accuracy before first update
            # 更新前的准确率
            with paddle.no_grad():
                # 去掉接下来关于Variable的梯度计算，减少时间和内存消耗，https://blog.csdn.net/jacke121/article/details/80597759
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True, dp_training=False)
                # logits_q = self.net.forward(x_qry[i], self.net.parameters(), bn_training=True, dp_training=False)  # 涉及x_qry的都dp_training=False，说明只有在元训练过程中的训练时有dropout/
                loss_q = F.cross_entropy(logits_q, y_qry_long)  # y_qry[i]是一个任务的query image
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, axis=1).argmax(axis=1)
                correct = paddle.equal(pred_q, y_qry_long).numpy().sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            # 第一次更新后
            # logits_q = self.net(x_qry[i], fast_weights, bn_training=True, dp_training=False)
            # loss_q = F.cross_entropy(logits_q, y_qry_long)
            # losses_q[1] += loss_q
            with paddle.no_grad():
                logits_q = self.net.forward(x_qry[i], fast_weights, bn_training=True, dp_training=False)
                # logits_q = self.net.forward(x_qry[i], fast_weights, bn_training=True, dp_training=False)
                loss_q = F.cross_entropy(logits_q, y_qry_long)
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, axis=1).argmax(axis=1)
                correct = paddle.equal(pred_q, y_qry_long).sum().item()
                corrects[1] = corrects[1] + correct

            # 第2次及后续更新
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net.forward(x_spt[i], fast_weights, bn_training=True, dp_training=True)
                loss = F.cross_entropy(logits, y_spt_long)
                # 2. compute grad on theta_pi
                grad = paddle.grad(outputs = loss, inputs = fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                if k < self.update_step - 1:
                    with paddle.no_grad():
                        logits_q = self.net.forward(x_qry[i], fast_weights, bn_training=True, dp_training=False)
                        # loss_q will be overwritten and just keep the loss_q on last update step.
                        loss_q = F.cross_entropy(logits_q, y_qry_long)
                        losses_q[k + 1] += loss_q
                else:
                    logits_q = self.net.forward(x_qry[i], fast_weights, bn_training=True, dp_training=False)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry_long)
                    losses_q[k + 1] += loss_q
                with paddle.no_grad():
                    pred_q = F.softmax(logits_q, axis = 1).argmax(axis=1)
                    correct = paddle.equal(pred_q, y_qry_long).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num  # 只用了最后一次的损失来进行元更新参数（一阶近似）
            # meta-update theta parameters
        self.meta_optim.clear_grad()
        # print("------------------------this is last-----------------------------")
        # this_parameters = self.net.parameters()
        # print(this_parameters[0])
        loss_q.backward()
        self.meta_optim.step()
        # print("------------------------this is next-------------------------------")
        # print(self.net.parameters()[0])
        # next_parameters = self.net.parameters()
        # if (this_parameters == next_parameters):
        #     print("the vars was donnt change")
            #
            # # 这里corrects累加了所有任务的正确数，长度为1（更新前）+update_step
        accs = np.array(corrects) / (querysz * task_num)
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, channel, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, channel, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.shape[0]

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt, dp_training=False)
        y_spt_long = paddle.cast(y_spt,'int64')
        y_qry_long = paddle.cast(y_qry,'int64')
        loss = F.cross_entropy(logits, y_spt_long)
        grad = paddle.grad(outputs=loss, inputs=net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with paddle.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True, dp_training=False)
            # [setsz]
            pred_q = F.softmax(logits_q, axis = 1).argmax(axis=1)
            # scalar
            correct = paddle.equal(pred_q, y_qry_long).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with paddle.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True, dp_training=False)
            # [setsz]
            pred_q = F.softmax(logits_q, axis=1).argmax(axis=1)
            # scalar
            correct = paddle.equal(pred_q, y_qry_long).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True, dp_training=False)
            loss = F.cross_entropy(logits, y_spt_long)
            # 2. compute grad on theta_pi
            grad = paddle.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            # with paddle.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True, dp_training=False)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                # loss_q = F.cross_entropy(logits_q, y_qry_long)

            with paddle.no_grad():
                pred_q = F.softmax(logits_q, axis = 1).argmax(axis=1)
                correct = paddle.equal(pred_q, y_qry_long).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--iternum', type=int, help='iteration number', default=60000)  # 10  000
    argparser.add_argument('--k_spt', type=int, help='k shot for support set per class', default=1)  # 5
    argparser.add_argument('--k_qry', type=int, help='k shot for query set per class', default=1)  # 15
    argparser.add_argument('--imgsz1', type=int, help='imgsz1', default=640)
    argparser.add_argument('--imgsz2', type=int, help='imgsz2', default=480)
    argparser.add_argument('--imgc', type=int, help='imgc', default=4)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=2)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)  # <<<<<<<<<<<<
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    meta = Meta(args,config)
    tmp = filter(lambda x: not (x.stop_gradient), meta.net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print("the num of parameters is{}".format(num))

