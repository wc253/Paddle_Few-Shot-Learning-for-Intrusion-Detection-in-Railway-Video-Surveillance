import os
import paddle
from paddle.io import Dataset
from paddle.vision import transforms
import numpy as np
from PIL import Image

class local_dataset_msk(Dataset):
    """
    引入label shuffuling和分割图mask后的数据加载方式
    """

    def __init__(self, root, data, mode, k_shot, k_query, all_batchsz, resize_h, resize_w, randnseed,shuffle_p):
        """
        :param root: root path
        :param data: 训练数据的名字
        :param mode: 始终为train，这个参数在这没意义，只指引了路径
        :param k_shot: num of support imgs per class
        :param k_query: num of query imgs per class
        :param all_batchsz: args.task_num * args.iternum
        :param resize_h:
        :param resize_w:
        :param randnseed:
        """

        self.batchsz = all_batchsz
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = 2 * self.k_shot
        self.querysz = 2 * self.k_query
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.path = root
        self.mode = mode
        self.sym = '\\'  # linux文件分隔符不是\\，是/
        self.randnseed = randnseed
        self.p = shuffle_p
        print('shuffle DB :%s, b:%d, %d-shot, %d-query, resize_h:%d, resize_w:%d' % (
        mode, all_batchsz, k_shot, k_query, resize_h, resize_w))

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean = [0.419, 0.417, 0.418],std = [0.248, 0.246, 0.246])
                                                 ])
        self.transform_msk = transforms.Compose([lambda x: Image.open(x).convert('1'),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.117,), (0.321,))
                                                 ])
        self.load_datainf(data)
        self.create_batch(self.batchsz)
        self.create_batch_label(self.batchsz)

    def load_datainf(self, data):
        self.sce_idx_arr = data[0]
        self.sob_num_arr = data[1]
        self.nob_num_arr = data[2]

    def create_batch_label(self, batchsz):
        """
        create batch label for meta-learning.
        """
        batchsz1 = int((1 - self.p) * batchsz)  # <<<<有多少batch依旧保留有入侵为1
        batchsz2 = int(self.p * batchsz)  # <<<<有多少batch shuffle了，即标签反置
        self.idx_y = []  # idx for label
        for b in range(batchsz1):  # for each batch
            # 1.select n_way classes randomly
            idx_y = []
            clx_index = [0, 1]
            # 2. select intrusive samples
            for i in range(2):
                idx_y.append(np.array(clx_index[i]).tolist())

            self.idx_y.append(idx_y)

        for b in range(batchsz2):  # for each batch
            # 1.select n_way classes randomly
            idx_y = []
            clx_index = [1, 0]
            # 2. select intrusive samples
            for i in range(2):
                idx_y.append(np.array(clx_index[i]).tolist())

            self.idx_y.append(idx_y)

        np.random.seed(222)
        np.random.shuffle(self.idx_y)

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        所有支撑集和访问集集合列表
        :param batchsz:
        """
        np.random.seed(self.randnseed)
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            support_x = []
            query_x = []
            select_idx = int(np.random.choice(len(self.sce_idx_arr), 1))
            sce_idx = self.sce_idx_arr[select_idx]     #存入选定的场景
            sce_num = np.zeros(2, dtype=np.int)
            sce_num[1] = self.sob_num_arr[select_idx]
            sce_num[0] = self.nob_num_arr[select_idx]
            # 2. select intrusive samples
            for i in range(2):
                support_x_sn = []
                query_x_sn = []
                # print(sce_num[i])
                selected_imgs_idx = 1 + np.random.choice(int(sce_num[i]), self.k_shot + self.k_query, False)

                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                for s in indexDtrain:
                    support_x_sn.append(sce_idx + '_' + str(i) + '_' + str(s) + '.jpg')  #
                support_x.append(support_x_sn)

                for s in indexDtest:
                    query_x_sn.append(sce_idx + '_' + str(i) + '_' + str(s) + '.jpg')
                query_x.append(query_x_sn)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        normailize = transforms.Normalize(mean = [0.419, 0.417, 0.418],std = [0.248, 0.246, 0.246])
        to_tensor1 = transforms.ToTensor()
        support_x = paddle.zeros(shape = [self.setsz, 4, self.resize_h, self.resize_w])
        support_y = np.zeros((self.setsz), dtype=np.int)
        query_x = paddle.zeros(shape = [self.querysz, 4, self.resize_h, self.resize_w])
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = []
        flatten_support_x_msk = []
        c = 0
        clx_index = self.idx_y[index]   # 并不一定0就是没有人1就是有人
        for name, sublist in enumerate(self.support_x_batch[index]):
            for item in sublist:
                if name == 0:
                    flatten_support_x.append(os.path.join(self.path, self.mode, item))
                    ind_path = item.split('.')
                    flatten_support_x_msk.append(os.path.join(self.path, self.mode + '_msk', ind_path[0] + '.png'))
                    support_y[c] = clx_index[0]  # 0
                    c = c+1
                else:
                    flatten_support_x.append(os.path.join(self.path, self.mode, item))
                    ind_path = item.split('.')
                    flatten_support_x_msk.append(os.path.join(self.path, self.mode + '_msk', ind_path[0] + '.png'))
                    support_y[c] = clx_index[1]  # 1
                    c = c + 1
        support_y.astype(np.int32)

        flatten_query_x = []
        flatten_query_x_msk = []
        c = 0
        for name, sublist in enumerate(self.query_x_batch[index]):
            for item in sublist:
                if name == 0:
                    flatten_query_x.append(os.path.join(self.path, self.mode, item))
                    ind_path = item.split('.')
                    flatten_query_x_msk.append(os.path.join(self.path, self.mode + '_msk' , ind_path[0] + '.png'))
                    query_y[c] = clx_index[0]  # 0
                    c = c + 1
                else:
                    flatten_query_x.append(os.path.join(self.path, self.mode, item))
                    ind_path = item.split('.')
                    flatten_query_x_msk.append(os.path.join(self.path, self.mode + '_msk', ind_path[0] + '.png'))
                    query_y[c] = clx_index[1]  # 1
                    c = c + 1
                query_y.astype(np.int32)

        for i, path in enumerate(flatten_support_x):
            # print("this is 2 {}".format(paddle.in_dynamic_mode()))
            img = Image.open(path).convert('RGB')
            paddle.disable_static()
            img = to_tensor1(img)
            img = normailize(img)
            # img = self.transform(path)

            img_msk = Image.open(flatten_support_x_msk[i]).convert('1')
            paddle.disable_static()
            img_msk = to_tensor1(img_msk)
            # print(img_msk)
            # print("this is 4 {}".format(paddle.in_dynamic_mode()))
            support_x[i] = paddle.concat(x=[img, img_msk], axis=0)
            # print("this is 5 {}".format(paddle.in_dynamic_mode()))

        for i, path in enumerate(flatten_query_x):
            # print("this is 6 {}".format(paddle.in_dynamic_mode()))
            img = Image.open(path).convert('RGB')
            paddle.disable_static()
            img = to_tensor1(img)
            img = normailize(img)

            img_msk = Image.open(flatten_query_x_msk[i]).convert('1')
            paddle.disable_static()
            img_msk = to_tensor1(img_msk)
            query_x[i] = paddle.concat(x=[img, img_msk], axis=0)
        # print("this is 7 {}".format(paddle.in_dynamic_mode()))
        a = paddle.to_tensor(support_y,dtype = 'int64')
        b = paddle.to_tensor(query_y,dtype = 'int64')
        # print("this is 8 {}".format(paddle.in_dynamic_mode()))
        return support_x, a, query_x, b

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

def getpath_onetime():
    data_train = []
    data_vali = []
    data_test = []

    path = 'local_scene_dataset'
    mode = 'test'
    txt_path = os.path.join(path, mode + '.txt')
    c = 0
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            linestrlist = line.split('\t')
            if c == 0:
                sce_idx_arr_tr = linestrlist
                c += 1
            elif c == 1:
                sob_num_arr_tr = linestrlist
                c += 1
            elif c == 2:
                nob_num_arr_tr = linestrlist

    data_test.append(sce_idx_arr_tr)  # 测试集
    data_test.append(sob_num_arr_tr)
    data_test.append(nob_num_arr_tr)

    mode = 'train'
    txt_path = os.path.join(path, mode + '.txt')
    c = 0
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            linestrlist = line.split('\t')
            if (c+3) % 3 == 0:
                sce_idx_arr_tr = linestrlist
                c += 1
                data_vali.append(np.array(sce_idx_arr_tr)[0:2].tolist())  # 验证集

                data_train.append(np.array(sce_idx_arr_tr)[2:20].tolist()) # 18个训练场景

            elif (c+2) % 3 == 1:
                sob_num_arr_tr = linestrlist
                c += 1
                data_vali.append(np.array(sob_num_arr_tr)[0:2].tolist())  # 验证集

                data_train.append(np.array(sob_num_arr_tr)[2:20].tolist())

            elif (c+1) % 3 == 2:
                nob_num_arr_tr = linestrlist
                c += 1
                data_vali.append(np.array(nob_num_arr_tr)[0:2].tolist())  # 验证集

                data_train.append(np.array(nob_num_arr_tr)[2:20].tolist())

    return data_train, data_test, data_vali
if __name__ == '__main__':
    data_train, all_test, all_vali = getpath_onetime()
    cur_data_train = data_train[0: (0 + 3)]
    mini = local_dataset_msk('local_scene_dataset',cur_data_train, mode='train', k_shot=5,
                             k_query=15, all_batchsz=1000, resize_h=480, resize_w=640,randnseed=222)
    print(mini)
