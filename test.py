import paddle, os
import numpy as np
from local_dataset_msk import local_dataset_msk
from paddle.io import DataLoader
import argparse
# from maml import MetaLearner
from meta import Meta

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


paddle.seed(222)
argparser = argparse.ArgumentParser()
argparser.add_argument('--iternum', type=int, help='iteration number', default=20000)  # 10  000
argparser.add_argument('--k_spt', type=int, help='k shot for support set per class', default=2)  # 5
argparser.add_argument('--k_qry', type=int, help='k shot for query set per class', default=2)  # 15
argparser.add_argument('--imgsz1', type=int, help='imgsz1', default=640)
argparser.add_argument('--imgsz2', type=int, help='imgsz2', default=480)
argparser.add_argument('--imgc', type=int, help='imgc', default=4)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=2)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)  # <<<<<<<<<<<<
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

args = argparser.parse_args()
paddle.device.set_device("gpu")
    # np.random.seed(222)
print(args)
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
    ('linear', [2,768])
]

maml = Meta(args, config)
tmp = filter(lambda x: not (x.stop_gradient), maml.net.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))
# print(maml)
print("----------------------------")
print('Total trainable tensors:', num)
tasks_batch = args.task_num * args.iternum
data_train, all_test, all_vali = getpath_onetime()
cur_data_vali = all_vali[0:(0 + 3)]
cur_data_train = data_train[0: (0 + 3)]
localdata = local_dataset_msk('local_scene_dataset', cur_data_train, mode='train',
                                  k_shot=args.k_spt,
                                  k_query=args.k_qry, all_batchsz=tasks_batch, resize_h=480, resize_w=640,
                                  randnseed=222)  # tasks_batch
localdata_test = local_dataset_msk('local_scene_dataset', all_test, mode='train',
                                       k_shot=args.k_spt,
                                       k_query=args.k_qry, all_batchsz=50, resize_h=480, resize_w=640,
                                       randnseed=222)  # 50
localdata_vali = local_dataset_msk('local_scene_dataset', cur_data_vali, mode='train',
                                       k_shot=args.k_spt,
                                       k_query=args.k_qry, all_batchsz=50, resize_h=480, resize_w=640,
                                       randnseed=222)

db_test = DataLoader(localdata_test, batch_size=1, shuffle=False, num_workers=0)
last_acc_va_final = 0
num_cores = 0

accs_tr_em = []  # 用于累计训练准确率
save_folder = 'maml_dif_shuffle_msk/'

saved_parameters  = maml.net.parameters()
paddle.save(saved_parameters,save_folder + 'checkpoints/maml_{}.pdparams'.format(1))

maml_params = paddle.load(save_folder + 'checkpoints/maml_{}.pdparams'.format(0))
maml.net.vars = [paddle.to_tensor(item, stop_gradient=False) for item in maml_params]
print("look at this:----------------------------------------")
tmp = filter(lambda x: not (x.stop_gradient), maml.net.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))
print('Total trainable tensors:', num)
print("over-----------------------------------------------------")
accs_all_test = []
for x_spt, y_spt, x_qry, y_qry in db_test:
    x_spt = x_spt.squeeze(0)
    y_spt = y_spt.squeeze(0)
    x_qry = x_qry.squeeze(0)
    y_qry = y_qry.squeeze(0)
    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
    accs_all_test.append(accs)

accs_te = np.array(accs_all_test).mean(axis=0).astype(np.float16)
for i in range(len(accs_te)):
    print(accs_te[i])


