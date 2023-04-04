import paddle, os
import numpy as np
from local_dataset_ds_msk import local_dataset_msk
from paddle.io import DataLoader
import argparse
from meta import Meta

def train_onetime(metatrain_data, data_test, data_vali, train_idx, r):

    # train_idx 是某实验下第几次实验，对应不同的训练数据量
    paddle.seed(222)
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
        ('dropout', [0.1]),
        ('linear', [2,768])
    ]

    maml = Meta(args, config)
    tmp = filter(lambda x: not (x.stop_gradient), maml.net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(maml)
    print("----------------------------")
    print('Total trainable tensors:', num)
    tasks_batch = args.task_num * args.iternum
    localdata = local_dataset_msk('local_scene_dataset', metatrain_data, mode='train',
                                      k_shot=args.k_spt,
                                      k_query=args.k_qry, all_batchsz=tasks_batch, resize_h=480, resize_w=640,
                                      randnseed=222,shuffle_p=0.7)  # tasks_batch
    localdata_test = local_dataset_msk('local_scene_dataset', data_test, mode='train',
                                           k_shot=args.k_spt,
                                           k_query=args.k_qry, all_batchsz=50, resize_h=480, resize_w=640,
                                           randnseed=222,shuffle_p=0)  # 50
    localdata_vali = local_dataset_msk('local_scene_dataset', data_vali, mode='train',
                                           k_shot=args.k_spt,
                                           k_query=args.k_qry, all_batchsz=50, resize_h=480, resize_w=640,
                                           randnseed=222,shuffle_p=0)
    last_acc_va_final = 0
    num_cores = 0
    db = DataLoader(localdata, batch_size=args.task_num, shuffle=False, num_workers=num_cores)  #构成迭代器，按照batch_size取出任务
    db_vali = DataLoader(localdata_vali, batch_size=1, shuffle=False, num_workers=num_cores)
    db_test = DataLoader(localdata_test, batch_size=1, shuffle=False, num_workers=num_cores)
    accs_tr_em = []  # 用于累计训练准确率
    save_folder = 'maml_msk/'

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt = paddle.to_tensor(x_spt)
        y_spt = paddle.to_tensor(y_spt)
        x_qry = paddle.to_tensor(x_qry)
        y_qry = paddle.to_tensor(y_qry)

        accs_tr = maml.forward(x_spt, y_spt, x_qry, y_qry)   #accs_tr 是长度为训练中需更新的次数
        accs_tr_em.append(accs_tr)

        if (step + 1) % 10 == 0:
            acc_tr_em_mean = np.array(accs_tr_em).mean(axis=0).astype(np.float16)
            accs_tr_em = []


        if (step + 1) % 200 == 0:
            doc = open(save_folder + 'checkacc/tr_{}.txt'.format(train_idx), 'a+')
            doc.write('\nIter:%d training acc:' % (step))
            print("every 200 epoch,the train_acc is :")
            for i in range(len(acc_tr_em_mean)):
                doc.write(' %f ' % (acc_tr_em_mean[i]))
                print(acc_tr_em_mean[i])
            doc.close()

            accs_all_vali = []

            for x_spt, y_spt, x_qry, y_qry in db_vali:
                x_spt = x_spt.squeeze(0)
                y_spt = y_spt.squeeze(0)
                x_qry = x_qry.squeeze(0)
                y_qry = y_qry.squeeze(0)
                accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                accs_all_vali.append(accs)

            # [b, update_step+1]
            accs_va = np.array(accs_all_vali).mean(axis=0).astype(np.float16)  #五十个任务的验证上取平均
            doc = open(save_folder + 'checkacc/va_{}.txt'.format(train_idx), 'a+')
            print("after 200 epoch,train on 50 vali-task:")
            doc.write('\nIter:%d vali acc:' % (step))
            for i in range(len(accs_va)):
                doc.write(' %f ' % (accs_va[i]))
                print(accs_va[i])
            doc.close()

            acc_va_final = accs_va[-1]
            if acc_va_final > last_acc_va_final:
                last_accs_va = accs_va
                last_accs_ta = acc_tr_em_mean
                last_acc_va_final = acc_va_final
                saved_parameters = maml.net.parameters()
                paddle.save(saved_parameters,save_folder + 'checkpoints/maml_{}.pdparams'.format(train_idx))
                count_num = 0

            count_num += 1
            if count_num > 10:
                break
    doc = open(save_folder + 'out_train_{}.txt'.format(train_idx), 'a+')
    doc.write('\nMean train acc:')
    for i in range(len(last_accs_ta)):
        doc.write(' %f ' % (last_accs_ta[i]))
    doc.close()
    acc_tr_final = last_accs_ta[-1]
    print('Train acc:', last_accs_ta)

    # 验证集准确率
    print('Iter num:', step)
    doc = open(save_folder + 'out_vali_{}.txt'.format(train_idx), 'a+')
    doc.write('\nMean vali acc:')
    for i in range(len(last_accs_va)):
        doc.write(' %f ' % (last_accs_va[i]))
    doc.close()

    del maml

    maml = Meta(args,config)
    maml_params = paddle.load(save_folder + 'checkpoints/maml_{}.pdparams'.format(train_idx))
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
    acc_te_final = accs_te[-1]
    doc = open(save_folder + 'checkacc/te_{}.txt'.format(train_idx), 'a+')
    doc.write('\ntest acc:')
    for i in range(len(accs_te)):
        doc.write(' %f ' % (accs_te[i]))
    doc.close()
    return acc_tr_final, last_acc_va_final, acc_te_final
    # print('Test acc:', accs_te)
    # torch.save(maml, 'checkpoints/maml_model_{}.pt'.format(step))



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

    #os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--iternum', type=int, help='iteration number', default=60000)  # 10  000
    argparser.add_argument('--k_spt', type=int, help='k shot for support set per class', default=5)  # 5
    argparser.add_argument('--k_qry', type=int, help='k shot for query set per class', default=15)  # 15
    argparser.add_argument('--imgsz1', type=int, help='imgsz1', default=640)
    argparser.add_argument('--imgsz2', type=int, help='imgsz2', default=480)
    argparser.add_argument('--imgc', type=int, help='imgc', default=4)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=2)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)  #<<<<<<<<<<<<
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    paddle.device.set_device("gpu")
    print("now the environment is {}".format(paddle.device.get_device()))
    fold_idx = 10  # 最终确定的数据划分方式，即第二折，无需改动
    paddle.disable_static()
    data_train, all_test, all_vali = getpath_onetime()   #共10折
    count = 24  # 第二折 =2*3
    randnseed = [22, 222]  # 两种随机初始化种子
    ccc = len(randnseed)
    all_vali_acc = np.zeros([fold_idx])
    all_train_acc = np.zeros([fold_idx])
    all_test_acc = np.zeros([fold_idx])
    save_folder = 'maml_msk/'
    for j in range(8,fold_idx):
        cur_data_vali = all_vali[count:(count + 3)]
        cur_data_train = data_train[count:(count + 3)]
        all_train_acc[j], all_vali_acc[j], all_test_acc[j] = train_onetime(cur_data_train, all_test,
                                                                                        cur_data_vali, j, 222)
        count = count + 3

    doc = open(save_folder + 'all_train_testnum{}.txt'.format(fold_idx), 'a+')
    for j in range(fold_idx):
        doc.write(' %f ' % (all_train_acc[j]))
        doc.write('\n')
    doc.close()

    doc = open(save_folder + 'all_vali_testnum{}.txt'.format(fold_idx), 'a+')
    for j in range(fold_idx):
        doc.write(' %f ' % (all_vali_acc[j]))
        doc.write('\n')
    doc.close()

    doc = open(save_folder + 'all_test_testnum{}.txt'.format(fold_idx), 'a+')
    for j in range(fold_idx):
        doc.write(' %f ' % (all_test_acc[j]))
        doc.write('\n')
    doc.close()