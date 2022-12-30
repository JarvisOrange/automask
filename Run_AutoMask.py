

import os
import time
import seaborn as sns
import numpy as np
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
from ray.tune.suggest import ConcurrencyLimiter
import json
import torch
import random
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
from AutoMask import AutoMask
import torch.nn as nn
import matplotlib.pyplot as plt

def node_train(node):
    class TrainedModel():
        def __init__(self):
            super(TrainedModel, self).__init__()

        def run_model(self, task=None, model_name=None, dataset_name=None, config_file=None,
                      saved_model=True, train=True, other_args=None):

            config = ConfigParser(task, model_name, dataset_name,
                                  config_file, saved_model, train, other_args)
            print(list(config))
            exp_id = config.get('exp_id', None)
            if exp_id is None:
                # Make a new experiment ID
                exp_id = int(random.SystemRandom().random() * 100000)
                config['exp_id'] = exp_id
            # logger
            logger = get_logger(config)
            logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                        format(str(task), str(model_name), str(dataset_name), str(exp_id)))
            logger.info(config.config)
            # seed
            seed = config.get('seed', 0)
            set_random_seed(seed)
            # 加载数据集
            dataset = get_dataset(config)

            # 转换数据，并划分数据集
            self.train_data, self.valid_data, self.test_data = dataset.get_data()

            data_feature = dataset.get_data_feature()
            adj_mx = data_feature['adj_mx']

            # 加载执行器
            model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
                exp_id, model_name, dataset_name)
            model = get_model(config, data_feature)



            executor = get_executor(config, model, data_feature)
            # 训练

            if train or not os.path.exists(model_cache_file):
                executor.train(train_dataloader, valid_dataloader)
                if saved_model:
                    executor.save_model(model_cache_file)
            else:
                self.model = executor.load_model(model_cache_file)

            # 评估，评估结果将会放在 cache/evaluate_cache 下
            # executor.evaluate(test_data)

        def get_model(self):
            return self.model

        def get_dataloader(self):
            return self.train_data, self.valid_data, self.test_data



    TrainedModel = TrainedModel()
    TrainedModel.run_model(task='traffic_state_pred', model_name = 'STGCN', dataset_name='METR_LA', train=False)
    model = TrainedModel.get_model()
    automask = AutoMask(TrainedModel.get_model(), method = 0)
    if torch.cuda.is_available():
       automask = automask.cuda()

    train_dataloader, valid_dataloader, test_dataloader = TrainedModel.get_dataloader()


    print("start training")
    min_val_loss = float('inf')
    wait = 0
    best_epoch = 0
    train_time = []
    eval_time = []
    device = 'cuda'
    num_batches = len(train_dataloader)
    epoch = 10
    node = node
    train_time = []
    optimizer = torch.optim.SGD(automask.maskModel.parameters(), lr=0.1)
    mae_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    class myLoss(nn.Module):
        def __init__(self, model):
            super(myLoss, self).__init__()
            self.model = model

        def forward(self, x1, x2, y):
            loss1 = mae_loss(x1[:,:,node,:], x2[:,:,node,:])
            return loss1
            loss2 = mae_loss(x1, y)
            return torch.add(loss1, loss2)

    loss_fn = myLoss(model)
    for i in range(epoch):
        print("-------epoch {}".format(i+1))

        losses = []
        losses_ = []
        LL1 = []
        LL1_= []
        for step, batch in enumerate(train_dataloader):
            batch.to_tensor(device='cuda')
            optimizer.zero_grad()
            with torch.no_grad():
                output_no_mask = model(batch)
            output_mask = automask(batch)

            y = torch.tensor(batch['y'])
            y = torch.tensor(y[:, 0:1, :, :], device='cuda', dtype=torch.float)

            loss = loss_fn(output_mask, output_no_mask, y)
            L1_ = l1(output_no_mask, y)
            L1 = l1(output_mask,y)
            LL1.append(L1.item())
            LL1_.append(L1_.item())
            print(" train epoch{} {} no mask{} mask{}".format(i+1, step, L1_.item(), L1.item()))
            print("train epoch {} batch {} loss {}".format(i+1,step,loss.item()))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()




        print('training epoch{} loss no mask{} mask{}'.format(i+1, np.mean(losses_), np.mean(losses) ))
        # print('training epoch{} loss no mask{} mask{}'.format(i+1, np.mean(LL1_), np.mean(LL1) ))
        print("epoch complete!")
        print("evaluating now!")
        automask.eval()
        losses_eval = []
        losses1 = []
        losses2 = []

        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                batch.to_tensor(device='cuda')
                with torch.no_grad():
                    output_no_mask_test = model(batch)
                output_mask_test = automask(batch)
                y_test = torch.tensor(batch['y'])
                y_test = torch.tensor(y_test[:, 0:1, :, :], device='cuda', dtype=torch.float)
                loss_test = loss_fn(output_mask_test, output_no_mask_test, y_test)
                print("eval epoch {} batch {} loss {}".format( i + 1, step, loss_test.item()))
                losses_eval.append(loss_test.item())

                l1_test = l1(output_mask_test, output_no_mask_test)
                l2_test = l2(output_mask_test, y_test)
                # losses1.append(l1_test.item())
                # losses2.append(l2_test.item())
        print("eval  epoch {} loss {} ".format(i+1, np.mean(losses_eval)))
        # print(list(automask.maskModel.parameters()))
    index = 0
    for valid in valid_dataloader:
        input = torch.tensor(valid['X'], device='cuda',dtype=torch.float)
        index = index + 1
        if index == 7:
            break

    fig = plt.figure(1, figsize=(160, 140))
    for i in range(8):

        plt.subplot(8, 1, i + 1)
        mask = automask.maskModel.get_mask(input[i].unsqueeze(dim=0))

        mask = mask.squeeze(dim = 0)
        mask = mask.squeeze(dim=2)
        mask = mask.cpu()
        mask = mask.detach().numpy()
        ax = sns.heatmap(data=mask, linecolor='grey')
        dirname = "pic"+str(node)
        if os.path.exists(dirname):
            pass
        else:
            os.mkdir(dirname)
        plt.plot()

    # plt.savefig("./pic" + str(node) + "/" + str(i) + "_unit.jpg")
    plt.savefig("./pic" + str(node) + "/" + "_unit.jpg")
    plt.show()




