import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import h5py,time,sys
import torch.optim as optim
from module.GateMechanism import GateMechanism
from module.CrossShareUnit import CrossShareUnit
from module.LSTHM import LSTHM

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from matplotlib import colors

print(torch.__version__)

def load_saved_data():

    h5f = h5py.File('./data/X_train.h5', 'r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/y_train.h5', 'r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/X_valid.h5', 'r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/y_valid.h5', 'r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/X_test.h5', 'r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('./data/y_test.h5', 'r')
    y_test = h5f['data'][:]
    h5f.close()

    return X_train, y_train, X_valid, y_valid, X_test, y_test

class Map_pred(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(Map_pred, self).__init__()
        self.dropout = dropout
        self.n1 = nn.Linear(in_size, hidden_size)
        self.n2 = nn.Linear(hidden_size, 1)

    def forward(self,input):
        y1 = torch.relu(self.dropout(self.n1(input)))
        y2 = self.n2(y1)
        y2 = torch.reshape(y2, shape=[-1])
        return y2

class Net(nn.Module):
    def __init__(self,config):
        super(Net, self).__init__()
        [self.d_l, self.d_a, self.d_v] = config['input_dims']
        [self.dh_l, self.dh_a, self.dh_v] = config["h_dims"]
        self.z_dim = config["z_dim"]

        self.K = config["K"]
        self.max_len = config["max_len"]

        self.final_dims = config["final_dims"]
        self.dropout = nn.Dropout(config["dropout"])

        self.CatShape = config["cat_shape"]
        self.z_fc1_dim = config["z_fc1"]
        self.z_fc2_dim = config["z_fc2"]
        self.z_drop1 = config["z_drop1"]
        self.z_drop2 = config["z_drop2"]

        self.VisionGate = GateMechanism(self.d_v)
        self.AcousticGate = GateMechanism(self.d_a)

        self.wordLSTHM = LSTHM(self.d_l, self.dh_l, self.z_dim)
        self.covarepLSTHM = LSTHM(self.d_a, self.dh_a, self.z_dim)
        self.facetLSTHM = LSTHM(self.d_v, self.dh_v, self.z_dim)

        self.crossshareunit1 = CrossShareUnit(self.dh_l, self.dh_v, self.K, self.max_len)
        self.crossshareunit2 = CrossShareUnit(self.dh_l, self.dh_a, self.K, self.max_len)

        self.Z_Hat_fc1 = nn.Linear(self.CatShape, self.z_fc1_dim)
        self.Z_Hat_fc2 = nn.Linear(self.z_fc1_dim, self.z_dim)
        self.Z_Hat_dropout = nn.Dropout(self.z_drop1)

        self.gamma_fc1 = nn.Linear(self.CatShape, self.z_fc2_dim)
        self.gamma_fc2 = nn.Linear(self.z_fc2_dim, self.z_dim)
        self.gamma_dropout = nn.Dropout(self.z_drop2)

        self.pred_model = Map_pred(self.z_dim, self.final_dims, self.dropout)

        self.l1_att_list = []
        self.l2_att_list = []
        self.v_att_list = []
        self.a_att_list = []

    def forward(self, x, IsLastBatch):
        x_l = x[:, :, :self.d_l]  # 20 686 300
        x_a = x[:, :, self.d_l:self.d_a + self.d_l]  # 20 686 5
        x_v = x[:, :, self.d_a + self.d_l:]  # 20 686 20

        x_v_gate = self.VisionGate(x_v)
        x_a_gate = self.AcousticGate(x_a)

        self.c_v = torch.zeros(x_v.shape[0], self.dh_v).cuda()
        self.c_a = torch.zeros(x_a.shape[0], self.dh_a).cuda()
        self.c_l = torch.zeros(x_l.shape[0], self.dh_l).cuda()

        self.h_v = torch.zeros(x_v.shape[0], self.dh_v).cuda()
        self.h_a = torch.zeros(x_a.shape[0], self.dh_a).cuda()
        self.h_l = torch.zeros(x_l.shape[0], self.dh_l).cuda()

        self.z = torch.zeros(x_l.shape[0], self.z_dim).cuda()

        l1_att = torch.zeros(x_l.shape[0], self.max_len, self.dh_l).cuda()
        l2_att = torch.zeros(x_l.shape[0], self.max_len, self.dh_l).cuda()
        v_att = torch.zeros(x_v.shape[0], self.max_len, self.dh_v).cuda()
        a_att = torch.zeros(x_a.shape[0], self.max_len, self.dh_a).cuda()

        for t in range(0, x_l.shape[1]):# timestamp->each word
            # processing each modality with a LSTM Hybrid
            #input_x[batch_size_sentence,each_word,dim_word_vector]
            c_v, h_v = self.facetLSTHM(x_v_gate[:, t, :], self.c_v, self.h_v, self.z)
            c_a, h_a = self.covarepLSTHM(x_a_gate[:, t, :], self.c_a, self.h_a, self.z)
            c_l, h_l = self.wordLSTHM(x_l[:, t, :], self.c_l, self.h_l, self.z)

            h_l = torch.unsqueeze(h_l,dim=1)
            h_a = torch.unsqueeze(h_a, dim=1)
            h_v = torch.unsqueeze(h_v, dim=1)

            l_hidden1,v_hidden,l_att_vec1,v_att_vec = self.crossshareunit1(h_l,h_v)
            l_hidden2, a_hidden, l_att_vec2, a_att_vec = self.crossshareunit2(h_l, h_a)

            if IsLastBatch==True:
                if t==0:
                    l1_att = l_att_vec1
                    l2_att = l_att_vec2
                    v_att = v_att_vec
                    a_att = a_att_vec
                else:
                    l1_att = torch.cat([l1_att, l_att_vec1], dim=1)
                    l2_att = torch.cat([l2_att, l_att_vec2], dim=1)
                    v_att = torch.cat([v_att, v_att_vec], dim=1)
                    a_att = torch.cat([a_att, a_att_vec], dim=1)

            l_hidden = 0.7 * l_hidden1 + 0.3 * l_hidden2
            h_cat = torch.cat([l_hidden, v_hidden, a_hidden], dim=-1)
            h_cat = torch.squeeze(h_cat, dim=1)
            Z_Hat = F.tanh(self.Z_Hat_fc2(self.Z_Hat_dropout(self.Z_Hat_fc1(h_cat))))
            gamma = F.sigmoid(self.gamma_fc2(self.gamma_dropout(self.gamma_fc1(h_cat))))

            Z = gamma * Z_Hat

            self.c_v = c_v
            self.c_a = c_a
            self.c_l = c_l

            h_l = torch.squeeze(h_l,dim=1)
            h_a = torch.squeeze(h_a, dim=1)
            h_v = torch.squeeze(h_v, dim=1)

            self.h_v = h_v
            self.h_a = h_a
            self.h_l = h_l

            self.z = Z

        if IsLastBatch==True:
            self.l1_att_list.append(l1_att.cpu().data.numpy())
            self.l2_att_list.append(l2_att.cpu().data.numpy())
            self.v_att_list.append(v_att.cpu().data.numpy())
            self.a_att_list.append(a_att.cpu().data.numpy())

        y = self.pred_model(Z)

        return y

    def GetAttWeight(self):

        return self.l1_att_list, self.l2_att_list, self.v_att_list, self.a_att_list

def train_net(X_train, y_train, X_valid, y_valid, X_test, y_test, config):
    torch.manual_seed(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(config)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.L1Loss()
    criterion = criterion.to(device)

    def train(model, batchsize, X_train, y_train, optimizer, criterion):
        epoch_loss = 0
        model.train()
        total_n = X_train.shape[0]
        num_batches = int(total_n / batchsize) + 1

        for batch in range(num_batches):
            islastbatch = False
            if batch==(num_batches-1):
                islastbatch = True
            start = batch * batchsize
            end = (batch + 1) * batchsize
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train[start:end, :, :]).cuda()
            batch_y = torch.Tensor(y_train[start:end]).cuda()
            predictions = model.forward(batch_X, islastbatch)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid, y_valid, criterion, batchsize=64):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            total_n = X_valid.shape[0]
            num_batches = int(total_n / batchsize) + 1
            for batch in range(num_batches):
                start = batch * batchsize
                end = (batch + 1) * batchsize
                batch_X = torch.Tensor(X_valid[start:end,:, :]).cuda()
                batch_y = torch.Tensor(y_valid[start:end]).cuda()
                predictions = model.forward(batch_X, False)
                loss = criterion(predictions, batch_y)
                epoch_loss += loss.item()
        return epoch_loss / num_batches

    def predict(model, X_test, batchsize=64):
        batch_preds = []
        model.eval()
        with torch.no_grad():
            total_n = X_test.shape[0]
            num_batches = int(total_n / batchsize) + 1
            for batch in range(num_batches):
                start = batch * batchsize
                end = (batch + 1) * batchsize
                batch_X = torch.Tensor(X_test[start:end,:, :]).cuda()
                predictions = model.forward(batch_X, False)
                predictions = predictions.cpu().data.numpy()
                batch_preds.append(predictions)
            batch_preds = np.concatenate(batch_preds, axis=0)
        return batch_preds

    def get_attweight(model):

        l1_att_list, l2_att_list, v_att_list, a_att_list = model.GetAttWeight()

        return l1_att_list, l2_att_list, v_att_list, a_att_list

    # timing
    start_time = time.time()

    end_time = time.time()
    print(end_time - start_time)

    best_valid = 999999.0
    ## train and valid
    for epoch in range(config["num_epochs"]):
        train_loss = train(model,config["batchsize"],X_train, y_train, optimizer, criterion)
        valid_loss = evaluate(model, X_valid, y_valid, criterion)
        # scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')
            model_name = './model_save/mosi/mosi_model_{}.pt'.format(epoch)
            torch.save(model, model_name)
        else:
            print(epoch, train_loss, valid_loss)
            model_name = './model_save/mosi/mosi_model_{}.pt'.format(epoch)
            torch.save(model, model_name)

        # attention weight visualization method 1
        # l1_att_list, l2_att_list, v_att_list, a_att_list = get_attweight(model)
        # batch_l1_att = l1_att_list[0]
        # l1_att = np.transpose(batch_l1_att[-1])
        # plt.imshow(l1_att)  # show image
        # plt.colorbar()
        # figname = "./att_visualization/l1/l1_att_{}.jpg".format(epoch)
        # plt.savefig(figname)
        # plt.clf()
        #
        # batch_v_att = v_att_list[0]
        # v_att = np.transpose(batch_v_att[-1])
        # plt.imshow(v_att)  # show image
        # plt.colorbar()
        # figname = "./att_visualization/v/v_att_{}.jpg".format(epoch)
        # plt.savefig(figname)
        # plt.clf()
        #
        # batch_l2_att = l2_att_list[0]
        # l2_att = np.transpose(batch_l2_att[-1])
        # plt.imshow(l2_att)  # show image
        # plt.colorbar()
        # figname = "./att_visualization/l2/l2_att_{}.jpg".format(epoch)
        # plt.savefig(figname)
        # plt.clf()
        #
        # batch_a_att = a_att_list[0]
        # a_att = np.transpose(batch_a_att[-1])
        # plt.imshow(a_att)  # show image
        # plt.colorbar()
        # figname = "./att_visualization/a/a_att_{}.jpg".format(epoch)
        # plt.savefig(figname)
        # plt.clf()

        # attention weight visualization method 2
        # batch_l1_att = l1_att_list[0]
        # l1_att = np.transpose(batch_l1_att[-1])
        # batch_v_att = v_att_list[0]
        # v_att = np.transpose(batch_v_att[-1])
        #
        # vmin1 = min(np.min(l1_att), np.min(v_att))
        # vmax1 = max(np.max(l1_att), np.max(v_att))
        # norm = colors.Normalize(vmin=vmin1, vmax=vmax1)
        #
        # fig1 = plt.figure()
        # ax = fig1.add_subplot(121)
        # bx = fig1.add_subplot(122)
        # a = ax.pcolormesh(l1_att, norm=norm, cmap=plt.get_cmap('rainbow'))
        # b = bx.pcolormesh(v_att, norm=norm, cmap=plt.get_cmap('rainbow'))
        # fig1.colorbar(a, ax=[ax, bx], shrink=0.5)
        # figname1 = "./att_visualization/l1&v_att_{}.jpg".format(epoch)
        # plt.savefig(figname1)
        #
        #
        # batch_l2_att = l2_att_list[0]
        # l2_att = np.transpose(batch_l2_att[-1])
        # batch_a_att = a_att_list[0]
        # a_att = np.transpose(batch_a_att[-1])
        #
        # vmin2 = min(np.min(l2_att), np.min(a_att))
        # vmax2 = max(np.max(l2_att), np.max(a_att))
        # norm = colors.Normalize(vmin=vmin2, vmax=vmax2)
        #
        # fig2 = plt.figure()
        # cx = fig2.add_subplot(121)
        # dx = fig2.add_subplot(122)
        # c = cx.pcolormesh(l2_att, norm=norm, cmap=plt.get_cmap('rainbow'))
        # d = dx.pcolormesh(a_att, norm=norm, cmap=plt.get_cmap('rainbow'))
        # fig2.colorbar(c, ax=[cx, dx], shrink=0.5)
        # figname2 = "./att_visualization/l2&a_att_{}.jpg".format(epoch)
        # plt.savefig(figname2)

    model = torch.load('./model_save/mosi/mosi_model_50.pt', map_location='cuda')
    predictions = predict(model, X_test)
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)


    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    print("F1 ", f1_score(true_label, predicted_label, average='weighted'))
    print("MAE: ", np.mean(np.absolute(predictions - y_test)))
    print("Corr: ", np.corrcoef(predictions, y_test)[0][1])
    mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
    print("7-class: ", mult)
    print(config)
    sys.stdout.flush()

    ##test print result
    for i in range(50):
        epoch = i
        model_name = './model_save/mosi/mosi_model_{}.pt'.format(epoch)
        model = torch.load(model_name, map_location='cuda')
        predictions = predict(model, X_test)
        true_label = (y_test >= 0)
        predicted_label = (predictions >= 0)
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label, digits=5))
        print("Accuracy ", accuracy_score(true_label, predicted_label))
        print("F1 ", f1_score(true_label, predicted_label, average='weighted'))
        print("MAE: ", np.mean(np.absolute(predictions - y_test)))
        print("Corr: ", np.corrcoef(predictions, y_test)[0][1])
        mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
        print("7-class: ", mult)
        print(config)
        sys.stdout.flush()
        print("-------------------epoch{}-----------------".format(epoch))

if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data()

    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    print(X_test.shape)
    print(y_test.shape)

    config = dict()
    config["input_dims"] = [300, 5, 20]
    hl = 128
    ha = 64
    hv = 32

    config["h_dims"] = [hl, ha, hv]
    config["z_dim"] = 128
    config["final_dims"] = 128
    config["batchsize"] = 16
    config["num_epochs"] = 100
    config["lr"] = 0.00006
    config['dropout'] = 0.3
    config["K"] = 2
    config["max_len"] = 1 #tiemstamp =1

    config["cat_shape"] = hl + hv + ha
    config["z_fc1"] = 256
    config["z_fc2"] = 256
    config["z_drop1"] = 0.3
    config["z_drop2"] = 0.3

    train_net(X_train, y_train, X_valid, y_valid,  X_test, y_test, config)

