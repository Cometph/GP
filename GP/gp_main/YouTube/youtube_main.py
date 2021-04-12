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

print(torch.__version__)

def load_saved_data():
    h5f = h5py.File('./data/text_train_emb.h5', 'r')
    X_train_emb = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/video_train.h5', 'r')
    X_train_vedio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/audio_train.h5', 'r')
    X_train_audio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/y_train.h5', 'r')
    y_train_onehot = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/text_valid_emb.h5', 'r')
    X_valid_emb = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/video_valid.h5', 'r')
    X_valid_vedio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/audio_valid.h5', 'r')
    X_valid_audio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/y_valid.h5', 'r')
    y_valid_onehot = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/text_test_emb.h5', 'r')
    X_test_emb = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/video_test.h5', 'r')
    X_test_vedio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/audio_test.h5', 'r')
    X_test_audio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('./data/y_test.h5', 'r')
    y_test_onehot = h5f['d1'][:]
    h5f.close()

    print(X_train_audio.shape, X_train_vedio.shape, y_train_onehot.shape)

    y_train = np.argmax(y_train_onehot, axis=1)
    y_valid = np.argmax(y_valid_onehot, axis=1)
    y_test = np.argmax(y_test_onehot, axis=1)
    print(y_train[:10])
    return X_train_emb, X_train_vedio, X_train_audio, y_train, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, \
           X_test_emb, X_test_vedio, X_test_audio, y_test, y_train_onehot, y_valid_onehot, y_test_onehot

class Map_pred(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(Map_pred, self).__init__()
        self.dropout = dropout
        self.n1 = nn.Linear(in_size, hidden_size)
        self.n2 = nn.Linear(hidden_size, 3)

    def forward(self,input):
        y1 = torch.relu(self.dropout(self.n1(input)))
        y2 = self.n2(y1)

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

    def forward(self, x_l, x_v, x_a, IsLastBatch):

        x_v_gate = self.VisionGate(x_v)
        x_a_gate = self.AcousticGate(x_a)

        self.c_v = torch.zeros(x_v.shape[0], self.dh_v).cuda()
        self.c_a = torch.zeros(x_a.shape[0], self.dh_a).cuda()
        self.c_l = torch.zeros(x_l.shape[0], self.dh_l).cuda()

        self.h_v = torch.zeros(x_v.shape[0], self.dh_v).cuda()
        self.h_a = torch.zeros(x_a.shape[0], self.dh_a).cuda()
        self.h_l = torch.zeros(x_l.shape[0], self.dh_l).cuda()

        self.z = torch.zeros(x_l.shape[0], self.z_dim).cuda()

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

        y = self.pred_model(Z)

        return y

    def GetAttWeight(self):

        return self.l1_att_list, self.l2_att_list, self.v_att_list, self.a_att_list

def train_net(X_train_emb, X_train_vedio, X_train_audio, y_train, X_valid_emb, X_valid_vedio,
              X_valid_audio, y_valid, X_test_emb, X_test_vedio, X_test_audio, y_test, config):
    torch.manual_seed(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(config)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    def train(model, batchsize, X_train_emb, X_train_vedio, X_train_audio, y_train, optimizer, criterion):
        epoch_loss = 0
        model.train()
        total_n = X_train_emb.shape[0]
        num_batches = int(total_n / batchsize) + 1
        for batch in range(num_batches):
            start = batch * batchsize
            end = (batch + 1) * batchsize
            optimizer.zero_grad()
            batch_X_embed = torch.Tensor(X_train_emb[start:end, :, :]).cuda()
            batch_X_v = torch.Tensor(X_train_vedio[start:end, :, :]).cuda()
            batch_X_a = torch.Tensor(X_train_audio[start:end, :, :]).cuda()
            batch_y = torch.LongTensor(y_train[start:end]).cuda()
            predictions = model.forward(batch_X_embed, batch_X_v, batch_X_a)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, criterion):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X_embed = torch.Tensor(X_valid_emb).cuda()
            batch_X_v = torch.Tensor(X_valid_vedio).cuda()
            batch_X_a = torch.Tensor(X_valid_audio).cuda()
            batch_y = torch.LongTensor(y_valid).cuda()
            predictions = model.forward(batch_X_embed, batch_X_v, batch_X_a)
            loss = criterion(predictions, batch_y)
            epoch_loss += loss.item()
        return epoch_loss

    def predict(model,X_test_emb, X_test_vedio, X_test_audio, batchsize=64):
        model.eval()
        with torch.no_grad():
            batch_X_embed = torch.Tensor(X_test_emb).cuda()
            batch_X_v = torch.Tensor(X_test_vedio).cuda()
            batch_X_a = torch.Tensor(X_test_audio).cuda()
            predictions = model.forward(batch_X_embed, batch_X_v, batch_X_a)
            predictions = F.softmax(predictions, 1)
            predictions = predictions.cpu().data.numpy()

        return predictions

    # timing
    start_time = time.time()
    best_valid = 999999.0
    # # rand = random.randint(0, 100000)
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, config["batchsize"], X_train_emb, X_train_vedio, X_train_audio, y_train, optimizer, criterion)
        valid_loss = evaluate(model, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, criterion)
        # scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')
            model_name = './model_save/youtube/youtube_model_{}.pt'.format(epoch)
            torch.save(model, model_name)
        else:
            print(epoch, train_loss, valid_loss)
            model_name = './model_save/youtube/youtube_model_{}.pt'.format(epoch)
            torch.save(model, model_name)

    model = torch.load('./model_save/youtube/youtube_model_50.pt', map_location='cuda')
    predictions = predict(model,X_test_emb, X_test_vedio, X_test_audio)
    true_label = y_test
    predicted_label = np.argmax(predictions, axis=1)

    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    print("F1 ", f1_score(true_label, predicted_label, average='weighted'))
    print(config)
    sys.stdout.flush()
    end_time = time.time()
    print("train time:",end_time - start_time)

    #test print result
    for i in range(50):
        epoch = i
        model_name = './model_save/youtube/youtube_model_{}.pt'.format(epoch)
        model = torch.load(model_name, map_location='cuda')
        # print("111111111111111111")
        # for name, param in model.named_parameters():
        #     print(name, '      ', param.size())
        # print("111111111111111111")
        predictions = predict(model, X_test_emb, X_test_vedio, X_test_audio)
        true_label = y_test
        predicted_label = np.argmax(predictions, axis=1)
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label, digits=5))
        print("Accuracy ", accuracy_score(true_label, predicted_label))
        print("F1 ", f1_score(true_label, predicted_label, average='weighted'))
        print(config)
        sys.stdout.flush()
        print("-------------------epoch{}-----------------".format(epoch))

if __name__ == '__main__':
    X_train_emb, X_train_vedio, X_train_audio, y_train, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, X_test_emb, \
    X_test_vedio, X_test_audio, y_test, y_train_onehot, y_valid_onehot, y_test_onehot = load_saved_data()
    #train:15290 valid: 2291 test: 4832
    #language:300 vedio:35 acoustic:74
    print(X_train_emb.shape)
    print(y_train.shape)
    print(X_valid_emb.shape)
    print(y_valid.shape)
    print(X_test_emb.shape)
    print(y_test.shape)
    print(X_train_vedio.shape)
    print(X_train_audio.shape)

    config = dict()
    config["input_dims"] = [300, 74, 36]
    hl = 128
    ha = 64
    hv = 32

    config["h_dims"] = [hl, ha, hv]
    config["z_dim"] = 128
    config["final_dims"] = 128
    config["batchsize"] = 16
    config["num_epochs"] = 50
    config["lr"] = 0.001
    config['dropout'] = 0.3
    config["K"] = 2
    config["max_len"] = 1 #tiemstamp =1

    config["cat_shape"] = hl + hv + ha
    config["z_fc1"] = 256
    config["z_fc2"] = 256
    config["z_drop1"] = 0.3
    config["z_drop2"] = 0.3

    train_net(X_train_emb, X_train_vedio, X_train_audio, y_train, X_valid_emb, X_valid_vedio,
              X_valid_audio,y_valid, X_test_emb, X_test_vedio, X_test_audio, y_test, config)

