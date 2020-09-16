from keras.utils import plot_model
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from  keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, UpSampling1D,\
    ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU,Lambda
from keras import backend as K
from keras.layers import Dense
from keras.regularizers import l2
import numpy as np
from keras import Input
from keras.models import Model
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.optimizers import Adam

a = 20  # 特征维度

#  数据读取 5数据 55标签b
x_train = np.loadtxt(open("CNN/xiudatas_train_de4_E.csv"),delimiter=",", skiprows=0)
y_train = np.loadtxt(open("CNN/xiulabel_train.csv"),delimiter=",", skiprows=0)
x_test = np.loadtxt(open("CNN/xiudatas_test_de4_E.csv"),delimiter=",", skiprows=0)
y_test = np.loadtxt(open("CNN/xiulabel_test.csv"),delimiter=",", skiprows=0)

label_train = y_train
label_test = y_test
y_train = np_utils.to_categorical(y_train, num_classes=2)    # 将整型的类别标签转为one-hot编码
y_test = np_utils.to_categorical(y_test, num_classes=2)

x_train = x_train.reshape(-1, a, 1)
x_test = x_test.reshape(-1, a, 1)

def Conv1d_BN(x, nb_filter,kernel_size, padding='same',strides=1,name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv1D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=2,name=bn_name)(x)

    return x
# CNN_inception块
def Inception(x,nb_filter):
    branch1x1 = Conv1d_BN(x, nb_filter, 1, padding='same',strides=1,name=None)

    branch3x3 = Conv1d_BN(x, nb_filter, 1, padding='same',strides=1,name=None)
    branch3x3 = Conv1d_BN(branch3x3, nb_filter, 3, padding='same',strides=1,name=None)

    branch5x5 = Conv1d_BN(x, nb_filter, 1, padding='same',strides=1,name=None)
    branch5x5 = Conv1d_BN(branch5x5, nb_filter, 1, padding='same',strides=1,name=None)

    branchpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    branchpool = Conv1d_BN(branchpool, nb_filter, 1, padding='same', strides=1, name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=2)
    return x  # 输出尺寸与输入一样 +pad

cell_size = 512 #隐藏层单元数
def LSTM_d(x):
    x=LSTM(units=cell_size,kernel_regularizer=l2(0.005))(x)
    return x


# GoogLeNet
inpt = Input(shape=(a, 1))
#padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
x = Conv1d_BN(inpt, 32, 7,strides=2, padding='same')
# x = MaxPooling1D(pool_size=3, strides=2,padding='same')(x)


x = Conv1d_BN(x, 64, 3, strides=1, padding='same')
# x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)

x = Inception(x, 8)#256
# x = LSTM_d(x)
# x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)

x = Dropout(0.5)(x)
# x = Inception(x,15)#480
# x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)
x = Inception(x, 16)#512
x = LSTM_d(x)
x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
# x = Inception(x,16)
# x = Inception(x,16)
# x = Inception(x,17)#528
# x = Inception(x,26)#832
# x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)
# x = Inception(x,16)
# x = Inception(x,32)#1024
# x = AveragePooling1D(pool_size=7,strides=7,padding='same')(x)
x = Dropout(0.5)(x)

x = Flatten()(x)#展开成一维
x = Dense(256, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
adam = Adam(lr=0.01) # alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）
# sgd = SGD(lr=0.01)
m=32
model = Sequential()
model = Model(inpt, x, name='inception')
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])     # 可改
#loss='categorical_crossentropy' 'binary_crossentropy'
#optimizer='sgd' 'adam' 'adagrad'
model.summary() # 通过model.summary()输出模型各层的参数状况

plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=False,dpi=100)
from keras.callbacks import ModelCheckpoint
#filepath = r'./MC/best_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
filepath = r'./MC/best_model.h5'
checkpoint = ModelCheckpoint(filepath,#(就是你准备存放最好模型的地方),
                             monitor='val_accuracy',#(或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
                             verbose=1,#(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
                             save_best_only='True',#(只保存最好的模型,也可以都保存),
                             save_weights_only='True',
                             mode='max',#(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                             period=1)#(checkpoints之间间隔的epoch数)

# lrreduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)  # 回调函数

import time
fit_start = time.perf_counter()
history = model.fit(x_train, y_train, batch_size=m, epochs=200,   verbose=2, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint])
fit_end = time.perf_counter()

print("train time is: ", fit_end-fit_start)
#model.save('model.h5')
#model.load_weights('model.h5')
model.load_weights(r'./MC/best_model.h5')
t_start = time.perf_counter()
# 训练集测试集结果
predict_train = model.predict(x_train)
predict_train = np.argmax(predict_train,axis=1)
predict_test = model.predict(x_test)
predict_test = np.argmax(predict_test,axis=1)
t_end = time.perf_counter()
# 混淆矩阵
def length(x,y):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y)):
        if x[i] == 1 and y[i] == 1:
            TP = TP+1
            continue
        else:
            if x[i] == 1 and y[i] == 0:
                FP = FP+1
                continue
            else:
                if x[i] == 0 and y[i] == 0:
                    TN = TN+1
                    continue
                else:
                    FN = FN+1
                    continue
    return TP,FP,TN,FN
train = length(predict_train,label_train)
test = length(predict_test,label_test)
# 精确率
train_p = train[0]/(train[0]+train[1])
test_p = test[0]/(test[0]+test[1])
# 敏感度
train_sen = train[0]/(train[0]+train[3])
test_sen = test[0]/(test[0]+test[3])
# 特异性
train_spe = train[2]/(train[1]+train[2])
test_spe = test[2]/(test[1]+test[2])
# loss&acc
loss_train,acc_train = model.evaluate(x_train, y_train, verbose=2)
loss_test, acc_test = model.evaluate(x_test, y_test, verbose=2)

##保存CNN结果
# lstm_train=np.transpose([predict_train])
# lstm_test=np.transpose([predict_test])
# np.save('E:/研究生论文_cff/20.7.19ADSNP_plos one_三区/20.4.29ADSNP_my paper_三区/程序/python_cnn&tu/SNP/五五五折/CNN/label_traincnn_snpe.npy',lstm_train)
# np.save('E:/研究生论文_cff/20.7.19ADSNP_plos one_三区/20.4.29ADSNP_my paper_三区/程序/python_cnn&tu/SNP/五五五折/CNN/label_testcnn_snpe.npy',lstm_test)

# ROC AUC
fpr_te,tpr_te,threshold_te = roc_curve(label_test, predict_test) ###计算真正率和假正率
roc_auc_te = auc(fpr_te,tpr_te) ###计算auc的值
fpr_tr,tpr_tr,threshold_tr = roc_curve(label_train, predict_train) ###计算真正率和假正率
roc_auc_tr = auc(fpr_tr,tpr_tr) ###计算auc的值

SVM_train = np.loadtxt(open("五五五折/CNN/label_train_snpe.csv"),delimiter=",", skiprows=0)
SVM_test = np.loadtxt(open("五五五折/CNN/label_test_snpe.csv"),delimiter=",", skiprows=0)
fpr_tesvm,tpr_tesvm,threshold_tesvm = roc_curve(label_test, SVM_test) ###计算真正率和假正率
roc_auc_tesvm = auc(fpr_tesvm,tpr_tesvm) ###计算auc的值
fpr_trsvm,tpr_trsvm,threshold_trsvm = roc_curve(label_train, SVM_train) ###计算真正率和假正率
roc_auc_trsvm = auc(fpr_trsvm,tpr_trsvm) ###计算auc的值

CNN_train = np.load("五五五折/CNN/label_traincnn_snpe.npy")
CNN_test = np.load("五五五折/CNN/label_testcnn_snpe.npy")
fpr_tecnn,tpr_tecnn,threshold_tecnn = roc_curve(label_test, CNN_test) ###计算真正率和假正率
roc_auc_tecnn = auc(fpr_tecnn,tpr_tecnn) ###计算auc的值
fpr_trcnn,tpr_trcnn,threshold_trcnn = roc_curve(label_train, CNN_train) ###计算真正率和假正率
roc_auc_trcnn = auc(fpr_trcnn,tpr_trcnn) ###计算auc的值



font={'family':'vani bold',

      # 'color':'red',
      'size':17
}
p=15

plt.figure()
plt.figure()
plt.figure()
lw = 4
plt.figure(figsize=(7,5))

plt.plot(fpr_tesvm, tpr_tesvm,
         lw=lw, label='ROC_SVM  (AUC = %0.2f)' % roc_auc_tesvm)
plt.plot(fpr_tecnn, tpr_tecnn,
         lw=lw, label='ROC_CNN  (AUC = %0.2f)' % roc_auc_tecnn)
plt.plot(fpr_te, tpr_te, color='darkorange',
         lw=lw, label='ROC_CNN-LSTM  (AUC = %0.2f)' % roc_auc_te)

y=[0,0]
x=[0,1]
z=[1,1]

plt.plot(x,y,'k',lw=2)
plt.plot(y,x,'k',lw=2)
# plt.plot(x,z,'k',lw=2)
# plt.plot(z,x,'k',lw=2)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1])
plt.ylim([0.0, 1])
plt.tick_params(axis='both',labelsize=p,size=12)
plt.ylabel('True Positive Rate',fontdict=font)
plt.xlabel('False Positive Rate',fontdict=font)
plt.subplots_adjust(left = 0.15,bottom = 0.128)
plt.legend(loc="lower right",prop={'size': 12})
plt.grid(False)
plt.savefig('./auc.png',bbox_inches = 'tight')
plt.show()

print('Train loss :', loss_train)
print('Train accuracy :', acc_train)
print('Train Precision :', train_p)
print('Train Sensitivity :', train_sen)
print('Train Specificity :', train_spe)
print('Train AUU :', roc_auc_tr)
print('Test loss :', loss_test)
print('Test accuracy :', acc_test)
print('Test Precision :', test_p)
print('Test Sensitivity :', test_sen)
print('Test Specificity :', test_spe)
print("test time is: ", t_end-t_start)
print('Train AUU :', roc_auc_te)


def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(history, 'accuracy', 'val_accuracy')
show_train_history(history, 'loss', 'val_loss')
scores = model.evaluate(x_test, y_test)
# 模型预测,输入测试集,输出预测结果
scores[1]