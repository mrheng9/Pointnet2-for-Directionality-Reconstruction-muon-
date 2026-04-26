import os
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deepsphere import HealpyGCNN
from deepsphere import healpy_layers as hp_layer

font = {'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:3","/gpu:2","/gpu:1","/gpu:0"])

class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.98
        self.model.optimizer.lr.assign(new_lr)
        
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15,
    restore_best_weights=True,
    verbose=1 
    )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./checkpoint_theta",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

def distloss(y_true, y_pred):
    custom_loss = tf.sqrt((y_true[:,0]-y_pred[:,0])**2 + (y_true[:,1]-y_pred[:,1])**2 + (y_true[:,2]-y_pred[:,2])**2)
    return custom_loss

def draw_performance(x,y,title):
    plt.clf()
    plt.figure(figsize=(12,8))
    plt.grid()
    plt.scatter(x,y, label="predictions", color='red')
    plt.plot([np.amin(x), np.amax(x)], [np.amin(x), np.amax(x)], 'k-', alpha=0.75, zorder=0, label="y=x")
    plt.legend()
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.savefig("%s.png" % title, dpi = 300)

def get_stacked_data(_folder_path, _name):
    # _file_num = len([file for file in os.listdir(_folder_path) if re.match('y_', file)])
    _file_num = 2000
    _x = []
    for _i in range(_file_num):
        if os.path.exists('%s/%s_%i.npy' % (_folder_path, _name, _i)):
            _x.append(np.load('%s/%s_%i.npy' % (_folder_path, _name, _i)))
    _x = np.concatenate(_x)
    if _name == 'fht_pix':
        _x[_x > 1024] = 1024
        _x[_x <=0 ] = 1024

    print(_name + ' has been loaded')
    return _x

if __name__ == "__main__":

    output_dir = "experiments_vertex/test3"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nside = 32 # this number should be the same as the number of the nside in the data
    npix = hp.nside2npix(nside)
    print("n pixel: %d" % npix)
    indices = np.arange(hp.nside2npix(nside))
    feature_num=2
    folder_path_y='/disk_pool1/weijsh/e+/y'
    folder_path_x='/disk_pool1/weijsh/e+/det_feat'
    feature_list=['fht_pix','npe_pix']
    all_features=[]
    for feature in feature_list:
        all_features.append(get_stacked_data(folder_path_x,feature))
    x_all=np.stack(all_features,axis=-1)   
    y_all=get_stacked_data(folder_path_y,'y')

    print('x_all.shape = ', x_all.shape)
    print('y_all.shape = ', y_all.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all[:,6:9], test_size=0.2, random_state=42)
    
    del x_all
    del y_all

    output_true_path = os.path.join(output_dir, 'true_xyz.npy')
    np.save(output_true_path, y_test)

    layers = [hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyPool(p=1),
              hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyChebyshev(K=10, Fout=48, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyPool(p=1),
              hp_layer.HealpyChebyshev(K=10, Fout=48, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyPool(p=1),
              hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyChebyshev(K=10, Fout=6, use_bias=True, use_bn=True, activation="relu"),
              hp_layer.HealpyPool(p=1),
              hp_layer.HealpyChebyshev(K=10, Fout=3),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(3)]

    tf.keras.backend.clear_session()
    with strategy.scope():
        model = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=40)
        batch_size = 32
        model.build(input_shape=(None, len(indices), feature_num))
        model.summary(110)

        model.compile(
                optimizer=tf.keras.optimizers.Adam(0.002),
                loss=tf.keras.losses.MeanSquaredError(),
                # loss=distloss,
                metrics=[tf.keras.metrics.MeanSquaredError()]
            )

        history = model.fit(
                x=x_train,
                y=y_train,
                batch_size=batch_size,
                epochs=150,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[LearningRateReducerCb(),es_callback]
            )

        model.save_weights('vertex_weigths')

        plt.figure(figsize=(12,8))
        plt.clf()
        plt.plot(history.history["loss"], label="training")
        plt.plot(history.history["val_loss"], label="validation")
        plt.grid()
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.savefig("learning_curve_theta.png", dpi = 300)

    print(model.evaluate(x_train, y_train))
    print(model.evaluate(x_test, y_test))

    predictions = model.predict(x_test)
    output_pred_path = os.path.join(output_dir, 'pred_xyz.npy')
    np.save('output_pred_path',predictions)