from pyexpat import model
from sqlite3 import Time
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#import keras
import tensorflow as tf
from tensorflow import keras
#from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
#from keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dropout
from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
import pandas as pd
from keras.regularizers import l1,l2
import warnings
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import datetime





warnings.filterwarnings('ignore')

## loading the time series dataset
def load_dataset (path):

    df = pd.read_csv(path)
    pd.set_option('display.max_columns', None)
    time = df['(PDH-CSV 4.0) (Coordinated Universal Time)(0)']
    cpu = df['\\\\zneudl1p52itb8\\processor(_total)\\% processor time']
    ram = df['\\\\zneudl1p52itb8\\memory\\% committed bytes in use']
    disk_bytes = df['\\\\zneudl1p52itb8\\logicaldisk(c:)\\disk write bytes/sec']
    disk_sec = df['\\\\zneudl1p52itb8\\logicaldisk(c:)\\avg. disk sec/write']
    df = df.replace(r'^\s*$', np.nan, regex=True)
    d0 = {'cpu':cpu}
    CPU=pd.DataFrame(data=d0)
    d1 = {'disk_sec':disk_sec}
    DISK_SEC=pd.DataFrame(data=d1)
    d2 = {'disk_bytes':disk_bytes}
    DISK_BYTES=pd.DataFrame(data=d2)
    d3 = {'ram':ram}
    RAM=pd.DataFrame(data=d3)
    df=pd.concat([CPU,DISK_SEC,DISK_BYTES,RAM],axis=1)

    data= np.concatenate([CPU,DISK_SEC,DISK_BYTES,RAM],axis=1)
    return  data

"""
path = 'logs.csv'
data = load_dataset(path)
"""



### windowing
def windowing(data, window_size):
    #window_size = 50
    windowed_data_size = round(data.shape[0]/50) -1
    variable_size = data.shape[1]
    windowed_data = np.zeros((windowed_data_size,window_size,variable_size))
    for i in range(windowed_data_size):

        sample = data[i*window_size:(i*window_size)+window_size,:]
        windowed_data[i,:,:] = sample
    return  windowed_data, windowed_data_size,variable_size

"""
window_size =50
[windowed_data,windowed_data_size,variable_size] = windowing(window_size)
"""


## minmax scale
from sklearn.preprocessing import MinMaxScaler

def scaling(windowed_data, scalling_type,windowed_data_size,window_size):

    if scalling_type =="minmax":
        scaler = MinMaxScaler()

    scaled_data = np.zeros((windowed_data_size,window_size,4))
    for i in range(windowed_data_size):
        sample = windowed_data[i,:,:]
        scaled_data_sample = scaler.fit_transform(sample)
        scaled_data[i,:,:] = scaled_data_sample
    return scaled_data

"""
scalling_type = "minmax"
scaled_data = scaling(windowed_data, scalling_type,windowed_data_size,window_size )

"""

### Generate training and testing
from sklearn.model_selection import train_test_split

def generate_trainingTesting (scaled_data, split_percentage, variable_size):
    Xx_train, Xx_test = train_test_split(scaled_data, test_size=split_percentage)
    Xxx_train = Xx_train[:, :, 0]
    Xxx_test = Xx_test[:, :, 0]
    for i in range(1,variable_size):
        #Xxx_train = np.concatenate([Xx_train[:, :, i], Xx_train[:, :, 1], Xx_train[:, :, 2], Xx_train[:, :, 3]], axis=1)
        Xxx_train = np.concatenate([Xxx_train, Xx_train[:, :, i]], axis=1)
        Xxx_test = np.concatenate([Xxx_test, Xx_test[:, :, i]], axis=1)

    Xx_train = Xxx_train.reshape(Xxx_train.shape[0], 1, scaled_data.shape[1]*variable_size)
    Xx_test = Xxx_test.reshape(Xxx_test.shape[0], 1, scaled_data.shape[1]*variable_size)
    return Xx_train, Xx_test

"""
split_percentage = 0.33
[Xx_train, Xx_test] = generate_trainingTesting (scaled_data,split_percentage, variable_size)
"""
class NGmodel_1(Model):
    
    def __init__(self):
        super(NGmodel_1, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu")])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(2, activation="relu"),
            layers.Dense(4, activation="sigmoid")])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def window_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
        
    return np.array(Xs), np.array(ys)

def datetime_to_float(d):
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

def datetime_to_float(d):
    return d.timestamp()



# def create_model_1(input_dim,
#                     timesteps,
#                     batch_size,
#                     intermediate_dim,
#                     latent_dim,
#                     epsilon_std = 1.,
#                     regularisation_used = 0,
#                     drop_out = 0):
    
    
    # FILE_PATH = "C:/Users/rsers/Documents/Python Environments/Data Reformatter/project/logs.csv"

    # df = pd.read_csv(FILE_PATH, names=('Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5'))

    # # Remove previous column titles and renindex
    # df = df[1:]
    # df = df.reset_index(drop=True)

    # # Rename columns for practice
    # df = df.rename(columns={'Data_1':'Date','Data_2':'Disk_bytes/sec','Data_3':'Disk_sec/write',
    #                 'Data_4':'Disk_bytes_in_use','Data_5':'Processor_Time'})

    
    # df[['Disk_bytes/sec','Disk_sec/write','Disk_bytes_in_use','Processor_Time']] = df[['Disk_bytes/sec','Disk_sec/write','Disk_bytes_in_use','Processor_Time']].astype('float')

    # df['Date'] = df['Date'].str.replace('/', '')
    # df['Date'] = df['Date'].str.replace(':', '')
    # df['Date'] = df['Date'].str.replace(' ', '')
    # df['Date'] = df['Date'].astype(float)
    
    # mask = np.random.choice([True, False], len(df),p=[0.8, 0.2])

    # train = df[mask]
    # test = df[~mask]

    # scaler = StandardScaler()
    # scaler = scaler.fit(train[['Disk_bytes/sec','Disk_sec/write','Disk_bytes_in_use','Processor_Time']])
    # x_train = pd.DataFrame(scaler.transform(train[['Disk_bytes/sec','Disk_sec/write','Disk_bytes_in_use','Processor_Time']]),columns=['Disk_bytes/sec','Disk_sec/write',
    #                'Disk_bytes_in_use','Processor_Time'])
    # x_test = pd.DataFrame(scaler.transform(test[['Disk_bytes/sec','Disk_sec/write','Disk_bytes_in_use','Processor_Time']]),columns=['Disk_bytes/sec','Disk_sec/write',
    #                'Disk_bytes_in_use','Processor_Time'])

    # TIME_STEPS = 50
    # X_train, y_train = window_dataset(x_train, train[['Disk_bytes/sec','Disk_sec/write','Disk_bytes_in_use','Processor_Time']], TIME_STEPS)
    # X_test, y_test = window_dataset(x_test, test[['Disk_bytes/sec','Disk_sec/write','Disk_bytes_in_use','Processor_Time']], TIME_STEPS)

    
    
    #x = Input(shape=(X_train.shape[1], X_train.shape[2]))
    #print(x)

   
    
    


### Define VAE Model
def create_lstm_vae(input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std = 1.,
                    regularisation_used = 0,
                    drop_out = 0
                    ):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.

    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma
    """
    x = Input(shape=(timesteps, input_dim,))

    print(x)
    # LSTM encoding
    h = LSTM(intermediate_dim)(x)
    

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    if regularisation_used == 1:

        z_mean = Dense(latent_dim,activity_regularizer=l2(0.01))(h)
        z_log_sigma = Dense(latent_dim, activity_regularizer=l2(0.01))(h)
    if drop_out > 0:

        z_mean = Dropout(drop_out)(z_mean)
        z_log_sigma = Dropout(drop_out)(z_log_sigma)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])


    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)
    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_log_sigma)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    def nll(y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    #vae.compile(optimizer=Adam(lr=0.001), loss=nll)
    vae.compile(optimizer=Adam(lr=0.001), loss=vae_loss,  experimental_run_tf_function = False)

    #vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator



 ## Hyperparameters
"""
input_dim = Xx_train.shape[-1] # 13
timesteps = Xx_train.shape[1] # 3
batch_size = 1
intermediate_dim = 250
latent_dim=100
epsilon_std=1
regularisation_used = 0
drop_out = 0.4

vae, enc, gen = create_lstm_vae(
    input_dim = Xx_train.shape[-1],
    timesteps=timesteps,
    batch_size=batch_size,
    intermediate_dim=intermediate_dim,
    latent_dim=latent_dim,
    epsilon_std=epsilon_std,
    regularisation_used=regularisation_used,
    drop_out=drop_out
    )
"""



## Define training method
from sklearn.model_selection import KFold

def Training_routine (vae, Xx_train, Xx_test, training_type=0, epochs=100, num_folds = 10, trainig_round = 3, batch_size=1):

    """
    Choose triaing type for the LSTM Variational Autoencoder (VAE). Returns Hisotry of training and testing validation matrix

    # Arguments
        Xx_train: training samples ( time series )
        Xx_test: testing samples ( time series )
        epochs: number of optimisation iterations
        training_type: int.[ 0 for no k-fold, 1 for k-fold]
        num_folds: For only training_type, define number of k-folds
        trainig_round: For only training_type, define number of k-folds iterations

    """
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print(Xx_train.shape)
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    
    if training_type == 0:
        histroy_sample = vae.fit(Xx_train, Xx_train, validation_data=(Xx_test, Xx_test), epochs=epochs)
    else:

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # K-fold Cross Validation model evaluation
        fold_no = 0
        inputs = np.concatenate((Xx_train, Xx_test), axis=0)
        targets = np.concatenate((Xx_train, Xx_test), axis=0)
        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        for trainig_round_session in range(trainig_round):
            for train, test in kfold.split(inputs, targets):
                # Generate a print
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no} ...')

                # Fit data to model
                histroy_sample = vae.fit(inputs[train], targets[train],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1)

                # Generate generalization metrics
                scores = vae.evaluate(inputs[test], targets[test], verbose=0)
                print('Score for fold', fold_no, scores)
                # acc_per_fold.append(scores)
                loss_per_fold.append(scores)

                # Increase fold number
                fold_no = fold_no + 1

    return histroy_sample
"""
training_type = 0
epochs = 1
num_folds = 10
trainig_round = 3
history_model = Training_routine (vae, Xx_train, Xx_test, training_type, epochs, num_folds , trainig_round , batch_size)
"""



### main



def Training_run( Xx_train,Xx_test,input_dim, timesteps, intermediate_dim,batch_size, latent_dim,
                  epsilon_std, regularisation_used, drop_out,training_type,epochs, num_folds):

    print('Start train')
    model_1  = create_model_1(input_dim=input_dim,
        timesteps=timesteps,
        batch_size=batch_size,
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        epsilon_std=epsilon_std,
        regularisation_used=regularisation_used,
        drop_out=drop_out)
    
    print('Start run')
    #vae.fit(Xx_train, Xx_train, validation_data=(Xx_test, Xx_test), epochs=epochs)
    model_1.fit(Xx_train, Xx_train, validation_data=(Xx_test, Xx_test), epochs=10)
    
    print('Finished')
    
    
    vae, enc, gen = create_lstm_vae(
        input_dim=input_dim,
        timesteps=timesteps,
        batch_size=batch_size,
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        epsilon_std=epsilon_std,
        regularisation_used=regularisation_used,
        drop_out=drop_out
    )

    history_model = Training_routine(vae, Xx_train, Xx_test, training_type=training_type, epochs=epochs, num_folds=num_folds, trainig_round=3, batch_size=1)
    return history_model, vae


def main(argv):

    ## Global variables
    path = 'project/logs.csv'
    #path = 'C:/Users/rsers/Documents/Python Environments/Data Reformatter/project/logs.csv'
    window_size = 50
    scalling_type = "minmax"
    split_percentage = 0.33
    input_dim = 200
    timesteps = 1
    batch_size = 1

    data = load_dataset(path)
    [windowed_data, windowed_data_size, variable_size] = windowing(data, window_size)
    scaled_data = scaling(windowed_data, scalling_type, windowed_data_size, window_size)
    [Xx_train, Xx_test] = generate_trainingTesting(scaled_data, split_percentage, variable_size)

    import json
    import json
    import sys
    file_path = sys.argv[1]#"data.json"
    #file_path = 'C:/Users/rsers/Documents/Python Environments/Data Reformatter/project/data.json'

    with open(file_path, 'r') as j:
        space = json.loads(j.read())
        print(space)

    for model in range(int(sys.argv[2])):
        ## choose random hyperspace parameters
        import random
        intermediate_dim = int(random.choice(list(space['intermediate_dim'][0].values())))
        latent_dim = int(random.choice(list(space['latent_dim'][0].values())))
        epsilon_std = int(random.choice(list(space['epsilon_std'][0].values())))
        regularisation_used = int(random.choice(list(space['regularisation_used'][0].values())))
        drop_out = float(random.choice(list(space['drop_out'][0].values())))
        training_type = int(random.choice(list(space['training_type'][0].values())))
        epochs = int(random.choice(list(space['epochs'][0].values())))
        num_folds = int(random.choice(list(space['num_folds'][0].values())))
        print('============================================================================================================')
        print('============================================================================================================')
        print('Selected Hyperparameters: ',
              f"inter_dim {intermediate_dim}, latent_dim {latent_dim}, epsilon_std {epsilon_std},",
              f"regularisation_used {regularisation_used}, drop_out {drop_out}, training_type {training_type},",
              f"epochs {epochs}, num_folds {num_folds}")
        print('============================================================================================================')
        print('============================================================================================================')
       
        model_outcome, vae = Training_run(Xx_train, Xx_test, input_dim, timesteps, intermediate_dim, batch_size, latent_dim,
                     epsilon_std, regularisation_used, drop_out, training_type, epochs, num_folds)

        df_columns = ['loss', 'val_loss',
                      'intermediate_dim', 'latent_dim', 'epsilon_std', 'regularisation_used',
                      'drop_out', 'training_type',
                      'epochs', 'num_folds'];
        
        # Name the model set
        name = f"Set-3_{model}"
        # Save the model and its weights
        vae.save_weights(f"{name}_weights.h5")
        vae.save(f"{name}_model.h5")
        
        # df = pd.DataFrame(columns = df_columns)
        csv_str = 'hypertest.csv';
        # df.to_csv(csv_str, sep=',', index  = False)
        df_new = pd.DataFrame([[str(model_outcome.history['loss'][-1]), str(model_outcome.history['val_loss'][-1]),
                                str(intermediate_dim), str(latent_dim),
                                str(epsilon_std), str(regularisation_used),
                                str(drop_out), str(training_type),str(epochs), str(num_folds),
                               ]],
                              columns=df_columns)
        with open(csv_str, 'a') as f:
            #df_new.to_csv(f, header=False, index=False)
            df_new.to_csv(f, header=False, encoding='utf8', line_terminator='\n', index=False)
        # if loss does not exist then
        # df = pd.DataFrame(columns = df

if __name__ == "__main__":
    import sys, getopt
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    main(sys.argv)
    ###
## load model
#vae.load_weights('LSTM_w4.h5')



"""


## Testing VAE
preds = vae.predict(Xx_train, batch_size=batch_size)
predst = vae.predict(Xx_test, batch_size=batch_size)

z_test = enc.predict(Xx_train, batch_size=batch_size)
#plt.scatter(z_test[:, 0], z_test[:, 1], s=50, cmap='viridis')

# pick a column to plot.

plt.plot(Xx_train[22, 0, :], linewidth=3.5,label="input X")
plt.plot(preds[22, 0, :],label="reconstructed X")#
plt.legend(loc="upper left")
plt.xlabel('time sec')
plt.ylabel('scaled between 0-1')
plt.legend()
plt.show()


plt.figure()
plt.plot(Xx_test[27, 0, :], linewidth=3.5)
plt.plot(predst[27, 0, :])
plt.legend()
plt.show()



    space = {
        'intermediate_dim': [
            {
                'value1': '250',
                'value2': '200',
                'value3': '150'
            }

        ],

        'latent_dim': [
            {
                'value1': '100',
                'value2': '10',
                'value3': '2'
            }

        ],

        'epsilon_std': [
            {
                'value1': '2',
                'value2': '1',
                'value3': '0'
            }

        ],

        'regularisation_used': [
            {
                'value1': '2',
                'value2': '1'
            }

        ],

        'drop_out': [
            {
                'value1': '0.3',
                'value2': '0'
            }

        ],

        'training_type': [
            {
                'value1': '1',
                'value2': '0'
            }

        ],

        'epochs': [
            {
                'value1': '200',
                'value2': '100'
            }

        ],

        'num_folds': [
            {
                'value1': '20',
                'value2': '10'
            }

        ]

    }

import json

with open('data.json', 'w') as fp:
    json.dump(space, fp,  indent=4)


json_object = json.dumps(space)



"""
