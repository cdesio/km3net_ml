


import numpy as np
import os
import pandas as pd




from network_models import train_neural_network, inference_step


# In[3]:


from network_models import TXYZnet


# In[4]:


model = TXYZnet(num_classes=2)


# In[5]:


model.summary()


# In[6]:


from data_loaders import data_generator, metadata_generator, get_n_iterations, get_class_weights

# In[7]:


# In[8]:


train_test_dir = os.path.join("train_test_files","cosz")
fnames_train =[os.path.join(train_test_dir, "Xy_train{}_sel5_doms.npz".format(i+1)) for i in range(100)]
fnames_test =[os.path.join(train_test_dir, "Xy_test{}_sel5_doms.npz".format(i+1)) for i in range(100)]
fnames_val =[os.path.join(train_test_dir, "Xy_val{}_sel5_doms.npz".format(i+1)) for i in range(100)]


# In[9]:


n_files=100
batch_size = 32
steps_per_epoch, n_events = get_n_iterations(fnames_train[:n_files], batch_size=batch_size)
print(steps_per_epoch, n_events)
validation_steps, n_evts_val = get_n_iterations(fnames_val[:n_files], batch_size=batch_size)
print(validation_steps, n_evts_val)
prediction_steps, n_evts_test = get_n_iterations(fnames_test[:n_files], batch_size=batch_size)
print(prediction_steps, n_evts_test)


# In[10]:


cls_weights = {i: v for i, v in enumerate(get_class_weights(fnames_train[:n_files]))}


# In[11]:


from keras.utils import to_categorical
def process_cosz(y):
    y[y>0]=1
    y[y<=0]=0
    return to_categorical(y)

def add_channel_dim(X):
    return X[:,:, np.newaxis, ...]
training_generator = data_generator(fnames_train[:n_files], batch_size=batch_size, 
                                    fdata=add_channel_dim, ftarget=process_cosz)


# In[12]:


validation_generator = data_generator(fnames_val[:n_files], batch_size=batch_size,
                                     fdata=add_channel_dim, ftarget=process_cosz)


# In[ ]:


train_neural_network(model, training_generator, steps_per_epoch, validation_generator, validation_steps,
                     batch_size=batch_size, class_weights=cls_weights, log_suffix="updown")


# In[ ]:




