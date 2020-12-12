#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


# In[ ]:





# In[2]:


import cv2


# In[3]:


dir='dataset'


# In[4]:


Categories=['with mask','without mask']


# In[5]:


features=[]
labels=[]
for categories in Categories:
  path=os.path.join(dir,categories)
  label=Categories.index(categories)
  for img in os.listdir(path):
    img_path=os.path.join(path,img)
    img_array=cv2.imread(img_path)
    gray=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    resize=cv2.resize(gray,(100,100))
    features.append(resize)
    labels.append(label)


# In[6]:


plt.imshow(features[0])
features[0].shape


# In[7]:


images=np.array(features)/255.0
images=np.reshape(images,(images.shape[0],100,100,1))


# In[8]:


images.shape


# In[9]:


labels=np.array(labels)


# In[10]:


labels=to_categorical(labels)
labels


# In[11]:


from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


# In[12]:


model = Sequential()


# In[13]:


model.add(Conv2D(200,(3,3),activation='relu',input_shape=images.shape[1:]))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(100,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))


# In[14]:


model.summary()


# In[15]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2)


# In[17]:


checkpoint=ModelCheckpoint('model-{epoch:03d}.model',save_best_only=True)


# In[18]:


history=model.fit(x_train,y_train,epochs=20,callbacks=[checkpoint],validation_split=0.2)


# In[19]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[20]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[21]:


model.evaluate(x_test,y_test)


# In[22]:


newmodel=tf.keras.models.load_model('model-012.model')


# In[35]:


face_clsfr=cv2.CascadeClassifier('haar_faces.xml')

source=cv2.VideoCapture(0)
source.set(3,640)
source.set(4,480)
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


# In[37]:


while False:

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=newmodel.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()


# In[ ]:





# In[ ]:




