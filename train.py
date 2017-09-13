# Set backend to Theano
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras import backend

# Create first network with Keras\   
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.models import load_model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=10, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)

#save then load the model
model.save('./outputs/my_model.h5')  # creates a HDF5 file 'my_model.h5'

# returns a compiled model
# identical to the previous one
model_reloaded = load_model('./outputs/my_model.h5')

# calculate predictions
predictions = model_reloaded.predict(X)
print('model reloaded')

# round predictions
#rounded = [round(x[0]) for x in predictions]

#predict with row of data
input1 = numpy.array([[1.,85.,66.,29.,0.,26.6,0.351,31.]])
print(model_reloaded.predict(input1))

import numpy
inputstring = "1.,85.,66.,29.,0.,26.6,0.351,31."
input2 = numpy.fromstring(inputstring,dtype=float, sep=',').reshape((1,8))
pred=model_reloaded.predict(input2)
print(pred[0][0])