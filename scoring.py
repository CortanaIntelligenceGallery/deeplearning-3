# Prepare the web service definition by authoring
# init() and run() functions. 
def init():
    import os
    #Set backend to tensorflow
    os.environ['KERAS_BACKEND']='tensorflow'
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import load_model
    
    # load the model
    global trainedmodel
    trainedmodel = load_model("my_model.h5")
    
def run(inputstring):
    import numpy
    input1 = numpy.fromstring(inputstring,dtype=float, sep=',').reshape((1,8))
    score=trainedmodel.predict(input1)
    return str(score[0][0])