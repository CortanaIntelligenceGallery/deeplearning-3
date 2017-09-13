Build a deep learning model using Keras and Tensorflow to predict the onset of diabetes using the famous Pima Indians Diabetes dataset.

This is based on a [blog post](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras) by Jason Brownlee.

A copy of the dataset is included. 

Install the following libraries at the command prompt in the Workbench command window:

```
conda install keras
conda install tensorflow
```

Run the code and create the model file. Then download the model and enter the following command to deploy to ACS or locally.

```
az ml service create realtime -f scoring.py -m my_model.h5 -n kerasdiab1 -r python -c aml_config/conda-dependencies.yml
```
After service is created, use the service id to test the service.

```
az ml service run realtime -i <sevice id> -d "1.,85.,66.,29.,0.,26.6,0.351,31."
```

