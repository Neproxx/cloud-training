# CloudTraining
Tutorial on training machine learning models on Azure spot instances as part of the KTH DevOps 2022 course

### Creating a training script with checkpointing

There are 4 main steps to the training script:

1.  Create a folder to save your model at some frequency (e.g after every epoch)
2.  Load your dataset of interest
3.  Specify the callback/checkpointing object (we used Keras callbacks)
4.  Check if model already exits in folder (that you save your model to) and simply resuming training if model exists

Firstly, we will create a folder called 'Saved_Model' in our working directory where we can store our saved models whenever training is interrupted. This can be made using:

```console
mkdir Saved_Model
```

For our training script example, we are going to use Tensorflow with Keras API to build a Convolutional Neural Network (CNN) on the following dataset (any other dataset of interest can be used): [horses_or_humans](https://www.tensorflow.org/datasets/catalog/horses_or_humans). 

We will also use the 'MobileNetV3Small' architecture as it has over a million parameters to learn but any other keras model instance can be used. You can find this architecture under:

```python
tf.keras.applications.MobileNetV3Small
```

Now we will load the dataset and split it into a training set and a validation set using:

```python
(train_ds, val_ds) = tfds.load(name='horses_or_humans', split=['train', 'test'],
                               as_supervised=True, batch_size=32)                           
```

The next part is to set the callback object in order to save the Keras model or model weights at some frequency. This can easily be done using [ModelCheckPoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint):


```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(os.getcwd(), 'Saved_Model', 'Models.{epoch}-{val_loss:.2f}.hdf5'),
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='min',
    save_freq='epoch',
    options=None,
    initial_value_threshold=None,
)                           
```

When specifying the name of the saved model, 'Models.{epoch}-{val_loss:.2f}' in our case, it is important to include the .{epoch} part as this will later on be used to inform us of the epoch number when our model gets terminated. We also saved the file using the '.hdf5' extension such that the whole model is contained in a single file. We also save the model at every epoch as can be set by the 'save_freq' setting.

Next is to check whether a model already exists in the 'Saved_Model' file and to simply resume training from there. We will also use a regular expression to extract the epoch number from the saved file name and then load the model to continue training from the last epoch before termination. 

This can be done in the following way:

```python
# If model already exists, continue training
if os.listdir(os.path.join(os.getcwd(), 'Saved_Model')):

    # Regular expression pattern to extract epoch number
    pattern = '[^0-9]+([0-9]+).+'
    filename = os.listdir(os.path.join(os.getcwd(), 'Saved_Model'))[-1]

    # Find epoch number
    last_epoch = int(re.findall(pattern=pattern, string=filename)[0])

    # Load model and continue training model from last epoch
    model = load_model(filepath=os.path.join(os.getcwd(), 'Saved_Model', filename))
    model.fit(x=train_ds, epochs=5, validation_data=val_ds, callbacks=[checkpoint], initial_epoch=last_epoch)                    
```

If no model exists already (i.e no training has been done yet), we simply define our model (MobileNetV3Small in our case) and compile then fit the model to the data for training.

Whenever our training gets interuptted, the script will simply refer to the 'Saved_Model' file and just reload the model from where it left off.

Check out the [main.py](https://github.com/Neproxx/cloud-training/blob/main/main.py) in the repository to see the whole training script.


### Building a container

... TODO ...

### Creating a VM
We want to use the Azure CLI to create, start and manage our VM. For this we either need to install it or start a docker container that already has it installed. We do the latter with the following command. Note that "-it ... bash" tells docker to glue our terminal to the container and start bash inside of it so that we can interact with the container. On the other hand "mcr.microsoft.com/azure-cli" is the image name that is downloaded from [dockerhub](https://hub.docker.com/_/microsoft-azure-cli).

```console
docker run -it mcr.microsoft.com/azure-cli bash
```

Once the container is up and running, login with the following command and follow the instructions.

```console
az login
```

After we are logged in, we want to create a VM. Note that you can also do this via the webbrowser. However, in this tutorial we want to stick only to the Azure CLI for simplicity. First of all, we need to create a resource group that Azure uses to group all resources that are related to each other. This could span a react.js frontend server, a python backend server and a Postgres database server that all relate to the same application. Below, we create a resource group named "devops-tutorial" the meta data of which are stored on EU West servers. If you prefer to choose your own names in this and the following commands, you will have to replace them in the rest of the tutorial as well.

```console
az group create --name devops-tutorial --location westeu
```



... TODO ...

### Setting up the VM
Since we are using spot instances, our VM may be shut down or restarted at any point in time. Therefore, we want to always execute a script on boot that starts the docker container and thus the training process. We can use the Azure CLI to interact with the VM. The following code creates an executable script in the folder /tutorial. Replace the image name in the echo line with your own name if you did build the image yourself or use leave it the same if you want to use our version.

Note: Again, we could generate a private key and directly login to the VM and execute code there, however we want to stick to the Azure CLI for simplicity. The syntax cat "<<EOF > file_name ... EOF" below is used to push a multiline string (indicated by "...") inside a file.

```console
az vm run-command invoke -g devops-tutorial -n cloud-training --command-id RunShellScript --scripts \
"mkdir /tutorial
touch /tutorial/startup-script.sh
cat <<EOF > /tutorial/startup-script.sh
#\!/bin/bash
docker container run neprox/cloud-training-app 
EOF
chmod +x /tutorial/startup-script.sh
cat /tutorial/startup-script.sh"
```

Cronjobs are jobs that are executed on a periodic schedule. We want to create a cronjob that executes the script we just created on boot:

```console
az vm run-command invoke -g devops-tutorial -n cloud-training --command-id RunShellScript --scripts \
"sudo crontab -l > cron_bkp
sudo echo '@reboot sudo /tutorial/startup-script.sh >/dev/null 2>&1' >> cron_bkp
sudo crontab cron_bkp
sudo rm cron_bkp"
```

### Model Training
It is time to train a model on our VM.

... TODO ...

Since it can be terminated / evicted at any moment, we will use the Azure CLI to simulate such an incident:  
```console
az vm simulate-eviction --name cloud-training --resource-group devops-tutorial
```

### Cleanup
... TODO ...
Explain how to delete the things again, give a hint that they can double check within the UI of the webbrowser.

```console
az group delete --name devops-tutorial
```

### Further Notes
In this tutorial, we stored the model on the VM directly. In reality you would prefer 
