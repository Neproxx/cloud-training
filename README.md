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

The next part is to set the callback object in order to save the Keras model or model weights at some frequency. This can easily be done using [ModelCheckPoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) (check the link for a deeper description of the parameter settings):


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

Now whenever our training gets interuptted, the script will simply refer to the 'Saved_Model' file and just reload the model from where it left off.

Check out the [main.py](https://github.com/Neproxx/cloud-training/blob/main/main.py) in the repository to see the whole training script.


### Building a container

... TODO ...

### Creating a VM
Setting up a VM programmatically is useful in production, however since this is a tutorial, it is easier and safer to use the UI in the webbrowser that informs you about prices, etc...

TODO: pictures

Azure automatically creates a resource group that it uses to group all resources that are related to each other. This is very useful, as the number of resources that are implicitly created can be very big. When creating a VM, you are for example creating a disk resource and IP address resource as well. When you want to delete your application, you can find all of them inside your resource group.  We create a resource group named "devops-tutorial". If you prefer to choose your own names in this and the following commands, you will have to replace them in the rest of the tutorial as well.

... TODO: Pictures and more explanation ...

### Setting up the VM
For managing the VM, we will use the Azure CLI. For this we either need to install it or start a docker container that already has it installed. We do the latter with the command below. 

```console
docker run -it mcr.microsoft.com/azure-cli bash
```

Note that "-it ... bash" tells docker to glue our terminal to the container and start bash inside of it so that we can interact with the container. On the other hand "mcr.microsoft.com/azure-cli" is the image name that is provided by Microsoft and downloaded from [dockerhub](https://hub.docker.com/_/microsoft-azure-cli).

Once the container is up and running, login with the following command and follow the instructions.

```console
az login
```

Since we are using spot instances, our VM may be shut down or restarted at any point in time. Therefore, we have to make sure that the training container always starts on bootup and continues training. We now create an executable script in the folder /tutorial which starts the container. Replace the image name in the echo line with your own name if you did build the image yourself or use leave it the same if you want to use our version.

```console
az vm run-command invoke -g devops-tutorial -n cloud-training --command-id RunShellScript --scripts \
"mkdir /tutorial
touch /tutorial/startup-script.sh
cat <<EOF > /tutorial/startup-script.sh
#\!/bin/bash
docker container run neprox/cloud-training-app 
EOF
chmod +x /tutorial/startup-script.sh"
```

The syntax cat "<<EOF > file_name ... EOF" below is commonly used to push a multiline string (indicated by "...") inside a file although it is not very intuitive. In addition, we need to explicitly tell the os with "chmod +x" that the file is allowed to be executed. Note also that we could log into the VM directly via ssh or install a software to log into the VM with a GUI, however for simplicity we stick to the Azure CLI.

After we have created a script that starts the container, we need to execute it on every bootup. The cronjobs package suits our purposes, as it executes jobs on a periodic schedule. In our case, we specify our schedule with the string "@reboot". Since the syntax can be hard to comprehend, the [crontab-generator] is very useful to generate the cronjob. The script that is executed below can be summarized by "Store all current cronjobs to cron_bkp, add a new cronjob, schedule all the cronjobs that are now in cron_bkp, delete the file cron_bkp".

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
