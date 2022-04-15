# CloudTraining
Tutorial on training machine learning models on Azure spot instances as part of the KTH DevOps 2022 course

TODO: Give background and big picture

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
Setting up a VM programmatically is useful in production, however since this is a tutorial, it is easier and safer to use the UI in the webbrowser that informs you about prices, etc... Login to the [Azure portal](portal.azure.com) and start the process of creating a VM as shown below.

![Create VM](./images/01-Azure-tutorial-Create-VM.gif)

First of all, we need to create a resource group that groups all related components of an application together. This is very useful, as many resources will be created implicitly and you will only know about them and find them by investigating the resource group. A VM for example requires disk space and an IP address which both represent implicitly created resources. When you want to delete your application, you can find all of them inside your resource group. We create a resource group named "tutorial-resource-group". If you prefer to choose your own names in this and the following commands, you will have to replace them in the rest of the tutorial as well.

![Create resource group](./images/02-Azure-tutorial-Resource-Group-and-VM-name.gif)

Next up, we need a VM image to run on our new server. For us, it is important that it has Docker installed so that we can start the Docker container when the system boots up. The tensorflow image from Nvidia fits our purposes.

![Select correct image](./images/03-Azure-tutorial-Image-selection.gif)

We tick the "spot instance" box and notice that prices go down by 80% to 90%. We can select an eviction policy that is either solely based on capacity, meaning that Azure will terminate our instance when they require the hardware capacity, or also on price. Prices for cloud servers are dynamic and thus it is reasonable to evict a machine once it passes a price threshold. You may set it to for example 0.5$. 

![Configure spot instance](./images/04-Azure-tutorial-Spot-Instance-configuration.gif)

Most of our costs will be determined by the server size / hardware we select. In a real use case, we would select a large server size including GPUs in order to speed up model training. For educational purposes however, it is sufficient to select a small server that is covered by Azure's free plan.

![Select hardware](./images/05-Azure-tutorial-Select-hardware.gif)

Finally, enter a user name and a key pair name for ssh authentication. This will not be needed in the scope of the tutorial, but you could use these credentials to connect to the VM. Click on Review + create and see how your VM instance is created. If you plan on ls


![Finish VM creation](./images/06-Azure-tutorial-click-on-create.gif)

### Setting up the VM
For managing the VM, we will use the Azure CLI. For this we either need to install it or start a docker container that already has it installed. We do the latter with the command below. 

```console
docker run -it mcr.microsoft.com/azure-cli bash
```

Note that "-it ... bash" tells docker to glue our terminal to the container and start bash inside of it so that we can interact with it. On the other hand "mcr.microsoft.com/azure-cli" is the image name that is provided by Microsoft and downloaded from [dockerhub](https://hub.docker.com/_/microsoft-azure-cli). Once the container is up and running, login by following the instructions inside the terminal. If you plan to login to your VM with ssh, download the private key file. You find instruction on what to do when you navigate to the VM in the Azure portal and click on the "connect" button. However, to execute the rest of the tutorial, you will not need ssh.

```console
az login
```

Since we are using spot instances, our VM may be shut down or restarted at any point in time. Therefore, we have to make sure that the container always starts on bootup and continues training. We now create an executable script in the folder "/tutorial" which starts the container. Replace the image name in the docker container run line with your own image name if you built it yourself and pushed it to dockerhub.

```console
az vm run-command invoke -g tutorial-resource-group -n cloud-training --command-id RunShellScript --scripts "
mkdir /tutorial
mkdir /tutorial/models
touch /tutorial/startup-script.sh
cat <<EOF > /tutorial/startup-script.sh
#\!/bin/bash
docker container -v /tutorial/models:/app/Saved_Model run neprox/cloud-training-app 
EOF
chmod +x /tutorial/startup-script.sh"
```

The syntax cat "<<EOF > file_name ... EOF" below is commonly used to push a multiline string (indicated by "...") into a file. In addition, we need to explicitly tell the os with "chmod +x" that the file is allowed to be executed. We use a -v flag to specify a bind mount. This makes the host folder specified on the left (/tutorial/models) be accessible inside the container under the path stated on the right (/app/Saved_Model). Thus, we can from inside the container store models on the VM machine. Note also that we could login to the VM directly via ssh or install a software to login to the VM with a GUI, however for simplicity we stick to the Azure CLI.

After we have created a script that starts the container, we need to execute it on every bootup. The cronjobs package suits our purposes, as it executes jobs on a periodic schedule. In our case, we specify our schedule with the string "@reboot". Since the syntax can be hard to comprehend, the [crontab-generator] is very useful to generate the cronjob. The script that is executed below can be summarized by "Store all current cronjobs to cron_bkp, add a new cronjob, schedule all the cronjobs that are now in cron_bkp, delete the file cron_bkp". It may be the case that you get the stderr out put "no crontab for root" which you can ignore, because the crontab gets created with this script.

```console
az vm run-command invoke -g tutorial-resource-group -n cloud-training --command-id RunShellScript --scripts "
sudo crontab -l > cron_bkp
sudo echo '@reboot sudo /tutorial/startup-script.sh >/dev/null 2>&1' >> cron_bkp
sudo crontab cron_bkp
sudo rm cron_bkp"
```

### Model Training
It is time to train a model on our VM. We can either start the script directly or reboot and let the machine start the script by itself. Let us first start the script directly.

```console
az vm run-command invoke -g tutorial-resource-group -n cloud-training --command-id RunShellScript --scripts "
/tutorial/startup-script.sh
```

The model is now training. If you want to see the progress live you can login to the server with ssh, check the id fo the running container with

```console
docker container ls 
```

and then watch the prints of the container (and thus the training progress) with:

```console
docker container logs -f <container-id> 
```

If you do not want to do this, just wait 5 minutes to make sure that some training progress was made (at least two epochs of training). We now want to simulate an eviction / termination of the machine. Unlike AWS, Azure does not give us any non-hacky way of automatically restarting the VM after eviction. In order to keep this tutorial simple, we will thus manually restart the VM.

The easiest way to do this is to use the webbrowser UI. You can check the status of the machine by clicking on refresh. The restarting process might take a couple of minutes.

![Restarting VM](./images/restart-vm.gif)

Alternatively, you can use the Azure CLI:

```console
az vm restart --name cloud-training --resource-group tutorial-resource-group
```

### Cleanup
Cleaning up our resources is made easy, as we can simply go to resource groups and delete both groups that we find there. It does take a while until Azure has removed all services, so you naturally have to wait 1-3 minutes until everything is gone.

![Deleting all resources](./images/delete-resources.gif)


### Further Notes
In this tutorial, we stored the model on the VM directly. In reality you would prefer to put it in some more persistent cloud storage like an AWS S3 bucket. If you want to use this approach to reduce costs of model training, we advise you to use AWS which is much better suited for it. For example, because it gives you an option to automatically request a new spot instance once resources become available (see [spot requests](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-requests.html)). In order to conform with KTH however, we developed this tutorial using Azure.
