# CloudTraining
Tutorial on training machine learning models on Azure spot instances as part of the KTH DevOps 2022 course

### Creating a training script with checkpointing

... TODO ...

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
