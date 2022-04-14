# CloudTraining
Tutorial on training machine learning models on Azure spot instances as part of the KTH DevOps 2022 course

### Creating a training script with checkpointing

... TODO ...

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
