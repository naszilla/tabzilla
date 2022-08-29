**Intended for internal use** 

This document describes how to set up a GCP instance and run a basic experiment with TabZilla.

1. Make sure you have GCP [command line tools](https://cloud.google.com/sdk/gcloud) installed 

2. Make sure you can authenticate on github over ssh (see [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)). You might need to create an SSH key and add it to your github account. To check, run:
```commandline
> ssh -T git@github.com
```

this should print something like:
```commandline
> Hi <username>! You've successfully authenticated, but GitHub does not
> provide shell access.
```

See [this link]([agent forwarding to authenticate your github account](https://docs.github.com/en/developers/overview/using-ssh-agent-forwarding)) for troubleshooting.

4. Create a GCP instance from the `tabzilla` family:

```
zone=us-central1-a
project=research-collab-naszilla
family=tabzilla

# machine type (change if you'd like)
machine_type=n1-highmem-2

# the name of the instance you will create
instance_name=<MY-INSTANCE-NAME>

# create an instance from the latest tabzilla-family image
gcloud beta compute instances create ${instance_name} \
--zone=${zone} \
--project=${project} \
--image-family=${family} \
--machine-type=${machine_type} \
--scopes=https://www.googleapis.com/auth/devstorage.read_write
```

This should return something like the following:

```
Created [https://www.googleapis.com/compute/v1/projects/research-collab-naszilla/zones/us-central1-a/instances/dcm-tabzilla].
NAME          ZONE           MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP      STATUS
############  us-central1-a  n1-standard-1               xx.xxx.x.xx  123.456.789.000  RUNNING
```

**Pro tip!!** if you're using an IDE that lets you ssh into a remote machine (like Visual Studio Code), **you can ssh into your GCP instance using the IDE, using the `EXTERNAL_IP` address printed above**. This allows you to use the instance more easily for development purposes.

2. SSH into the instance, using flag `-A` for agent forwarding

```
gcloud compute ssh --ssh-flag="-A" --zone ${zone} ${instance_name}  --project ${project}
```

All conda environments used by TabSurvey should already be prepared:

```commandline
> conda info --envs
```

should return:
```
# conda environments:
#
base                  *  /opt/conda
gbdt                     /opt/conda/envs/gbdt
openml                   /opt/conda/envs/openml
sklearn                  /opt/conda/envs/sklearn
tensorflow               /opt/conda/envs/tensorflow
torch                    /opt/conda/envs/torch
```

3. Go to the shared tabzilla directory, and update it if needed.

```
cd /home/shared/tabzilla/
git pull
```

4. Look at the datasets that are currently pre-processed on this image:

```
ls /home/shared/tabzilla/TabSurvey/datasets
```

this should print something like:

```
openml__APSFailure__168868                        openml__ilpd__9971
openml__Amazon_employee_access__34539             openml__isolet__3481
openml__Australian__146818                        openml__jannis__168330
openml__Bioresponse__9910                         openml__jasmine__168911
openml__CIFAR_10__167124                          openml__jm1__3904
...
```

5. Run an experiment!

First, modify the script `/home/shared/tabzilla/scripts/test_tabzilla_on_instance.sh` to specify an ML model, conda environment, and dataset name. You can use any dataset already on the instance (see previous step). You need to modify the following three lines:

```
# define the conda env that should be used {sklearn | gbdt | torch | tensorflow}
ENV_NAME=sklearn

# name of the model/algorithm
MODEL_NAME=KNN

# name of the dataset
DATASET_NAME=openml__california__361089
```

```
cd /home/shared/tabzilla/scripts
```


(you might need to chmod it first..):

```commandline
> cd ~/tabzilla
> chmod +x ./scripts/test_tabsurvey.sh 
> ./scripts/test_tabsurvey.sh
```



