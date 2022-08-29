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
base                     /opt/conda/envs/openml
gbdt                     /opt/conda/envs/gbdt
sklearn                  /opt/conda/envs/sklearn
tensorflow               /opt/conda/envs/tensorflow
torch                    /opt/conda/envs/torch
```

3. Run a test script:

(you might need to chmod it first..):

```commandline
> cd ~/tabzilla
> chmod +x ./scripts/test_tabsurvey.sh 
> ./scripts/test_tabsurvey.sh
```

This should print output from the TabSurvey train/test cycles:

```commandline
----------------------------------------------------------------------------
Training DecisionTree with config/adult.yml in env sklearn

Namespace(config='config/adult.yml', model_name='DecisionTree', dataset='Adult', objective='binary', use_gpu=False, gpu_ids=[0, 1], data_parallel=True, optimize_hyperparameters=False, n_trials=2, direction='maximize', num_splits=5, shuffle=True, seed=221, scale=True, target_encode=True, one_hot_encode=False, batch_size=128, val_batch_size=256, early_stopping_rounds=20, epochs=3, logging_period=100, num_features=14, num_classes=1, cat_idx=[1, 3, 5, 6, 7, 8, 9, 13], cat_dims=[9, 16, 7, 15, 6, 5, 2, 42])
Train model with given hyperparameters
Loading dataset Adult...
Dataset loaded!
(32561, 14)
Scaling the data...
{'Log Loss - mean': 0.4087199772743019, 'Log Loss - std': 0.0, 'AUC - mean': 0.8968605810185616, 'AUC - std': 0.0, 
...
```


