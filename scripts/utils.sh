#! /bin/bash
# functions for running batch jobs
# load these functions by running 'source utils.sh'

# constants
image_family=tabzilla
service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com
zone=us-central1-a
project=research-collab-naszilla
machine_type=n1-highmem-2

# GPU specs
ACCELERATOR_TYPE=nvidia-tesla-t4
ACCELERATOR_COUNT=1

# conda envs
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

run_experiment() {

  # $1 = model name
  # $2 = dataset name
  # $3 = env name
  # $4 = instance name
  # $5 = experiment name
  # $6 = config file
  model_name="$1"
  dataset_name="$2"
  env_name="$3"
  instance_name="$4"
  experiment_name="$5"
  config_file="$6"

  echo "run_experiment: model_name: ${model_name}"
  echo "run_experiment: dataset_name: ${dataset_name}"
  echo "run_experiment: env_name: ${env_name}"
  echo "run_experiment: instance_name: ${instance_name}"
  echo "run_experiment: experiment_name: ${experiment_name}"
  echo "run_experiment: config_file: ${config_file}"


  # set a return trap to delete the instance when this function returns
  trap "echo deleting instance ${instance_name}...; printf \"Y\" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}" RETURN

  # maximum number of attempts at creating gcloud instance and ssh
  MAX_TRIES=3

  echo "launching instance ${instance_name}..."

  COUNT=1
  while [ $COUNT -le $MAX_TRIES ]; do

    # attempt to create instance
    gcloud compute instances create $instance_name --zone=$zone \
    --project=$project --image-family=$image_family \
    --machine-type=${machine_type} \
    --service-account $service_account \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write

    # keep this for later
    INSTANCE_RETURN_CODE=$?

    if [ $INSTANCE_RETURN_CODE -ne 0 ]; then
      # failed to create instance
      let COUNT=COUNT+1
      echo "failed to create instance during attempt ${COUNT}... (exit code: ${INSTANCE_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES + 1 )) ]]; then
        echo "too many create-instance attempts. giving up."
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully created instance: ${instance_name}"

  # ssh and run the experiment. steps:
  # 1. set environment variables used by script tabzilla_experiment.sh
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/shared/tabzilla
  instance_script=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

  COUNT=1
  MAX_TRIES_SSH=2
  while [ $COUNT -le $MAX_TRIES_SSH ]; do

    # attempt to run experiment
    gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
      --command="\
      export ENV_NAME=\"${env_name}\"; \
      export MODEL_NAME=${model_name}; \
      export DATASET_NAME=${dataset_name}; \
      export EXPERIMENT_NAME=${experiment_name}; \
      export CONFIG_FILE=${config_file}; \
      chmod +x ${instance_script}; \
      /bin/bash ${instance_script}"

    SSH_RETURN_CODE=$?

    # was this instance preempted?
    echo gcloud compute operations list \
      --filter="operationType=compute.instances.preempted AND targetLink:instances/${instance_name}"

    if [ $SSH_RETURN_CODE -ne 0 ]; then
      # failed to run experiment
      let COUNT=COUNT+1
      echo "failed to run experiment during attempt ${COUNT}... (exit code: ${SSH_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES_SSH + 1 )) ]]; then
        echo "too many SSH attempts. giving up and deleting instance."
        printf "Y" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully ran experiment"

  # we don't need to delete the instance here, because we set a return trap
}

run_experiment_gpu() {

  # identical to run_experiment(), but attaches a teslsa T4 GPU to the instance and uses a 200GB disk rather than default 100GB

  # $1 = model name
  # $2 = dataset name
  # $3 = env name
  # $4 = instance name
  # $5 = experiment name
  # $6 = config file
  model_name="$1"
  dataset_name="$2"
  env_name="$3"
  instance_name="$4"
  experiment_name="$5"
  config_file="$6"

  echo "run_experiment: model_name: ${model_name}"
  echo "run_experiment: dataset_name: ${dataset_name}"
  echo "run_experiment: env_name: ${env_name}"
  echo "run_experiment: instance_name: ${instance_name}"
  echo "run_experiment: experiment_name: ${experiment_name}"
  echo "run_experiment: config_file: ${config_file}"


  # set a return trap to delete the instance when this function returns
  trap "echo deleting instance ${instance_name}...; printf \"Y\" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}" RETURN

  # maximum number of attempts at creating gcloud instance and ssh
  MAX_TRIES=3

  echo "launching instance ${instance_name}..."

  COUNT=1
  while [ $COUNT -le $MAX_TRIES ]; do

    # attempt to create instance
    gcloud compute instances create $instance_name --zone=$zone \
    --project=$project --image-family=$image_family \
    --machine-type=${machine_type} \
    --service-account $service_account \
    --maintenance-policy TERMINATE \
    --boot-disk-size=200GB \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write \
    --accelerator type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}

    # keep this for later
    INSTANCE_RETURN_CODE=$?

    if [ $INSTANCE_RETURN_CODE -ne 0 ]; then
      # failed to create instance
      let COUNT=COUNT+1
      echo "failed to create instance during attempt ${COUNT}... (exit code: ${INSTANCE_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES + 1 )) ]]; then
        echo "too many create-instance attempts. giving up."
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully created instance: ${instance_name}"

  # ssh and run the experiment. steps:
  # 1. set environment variables used by script tabzilla_experiment.sh
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/shared/tabzilla
  instance_script=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

  COUNT=1
  MAX_TRIES_SSH=2
  while [ $COUNT -le $MAX_TRIES_SSH ]; do

    # attempt to run experiment
    gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
      --command="\
      export ENV_NAME=\"${env_name}\"; \
      export MODEL_NAME=${model_name}; \
      export DATASET_NAME=${dataset_name}; \
      export EXPERIMENT_NAME=${experiment_name}; \
      export CONFIG_FILE=${config_file}; \
      chmod +x ${instance_script}; \
      /bin/bash ${instance_script}"

    SSH_RETURN_CODE=$?

    # was this instance preempted?
    echo gcloud compute operations list \
      --filter="operationType=compute.instances.preempted AND targetLink:instances/${instance_name}"


    if [ $SSH_RETURN_CODE -ne 0 ]; then
      # failed to run experiment
      let COUNT=COUNT+1
      echo "failed to run experiment during attempt ${COUNT}... (exit code: ${SSH_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES_SSH + 1 )) ]]; then
        echo "too many SSH attempts. giving up and deleting instance."
        printf "Y" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully ran experiment"

  # we don't need to delete the instance here, because we set a return trap
}


run_experiment_a100() {

  # identical to run_experiment(), but uses an A100 machine image, and uses a 200GB disk rather than default 100GB

  # $1 = model name
  # $2 = dataset name
  # $3 = env name
  # $4 = instance name
  # $5 = experiment name
  # $6 = config file
  model_name="$1"
  dataset_name="$2"
  env_name="$3"
  instance_name="$4"
  experiment_name="$5"
  config_file="$6"

  echo "run_experiment: model_name: ${model_name}"
  echo "run_experiment: dataset_name: ${dataset_name}"
  echo "run_experiment: env_name: ${env_name}"
  echo "run_experiment: instance_name: ${instance_name}"
  echo "run_experiment: experiment_name: ${experiment_name}"
  echo "run_experiment: config_file: ${config_file}"


  # set a return trap to delete the instance when this function returns
  trap "echo deleting instance ${instance_name}...; printf \"Y\" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}" RETURN

  # maximum number of attempts at creating gcloud instance and ssh
  MAX_TRIES=3

  echo "launching instance ${instance_name}..."

  COUNT=1
  while [ $COUNT -le $MAX_TRIES ]; do

    # attempt to create instance
    gcloud compute instances create $instance_name --zone=$zone \
    --project=$project --image-family=$image_family \
    --machine-type=a2-highgpu-1g \
    --service-account $service_account \
    --maintenance-policy TERMINATE \
    --boot-disk-size=200GB \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write

    # keep this for later
    INSTANCE_RETURN_CODE=$?

    if [ $INSTANCE_RETURN_CODE -ne 0 ]; then
      # failed to create instance
      let COUNT=COUNT+1
      echo "failed to create instance during attempt ${COUNT}... (exit code: ${INSTANCE_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES + 1 )) ]]; then
        echo "too many create-instance attempts. giving up."
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully created instance: ${instance_name}"

  # ssh and run the experiment. steps:
  # 1. set environment variables used by script tabzilla_experiment.sh
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/shared/tabzilla
  instance_script=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

  COUNT=1
  MAX_TRIES_SSH=2
  while [ $COUNT -le $MAX_TRIES_SSH ]; do

    # attempt to run experiment
    gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
      --command="\
      export ENV_NAME=\"${env_name}\"; \
      export MODEL_NAME=${model_name}; \
      export DATASET_NAME=${dataset_name}; \
      export EXPERIMENT_NAME=${experiment_name}; \
      export CONFIG_FILE=${config_file}; \
      chmod +x ${instance_script}; \
      /bin/bash ${instance_script}"

    SSH_RETURN_CODE=$?

    # was this instance preempted?
    echo gcloud compute operations list \
      --filter="operationType=compute.instances.preempted AND targetLink:instances/${instance_name}"


    if [ $SSH_RETURN_CODE -ne 0 ]; then
      # failed to run experiment
      let COUNT=COUNT+1
      echo "failed to run experiment during attempt ${COUNT}... (exit code: ${SSH_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES_SSH + 1 )) ]]; then
        echo "too many SSH attempts. giving up and deleting instance."
        printf "Y" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}
        exit 1
      fi
      echo "trying again in 30 seconds..."
      sleep 30
    else
      # success!
      break
    fi
  done
  echo "successfully ran experiment"

  # we don't need to delete the instance here, because we set a return trap
}


sync_logs(){
  # $1 = path to log files. the contents of this directory will be added to the gcloud bucket tabzilla-results/logs
  # $2 = experiment name
  echo "syncing log files from $1 for experiment $2 to gcloud..."

  # new name for the zip file
  zip_name=$2_logs_$(date +"%m%d%y_%H%M%S").zip

  # zip logs
  zip -jr ${zip_name} $1

  # copy to gcloud
  gsutil cp ${zip_name} gs://tabzilla-results/logs/${zip_name}
}

delete_instances() {
  # deletes all instances in global variable INSTANCE_LIST
  echo "attempting to delete all instances..."
  for i in "${INSTANCE_LIST[@]}";
    do
        echo "deleting instance: $i"
        printf "Y" | gcloud compute instances delete $i --zone=${zone} --project=${project}
    done
}

wait_until_processes_finish() {
  # only takes one arg: the maximum number of processes that can be running
  # print a '.' every 60 iterations
  counter=0
  while [ `jobs -r | wc -l | tr -d " "` -gt $1 ]; do
    sleep 1
    counter=$((counter+1))
    if (($counter % 60 == 0))
    then
      echo -n "."     # no trailing newline
    fi
  done
  echo "no more than $1 jobs are running. moving on."
}
