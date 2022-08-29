#! /bin/bash
# functions for running batch jobs
# load these functions by running 'source utils.sh'

# constants
image_family=tabzilla
service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com
zone=us-central1-a
project=research-collab-naszilla
machine_type=n1-highmem-2

run_experiment() {

  # $1 = model name
  # $2 = dataset name
  # $3 = env name
  # $4 = instance name
  model_name="$1"
  dataset_name="$2"
  env_name="$3"
  instance_name="$4"

  echo "run_experiment: model_name: ${model_name}"
  echo "run_experiment: dataset_name: ${dataset_name}"
  echo "run_experiment: env_name: ${env_name}"
  echo "run_experiment: instance_name: ${instance_name}"

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
      chmod +x ${instance_script}; \
      /bin/bash ${instance_script}"

    SSH_RETURN_CODE=$?

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
  # $1 = path to log files. the contents of this directory will be added to the gcloud bucket reczilla-results/inbox/logs
  echo "syncing log files from $1 to gcloud..."
  gsutil -m rsync $1 gs://tabzilla-results/logs
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
