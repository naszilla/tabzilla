#! /bin/bash
# functions for running batch jobs
# load these functions by running 'source utils.sh'

# constants
image_family=reczilla
service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com
zone=us-central1-a
project=research-collab-naszilla

# set of algorithms
alg_list="ItemKNNCF_asymmetric
ItemKNNCF_tversky
ItemKNNCF_euclidean
ItemKNNCF_cosine
ItemKNNCF_jaccard
ItemKNNCF_dice
UserKNNCF_asymmetric
UserKNNCF_tversky
UserKNNCF_euclidean
UserKNNCF_cosine
UserKNNCF_jaccard
UserKNNCF_dice
TopPop
GlobalEffects
Random
P3alphaRecommender
RP3betaRecommender
MatrixFactorization_FunkSVD_Cython
MatrixFactorization_AsySVD_Cython
MatrixFactorization_BPR_Cython
IALSRecommender
PureSVDRecommender
NMFRecommender
SLIM_BPR_Cython
SLIMElasticNetRecommender
EASE_R_Recommender
Mult_VAE_RecommenderWrapper
DELF_EF_RecommenderWrapper
CoClustering
SlopeOne"

dataset_list="Anime
BookCrossing
CiaoDVD
Dating
Epinions
FilmTrust
Frappe
GoogleLocalReviews
Gowalla
Jester2
LastFM
MarketBiasAmazon
MarketBiasModCloth
MovieTweetings
Movielens100K
Movielens10M
Movielens1M
Movielens20M
MovielensHetrec2011
NetflixPrize
Recipes
Wikilens"

dataset_array=(
Anime
BookCrossing
CiaoDVD
Dating
Epinions
FilmTrust
Frappe
GoogleLocalReviews
Gowalla
Jester2
LastFM
MarketBiasAmazon
MarketBiasModCloth
MovieTweetings
Movielens100K
Movielens10M
Movielens1M
Movielens20M
MovielensHetrec2011
NetflixPrize
Recipes
Wikilens
)

random_alg() {
  # return a random algorithm name, drawn from alg_list.txt
  echo $(sort --random-sort <<<"$alg_list" | head -1)
}

random_dataset() {
  # return a random algorithm name, drawn from dataset_list.txt
  echo $(sort --random-sort <<<"$dataset_list" | head -1)
}

delete_instances() {
  # $1 = name of list of instance names. all of these instances will be deleted
  echo "attempting to delete all instances..."
  local instance_list=$1
  for i in "${instance_list[@]}";
    do
        echo "deleting instance: $i"
        printf "Y" | gcloud compute instances delete $i --zone=${zone} --project=${project}
    done
}

sync_logs(){
  # $1 = path to log files. the contents of this directory will be added to the gcloud bucket reczilla-results/inbox/logs
  echo "syncing log files from $1 to gcloud..."
  gsutil -m rsync $1 gs://reczilla-results/inbox/logs
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

run_experiment() {

  # $1 = argument string passed to Experiment_handler.run_experiment
  # $2 = full path to the split data on the gcloud bucket (should start with gc://reczilla...)
  # $3 = instance name
  args_str="$1"
  split_path="$2"
  instance_name="$3"

  echo "run_experiment: args_str: ${args_str}"
  echo "run_experiment: split_path: ${split_path}"
  echo "run_experiment: instance_name: ${instance_name}"

  # set a return trap to delete the instance when this function returns
  trap "echo deleting instance ${instance_name}...; printf \"Y\" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}" RETURN

  # maximum number of attempts at creating gcloud instance and ssh
  MAX_TRIES=5

  echo "launching instance ${instance_name}..."

  COUNT=1
  while [ $COUNT -le $MAX_TRIES ]; do

    # attempt to create instance
    gcloud compute instances create $instance_name --zone=$zone \
    --project=$project --image-family=$image_family \
    --machine-type=n1-highmem-2 \
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
      echo "trying again in 5 seconds..."
      sleep 5
    else
      # success!
      break
    fi
  done
  echo "successfully created instance: ${instance_name}"

  # ssh and run the experiment. steps:
  # 1. set environment variables used by script run_experiment_on_instance.sh
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/shared/reczilla
  instance_script_location=${instance_repo_dir}/reczilla_scripts/run_experiment_on_instance.sh

  sleep 10

  COUNT=1
  MAX_TRIES_SSH=2
  while [ $COUNT -le $MAX_TRIES_SSH ]; do

    # attempt to run experiment
    gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
      --command="\
      export ARGS=\"${args_str}\"; \
      export SPLIT_PATH_ON_BUCKET=${split_path}; \
      chmod +x ${instance_script_location}; \
      /bin/bash ${instance_script_location}"

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

  # remember we don't need to delete the instance here, because we set a return trap
}


run_experiment_GPU() {
  # NOTE: this should be identical to run_experiment, but use a GPU instance rather than CPU

  # $1 = argument string passed to Experiment_handler.run_experiment
  # $2 = full path to the split data on the gcloud bucket (should start with gc://reczilla...)
  # $3 = instance name
  args_str="$1"
  split_path="$2"
  instance_name="$3"

  echo "run_experiment: args_str: ${args_str}"
  echo "run_experiment: split_path: ${split_path}"
  echo "run_experiment: instance_name: ${instance_name}"

  # set a return trap to delete the instance when this function returns
  trap "echo deleting GPU instance ${instance_name}...; printf \"Y\" | gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}" RETURN

  # maximum number of attempts at creating gcloud instance and ssh
  MAX_TRIES=5

  echo "launching GPU instance ${instance_name}..."

  COUNT=1
  while [ $COUNT -le $MAX_TRIES ]; do

    ACCELERATOR_TYPE=nvidia-tesla-t4
    ACCELERATOR_COUNT=1

    # attempt to create instance
    gcloud compute instances create $instance_name --zone=$zone \
    --project=$project --image-family=$image_family \
    --machine-type=n1-highmem-2 \
    --accelerator type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT} \
    --service-account $service_account \
    --maintenance-policy TERMINATE \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write

### example from https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus:
#gcloud compute instances create VM_NAME \
#    --machine-type MACHINE_TYPE \
#    --zone ZONE \
#    --boot-disk-size DISK_SIZE \
#    --accelerator type=ACCELERATOR_TYPE,count=ACCELERATOR_COUNT \
#    [--image IMAGE | --image-family IMAGE_FAMILY] \
#    --image-project IMAGE_PROJECT \
#    --maintenance-policy TERMINATE --restart-on-failure \
#    [--preemptible]

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
      echo "trying again in 5 seconds..."
      sleep 5
    else
      # success!
      break
    fi
  done
  echo "successfully created instance: ${instance_name}"

  # ssh and run the experiment. steps:
  # 1. set environment variables used by script run_experiment_on_instance.sh
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/shared/reczilla
  instance_script_location=${instance_repo_dir}/reczilla_scripts/run_experiment_on_instance.sh

  sleep 10

  COUNT=1
  MAX_TRIES_SSH=2
  while [ $COUNT -le $MAX_TRIES_SSH ]; do

    # attempt to run experiment
    gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
      --command="\
      export ARGS=\"${args_str}\"; \
      export SPLIT_PATH_ON_BUCKET=${split_path}; \
      chmod +x ${instance_script_location}; \
      /bin/bash ${instance_script_location}"

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
  echo "successfully ran GPU experiment"

  # remember we don't need to delete the instance here, because we set a return trap
}