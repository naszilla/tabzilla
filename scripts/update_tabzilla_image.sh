#! /bin/bash

# this script creates a new image in the tabzilla family, with the latest codebase. from naszilla/tabzilla

# this assumes that the disk name is the same as the instance name, which is usually true.

zone=us-central1-a
instance=update-tabzilla
project=research-collab-naszilla
family=tabzilla
service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com

echo "creating instance ${instance}..."
# create an instance from the latest tabzilla-family image
gcloud compute instances create $instance --zone=$zone \
--project=$project --image-family=$family \
--service-account $service_account \
--scopes=https://www.googleapis.com/auth/devstorage.read_write


sleep 10

echo "finished creating instance ${instance}."

echo "updating code on  instance ${instance}..."

# ssh in, and update the code
gcloud compute ssh --ssh-flag="-A" ${instance} --zone=${zone} --project=${project} \
  --command="\
  cd /home/shared/tabzilla; \
  git pull"


echo "finished updating code."

# stop the instance
gcloud compute instances stop $instance

sleep 10

# create a name for the new image with today's date
new_image_name=tabzilla-$(date +"%m%d%y")

echo "creating image ${new_image_name}..."

# create a new image from this instance, and add it to the tabzilla family
gcloud compute images create $new_image_name \
    --source-disk $instance \
    --project=$project \
    --source-disk-zone $zone \
    --family $family --force

echo "finished creating image, deleting instance"

# delete the instance
printf "Y" | gcloud compute instances delete ${instance} --zone=${zone} --project=$project
