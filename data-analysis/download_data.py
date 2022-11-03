import openml

dataset = openml.tasks.get_task(2285, download_data=True)
print(dataset)