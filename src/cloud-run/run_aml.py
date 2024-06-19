from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
from src.nima.giiaa.evaluation_giiaa_histogram import evaluate_nima


def register_dataset(aml_workspace: Workspace, dataset_name: str, datastore_name: str, file_path: str) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.from_binary_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace, name=dataset_name, create_new_version=True)
    return dataset

if __name__ == "__main__":

    run = Run.get_context()

    # dataset = register_dataset(aml_workspace="xiaa-aml", dataset_name="inputdata-1", datastore_name="datastore_inputdata", file_path="")

    run.log("Dataset registered", 1)

    run.complete()
    