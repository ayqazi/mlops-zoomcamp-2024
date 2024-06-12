import mlflow
import numpy as np

mlflow.set_tracking_uri(uri="http://localhost:5000/")

client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name("Iris Classifier")
dev_model = client.search_registered_models("name='dev.iris_regressor'")[0]
# print(dev_model)
dev_mv = dev_model.latest_versions[0]
loaded_model = mlflow.pyfunc.load_model(dev_mv.source)
# print(loaded_model)

test_result = loaded_model.predict(np.array([[6.1, 2.8, 4.7, 1.2]]))
if test_result[0] == 1:
    dev_model_uri = f'models:/{dev_mv.name}/{dev_mv.version}'
    print(f"Promoting {dev_model_uri} (run_id {dev_mv.run_id}) to QA")
    qa_mv = client.copy_model_version(dev_model_uri, 'qa.iris_regressor')

# # Load the Iris dataset
# X, y = datasets.load_iris(return_X_y=True)
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# predictions = loaded_model.predict(X_test)
#
# iris_feature_names = datasets.load_iris().feature_names
#
# result = pd.DataFrame(X_test, columns=iris_feature_names)
# result["actual_class"] = y_test
# result["predicted_class"] = predictions

# client.copy_model_version(dev_model_uri, 'qa.iris_regressor')
