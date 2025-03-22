from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    InputSchema,
    InputType,
    TaskSchema,
    ResponseBody,
    TextResponse,
)
from typing import TypedDict
from .model import AgeGenderDetector
from logging import getLogger
import json
import argparse


logger = getLogger(__name__)

# Configure UI Elements in RescueBox Desktop
def task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="image_directory",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema], parameters=[])


# Specify the input and output types for the task
class Inputs(TypedDict):
    image_directory: DirectoryInput


class Parameters(TypedDict):
    pass


server = MLServer(__name__)
server.add_app_metadata(
    name="Age and Gender Classifier",
    author="UMass Rescue",
    version="0.1.0",
    info=load_file_as_string("age-and-gender-detection/img-app-info.md"),
)
model = AgeGenderDetector()


@server.route("/predict", task_schema_func=task_schema)
def predict(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["image_directory"].path
    logger.info(f"Input path: {input_path}")
    res_list = model.predict_age_and_gender_on_dir(input_path)
    logger.info(f"Response: {res_list}")
    response = TextResponse(value=json.dumps(res_list))
    return ResponseBody(root=response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument(
        "--port", type=int, help="Port number to run the server", default=5000
    )
    args = parser.parse_args()
    server.run(port=args.port)