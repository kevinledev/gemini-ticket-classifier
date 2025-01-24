import os
from kfp import dsl
from kfp import compiler
from dotenv import load_dotenv
from google.cloud import aiplatform

from components.prepare_batch_data import prepare_batch_data
from components.run_batch_prediction import run_batch_prediction
from components.process_results import process_results

load_dotenv()


@dsl.pipeline(
    name="ticket-classification-pipeline",
    description="Pipeline to classify support tickets using Gemini batch processing",
)
def ticket_classification_pipeline(
    project_id: str = os.getenv("PROJECT_ID"),
    location: str = os.getenv("LOCATION"),
    gcs_input_path: str = os.getenv("GCS_INPUT_PATH"),
    gcs_batch_path: str = os.getenv("GCS_BATCH_PATH"),
    gcs_output_path: str = os.getenv("GCS_OUTPUT_PATH"),
    bq_destination: str = os.getenv("BQ_DESTINATION"),
):
    # Prepare data for batch processing
    batch_data = prepare_batch_data(
        project_id=project_id,
        gcs_input_path=gcs_input_path,
        gcs_batch_path=gcs_batch_path,
    ).set_caching_options(True)

    # Run batch prediction
    prediction_results = run_batch_prediction(
        project_id=project_id,
        location=location,
        gcs_batch_path=batch_data.output,
        gcs_output_path=gcs_output_path,
    ).set_caching_options(False)

    # Process results and write to BigQuery
    process_results(
        project_id=project_id,
        gcs_input_path=gcs_input_path,
        gcs_output_path=prediction_results.output,
        bq_destination=bq_destination,
    ).set_caching_options(False)


def main():
    """Compile and run the pipeline"""
    from kfp.compiler import Compiler
    from google.cloud import aiplatform

    # Compile pipeline
    Compiler().compile(
        pipeline_func=ticket_classification_pipeline, package_path="pipeline.json"
    )

    # Initialize Vertex AI
    aiplatform.init(project="kevinle-ticket-classifier", location="us-central1")

    # Create and run pipeline job
    job = aiplatform.PipelineJob(
        display_name="ticket-classification-batch",
        template_path="pipeline.json",
        pipeline_root="gs://kevinle-ticket-pipeline",
    )

    job.run()


if __name__ == "__main__":
    main()
