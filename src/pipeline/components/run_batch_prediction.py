from kfp import dsl

@dsl.component(
    packages_to_install=[
        "google-cloud-aiplatform>=1.71.1",
        "vertexai>=0.0.1",
        "pandas>=2.0.0",
        "google-cloud-bigquery>=3.0.0",
        "google-cloud-storage>=2.0.0",
        "protobuf<5.0.0",
        "urllib3<2.0.0",
        "kfp>=2.0.0",
        "kfp-pipeline-spec==0.6.0",
        "kfp-server-api>=2.1.0,<2.4.0",
        "kubernetes>=8.0.0,<31.0.0",
        "PyYAML>=5.3,<7.0",
        "requests-toolbelt>=0.8.0,<1.0.0",
        "tabulate>=0.8.6,<1.0.0",
        "pyarrow",
    ],
    base_image="python:3.9-slim",
)
def run_batch_prediction(
    project_id: str, location: str, gcs_batch_path: str, gcs_output_path: str
) -> str:
    """Run batch prediction using Gemini"""
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob
    import logging
    import time

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting batch prediction with inputs:")
        logger.info(f"Input path: {gcs_batch_path}")
        logger.info(f"Output path: {gcs_output_path}")

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        # Submit batch prediction job
        batch_prediction_job = BatchPredictionJob.submit(
            source_model="gemini-1.5-flash-002",
            input_dataset=gcs_batch_path,
            output_uri_prefix=gcs_output_path,
        )

        logger.info(f"Job resource name: {batch_prediction_job.resource_name}")
        logger.info(f"Model name: {batch_prediction_job.model_name}")
        logger.info(f"Initial state: {batch_prediction_job.state.name}")

        # Monitor until complete
        while not batch_prediction_job.has_ended:
            time.sleep(10)
            batch_prediction_job.refresh()
            logger.info(f"Job state: {batch_prediction_job.state.name}")

        # Check final status
        if batch_prediction_job.has_succeeded:
            logger.info("Job succeeded!")
            logger.info(f"Output location: {batch_prediction_job.output_location}")
            return batch_prediction_job.output_location
        else:
            error_msg = f"Job failed: {batch_prediction_job.error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    except Exception as e:
        logger.error("Error in batch prediction:", exc_info=True)
        raise
