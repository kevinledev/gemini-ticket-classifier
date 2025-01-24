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
def prepare_batch_data(
    project_id: str, gcs_input_path: str, gcs_batch_path: str
) -> str:
    """Prepare data for batch prediction"""
    from google.cloud import storage
    import pandas as pd
    import json
    import logging
    import sys
    from io import StringIO

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting data preparation")

        # Read input CSV
        storage_client = storage.Client(project=project_id)
        input_bucket = storage_client.bucket(gcs_input_path.split("/")[2])
        input_blob = input_bucket.blob("/".join(gcs_input_path.split("/")[3:]))

        df = pd.read_csv(StringIO(input_blob.download_as_text()))
        logger.info(f"Read {len(df)} rows from CSV")

        # Prepare JSONL for Gemini
        batch_instances = []
        prompt_template = """Classify this support ticket into one of these categories:
- Technical Issue
- Billing Inquiry
- Product Inquiry
- Cancellation Request
- Refund Request

Ticket:
{ticket_text}

Respond with only the category name and confidence score (0-1) separated by a comma.
Example: Technical Issue,0.95"""

        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing row {idx}")

            ticket_text = f"Subject: {row['Ticket Subject']}\nDescription: {row['Ticket Description']}"

            # Format matching the example JSONL structure
            instance = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": prompt_template.format(
                                        ticket_text=ticket_text
                                    )
                                }
                            ],
                        }
                    ]
                }
            }
            batch_instances.append(json.dumps(instance))

        # Write JSONL to GCS
        logger.info(f"Writing {len(batch_instances)} instances to {gcs_batch_path}")
        output_bucket = storage_client.bucket(gcs_batch_path.split("/")[2])
        output_blob = output_bucket.blob("/".join(gcs_batch_path.split("/")[3:]))

        # Log sample instance for debugging
        logger.info(f"Sample JSONL instance:\n{batch_instances[0]}")

        output_content = "\n".join(batch_instances)
        output_blob.upload_from_string(output_content)

        logger.info("Successfully wrote JSONL file")
        return gcs_batch_path

    except Exception as e:
        logger.error("Error in prepare_batch_data:", exc_info=True)
        raise
