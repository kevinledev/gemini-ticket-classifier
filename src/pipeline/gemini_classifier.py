from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform

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
        "pyarrow"
    ],
    base_image="python:3.9-slim"
)
def prepare_batch_data(
    project_id: str,
    gcs_input_path: str,
    gcs_batch_path: str
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
        input_bucket = storage_client.bucket(gcs_input_path.split('/')[2])
        input_blob = input_bucket.blob('/'.join(gcs_input_path.split('/')[3:]))
        
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
                    "contents": [{
                        "role": "user",
                        "parts": [{
                            "text": prompt_template.format(ticket_text=ticket_text)
                        }]
                    }]
                }
            }
            batch_instances.append(json.dumps(instance))
        
        # Write JSONL to GCS
        logger.info(f"Writing {len(batch_instances)} instances to {gcs_batch_path}")
        output_bucket = storage_client.bucket(gcs_batch_path.split('/')[2])
        output_blob = output_bucket.blob('/'.join(gcs_batch_path.split('/')[3:]))
        
        # Log sample instance for debugging
        logger.info(f"Sample JSONL instance:\n{batch_instances[0]}")
        
        output_content = '\n'.join(batch_instances)
        output_blob.upload_from_string(output_content)
        
        logger.info("Successfully wrote JSONL file")
        return gcs_batch_path
        
    except Exception as e:
        logger.error("Error in prepare_batch_data:", exc_info=True)
        raise

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
        "pyarrow"
    ],
    base_image="python:3.9-slim"
)
def run_batch_prediction(
    project_id: str,
    location: str,
    gcs_batch_path: str,
    gcs_output_path: str
) -> str:
    """Run batch prediction using Gemini"""
    import vertexai
    from vertexai.batch_prediction import BatchPredictionJob
    import logging
    import sys
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
            output_uri_prefix=gcs_output_path
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
        "pyarrow"
    ],
    base_image="python:3.9-slim"
)
def process_results(
    project_id: str,
    gcs_input_path: str,
    gcs_output_path: str,
    bq_destination: str
):
    """Process batch prediction results and load to BigQuery"""
    from google.cloud import storage, bigquery
    import pandas as pd
    import json
    import logging
    import sys
    from io import StringIO
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Initial setup and validation
        logger.info("=== Starting Component Setup ===")
        logger.info(f"Project ID: {project_id}")
        logger.info(f"Input path: {gcs_input_path}")
        logger.info(f"Predictions path: {gcs_output_path}")
        logger.info(f"BigQuery destination: {bq_destination}")
        
        # 2. Validate paths
        logger.info("=== Validating GCS paths ===")
        if not gcs_input_path.startswith('gs://'):
            raise ValueError(f"Invalid input path format: {gcs_input_path}")
        if not gcs_output_path.startswith('gs://'):
            raise ValueError(f"Invalid output path format: {gcs_output_path}")
            
        # 3. Initialize clients
        logger.info("=== Initializing Clients ===")
        try:
            storage_client = storage.Client(project=project_id)
            logger.info("Storage client initialized")
        except Exception as e:
            logger.error("Failed to initialize storage client")
            raise
            
        # 4. Check input file with detailed error handling
        logger.info("=== Checking Input File ===")
        try:
            input_bucket_name = gcs_input_path.replace('gs://', '').split('/')[0]
            input_blob_path = '/'.join(gcs_input_path.replace('gs://', '').split('/')[1:])
            
            logger.info(f"Input bucket: {input_bucket_name}")
            logger.info(f"Input blob path: {input_blob_path}")
            
            # Check bucket exists
            input_bucket = storage_client.bucket(input_bucket_name)
            if not input_bucket.exists():
                logger.error(f"Bucket does not exist: {input_bucket_name}")
                raise FileNotFoundError(f"Bucket not found: {input_bucket_name}")
            logger.info("Bucket exists")
            
            # List files in directory to verify access
            blobs = list(input_bucket.list_blobs(prefix='raw/'))
            logger.info(f"Found {len(blobs)} files in raw/ directory")
            for blob in blobs:
                logger.info(f"Found file: {blob.name}")
            
            # Try to get the specific blob
            input_blob = input_bucket.blob(input_blob_path)
            try:
                metadata = input_blob.metadata
                logger.info(f"Blob metadata: {metadata}")
            except Exception as e:
                logger.warning(f"Could not get metadata: {str(e)}")
            
            # Check if file exists
            exists = input_blob.exists()
            logger.info(f"Blob exists: {exists}")
            
            if not exists:
                raise FileNotFoundError(f"Input CSV not found: {gcs_input_path}")
            
            # Try to get file stats
            try:
                size = input_blob.size
                updated = input_blob.updated
                logger.info(f"File size: {size} bytes")
                logger.info(f"Last updated: {updated}")
            except Exception as e:
                logger.warning(f"Could not get file stats: {str(e)}")
            
            # Try to read first few bytes
            try:
                sample = input_blob.download_as_string(start=0, end=100)
                logger.info(f"First 100 bytes: {sample}")
            except Exception as e:
                logger.warning(f"Could not read file sample: {str(e)}")
            
            logger.info("Input file checks completed")
            
        except Exception as e:
            logger.error("Failed to access input file", exc_info=True)
            raise
            
        # 5. Check predictions directory
        logger.info("=== Checking Predictions Directory ===")
        pred_bucket_name = gcs_output_path.replace('gs://', '').split('/')[0]
        pred_prefix = '/'.join(gcs_output_path.replace('gs://', '').split('/')[1:])
        
        logger.info(f"Predictions bucket: {pred_bucket_name}")
        logger.info(f"Predictions prefix: {pred_prefix}")
        
        pred_bucket = storage_client.bucket(pred_bucket_name)
        
        # Specifically look for predictions.jsonl
        predictions_path = f"{pred_prefix}/predictions.jsonl"
        predictions_blob = pred_bucket.blob(predictions_path)
        
        if not predictions_blob.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
            
        logger.info(f"Found predictions file: {predictions_path}")
        logger.info(f"File size: {predictions_blob.size} bytes")
        
        # Read predictions
        logger.info("Reading predictions file")
        predictions = []
        content = predictions_blob.download_as_text()
        
        for line_num, line in enumerate(content.splitlines(), 1):
            try:
                data = json.loads(line)
                if data.get('response') and data['response'].get('candidates'):
                    pred_text = data['response']['candidates'][0]['content']['parts'][0]['text']
                    category, confidence = pred_text.strip().split(',')
                    predictions.append({
                        'processed_time': data.get('processed_time'),
                        'category': category.strip(),
                        'confidence': float(confidence)
                    })
                else:
                    logger.warning(f"No prediction in line {line_num}")
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(predictions)} predictions")
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame(predictions)
        logger.info(f"Predictions shape: {pred_df.shape}")
        
        # Read and process input data
        logger.info("Reading input CSV")
        input_df = pd.read_csv(StringIO(input_blob.download_as_text()))
        logger.info(f"Input shape: {input_df.shape}")
        
        # Merge data
        result_df = pd.concat([
            input_df.reset_index(drop=True),
            pred_df.reset_index(drop=True)
        ], axis=1)
        
        logger.info(f"Final dataset shape: {result_df.shape}")
        
        # Load to BigQuery
        logger.info("Loading to BigQuery")
        client = bigquery.Client(project=project_id)
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE"
        )
        
        job = client.load_table_from_dataframe(
            result_df, bq_destination, job_config=job_config
        )
        job.result()
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error("Error in process_results:", exc_info=True)
        raise

@dsl.pipeline(
    name="ticket-classification-pipeline",
    description="Pipeline to classify support tickets using Gemini batch processing"
)
def ticket_classification_pipeline(
    project_id: str = "kevinle-ticket-classifier",
    location: str = "us-central1",
    gcs_input_path: str = "gs://kevinle-ticket-pipeline/raw/customer_support_tickets.csv",
    gcs_batch_path: str = "gs://kevinle-ticket-pipeline/batch/batch_input.jsonl",
    gcs_output_path: str = "gs://kevinle-ticket-pipeline/batch/predictions",
    bq_destination: str = "kevinle-ticket-classifier.ticket_classification.classified_tickets"
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
        pipeline_func=ticket_classification_pipeline,
        package_path='pipeline.json'
    )
    
    # Initialize Vertex AI
    aiplatform.init(
        project='kevinle-ticket-classifier',
        location='us-central1'
    )
    
    # Create and run pipeline job
    job = aiplatform.PipelineJob(
        display_name='ticket-classification-batch',
        template_path='pipeline.json',
        pipeline_root='gs://kevinle-ticket-pipeline',
    )
    
    job.run()

if __name__ == "__main__":
    main()
