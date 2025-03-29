import logging
from itertools import batched

from embedding import SentenceEmbedder
from utils import supa_connx_str

import psycopg
import torch

# TODO: Consider refactoring database operations to reduce code duplication
# - Extract common try-except pattern from f'etch_data()', 'create_vecdb()', and 'insert_vecdb_data()' functions
# - Implement a unified execute_db_query helper function to standardize error handling

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_data(schema: str, data_table: str, connection_string: str) -> list:
    """Fetch data from the content database table."""
    logging.info(f"Fetching data from content database table '{schema}.{data_table}' ...")
    query = f"""
        SELECT id, content
        FROM {schema}.{data_table}
    """
    try:
        with psycopg.connect(connection_string) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                logging.info(f"Successfully fetched {len(results)} records.")
                return results

    except psycopg.Error as error:
        logger.exception(f"Database error occurred: {error}")
        return []

    except Exception as exception:
        logger.exception(f"An exception occurred: {exception}")
        return []


def prepare_vecdb_data(results, model_name: str, device: str, chunk_params: dict, batch_size: int):
    """Process data and create embeddings."""
    logging.info(f"Preparing (id, embeddings) records for the vector database ...")

    if not results:
        logging.warning("No data to process")
        return [], []

    embedder = SentenceEmbedder(model_name, device, chunk_params=chunk_params)
    all_encoded_ids = []
    all_embeddings = []

    for i, batch in enumerate(batched(results, n=batch_size)):
        logging.info(f"Processing batch {i + 1}")
        # Convert batch to appropriate format
        ids = [item[0] for item in batch]
        sentences = [item[1] for item in batch]
        encoded_ids, embeddings = embedder(ids, sentences)
        all_encoded_ids.extend(encoded_ids)
        all_embeddings.extend(embeddings)

    logging.info("Finished processing all batches")
    return all_encoded_ids, all_embeddings


def create_vecdb(schema: str, vec_table: str, data_table: str, connection_string: str):
    """Create the vector database table."""
    vec_table = vec_table.replace('-', '_')
    logging.info(f"Creating vector database '{schema}.{vec_table}' ...")
    query = f"""
        CREATE TABLE IF NOT EXISTS {schema}.{vec_table} (
            id INT NOT NULL REFERENCES {schema}.{data_table}(id) ON DELETE CASCADE,
            embedding VECTOR(384)
        );
    """
    try:
        with psycopg.connect(connection_string) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                connection.commit()
                logging.info(f"Vector table '{schema}.{vec_table}' created or already exists")

    except psycopg.Error as error:
        logger.exception(f"Database error occurred: {error}")
        raise

    except Exception as exception:
        logger.exception(f"An exception occurred: {exception}")
        raise


def insert_vecdb_data(schema: str, vec_table: str, connection_string: str, ids, embeddings):
    """Insert the embedding data into the vector database."""
    if not ids or not embeddings:
        logging.warning("No data to insert")
        return

    logging.info(f"Populating vector database '{schema}.{vec_table}' with {len(ids)} records...")
    embeddings_ = [embedding.tolist() for embedding in embeddings]
    data = list(zip(ids, embeddings_))

    query = f"""
        INSERT INTO {schema}.{vec_table} (id, embedding)
        VALUES (%s, %s);
    """

    try:
        with psycopg.connect(connection_string) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(query, data)
                connection.commit()
                logging.info(f"Successfully inserted {len(ids)} embedding records.")

    except psycopg.Error as error:
        logger.exception(f"Database error occurred: {error}")
        raise

    except Exception as exception:
        logger.exception(f"An exception occurred: {exception}")
        raise


def main(model_path, schema, data_table, chunk_params={}, batch_size=1000):
    """Main function to orchestrate the vector database creation process."""
    model_name = model_path.split('/')[-1]
    logging.info(f"Starting main function: Create vector database using model '{model_name}'")

    connection_string = supa_connx_str()

    # Fix device detection logic
    device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() else 'cpu'
    logging.info(f"Using torch device: {device}")

    # Fetch data, prepare embeddings, create and populate the vector database
    results = fetch_data(schema=schema, data_table=data_table, connection_string=connection_string)
    if not results:
        logging.error("Failed to fetch data. Exiting.")
        return None, None

    ids, embeddings = prepare_vecdb_data(results, model_path, device, chunk_params=chunk_params, batch_size=batch_size)
    if not ids or not embeddings:
        logging.error("Failed to generate embeddings. Exiting.")
        return None, None

    try:
        create_vecdb(schema=schema, vec_table=model_name.replace('-', '_'), data_table=data_table, connection_string=connection_string)
        insert_vecdb_data(schema=schema, vec_table=model_name.replace('-', '_'), connection_string=connection_string, ids=ids, embeddings=embeddings)
        return ids, embeddings
    except Exception as e:
        logging.error(f"Failed to create or populate vector database: {e}")
        return ids, embeddings  # Return the generated data even if DB operations failed



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector database creation script')
    
    parser.add_argument('--model-path', type=str, 
                        # default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Path to the sentence transformer model')
    parser.add_argument('--schema', type=str, 
                        default='abaqus',
                        help='Database schema name')
    parser.add_argument('--data-table', type=str, 
                        default='doc_embed_v1',
                        help='Source data table name')
    parser.add_argument('--batch-size', type=int, 
                        default=32,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    ids, embeddings = main(
        model_path      =   args.model_path,
        schema          =   args.schema,
        data_table      =   args.data_table,
        chunk_params    =   {}, 
        batch_size      =   args.batch_size
    )

    # python your_script.py --model-path="sentence-transformers/all-MiniLM-L6-v2" --schema="abaqus" --data-table="doc_embed_v1" --batch-size=64