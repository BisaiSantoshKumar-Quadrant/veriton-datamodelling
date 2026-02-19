"""
HTTP-triggered function that does EVERYTHING:
1. Discovers files in folder
2. Extracts schemas (like jobProcessor)
3. Detects relationships (like AIAnalyzer)
4. Saves all results
"""

import azure.functions as func
import json
import logging
from datetime import datetime
import sys
import os
import io

logging.basicConfig(level=logging.INFO)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
from azure.storage.blob import BlobServiceClient

try:
    from shared.shared import (
        extract_schema_metadata,
        extract_schema_from_json_file,
        detect_relationships,
        NumpyEncoder,
        STORAGE_ACCOUNT_NAME,
        STORAGE_ACCOUNT_KEY
    )
    logging.info("✅ Shared module imported")
except Exception as e:
    logging.error(f"❌ Failed to import shared: {e}")
    raise

SUPPORTED_EXTENSIONS = ['.csv', '.parquet', '.json']


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP-triggered function that processes job folder.
    Does EXACTLY what your blob-triggered functions do.
    """
    try:
        logging.info("=" * 80)
        logging.info("🚀 PROCESS JOB - HTTP TRIGGER")
        logging.info("=" * 80)
        
        # ============================================================
        # GET REQUEST PARAMETERS
        # ============================================================
        try:
            body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Invalid JSON"}),
                status_code=400,
                mimetype="application/json"
            )
        
        user_id = body.get('user_id')
        job_id = body.get('job_id')
        container_name = body.get('container_name', 'datamodelling')
        
        if not user_id or not job_id:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "user_id and job_id required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logging.info(f"📦 Container: {container_name}")
        logging.info(f"👤 User: {user_id}")
        logging.info(f"📋 Job: {job_id}")
        
        # ============================================================
        # CONNECT TO STORAGE
        # ============================================================
        conn_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={STORAGE_ACCOUNT_NAME};"
            f"AccountKey={STORAGE_ACCOUNT_KEY};"
            f"EndpointSuffix=core.windows.net"
        )
        
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container_name)
        
        job_folder = f"{user_id}/{job_id}/"
        job_status_path = f"{job_folder}metadata/job_status.json"
        
        
        job_data = {
            "user_id": user_id,
            "job_id": job_id,
            "container_name": container_name,
            "status": "processing",
            "created_at": datetime.utcnow().isoformat(),
            "processing_started_at": datetime.utcnow().isoformat()
        }
        
        job_status_blob = container_client.get_blob_client(job_status_path)
        job_status_blob.upload_blob(
            json.dumps(job_data, indent=2),
            overwrite=True
        )
        
        # ============================================================
        # STEP 1: DISCOVER FILES
        # ============================================================
        logging.info("=" * 80)
        logging.info("📂 STEP 1: DISCOVERING FILES")
        logging.info("=" * 80)
        
        all_blobs = list(container_client.list_blobs(name_starts_with=job_folder))
        
        processable_files = []
        for blob in all_blobs:
            blob_path = blob.name
            filename = blob_path.split('/')[-1]
            
            # Skip metadata and relationships folders
            if '/metadata/' in blob_path or '/relationships/' in blob_path:
                continue
            
            # Skip internal files
            if filename.startswith(('schema_', 'job_', 'batch_', 'analysis_', 'relationship')):
                continue
            
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                processable_files.append({
                    'blob': blob,
                    'path': blob_path,
                    'name': filename,
                    'extension': file_ext,
                    'size': blob.size
                })
        
        if not processable_files:
            job_data['status'] = 'failed'
            job_data['error'] = 'No processable files found'
            job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
            
            return func.HttpResponse(
                json.dumps({"status": "failed", "message": "No processable files"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logging.info(f"✅ Found {len(processable_files)} files")
        for f in processable_files:
            logging.info(f"   📄 {f['name']} ({f['extension']}, {f['size']} bytes)")
        
        # ============================================================
        # STEP 2: EXTRACT SCHEMAS
        # ============================================================
        logging.info("=" * 80)
        logging.info("📋 STEP 2: EXTRACTING SCHEMAS")
        logging.info("=" * 80)
        
        schemas = []
        processed_count = 0
        failed_files = []
        
        for file_info in processable_files:
            try:
                filename = file_info['name']
                file_ext = file_info['extension']
                blob_path = file_info['path']
                
                logging.info(f"🔍 Processing: {filename}")
                
                blob_client = container_client.get_blob_client(blob_path)
                file_data = blob_client.download_blob().readall()
                
                metadata = None
                
                # Same extraction logic as your code
                if file_ext == '.json':
                    metadata = extract_schema_from_json_file(file_data, filename, blob_path)
                
                elif file_ext == '.csv':
                    try:
                        df = pd.read_csv(io.BytesIO(file_data))
                    except:
                        df = pd.read_csv(io.BytesIO(file_data), engine="python", on_bad_lines="skip")
                    
                    if df is not None and not df.empty:
                        metadata = extract_schema_metadata(df, filename, blob_path)
                
                elif file_ext == '.parquet':
                    df = pd.read_parquet(io.BytesIO(file_data))
                    if df is not None and not df.empty:
                        metadata = extract_schema_metadata(df, filename, blob_path)
                
                if metadata:
                    # Add job context
                    metadata['user_id'] = user_id
                    metadata['job_id'] = job_id
                    metadata['blob_size_bytes'] = file_info['size']
                    
                    # Save schema filebase_name = os.path.splitext(filename)[0]
                    base_name = os.path.splitext(filename)[0]
                    schema_filename = f"schema_{base_name}.json"
                    schema_path = f"{job_folder}metadata/{schema_filename}"
                    
                    schema_blob = container_client.get_blob_client(schema_path)
                    schema_blob.upload_blob(
                        json.dumps(metadata, indent=2, cls=NumpyEncoder),
                        overwrite=True
                    )
                    
                    schemas.append(metadata)
                    processed_count += 1
                    
                    logging.info(f"   ✅ {filename}: {metadata['column_count']} cols, {metadata['row_count']} rows")
                else:
                    failed_files.append(filename)
                    logging.error(f"   ❌ Failed: {filename}")
                    
            except Exception as e:
                failed_files.append(filename)
                logging.exception(f"   ❌ Error: {filename}: {e}")
        
        if not schemas:
            job_data['status'] = 'failed'
            job_data['error'] = 'Schema extraction failed'
            job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
            
            return func.HttpResponse(
                json.dumps({"status": "failed", "message": "Schema extraction failed"}),
                status_code=500,
                mimetype="application/json"
            )
        
        logging.info("=" * 80)
        logging.info(f"✅ Schema extraction complete:")
        logging.info(f"   - Processed: {processed_count}/{len(processable_files)}")
        logging.info(f"   - Failed: {len(failed_files)}")
        logging.info("=" * 80)
        
        # Update status
        job_data['status'] = 'schema_extraction_complete'
        job_data['schemas_extracted'] = processed_count
        job_data['failed_files'] = failed_files
        job_data['schema_extraction_completed_at'] = datetime.utcnow().isoformat()
        job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
        
        # ============================================================
        # STEP 3: DETECT RELATIONSHIPS
        # ============================================================
        logging.info("=" * 80)
        logging.info("🧠 STEP 3: DETECTING RELATIONSHIPS")
        logging.info("=" * 80)
        
        job_data['status'] = 'analyzing_relationships'
        job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
        
        relationship_info = detect_relationships(schemas)
        
        # Initialize dimension tables list for use across steps
        all_dim_tables = []
        
        num_tables = len(relationship_info.get('tables', []))
        num_rels = len(relationship_info.get('relationships', []))
        
        logging.info(f"✅ Relationship detection complete:")
        logging.info(f"   - Tables: {num_tables}")
        logging.info(f"   - Relationships: {num_rels}")
        
        if num_rels > 0:
            logging.info("🔗 Detected relationships:")
            for rel in relationship_info['relationships'][:10]:
                from_t = rel.get('from_table', '')
                to_t = rel.get('to_table', '')
                rel_type = rel.get('relationship_type', '')
                logging.info(f"   {from_t} → {to_t} ({rel_type})")

        # ============================================================
        # STEP 3.5: BUILD VIRTUAL FACT TABLE
        # ============================================================
        logging.info("=" * 80)
        logging.info("🏗️ STEP 3.5: BUILDING VIRTUAL FACT TABLE")
        logging.info("=" * 80)

        virtual_fact_columns = []
        virtual_fact_fks = []
        existing_columns = []  # For deduplication
        all_dim_tables = []  # Track all dimension tables

        # Process ALL schemas as dimension tables
        for schema in schemas:
            table_name = schema.get("table_name", "")
            columns = schema.get("columns", [])
            
            # Find table info from relationship detection
            table_info = next(
                (t for t in relationship_info.get("tables", []) if t.get("table") == table_name),
                None
            )
            
            # Get primary keys (or detect from schema if not found)
            primary_keys = table_info.get("primary_keys", []) if table_info else []
            
            # Fallback: if no PK detected, look for potential keys in schema
            if not primary_keys:
                for col in columns:
                    if col.get("is_potential_key", False):
                        primary_keys.append(col.get("column_name"))
                        break
            
            # Skip tables without ANY primary key
            if not primary_keys:
                logging.warning(f"⚠️ Skipping {table_name} - No primary key detected")
                continue
            
            # Add to dimension tables list
            all_dim_tables.append(table_name)
            
            # Add PK(s) as foreign key columns in fact table
            for pk in primary_keys:
                # Handle duplicate column names
                col_name = pk
                if col_name in existing_columns:
                    col_name = f"{table_name}_{pk}"
                    logging.info(f"   ⚠️ Renamed duplicate: {pk} → {col_name}")
                
                existing_columns.append(col_name)
                
                # Find original column metadata
                original_col = next((c for c in columns if c.get("column_name") == pk), None)
                
                if original_col:
                    virtual_fact_columns.append({
                        "column_name": col_name,
                        "data_type": original_col.get("data_type", "int"),
                        "nullable": False,
                        "null_count": 0,
                        "null_percentage": 0.0,
                        "distinct_count": original_col.get("distinct_count", 0),
                        "cardinality_percentage": original_col.get("cardinality_percentage", 0.0),
                        "is_potential_key": False,
                        "is_foreign_key": True,
                        "sample_values": []
                    })
                    
                    # Track foreign key relationship
                    virtual_fact_fks.append({
                        "column": col_name,
                        "references_table": table_name,
                        "references_column": pk
                    })

        # Calculate total rows (sum of ALL dimension tables)
        total_dim_rows = sum(s.get("row_count", 0) for s in schemas)

        # Create virtual fact table metadata
        virtual_fact_metadata = {
            "table_name": "fact_table",
            "file_path": "virtual",
            "row_count": int(total_dim_rows),
            "column_count": len(virtual_fact_columns),
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "columns": virtual_fact_columns,
            "user_id": user_id,
            "job_id": job_id,
            "blob_size_bytes": 0,
            "is_virtual": True
        }

        # Save virtual fact table schema
        fact_schema_path = f"{job_folder}metadata/schema_fact_table.json"
        fact_schema_blob = container_client.get_blob_client(fact_schema_path)
        fact_schema_blob.upload_blob(
            json.dumps(virtual_fact_metadata, indent=2, cls=NumpyEncoder),
            overwrite=True
        )

        logging.info(f"✅ Virtual fact table created:")
        logging.info(f"   - Columns: {len(virtual_fact_columns)}")
        logging.info(f"   - Total rows: {total_dim_rows}")
        logging.info(f"   - Dimension tables: {len(all_dim_tables)}")

        # Override relationship detection - force star schema model
        relationship_info["fact_tables"] = ["fact_table"]
        relationship_info["dimension_tables"] = all_dim_tables

        # Update relationships to reflect fact → dimension connections
        relationship_info["tables"].append({
            "table": "fact_table",
            "primary_keys": [],
            "foreign_keys": virtual_fact_fks
        })
        
        # ============================================================
        # ============================================================
        # STEP 4: BUILD ENRICHED OUTPUT
        # ============================================================
        logging.info("=" * 80)
        logging.info("📊 STEP 4: BUILDING ENRICHED OUTPUT")
        logging.info("=" * 80)

        enriched_tables = []

        # Add all dimension tables (force DIM type)
        for schema in schemas:
            table_name = schema.get("table_name", "")
            columns = schema.get("columns", [])
            row_count = schema.get("row_count", 0)
            
            # Find primary and foreign keys
            table_info = next(
                (t for t in relationship_info.get("tables", []) if t["table"] == table_name),
                None
            )
            primary_keys = table_info.get("primary_keys", []) if table_info else []
            foreign_keys = table_info.get("foreign_keys", []) if table_info else []
            
            # Force all uploaded tables to be dimensions
            table_type = "DIM"
            
            enriched_tables.append({
                "table_name": table_name,
                "table_type": table_type,
                "row_count": row_count,
                "column_count": len(columns),
                "null_percentage": round(
                    sum(col.get("null_percentage", 0) for col in columns) / len(columns), 1
                ) if columns else 0,
                "primary_keys": primary_keys,
                "foreign_keys": [],  # Dimension tables have no FKs in star schema
                "columns": [
                    {
                        "name": col.get("column_name", ""),
                        "data_type": col.get("data_type", ""),
                        "null_percentage": round(col.get("null_percentage", 0), 1),
                        "distinct_count": col.get("distinct_count", 0),
                        "is_primary_key": col.get("column_name") in primary_keys,
                        "is_foreign_key": False  # Dimensions don't have FKs
                    }
                    for col in columns
                ]
            })

        # Add virtual fact table
        enriched_tables.append({
            "table_name": "fact_table",
            "table_type": "FACT",
            "row_count": virtual_fact_metadata["row_count"],
            "column_count": virtual_fact_metadata["column_count"],
            "null_percentage": 0.0,
            "primary_keys": [],
            "foreign_keys": virtual_fact_fks,
            "columns": [
                {
                    "name": col.get("column_name", ""),
                    "data_type": col.get("data_type", ""),
                    "null_percentage": 0.0,
                    "distinct_count": col.get("distinct_count", 0),
                    "is_primary_key": False,
                    "is_foreign_key": True
                }
                for col in virtual_fact_columns
            ]
        })

        # Build relationships: fact_table → each dimension
        enriched_relationships = []

        for fk in virtual_fact_fks:
            enriched_relationships.append({
                "from_table": "fact_table",
                "from_column": fk.get("column", ""),
                "to_table": fk.get("references_table", ""),
                "to_column": fk.get("references_column", ""),
                "from_table_role": "FACT",
                "to_table_role": "DIM",
                "relationship_type": "M:1",
                "join_type": "INNER",
                "cardinality": {
                    "from": "MANY",
                    "to": "ONE"
                }
            })

        complete_analysis = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "job_id": job_id,
            "schemas_analyzed": len(schemas),
            
            # EXPLICIT STAR SCHEMA MODEL
            "model": {
                "type": "STAR_SCHEMA",
                "fact_table": "fact_table",
                "dimension_tables": all_dim_tables
            },
            
            "tables": enriched_tables,
            "relationships": enriched_relationships,
            
            "summary": {
                "total_tables": len(enriched_tables),
                "fact_tables": ["fact_table"],
                "dimension_tables": all_dim_tables,
                "total_relationships": len(enriched_relationships),
                "total_rows": sum(t["row_count"] for t in enriched_tables)
            }
        }

        # ============================================================
        # STEP 5: SAVE RESULTS
        # ============================================================
        logging.info("=" * 80)
        logging.info("💾 STEP 5: SAVING RESULTS")
        logging.info("=" * 80)
        
        relationship_path = f"{job_folder}relationships/relationship.json"
        relationship_blob = container_client.get_blob_client(relationship_path)
        
        relationship_blob.upload_blob(
            json.dumps(complete_analysis, indent=2, cls=NumpyEncoder),
            overwrite=True
        )
        
        logging.info(f"✅ Saved: {relationship_path}")
        
        # Update job status to completed
        job_data['status'] = 'completed'
        job_data['completed_at'] = datetime.utcnow().isoformat()
        job_data['relationship_file'] = relationship_path
        job_data['total_tables'] = len(enriched_tables)
        job_data['total_relationships'] = len(enriched_relationships)
        
        job_status_blob.upload_blob(
            json.dumps(job_data, indent=2),
            overwrite=True
        )
        
        logging.info("=" * 80)
        logging.info("🎉 PROCESSING COMPLETED!")
        logging.info(f"   Files: {processed_count}")
        logging.info(f"   Tables: {len(enriched_tables)}")
        logging.info(f"   Relationships: {len(enriched_relationships)}")
        logging.info("=" * 80)
        
        # ============================================================
        # RETURN RESULTS
        # ============================================================
        return func.HttpResponse(
            json.dumps({
                "status": "completed",
                "message": "Processing completed successfully",
                "stage": "completed",
                "data": complete_analysis
            }, cls=NumpyEncoder),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.exception(f"🔥 Error: {e}")
        
        # Update status to failed
        try:
            job_data['status'] = 'failed'
            job_data['error'] = str(e)
            job_data['failed_at'] = datetime.utcnow().isoformat()
            job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
        except:
            pass
        
        return func.HttpResponse(
            json.dumps({
                "status": "failed",
                "message": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )