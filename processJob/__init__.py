"""
HTTP-triggered Azure Function that does EVERYTHING:
1. Discovers files in folder
2. Extracts schemas (like jobProcessor)
3. Detects relationships / builds normalized ER model (like AIAnalyzer)
4. Validates model before saving
5. Saves all results

model_source in output:
  "ai"                 → AI ER model succeeded
  "rule_based_fallback"→ AI failed; basic structure shown; retry available
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
root_dir    = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
from azure.storage.blob import BlobServiceClient

# Define these at the top level
USER_CONTAINER_NAME = os.getenv("USER_CONTAINER_NAME", "userdata")
RELATIONSHIPS_CONTAINER_NAME = os.getenv("RELATIONSHIPS_CONTAINER_NAME", "relationships")
NORMALIZED_CONTAINER_NAME = os.getenv("NORMALIZED_CONTAINER_NAME", "normalized")
METADATA_CONTAINER_NAME = os.getenv("METADATA_CONTAINER_NAME", "metadata")

try:
    from shared.shared import (
        extract_schema_metadata,
        extract_schema_from_json_file,
        detect_relationships,
        validate_er_model,
        NumpyEncoder,
        NULL_THRESHOLD_FOR_KEY,
        STORAGE_ACCOUNT_NAME,
        STORAGE_ACCOUNT_KEY,
    )
    logging.info("✅ Shared module imported")
except Exception as e:
    logging.error(f"❌ Failed to import shared: {e}")
    raise

SUPPORTED_EXTENSIONS = [".csv", ".parquet", ".json"]


# ====================================================================
# MAIN ENTRY POINT
# ====================================================================
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP-triggered function that processes a job folder.

    Accepted body params:
      user_id        (required)
      job_id         (required)
      container_name (optional, default: datamodelling)
      ai_only        (optional bool) — skip schema extraction,
                      re-run only AI ER modeling on existing schemas.
                      Used by the frontend "Retry AI Modeling" button.
    """
    job_data        = {}
    job_status_blob = None

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
            return _error_response("Invalid JSON in request body", 400)

        user_id        = body.get("user_id")
        job_id         = body.get("job_id")
        container_name = USER_CONTAINER_NAME
        ai_only        = bool(body.get("ai_only", False))

        if not user_id or not job_id:
            return _error_response("user_id and job_id are required", 400)

        logging.info(f"📦 Container : {container_name}")
        logging.info(f"👤 User      : {user_id}")
        logging.info(f"📋 Job       : {job_id}")
        logging.info(f"🔁 AI-only   : {ai_only}")

        # ============================================================
        # CONNECT TO STORAGE
        # ============================================================
        conn_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={STORAGE_ACCOUNT_NAME};"
            f"AccountKey={STORAGE_ACCOUNT_KEY};"
            f"EndpointSuffix=core.windows.net"
        )
        blob_service     = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container_name)
        metadata_client = blob_service.get_container_client(METADATA_CONTAINER_NAME)
        rel_client = blob_service.get_container_client(RELATIONSHIPS_CONTAINER_NAME)
        norm_client = blob_service.get_container_client(NORMALIZED_CONTAINER_NAME)

        job_folder       = f"{user_id}/{job_id}/"
        job_status_path  = f"{user_id}/{job_id}/job_status.json"

        job_data = {
            "user_id":                user_id,
            "job_id":                 job_id,
            "container_name":         container_name,
            "status":                 "processing",
            "created_at":             datetime.utcnow().isoformat(),
            "processing_started_at":  datetime.utcnow().isoformat(),
            "ai_only":                ai_only
        }
        job_status_blob = metadata_client.get_blob_client(f"{user_id}/{job_id}/job_status.json")
        job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)

        # ============================================================
        # AI-ONLY MODE — skip directly to Step 3
        # (used by "Retry AI Modeling" frontend button)
        # ============================================================
        if ai_only:
            return _handle_ai_only_retry(
                container_client, metadata_client, rel_client, norm_client, job_folder, job_data,
                job_status_blob, user_id, job_id
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
            filename  = blob_path.split("/")[-1]

            # Skip metadata-like files
            if filename.startswith(("schema_", "job_", "batch_", "analysis_", "relationship")):
                continue

            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                processable_files.append({
                    "blob":      blob,
                    "path":      blob_path,
                    "name":      filename,
                    "extension": file_ext,
                    "size":      blob.size
                })

        if not processable_files:
            job_data["status"] = "failed"
            job_data["error"]  = "No processable files found"
            job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
            return _error_response("No processable files found", 400)

        logging.info(f"✅ Found {len(processable_files)} file(s)")
        for f in processable_files:
            logging.info(f"   📄 {f['name']} ({f['extension']}, {f['size']} bytes)")

        # ============================================================
        # STEP 2: EXTRACT SCHEMAS
        # ============================================================
        logging.info("=" * 80)
        logging.info("📋 STEP 2: EXTRACTING SCHEMAS")
        logging.info("=" * 80)

        schemas         = []
        processed_count = 0
        failed_files    = []

        for file_info in processable_files:
            try:
                filename  = file_info["name"]
                file_ext  = file_info["extension"]
                blob_path = file_info["path"]

                logging.info(f"🔍 Processing: {filename}")

                blob_client = container_client.get_blob_client(blob_path)
                file_data   = blob_client.download_blob().readall()

                metadata = None

                if file_ext == ".json":
                    metadata = extract_schema_from_json_file(file_data, filename, blob_path)

                elif file_ext == ".csv":
                    try:
                        df = pd.read_csv(io.BytesIO(file_data))
                    except Exception:
                        df = pd.read_csv(
                            io.BytesIO(file_data), engine="python", on_bad_lines="skip"
                        )
                    if df is not None and not df.empty:
                        metadata = extract_schema_metadata(df, filename, blob_path)

                elif file_ext == ".parquet":
                    df = pd.read_parquet(io.BytesIO(file_data))
                    if df is not None and not df.empty:
                        metadata = extract_schema_metadata(df, filename, blob_path)

                if metadata:
                    metadata["user_id"]         = user_id
                    metadata["job_id"]          = job_id
                    metadata["blob_size_bytes"] = file_info["size"]

                    base_name       = os.path.splitext(filename)[0]
                    schema_filename = f"schema_{base_name}.json"
                    schema_path     = f"{user_id}/{job_id}/schema_{base_name}.json"

                    schema_blob = metadata_client.get_blob_client(schema_path)
                    schema_blob.upload_blob(
                        json.dumps(metadata, indent=2, cls=NumpyEncoder), overwrite=True
                    )

                    schemas.append(metadata)
                    processed_count += 1
                    logging.info(
                        f"   ✅ {filename}: {metadata['column_count']} cols, "
                        f"{metadata['row_count']} rows"
                    )
                else:
                    failed_files.append(filename)
                    logging.error(f"   ❌ Failed: {filename}")

            except Exception as e:
                failed_files.append(file_info.get("name", "unknown"))
                logging.exception(f"   ❌ Error processing file: {e}")

        if not schemas:
            job_data["status"] = "failed"
            job_data["error"]  = "Schema extraction failed for all files"
            job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
            return _error_response("Schema extraction failed", 500)

        logging.info("=" * 80)
        logging.info(f"✅ Schema extraction complete:")
        logging.info(f"   - Processed : {processed_count}/{len(processable_files)}")
        logging.info(f"   - Failed    : {len(failed_files)}")
        logging.info("=" * 80)

        job_data["status"]                         = "schema_extraction_complete"
        job_data["schemas_extracted"]              = processed_count
        job_data["failed_files"]                   = failed_files
        job_data["schema_extraction_completed_at"] = datetime.utcnow().isoformat()
        job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)

        # ============================================================
        # STEP 3 + 4 + 5: ER MODELING, VIRTUAL FACT, SAVE
        # ============================================================
        return _run_er_modeling_and_save(
            schemas, container_client, job_folder,
            job_data, job_status_blob,
            user_id, job_id, processed_count, failed_files,
            rel_client, norm_client
        )

    except Exception as e:
        logging.exception(f"🔥 Unhandled error: {e}")
        try:
            job_data["status"]    = "failed"
            job_data["error"]     = str(e)
            job_data["failed_at"] = datetime.utcnow().isoformat()
            if job_status_blob:
                job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
        except Exception:
            pass
        return _error_response(str(e), 500)


# ====================================================================
# AI-ONLY RETRY  (called when frontend clicks "Retry AI Modeling")
# ====================================================================
def _handle_ai_only_retry(
    container_client, metadata_client, rel_client, norm_client, job_folder, job_data, job_status_blob, user_id, job_id
):
    """
    Re-loads existing schemas from blob storage and re-runs only the
    AI ER modeling step. Does not re-extract schemas from source files.
    """
    logging.info("=" * 80)
    logging.info("🔁 AI-ONLY RETRY MODE")
    logging.info("=" * 80)

    job_data["status"] = "analyzing_relationships"
    job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)

    # Load existing schema files from metadata folder
    metadata_prefix = f"{user_id}/{job_id}/"
    schema_blobs    = [
        b for b in metadata_client.list_blobs(name_starts_with=metadata_prefix)
        if b.name.split("/")[-1].startswith("schema_")
    ]

    schemas = []
    for blob in schema_blobs:
        blob_client = metadata_client.get_blob_client(blob.name)
        data        = json.loads(blob_client.download_blob().readall())
        schemas.append(data)

    if not schemas:
        job_data["status"] = "failed"
        job_data["error"]  = "No existing schemas found for AI retry"
        job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)
        return _error_response("No schemas found to retry", 400)

    logging.info(f"✅ Loaded {len(schemas)} existing schema(s) for AI retry")

    return _run_er_modeling_and_save(
        schemas, container_client, job_folder,
        job_data, job_status_blob,
        user_id, job_id,
        processed_count=len(schemas), failed_files=[],
        rel_client=rel_client, norm_client=norm_client
    )


# ====================================================================
# SHARED ER MODELING + VIRTUAL FACT + SAVE PIPELINE
# ====================================================================
def _run_er_modeling_and_save(
    schemas, container_client, job_folder,
    job_data, job_status_blob,
    user_id, job_id,
    processed_count, failed_files,
    rel_client, norm_client
):
    """
    Steps 3–5: ER modeling, virtual fact table, validation, save.
    Shared between normal flow and AI-only retry.
    """

    # ============================================================
    # STEP 3: DETECT RELATIONSHIPS / BUILD ER MODEL
    # ============================================================
    logging.info("=" * 80)
    logging.info("🧠 STEP 3: BUILDING ER MODEL")
    logging.info("=" * 80)

    job_data["status"] = "analyzing_relationships"
    job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)

    relationship_info = detect_relationships(schemas)

    model_source       = relationship_info.get("model_source", "ai")
    ai_retry_available = relationship_info.get("ai_retry_available", False)

    logging.info(f"   model_source       = {model_source}")
    logging.info(f"   ai_retry_available = {ai_retry_available}")
    logging.info(f"   entities           = {len(relationship_info.get('tables', []))}")
    logging.info(f"   relationships      = {len(relationship_info.get('relationships', []))}")

    # ============================================================
    # STEP 3.5: BUILD STAR SCHEMA FROM NORMALIZED ENTITIES
    # ============================================================
    logging.info("=" * 80)
    logging.info("🏗️  STEP 3.5: BUILDING STAR SCHEMA")
    logging.info("=" * 80)

    virtual_fact_columns = []
    virtual_fact_fks     = []
    existing_columns     = []
    all_dim_tables       = []

    source_names = {s.get("table_name", "") for s in schemas}
    normalized_tables = [
        t for t in relationship_info.get("tables", [])
        if t.get("table").lower() not in source_names
        and t.get("table").lower() != "fact_table"
        and t.get("table")  # not empty
    ]
    # Identify candidate fact table:
    # Entity with the most outgoing FKs = transactional grain
    def outgoing_fk_count(t):
        return len(t.get("foreign_keys", []))

    # All entities with outgoing FKs are candidates
    candidates = [t for t in normalized_tables if outgoing_fk_count(t) > 0]

    # Pick the one with most outgoing FKs as fact
    fact_entity = max(candidates, key=outgoing_fk_count) if candidates else None

    fact_entity_name = fact_entity.get("table").lower() if fact_entity else None
    logging.info(f"   🎯 Candidate fact entity: {fact_entity_name}")

    # Build fact table FKs from the fact entity's FK columns
    if fact_entity:
        # 1. Add the fact entity's own PK
        fact_pk = fact_entity.get("primary_keys", [])
        for pk_col in fact_pk:
            if pk_col and pk_col not in existing_columns:
                existing_columns.append(pk_col)
                virtual_fact_columns.append({
                    "column_name":            pk_col,
                    "data_type":              "string",
                    "nullable":               False,
                    "null_count":             0,
                    "null_percentage":        0.0,
                    "distinct_count":         0,
                    "cardinality_percentage": 0.0,
                    "is_potential_key":       True,
                    "is_foreign_key":         False,
                    "is_primary_key":         True,
                    "sample_values":          []
                })

        # 2. Add FK columns (links to dimension tables)
        for fk in fact_entity.get("foreign_keys", []):
            col_name = fk.get("column")
            if col_name and col_name not in existing_columns:
                existing_columns.append(col_name)
                virtual_fact_columns.append({
                    "column_name":            col_name,
                    "data_type":              "string",
                    "nullable":               False,
                    "null_count":             0,
                    "null_percentage":        0.0,
                    "distinct_count":         0,
                    "cardinality_percentage": 0.0,
                    "is_potential_key":       False,
                    "is_foreign_key":         True,
                    "is_primary_key":         False,
                    "sample_values":          []
                })
                virtual_fact_fks.append({
                    "column":            col_name,
                    "references_table":  fk.get("references_table").lower(),
                    "references_column": fk.get("references_column")
                })
        # 3. Add numeric measure columns from the fact entity
        # These are the actual measurable values — quantity, amount, price, etc.
        numeric_types = {"int", "double", "float", "decimal", "bigint", "long", "number"}
        for attr in fact_entity.get("attributes", []):
            attr_name  = attr.get("name", "")
            attr_type  = attr.get("data_type", "").lower()
            is_pk      = attr_name in fact_pk
            is_fk      = attr.get("is_foreign_key", False)
            is_numeric = any(t in attr_type for t in numeric_types)
            is_high_null = attr.get("null_percentage", 0.0) > 5.0

            # Skip: already added as PK/FK, non-numeric, or high-null
            if attr_name in existing_columns:
                continue
            if not is_numeric:
                continue
            if is_high_null:
                continue

            existing_columns.append(attr_name)
            virtual_fact_columns.append({
                "column_name":            attr_name,
                "data_type":              attr.get("data_type", "double"),
                "nullable":               True,
                "null_count":             0,
                "null_percentage":        0.0,
                "distinct_count":         0,
                "cardinality_percentage": 0.0,
                "is_potential_key":       False,
                "is_foreign_key":         False,
                "is_primary_key":         False,
                "is_measure":             True,
                "sample_values":          []
            })

    # Dimension tables = all normalized entities EXCEPT the fact entity
    all_dim_tables = [
        t.get("table").lower() for t in normalized_tables
        if t.get("table") != fact_entity_name
    ]

    # Row count = row count of the source file the fact entity came from
    derived_from = fact_entity.get("derived_from", "") if fact_entity else ""
    fact_row_count = next(
        (s.get("row_count", 0) for s in schemas
        if s.get("table_name") == derived_from),
        0
    )

    logging.info(f"   ✅ Fact entity : {fact_entity_name}")
    logging.info(f"   ✅ Dimensions  : {all_dim_tables}")
    logging.info(f"   ✅ FK columns  : {len(virtual_fact_columns)}")
    logging.info(f"   ✅ Row count   : {fact_row_count}")

    # ============================================================
    # STEP 4: BUILD ENRICHED OUTPUT
    # ============================================================
    logging.info("=" * 80)
    logging.info("📊 STEP 4: BUILDING ENRICHED OUTPUT")
    logging.info("=" * 80)

    enriched_tables = []

    # Dimension tables
    for schema in schemas:
        table_name = schema.get("table_name", "")
        columns    = schema.get("columns", [])
        row_count  = schema.get("row_count", 0)

        table_info   = next(
            (t for t in relationship_info.get("tables", []) if t.get("table") == table_name),
            None
        )
        primary_keys = table_info.get("primary_keys", []) if table_info else []
        surrogate_keys = table_info.get("surrogate_keys", []) if table_info else []

        avg_null_pct = (
            round(sum(col.get("null_percentage", 0) for col in columns) / len(columns), 1)
            if columns else 0
        )
        

        enriched_tables.append({
            "table_name":     table_name,
            "table_type":    "SOURCE",   # was "DIM"
            "is_source_only": True,      # ADD THIS
            "is_normalized":  False,
            "row_count":      row_count,
            "column_count":   len(columns),
            "null_percentage": avg_null_pct,
            "primary_keys":   primary_keys,
            "surrogate_keys": surrogate_keys,
            "foreign_keys":   [],
            "columns": [
                {
                    "name":           col.get("column_name", ""),
                    "data_type":      col.get("data_type", ""),
                    "null_percentage": round(col.get("null_percentage", 0), 1),
                    "distinct_count": col.get("distinct_count", 0),
                    "is_primary_key": col.get("column_name") in primary_keys,
                    "is_surrogate":   col.get("column_name") in surrogate_keys,
                    "is_foreign_key": False,
                    "display_label":  (
                        f"{col.get('column_name', '')} ✨"
                        if col.get("column_name") in surrogate_keys
                        else col.get("column_name", "")
                    ),
                    "tooltip": (
                        "AI-generated surrogate key — not present in source file."
                        if col.get("column_name") in surrogate_keys
                        else f"Source column from {table_name}"
                    )
                }
                for col in columns
            ]
        })

    # Virtual fact table
    enriched_tables.append({
        "table_name":     "fact_table",
        "table_type":     "FACT",
        "row_count": fact_row_count,
        "column_count": len(virtual_fact_columns),
        "null_percentage": 0.0,
        "primary_keys":   fact_entity.get("primary_keys", []) if fact_entity else [],
        "surrogate_keys": fact_entity.get("surrogate_keys", []) if fact_entity else [],
        "foreign_keys":   virtual_fact_fks,
        "columns": [
            {
                "name":           col.get("column_name", ""),
                "data_type":      col.get("data_type", ""),
                "null_percentage": 0.0,
                "distinct_count": col.get("distinct_count", 0),
                "is_primary_key": col.get("is_primary_key", False),
                "is_surrogate": False,
                "is_foreign_key": col.get("is_foreign_key", False),
                "display_label":  col.get("column_name", ""),
                "tooltip":        "Foreign key in virtual fact table"
            }
            for col in virtual_fact_columns
        ]
    })
    # Add normalized logical entities (AI-decomposed from source tables)
    # These are the real output when model_source == "ai" and decomposition happened
    if model_source == "ai":
        for norm_table in relationship_info.get("tables", []):
            entity_name  = norm_table.get("table", "").lower()
            derived_from = norm_table.get("derived_from", "")

            # Skip physical source tables (already added above) and fact_table
            if not entity_name or entity_name == "fact_table":
                continue
            # Skip if it's a physical source table name (not a derived entity)
            source_names = [s.get("table_name", "") for s in schemas]
            if entity_name in source_names:
                continue

            enriched_tables.append({
                "table_name":      entity_name,
                "table_type":      "DIM",
                "derived_from":    derived_from,
                "is_normalized":   True,
                "row_count":       0,
                "column_count":    len(norm_table.get("attributes", [])),
                "null_percentage": 0.0,
                "primary_keys":    norm_table.get("primary_keys", []),
                "surrogate_keys":  norm_table.get("surrogate_keys", []),
                "foreign_keys":    norm_table.get("foreign_keys", []),
                "columns": [
                    {
                        "name":          attr.get("name", ""),
                        "data_type":     attr.get("data_type", "string"),
                        "null_percentage": 0.0,
                        "distinct_count": 0,
                        "is_primary_key": attr.get("name") in norm_table.get("primary_keys", []),
                        "is_surrogate":  attr.get("is_surrogate", False),
                        "is_foreign_key": attr.get("is_foreign_key", False),
                        "display_label": attr.get("display_label", attr.get("name", "")),
                        "tooltip":       attr.get("tooltip", "")
                    }
                    for attr in norm_table.get("attributes", [])
                ]
            })

    # Relationships
    enriched_relationships = []
    for fk in virtual_fact_fks:
        enriched_relationships.append({
            "from_table":        "fact_table",
            "from_column":       fk.get("column", ""),
            "to_table":          fk.get("references_table", ""),
            "to_column":         fk.get("references_column", ""),
            "from_table_role":   "FACT",
            "to_table_role":     "DIM",
            "relationship_type": "M:1",
            "join_type":         "INNER",
            "confidence":        None,
            "cardinality": {
                "from": "MANY",
                "to":   "ONE"
            }
        })

    # Also carry through any AI-detected entity-level relationships
    for rel in relationship_info.get("relationships", []):
        # Avoid duplicating fact→dim ones already added above
        if rel.get("from_table") != "fact_table":
            enriched_relationships.append(rel)
    normalized_table_names = [
            t["table_name"] for t in enriched_tables
            if t.get("is_normalized") and t["table_name"] != "fact_table"
        ]
    physical_table_names = [
            t["table_name"] for t in enriched_tables
            if not t.get("is_source_only") and t["table_name"] != "fact_table"
        ]

    complete_analysis = {
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "user_id":            user_id,
        "job_id":             job_id,
        "schemas_analyzed":   len(schemas),

        # Model metadata
        "model": {
            "type":             "STAR_SCHEMA",
            "fact_table":       "fact_table",
            "dimension_tables": all_dim_tables
        },

        # ER model extras (present when model_source = "ai")
        "raw_entity_analysis": relationship_info.get("raw_entity_analysis", {}),
        "cardinality_diagram": relationship_info.get("cardinality_diagram", ""),
        "standalone_entities": relationship_info.get("standalone_entities", []),
        "observations":        relationship_info.get("observations", []),

        # Model provenance — frontend uses this
        "model_source":        model_source,
        "ai_retry_available":  ai_retry_available,

        "tables":              enriched_tables,
        "relationships":       enriched_relationships,

        

        "summary": {
            "total_tables": len([t for t in enriched_tables if not t.get("is_source_only")]),
            "fact_tables":            ["fact_table"],
            "dimension_tables":       all_dim_tables,
            "normalized_entities":    normalized_table_names,
            "physical_source_tables": physical_table_names,
            "total_relationships":    len(enriched_relationships),
            "total_rows":             sum(t["row_count"] for t in enriched_tables)
        },
    }

    # ============================================================
    # NORMALIZE ENTITY NAMES (BEFORE VALIDATION)
    # ============================================================

    normalized_name_map = {}

    # 1️⃣ Normalize table names
    for table in enriched_tables:
        original = table["table_name"]
        lower    = original.lower()
        normalized_name_map[original] = lower
        normalized_name_map[lower] = lower 
        table["table_name"] = lower

    # 2️⃣ Normalize relationship references
    for rel in enriched_relationships:
        rel["from_table"] = normalized_name_map.get(
            rel.get("from_table"),
            rel.get("from_table", "").lower()
        )
        rel["to_table"] = normalized_name_map.get(
            rel.get("to_table"),
            rel.get("to_table", "").lower()
        )

    # 3️⃣ Normalize summary metadata (CRITICAL FIX)
    complete_analysis["model"]["dimension_tables"] = [
        normalized_name_map.get(t, t.lower())
        for t in complete_analysis["model"]["dimension_tables"]
    ]

    complete_analysis["summary"]["dimension_tables"] = [
        normalized_name_map.get(t, t.lower())
        for t in complete_analysis["summary"]["dimension_tables"]
    ]

    complete_analysis["summary"]["normalized_entities"] = [
        normalized_name_map.get(t, t.lower())
        for t in complete_analysis["summary"]["normalized_entities"]
    ]

    complete_analysis["summary"]["physical_source_tables"] = [
        normalized_name_map.get(t, t.lower())
        for t in complete_analysis["summary"]["physical_source_tables"]
    ]
    # ============================================================
    # STEP 5: VALIDATE BEFORE SAVING
    # ============================================================
    logging.info("=" * 80)
    logging.info("🔍 STEP 5: VALIDATING ER MODEL")
    logging.info("=" * 80)

    validation_errors = validate_er_model(complete_analysis)
    critical_errors   = [e for e in validation_errors if e.startswith("CRITICAL")]
    warnings          = [e for e in validation_errors if e.startswith("WARNING")]

    for warn in warnings:
        logging.warning(f"   ⚠️  {warn}")

    complete_analysis["validation_errors"]   = validation_errors
    complete_analysis["validation_warnings"] = warnings

    if critical_errors:
        logging.error(f"❌ {len(critical_errors)} critical validation error(s):")
        for err in critical_errors:
            logging.error(f"   🔴 {err}")

        # Do not save a critically broken model — return error with details
        job_data["status"] = "failed"
        job_data["error"]  = f"ER model validation failed: {critical_errors}"
        job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)

        return func.HttpResponse(
            json.dumps({
                "status":            "failed",
                "message":           "ER model failed critical validation. Not saved.",
                "critical_errors":   critical_errors,
                "warnings":          warnings,
                "model_source":      model_source,
                "ai_retry_available": ai_retry_available
            }, cls=NumpyEncoder),
            mimetype="application/json",
            status_code=422
        )

    logging.info(
        f"✅ Validation passed "
        f"({len(warnings)} warning(s), 0 critical errors)"
    )

    # ============================================================
    # STEP 6: SAVE RESULTS
    # ============================================================
    logging.info("=" * 80)
    logging.info("💾 STEP 6: SAVING RESULTS")
    logging.info("=" * 80)

    relationship_path = f"{user_id}/{job_id}/relationship.json" 
    relationship_blob = rel_client.get_blob_client(f"{user_id}/{job_id}/relationship.json")
    relationship_blob.upload_blob(
        json.dumps(complete_analysis, indent=2, cls=NumpyEncoder), overwrite=True
    )
    logging.info(f"✅ Saved: {relationship_path}")

    # ============================================================
    # STEP 6.5: WRITE NORMALIZED SCHEMA FILES
    # Only when AI modeling succeeded — fallback has no normalized entities
    # ============================================================
    if model_source == "ai":
        normalized_entities = relationship_info.get("raw_entity_analysis", {})
        # Use normalized_entities from the tables list instead
        for table in relationship_info.get("tables", []):
            entity_name = table.get("table")
            if not entity_name or entity_name == "fact_table":
                continue

            normalized_schema = {
                "table_name":           entity_name,
                "source":               "normalized",
                "derived_from":         table.get("derived_from", ""),
                "is_normalized":        True,
                "is_surrogate_entity":  bool(table.get("surrogate_keys")),
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "user_id":              user_id,
                "job_id":               job_id,
                "primary_keys":         table.get("primary_keys", []),
                "surrogate_keys":       table.get("surrogate_keys", []),
                "foreign_keys":         table.get("foreign_keys", []),
                "row_count":            0,      # logical — no physical rows yet
                "column_count":         len(table.get("attributes", [])),
                "columns": [
                    {
                        "column_name":   attr.get("name", ""),
                        "data_type":     attr.get("data_type", "string"),
                        "is_primary_key": attr.get("name") in table.get("primary_keys", []),
                        "is_foreign_key": attr.get("is_foreign_key", False),
                        "is_surrogate":  attr.get("is_surrogate", False),
                        "references":    attr.get("references"),
                        "source":        attr.get("source", "unknown"),
                        "display_label": attr.get("display_label", attr.get("name", "")),
                        "tooltip":       attr.get("tooltip", ""),
                        "nullable":      not attr.get("name") in table.get("primary_keys", [])
                    }
                    for attr in table.get("attributes", [])
                ]
            }

            normalized_path = f"{user_id}/{job_id}/schema_{entity_name}.json"
            norm_blob = norm_client.get_blob_client(normalized_path)
            norm_blob.upload_blob(
                json.dumps(normalized_schema, indent=2, cls=NumpyEncoder),
                overwrite=True
            )
            logging.info(f"   📄 Normalized schema saved: {entity_name}")

        logging.info(f"✅ Normalized schemas written to {job_folder}normalized/")

    job_data["status"]              = "completed"
    job_data["completed_at"]        = datetime.utcnow().isoformat()
    job_data["relationship_file"]   = relationship_path
    job_data["total_tables"]        = len(enriched_tables)
    job_data["total_relationships"] = len(enriched_relationships)
    job_data["model_source"]        = model_source
    job_data["ai_retry_available"]  = ai_retry_available
    job_status_blob.upload_blob(json.dumps(job_data, indent=2), overwrite=True)

    logging.info("=" * 80)
    logging.info("🎉 PROCESSING COMPLETED!")
    logging.info(f"   Files         : {processed_count}")
    logging.info(f"   Tables        : {len(enriched_tables)}")
    logging.info(f"   Relationships : {len(enriched_relationships)}")
    logging.info(f"   Model source  : {model_source}")
    logging.info(f"   AI retry btn  : {ai_retry_available}")
    logging.info("=" * 80)

    return func.HttpResponse(
        json.dumps({
            "status":  "completed",
            "message": "Processing completed successfully",
            "stage":   "completed",
            "data":    complete_analysis
        }, cls=NumpyEncoder),
        mimetype="application/json",
        status_code=200
    )


# ====================================================================
# HELPERS
# ====================================================================
def _error_response(message: str, status_code: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"status": "error", "message": message}),
        status_code=status_code,
        mimetype="application/json"
    )