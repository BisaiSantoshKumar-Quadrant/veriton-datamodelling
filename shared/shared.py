"""
Shared utility functions for Azure Functions
"""

import logging
import json
import re
from datetime import datetime, timezone
import pandas as pd
import io
import os
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    from openai import AzureOpenAI
except ImportError as e:
    logger.error(f"Failed to import Azure dependencies: {e}")
    raise

# ====================================================================
# ENVIRONMENT VARIABLES
# ====================================================================
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY  = os.environ.get("STORAGE_ACCOUNT_KEY")
LAKEHOUSE_CONTAINER  = os.environ.get("LAKEHOUSE_CONTAINER", "lakehouse-data")
DDL_CONTAINER        = os.environ.get("DDL_CONTAINER", "ddl-scripts")

USER_CONTAINER_NAME = os.getenv("USER_CONTAINER_NAME", "userdata")
RELATIONSHIPS_CONTAINER_NAME = os.getenv("RELATIONSHIPS_CONTAINER_NAME", "relationships")
NORMALIZED_CONTAINER_NAME    = os.getenv("NORMALIZED_CONTAINER_NAME", "normalized")
METADATA_CONTAINER_NAME      = os.getenv("METADATA_CONTAINER_NAME", "metadata")

AI_ENDPOINT = os.environ.get("AI_ENDPOINT")
AI_API_KEY  = os.environ.get("AI_API_KEY")
AI_MODEL    = os.environ.get("AI_MODEL", "gpt-4.1-mini")

WORKSPACE_ID = os.environ.get("WORKSPACE_ID")
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID")

SUPPORTED_FORMATS = [".csv", ".parquet", ".json"]

if not STORAGE_ACCOUNT_NAME:
    logger.warning("STORAGE_ACCOUNT_NAME not set - this may cause issues")
if not STORAGE_ACCOUNT_KEY:
    logger.warning("STORAGE_ACCOUNT_KEY not set - this may cause issues")

BLOB_CONN_STR = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT_NAME};"
    f"AccountKey={STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# ====================================================================
# CONSTANTS
# ====================================================================
# A column with null_percentage above this threshold is NEVER a key
NULL_THRESHOLD_FOR_KEY = 5.0

# Relationships below this confidence score are rejected
CONFIDENCE_THRESHOLD = 0.7


# ====================================================================
# BLOB OPERATIONS
# ====================================================================
def list_blobs(container_name, prefix=""):
    """List blobs in a container with optional prefix."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
        container_client = blob_service_client.get_container_client(container_name)
        return list(container_client.list_blobs(name_starts_with=prefix))
    except Exception as e:
        logging.error(f"Error listing blobs: {e}")
        return []


def load_blob_json(container, blob_name):
    client = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    blob_client = client.get_container_client(container).get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    return json.loads(data)


# ====================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# ====================================================================
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def read_blob_if_exists(blob_name):
    """Read a blob if it exists, otherwise return None."""
    
    
    service   = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    container = service.get_container_client(METADATA_CONTAINER_NAME)
    client    = container.get_blob_client(blob_name)
    try:
        data = client.download_blob().readall().decode()
        return json.loads(data)
    except Exception:
        return None


def save_to_blob(content, blob_name, container_name):
    """Save content to Azure Blob Storage."""
    
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name
    )
    blob_client.upload_blob(content, overwrite=True)


# ====================================================================
# ER MODEL VALIDATION
# ====================================================================
def validate_er_model(er_model: dict) -> list:
    """
    Validates the ER model output before it is saved or consumed.

    Returns a list of error strings. Empty list = valid.
    Critical errors (prefixed CRITICAL) should block saving.
    Warnings (prefixed WARNING) are logged but do not block.
    """
    errors = []

    tables = er_model.get("tables", [])
    if not tables:
        errors.append("CRITICAL: No entities found in ER model.")
        return errors  # No point continuing

    entity_names = set()
    for t in tables:
        name = t.get("table_name") or t.get("table") or "<unnamed>"
        entity_names.add(name)
        
        if not t.get("primary_keys"):
            if not t.get("surrogate_keys"):
                # Silently skip expected cases
                if name == "fact_table" or t.get("table_type") == "FACT":
                    pass
                elif t.get("is_normalized") is False and any(
                    e.get("derived_from") == name
                    for e in er_model.get("tables", [])
                ):
                    pass  # Source table that was decomposed - no PK expected
                else:
                    errors.append(f"WARNING: Entity '{name}' has no primary key or surrogate key.")

        # Surrogate keys must NOT appear in source_columns
        surrogate_set = set(t.get("surrogate_keys", []))
        source_set    = set(t.get("source_columns", []))
        overlap = surrogate_set & source_set
        if overlap:
            errors.append(
                f"WARNING: Entity '{name}' has columns listed as both surrogate and source: {overlap}"
            )

    # Relationships must reference known entities
    for rel in er_model.get("relationships", []):
        from_e = rel.get("from_table", "")
        to_e   = rel.get("to_table", "")
        conf   = rel.get("confidence", 1.0)

        if from_e not in entity_names:
            errors.append(f"CRITICAL: Relationship references unknown entity '{from_e}'")
        if to_e not in entity_names:
            errors.append(f"CRITICAL: Relationship references unknown entity '{to_e}'")
        if not rel.get("relationship_type"):
            errors.append(f"WARNING: Relationship {from_e}→{to_e} is missing type.")
        if conf is not None and conf < CONFIDENCE_THRESHOLD:
            errors.append(
                f"CRITICAL: Relationship {from_e}→{to_e} has confidence {conf} below threshold "
                f"{CONFIDENCE_THRESHOLD} — should have been filtered before saving."
            )

    return errors


# ====================================================================
# RELATIONSHIP DETECTION  (AI-powered with fallback)
# ====================================================================
def detect_relationships(schemas):
    """
    Detect relationships and produce a normalized ER model using AI.
    Falls back to rule-based detection if AI fails or returns bad JSON.
    """
    logging.info(f"🤖 Using AI to build ER model for {len(schemas)} tables")

    client = AzureOpenAI(
        api_key=AI_API_KEY,
        api_version="2024-05-01-preview",
        azure_endpoint=AI_ENDPOINT
    )

    # Build a richer input for the AI — include null_percentage so it can
    # honour the high-null-column rule itself as well
    formatted_schemas = []
    for schema in schemas:
        table_name = schema.get("table_name", "")
        columns    = schema.get("columns", [])
        formatted_schemas.append({
            "table_name": table_name,
            "row_count":  schema.get("row_count", 0),
            "columns": [
                {
                    "name":            col.get("column_name", ""),
                    "data_type":       col.get("data_type", ""),
                    "null_percentage": round(col.get("null_percentage", 0.0), 2),
                    "distinct_count":  col.get("distinct_count", 0),
                    "cardinality_pct": round(col.get("cardinality_percentage", 0.0), 2),
                    "is_potential_key": col.get("is_potential_key", False)
                }
                for col in columns
            ]
        })

    prompt = (
        "You are a Senior Data Architect specializing in data modeling and normalization.\n\n"
        "Analyze the provided database schemas and return a normalized ER model.\n\n"

        "🔒 Hard Rules:\n"
        "1. Use ONLY columns present in the input schemas.\n"
        "2. Do NOT invent attributes not derivable from the source.\n"
        "3. NEVER use a column with null_percentage > 5 as a primary key.\n"
        "4. If a table is denormalized (multiple conceptual entities in one table), decompose it.\n"
        "5. For decomposed entities with no natural PK, assign a surrogate key "
        "   Every entity MUST have all its columns listed in attributes[], "
        "including composite key columns. Never leave attributes[] empty.\n"
        "(e.g. StartupID, InvestorID). Mark these with is_surrogate: true.\n"
        "6. Surrogate key names must NOT match any existing source column name.\n"
        "6b. FK relationships MUST use the FK column on the child side as from_column, "
        "never the PK of the child. "
        "Example: Order.customer_code → Customer.customer_code is correct. "
        "Order.order_id → Customer.customer_code is WRONG.\n"
        "7. For every relationship, assign a confidence score 0.0–1.0. "
        "OMIT any relationship where confidence < 0.7.\n"
        "8. Output ONLY valid JSON. No markdown. No explanation outside the JSON.\n\n"

        "Steps:\n"
        "Step 1 — Analyse each table: identify if it is denormalized and what conceptual "
        "entities are embedded in it.\n"
        "Step 2 — Decompose denormalized tables into normalized entities with proper PKs/FKs.\n"
        "Step 3 — Determine relationships (1:1, 1:M, M:N) between entities.\n"
        "Step 4 — Assign confidence scores to each relationship.\n"
        "Step 5 — Build a single-line cardinality diagram.\n\n"

        "Output format (JSON ONLY, no other text):\n"
        "{\n"
        '  "raw_entity_analysis": {\n'
        '    "TableName": {\n'
        '      "source_file": "filename.csv",\n'
        '      "row_count": 0,\n'
        '      "is_denormalized": true,\n'
        '      "embedded_concepts": ["Concept1", "Concept2"],\n'
        '      "columns": [\n'
        '        {\n'
        '          "name": "col",\n'
        '          "data_type": "string",\n'
        '          "null_percentage": 0.0,\n'
        '          "observations": "e.g. mostly null — excluded from key inference"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  },\n'
        '  "normalized_entities": {\n'
        '    "EntityName": {\n'
        '      "derived_from": "OriginalTableName",\n'
        '      "primary_key": ["ColumnName"],\n'
        '      "composite_key": [],\n'
        '      "attributes": [\n'
        '        {\n'
        '          "name": "ColName",\n'
        '          "data_type": "string",\n'
        '          "is_foreign_key": false,\n'
        '          "is_surrogate": false,\n'
        '          "references": null\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  },\n'
        '  "relationships": [\n'
        '    {\n'
        '      "from_entity": "ChildEntity",\n'
        '      "to_entity": "ParentEntity",\n'
        '      "type": "1:M",\n'
        '      "from_column": "FKCol",\n'
        '      "to_column": "PKCol",\n'
        '      "description": "One ParentEntity has many ChildEntities",\n'
        '      "confidence": 0.95\n'
        '    }\n'
        '  ],\n'
        '  "cardinality_diagram": "Industry 1 ────< Startup 1 ────< Investment >──── 1 Investor",\n'
        '  "standalone_entities": [],\n'
        '  "observations": [\n'
        '    "Free-text notes on data quality, missing PKs, high-null columns, or modeling decisions"\n'
        '  ]\n'
        "}\n\n"

        "Decision Policy:\n"
        "- When uncertain → do NOT create a relationship.\n"
        "- Confidence < 0.7 → omit the relationship entirely.\n"
        "- High-null columns (null_percentage > 5) → never use as PK.\n"
        "- Note all decisions and data quality issues in the observations array.\n\n"

        "SCHEMAS:\n"
        f"{json.dumps(formatted_schemas, indent=2, cls=NumpyEncoder)}"
    )

    try:
        logging.info("📤 Sending schemas to Azure OpenAI for ER modeling...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=8000
                )
                content   = response.choices[0].message.content.strip()
                ai_result = _extract_json_from_text(content)

                logging.info("✅ AI ER model received — validating structure...")
                return _transform_ai_result_to_standard_format(ai_result, schemas)

            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        logging.warning(
                            f"⚠️ Rate limited. Waiting {wait_time}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        logging.error("❌ Max retries reached. Using fallback.")
                        raise
                else:
                    raise

    except Exception as e:
        logging.exception("❌ AI ER modeling failed. Using rule-based fallback.")
        return _fallback_relationship_detection(schemas)


def _transform_ai_result_to_standard_format(ai_result, schemas):
    """
    Validates and transforms the AI ER output into the pipeline's standard format.

    Key guarantees:
    - Required top-level keys are present (raises ValueError if not)
    - Relationships below CONFIDENCE_THRESHOLD are rejected
    - Surrogate keys are cross-checked against actual source columns
    - is_surrogate is set deterministically, not trusted blindly from AI
    - model_source is set to "ai"
    """

    # ------------------------------------------------------------------
    # 1. Validate required top-level keys
    # ------------------------------------------------------------------
    required_keys = {"normalized_entities", "relationships"}
    missing = required_keys - set(ai_result.keys())
    if missing:
        raise ValueError(
            f"AI response is missing required keys: {missing}. "
            f"Triggering fallback."
        )

    normalized_entities = ai_result.get("normalized_entities", {})
    raw_relationships   = ai_result.get("relationships", [])
    observations        = ai_result.get("observations", [])

    if not isinstance(normalized_entities, dict):
        raise ValueError("'normalized_entities' must be a dict. Triggering fallback.")
    if not isinstance(raw_relationships, list):
        raise ValueError("'relationships' must be a list. Triggering fallback.")

    # ------------------------------------------------------------------
    # 2. Build ground-truth set of real source column names
    #    (used to deterministically identify surrogate keys)
    # ------------------------------------------------------------------
    source_column_names = {
        col.get("column_name", "").strip().lower()
        for schema in schemas
        for col in schema.get("columns", [])
    }

    # ------------------------------------------------------------------
    # 3. Build tables_info from normalized entities
    # ------------------------------------------------------------------
    tables_info = []

    for entity_name, entity_data in normalized_entities.items():
        if not isinstance(entity_data, dict):
            logging.warning(f"⚠️ Skipping malformed entity: {entity_name}")
            observations.append(f"WARNING: Skipped malformed entity definition for '{entity_name}'")
            continue

        primary_key = entity_data.get("primary_key", [])
        attributes  = entity_data.get("attributes", [])

        if not isinstance(attributes, list):
            attributes = []

        surrogate_keys = []
        source_columns = []
        foreign_keys   = []

        for attr in attributes:
            if not isinstance(attr, dict):
                continue

            attr_name      = attr.get("name", "")
            attr_name_low  = attr_name.strip().lower()

            # Deterministic surrogate detection:
            # If the attribute name does not exist in any source schema column → surrogate
            is_surrogate_deterministic = attr_name_low not in source_column_names

            # Override whatever the AI said with our deterministic check
            attr["is_surrogate"] = is_surrogate_deterministic

            # Add display metadata for frontend
            if is_surrogate_deterministic:
                attr["source"]        = "ai_generated"
                attr["display_label"] = f"{attr_name} ✨"
                attr["tooltip"]       = (
                    "This key was generated by AI during normalization. "
                    "It does not exist in your source file."
                )
                surrogate_keys.append(attr_name)
            else:
                # Find which source file this column came from
                source_file = _find_source_file(attr_name_low, schemas)
                attr["source"]        = source_file
                attr["display_label"] = attr_name
                attr["tooltip"]       = f"Source column from {source_file}"
                source_columns.append(attr_name)

            # Build FK list
            if attr.get("is_foreign_key", False):
                raw_ref = attr.get("references", "") or ""
                if not raw_ref:  # skip FK with no reference
                    continue
                if "." in raw_ref:
                    ref_table, ref_col = raw_ref.split(".", 1)
                else:
                    ref_table, ref_col = raw_ref, None

                foreign_keys.append({
                    "column":            attr_name,
                    "references_table":  ref_table.strip().lower(),
                    "references_column": ref_col
                })

        tables_info.append({
            "table":          entity_name,
            "derived_from":   entity_data.get("derived_from", ""),
            "primary_keys":   primary_key,
            "surrogate_keys": surrogate_keys,
            "source_columns": source_columns,
            "foreign_keys":   foreign_keys,
            "attributes":     attributes
        })

    # ------------------------------------------------------------------
    # 4. Filter relationships by confidence
    # ------------------------------------------------------------------
    relationships = []
    rejected_count = 0

    for rel in raw_relationships:
        if not isinstance(rel, dict):
            continue

        confidence = float(rel.get("confidence", 0.0))

        if confidence < CONFIDENCE_THRESHOLD:
            logging.warning(
                f"⚠️ Rejected low-confidence relationship: "
                f"{rel.get('from_entity')} → {rel.get('to_entity')} "
                f"(confidence={confidence:.2f})"
            )
            observations.append(
                f"Relationship '{rel.get('from_entity')}' → '{rel.get('to_entity')}' "
                f"rejected: confidence {confidence:.2f} < threshold {CONFIDENCE_THRESHOLD}"
            )
            rejected_count += 1
            continue

        relationships.append({
            "from_table":        rel.get("from_entity", ""),
            "from_column":       rel.get("from_column", ""),
            "to_table":          rel.get("to_entity", ""),
            "to_column":         rel.get("to_column", ""),
            "relationship_type": rel.get("type", ""),
            "description":       rel.get("description", ""),
            "confidence":        confidence
        })

    # ------------------------------------------------------------------
    # 5. Log summary
    # ------------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("✅ AI ER Model Transformation Complete:")
    logging.info(f"   - Entities:              {len(tables_info)}")
    logging.info(f"   - Relationships accepted: {len(relationships)}")
    logging.info(f"   - Relationships rejected: {rejected_count}")
    logging.info(f"   - Surrogate keys created: "
                 f"{sum(len(t['surrogate_keys']) for t in tables_info)}")
    for obs in observations:
        logging.info(f"   📝 {obs}")
    logging.info("=" * 60)

    return {
        "tables":              tables_info,
        "relationships":       relationships,
        "raw_entity_analysis": ai_result.get("raw_entity_analysis", {}),
        "cardinality_diagram": ai_result.get("cardinality_diagram", ""),
        "standalone_entities": ai_result.get("standalone_entities", []),
        "observations":        observations,
        "fact_tables":         [],
        "dimension_tables":    [],
        "model_source":        "ai",
        "ai_retry_available":  False
    }


def _find_source_file(col_name_lower: str, schemas: list) -> str:
    """Return the table_name that contains this column, or 'unknown'."""
    for schema in schemas:
        for col in schema.get("columns", []):
            if col.get("column_name", "").strip().lower() == col_name_lower:
                return schema.get("table_name", "unknown")
    return "unknown"


# ====================================================================
# RULE-BASED FALLBACK
# ====================================================================
def _fallback_relationship_detection(schemas):
    """
    Rule-based fallback when AI fails or returns invalid JSON.
    Does NOT attempt normalization — returns raw table structure.
    Sets model_source = "rule_based_fallback" and ai_retry_available = True
    so the frontend can offer the user a retry button.
    """
    logging.info("⚙️ Running rule-based fallback relationship detection")

    relationships = []
    tables_info   = []

    table_columns         = {}
    original_to_normalized = {}

    for schema in schemas:
        original_name  = schema.get("table_name", "").strip()
        normalized_name = _normalize_table_name(original_name)
        columns         = schema.get("columns", [])

        if normalized_name and columns:
            table_columns[normalized_name]          = columns
            original_to_normalized[normalized_name] = original_name
            logging.info(f"📋 Mapped: '{original_name}' → '{normalized_name}'")

    if not table_columns:
        logging.warning("⚠️ No valid tables found in fallback")
        return {
            "tables":             [],
            "relationships":      [],
            "model_source":       "rule_based_fallback",
            "ai_retry_available": True,
            "observations":       ["No valid tables found during fallback detection."]
        }

    all_table_names = list(table_columns.keys())

    # Step 1: Identify primary keys (with null guard)
    pk_map = {}
    for table_name, columns in table_columns.items():
        for col in columns:
            col_name       = col.get("column_name", "").lower().strip()
            is_potential   = col.get("is_potential_key", False)
            null_pct       = col.get("null_percentage", 100.0)

            # Hard null guard — never accept high-null column as PK
            if null_pct > NULL_THRESHOLD_FOR_KEY:
                continue

            if is_potential:
                if col_name.endswith("_id"):
                    base = col_name.replace("_id", "")
                    if base == table_name or base + "s" == table_name or base == table_name + "s":
                        pk_map[table_name] = col_name
                        logging.info(f"✓ PK (fallback): {table_name}.{col_name}")
                        break
                elif col_name == "id":
                    pk_map[table_name] = col_name
                    logging.info(f"✓ PK (fallback): {table_name}.{col_name}")
                    break

    logging.info(f"📊 Fallback found {len(pk_map)} primary keys")

    # Step 2: Detect foreign keys
    for table_name, columns in table_columns.items():
        primary_keys = [pk_map[table_name]] if pk_map.get(table_name) else []
        foreign_keys = []

        for col in columns:
            col_name = col.get("column_name", "").lower().strip()
            null_pct = col.get("null_percentage", 100.0)

            if col_name == pk_map.get(table_name):
                continue
            if null_pct > NULL_THRESHOLD_FOR_KEY:
                continue

            if col_name.endswith("_id"):
                referenced_table = _find_matching_table(col_name, all_table_names)
                if referenced_table and referenced_table != table_name:
                    referenced_pk = pk_map.get(referenced_table)
                    if referenced_pk:
                        foreign_keys.append({
                            "column":            col_name,
                            "references_table":  referenced_table,
                            "references_column": referenced_pk
                        })
                        relationships.append({
                            "from_table":        table_name,
                            "from_column":       col_name,
                            "to_table":          referenced_table,
                            "to_column":         referenced_pk,
                            "relationship_type": "M:1",
                            "description":       f"Many {table_name} to One {referenced_table}",
                            "confidence":        None  # rule-based, no confidence score
                        })
                        logging.info(
                            f"🔗 Fallback relationship: "
                            f"{table_name}.{col_name} → {referenced_table}.{referenced_pk}"
                        )

        original_name = original_to_normalized.get(table_name, table_name)

        # In fallback all columns are treated as source columns (no normalization)
        source_columns_list = [
            col.get("column_name", "")
            for col in columns
            if col.get("null_percentage", 100.0) <= NULL_THRESHOLD_FOR_KEY
            or col.get("column_name", "") in primary_keys
        ]

        tables_info.append({
            "table":          original_name,
            "derived_from":   original_name,
            "primary_keys":   primary_keys,
            "surrogate_keys": [],           # Fallback never creates surrogates
            "source_columns": source_columns_list,
            "foreign_keys":   foreign_keys,
            "attributes": [
                {
                    "name":          col.get("column_name", ""),
                    "data_type":     col.get("data_type", ""),
                    "is_foreign_key": col.get("column_name", "").lower() in
                                      {fk["column"] for fk in foreign_keys},
                    "is_surrogate":  False,
                    "source":        original_name,
                    "display_label": col.get("column_name", ""),
                    "tooltip":       f"Source column from {original_name}",
                    "references":    None
                }
                for col in columns
            ]
        })

    logging.info("=" * 60)
    logging.info("✅ Rule-based fallback complete:")
    logging.info(f"   - Tables:        {len(tables_info)}")
    logging.info(f"   - Relationships: {len(relationships)}")
    logging.info("=" * 60)

    return {
        "tables":              tables_info,
        "relationships":       relationships,
        "raw_entity_analysis": {},
        "cardinality_diagram": "",
        "standalone_entities": [],
        "observations":        [
            "AI modeling failed. Showing raw table structure without normalization.",
            "Use the Retry AI Modeling button to attempt AI-based ER generation again."
        ],
        "fact_tables":         [t["table"] for t in tables_info if t["foreign_keys"]],
        "dimension_tables":    [
            t["table"] for t in tables_info
            if not t["foreign_keys"] and t["primary_keys"]
        ],
        "model_source":        "rule_based_fallback",
        "ai_retry_available":  True
    }


# ====================================================================
# TABLE NAME HELPERS
# ====================================================================
def _normalize_table_name(table_name):
    """Normalize table names by removing common suffixes and standardizing format."""
    name = table_name.lower().strip()
    for suffix in [" - copy", "_copy", " copy", "_backup", " backup"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def _find_matching_table(column_name, table_names):
    """Find best matching table for a foreign key column."""
    if not column_name.endswith("_id"):
        return None

    base_name = column_name.replace("_id", "")

    if base_name in table_names:
        return base_name

    for pattern in [base_name + "s", base_name + "es", base_name[:-1] + "ies"]:
        if pattern in table_names:
            return pattern

    if base_name.endswith("s"):
        singular = base_name[:-1]
        if singular in table_names:
            return singular

    for table in table_names:
        if base_name in table or table in base_name:
            return table

    return None


# ====================================================================
# JSON EXTRACTION (Robust AI Cleanup)
# ====================================================================
def _extract_json_from_text(text):
    """Extracts JSON from noisy AI output. Raises ValueError if nothing found."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip markdown fences
    cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Find the largest JSON object in the text
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError("No JSON object found in AI response")

    candidate = max(matches, key=len)
    return json.loads(candidate)


# ====================================================================
# DATATYPE HELPERS
# ====================================================================
def _datatypes_compatible(d1, d2):
    """Basic datatype compatibility check."""
    d1, d2 = d1.lower(), d2.lower()
    if "int"     in d1 and "int"     in d2: return True
    if "float"   in d1 and "float"   in d2: return True
    if "decimal" in d1 and "decimal" in d2: return True
    if "nvarchar" in d1 and "nvarchar" in d2: return True
    return False


# ====================================================================
# SCHEMA EXTRACTION
# ====================================================================
def extract_schema_from_json_file(data_bytes, file_name, file_path):
    """Extract schema from JSON file with proper nested structure handling."""
    try:
        content = data_bytes.decode("utf-8")
        data    = json.loads(content)

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    logging.info(f"📦 Found nested array '{key}' with {len(value)} records")
                    df = pd.json_normalize(value, max_level=0)
                    return extract_schema_metadata(df, key, file_path)
            logging.info("📄 Found single JSON object")
            df         = pd.json_normalize([data], max_level=0)
            table_name = file_name.rsplit(".", 1)[0]
            return extract_schema_metadata(df, table_name, file_path)

        elif isinstance(data, list):
            logging.info(f"📦 Found JSON array with {len(data)} records")
            df         = pd.json_normalize(data, max_level=0)
            table_name = file_name.rsplit(".", 1)[0]
            return extract_schema_metadata(df, table_name, file_path)

        else:
            logging.error(f"❌ Unsupported JSON structure in {file_name}")
            return None

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON parse error: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Error processing JSON: {e}")
        return None


def infer_enhanced_datatype(series, column_name):
    """Enhanced datatype inference including arrays and structs."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return "string"

    first_value = non_null.iloc[0]

    if isinstance(first_value, list):
        if len(first_value) > 0:
            first_elem = first_value[0]
            if isinstance(first_elem, dict):
                fields = [f"{k} {infer_simple_type(v)}" for k, v in first_elem.items()]
                return f"array<struct<{', '.join(fields)}>>"
            else:
                return f"array<{infer_simple_type(first_elem)}>"
        return "array<string>"

    elif isinstance(first_value, dict):
        fields = [f"{k} {infer_simple_type(v)}" for k, v in first_value.items()]
        return f"struct<{', '.join(fields)}>"

    raw_dtype = str(series.dtype).lower()
    if "int"      in raw_dtype: return "int"
    if "float"    in raw_dtype or "double" in raw_dtype: return "double"
    if "bool"     in raw_dtype: return "boolean"
    if "datetime" in raw_dtype: return "date"
    return "string"


def infer_simple_type(value):
    """Helper to infer simple types for nested structures."""
    if isinstance(value, bool):  return "boolean"
    if isinstance(value, int):   return "int"
    if isinstance(value, float): return "double"
    if isinstance(value, list):  return "array"
    if isinstance(value, dict):  return "struct"
    return "string"


def extract_schema_metadata(df, table_name, file_path):
    """
    Extract comprehensive schema metadata from a pandas DataFrame.
    Includes null guard on is_potential_key:
      - Columns with null_percentage > NULL_THRESHOLD_FOR_KEY are NEVER keys.
    """
    total_rows      = len(df)
    table_name_clean = table_name.rsplit(".", 1)[0] if "." in table_name else table_name

    schema_info = {
        "table_name":           table_name_clean,
        "file_path":            file_path,
        "row_count":            int(total_rows),
        "column_count":         int(len(df.columns)),
        "extraction_timestamp": datetime.utcnow().isoformat(),
        "columns":              []
    }

    for column in df.columns:
        series = df[column]
        dtype  = infer_enhanced_datatype(series, column)

        series_for_stats = series.copy()
        if series_for_stats.apply(lambda x: isinstance(x, (dict, list))).any():
            series_for_stats = series_for_stats.apply(
                lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x
            )

        null_count = int(series_for_stats.isna().sum())
        null_pct   = float(round((null_count / total_rows) * 100, 2)) if total_rows else 0.0

        try:
            distinct_count = int(series_for_stats.nunique())
        except Exception:
            try:
                distinct_count = len(set(str(x) for x in series_for_stats.dropna().tolist()))
            except Exception:
                distinct_count = 0

        is_nullable         = bool(series_for_stats.isna().any())
        cardinality_pct     = float(round((distinct_count / total_rows) * 100, 2)) if total_rows else 0.0

        col_info = {
            "column_name":          str(column),
            "data_type":            dtype,
            "nullable":             is_nullable,
            "null_count":           int(null_count),
            "null_percentage":      null_pct,
            "distinct_count":       int(distinct_count),
            "cardinality_percentage": cardinality_pct,
            "is_potential_key":     False
        }

        # Key detection — HARD NULL GUARD FIRST
        # A column with > NULL_THRESHOLD_FOR_KEY % nulls is never a key, period.
        if null_pct <= NULL_THRESHOLD_FOR_KEY:
            if cardinality_pct == 100.0 and null_count == 0:
                if "id" in str(column).lower():
                    col_info["is_potential_key"] = True
            elif cardinality_pct > 95 and total_rows > 100:
                if "id" in str(column).lower() or "key" in str(column).lower():
                    col_info["is_potential_key"] = True
        else:
            logging.debug(
                f"   ⛔ '{column}' excluded from key inference "
                f"(null_percentage={null_pct}% > {NULL_THRESHOLD_FOR_KEY}%)"
            )

        if not dtype.startswith(("array", "struct")):
            try:
                sample_vals = series_for_stats.dropna().unique()[:5].tolist()
                col_info["sample_values"] = [str(v) for v in sample_vals]
            except Exception:
                col_info["sample_values"] = []
        else:
            col_info["sample_values"] = []

        schema_info["columns"].append(col_info)

    return schema_info


# ====================================================================
# BATCH SCHEMA LOADER
# ====================================================================
def load_batch_schemas():
    """Loads ONLY schema files created after batch_start timestamp."""
    account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    account_key  = os.environ["STORAGE_ACCOUNT_KEY"]
    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix=core.windows.net"
    )
    service   = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(METADATA_CONTAINER_NAME)

    batch_info_raw = (
        container.get_blob_client("batch_info.json")
        .download_blob().readall().decode()
    )
    batch_start_str = json.loads(batch_info_raw)["batch_start"]
    batch_start     = datetime.fromisoformat(batch_start_str)
    if batch_start.tzinfo is None:
        batch_start = batch_start.replace(tzinfo=timezone.utc)

    schemas = []
    for blob in container.list_blobs(name_starts_with="schema_"):
        try:
            ts_str  = blob.name.rsplit("_", 1)[-1].replace(".json", "")
            file_ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if file_ts >= batch_start:
            data = container.get_blob_client(blob.name).download_blob().readall()
            schemas.append(json.loads(data))

    return schemas


# ====================================================================
# DDL UTILITIES  (unchanged from original)
# ====================================================================
def prepare_schemas_for_relationship_detection(schema_files, ddl_result=None):
    schemas = []
    for schema_file in schema_files:
        with open(schema_file, "r") as f:
            schema_data = json.load(f)

        table_name   = schema_data.get("table_name", "")
        columns_data = schema_data.get("columns", [])

        if ddl_result and "ddl_scripts" in ddl_result:
            ddl_script = ddl_result["ddl_scripts"].get(table_name, "")
            columns    = extract_columns_from_ddl(ddl_script) if ddl_script else [
                {"name": c.get("column_name", ""), "type": c.get("data_type", "")}
                for c in columns_data
            ]
        else:
            columns = [
                {"name": c.get("column_name", ""), "type": c.get("data_type", "")}
                for c in columns_data
            ]

        schemas.append({"table_name": table_name, "columns": columns})
    return schemas


def extract_columns_from_ddl(ddl_script):
    """Extract column information from DDL CREATE TABLE statement."""
    columns = []
    match   = re.search(r"\((.*)\)", ddl_script, re.DOTALL)
    if not match:
        return columns

    columns_text       = match.group(1)
    column_definitions = []
    current_def        = ""
    bracket_depth      = 0

    for char in columns_text:
        if char in "<[":
            bracket_depth += 1
        elif char in ">]":
            bracket_depth -= 1
        elif char == "," and bracket_depth == 0:
            column_definitions.append(current_def.strip())
            current_def = ""
            continue
        current_def += char

    if current_def.strip():
        column_definitions.append(current_def.strip())

    for col_def in column_definitions:
        parts = col_def.split()
        if len(parts) >= 2:
            col_name   = parts[0]
            type_parts = []
            for part in parts[1:]:
                if part.upper() in ["NOT", "PRIMARY", "UNIQUE", "NULL"]:
                    break
                type_parts.append(part)
            col_type = " ".join(type_parts)
            if col_type.startswith("ARRAY") or col_type.startswith("STRUCT"):
                col_type = col_type.split("<")[0]
            columns.append({"name": col_name, "type": col_type})

    return columns


def run_relationship_detection(metadata_folder, ddl_analysis_file):
    """Run relationship detection using metadata and DDL analysis."""
    with open(ddl_analysis_file, "r") as f:
        ddl_result = json.load(f)

    schema_files = [
        os.path.join(metadata_folder, f)
        for f in os.listdir(metadata_folder)
        if f.startswith("schema_") and f.endswith(".json")
    ]
    schemas = prepare_schemas_for_relationship_detection(schema_files, ddl_result)

    logging.info("=" * 60)
    logging.info("📋 Prepared Schemas for Relationship Detection:")
    for schema in schemas:
        logging.info(f"\nTable: {schema['table_name']}")
        for col in schema["columns"]:
            logging.info(f"  - {col['name']}: {col['type']}")
    logging.info("=" * 60)

    return detect_relationships(schemas)


# ====================================================================
# AI-BASED SCHEMA → DDL GENERATION  (unchanged from original)
# ====================================================================
def analyze_schemas_with_ai(schemas):
    """Sends schemas to Azure OpenAI and returns DDL JSON."""
    client = AzureOpenAI(
        api_key=AI_API_KEY,
        api_version="2024-05-01-preview",
        azure_endpoint=AI_ENDPOINT
    )

    prompt = (
        "You are a strict JSON generator.\n"
        "Return ONLY valid JSON. No explanations, no markdown.\n\n"
        "Output format:\n"
        "{\n"
        '  "ddl_scripts": {\n'
        '     "<table>": "CREATE TABLE dbo.<table>(...);"  \n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Fabric-compatible datatypes only\n"
        "- Exactly ONE PRIMARY KEY per table\n"
        "- No GO inside JSON\n"
        "- No markdown\n\n"
        "SCHEMAS:\n"
        f"{json.dumps(schemas, indent=2, cls=NumpyEncoder)}"
    )

    try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=8000
                )
                content = response.choices[0].message.content.strip()
                return _extract_json_from_text(content)
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        logging.warning(f"⚠️ Rate limited. Waiting {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
    except Exception:
        logging.exception("AI DDL generation failed. Using fallback.")
        return _local_ddl_fallback(schemas)


def _local_ddl_fallback(schemas):
    """Generate DDL scripts locally without AI."""
    ddl = {"ddl_scripts": {}}

    if isinstance(schemas, dict) and "schemas" in schemas:
        schemas = schemas["schemas"]

    for schema in schemas:
        tname        = schema["table_name"]
        cols         = []
        pk           = None
        pk_candidates = []

        for col in schema["columns"]:
            cname    = col["column_name"]
            dtype    = col["data_type"].lower()
            nullable = col["nullable"]

            if "int"  in dtype:                        sql_type = "INT"
            elif "float" in dtype:                     sql_type = "FLOAT"
            elif "date" in dtype or "time" in dtype:   sql_type = "DATETIME2"
            elif "bool" in dtype:                      sql_type = "BIT"
            else:                                      sql_type = "NVARCHAR(255)"

            cols.append(f"    [{cname}] {sql_type} {'NULL' if nullable else 'NOT NULL'}")

            if col["is_potential_key"]:
                pk_candidates.append(cname)

        if pk_candidates:
            pk = pk_candidates[0]

        sql = f"CREATE TABLE dbo.{tname} (\n"
        sql += ",\n".join(cols)
        if pk:
            sql += f",\n    PRIMARY KEY([{pk}])"
        sql += "\n);"

        ddl["ddl_scripts"][tname] = sql

    return ddl


def generate_fabric_compatible_ddl(analysis):
    """Generate final DDL script with GO statements."""
    ddl_parts = []
    for tname, script in analysis.get("ddl_scripts", {}).items():
        sanitized = _sanitize_sql_script(script)
        ddl_parts.append(sanitized + "\nGO\n")
    return "".join(ddl_parts)


def _sanitize_sql_script(script: str) -> str:
    """Remove markdown and formatting from SQL scripts."""
    if not script:
        return ""
    s = script.strip()
    s = re.sub(r"```sql", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```", "", s)
    s = re.sub(r"\bGO\b", "", s, flags=re.IGNORECASE)
    s = s.replace("`", "")
    if not s.endswith(";"):
        s += ";"
    return s