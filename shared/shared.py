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

# Environment variables with defaults
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.environ.get("STORAGE_ACCOUNT_KEY")
LAKEHOUSE_CONTAINER = os.environ.get("LAKEHOUSE_CONTAINER", "lakehouse-data")
METADATA_CONTAINER = os.environ.get("METADATA_CONTAINER", "metadata")
DDL_CONTAINER = os.environ.get("DDL_CONTAINER", "ddl-scripts")
RELATIONSHIPS_CONTAINER = os.getenv("RELATIONSHIPS_CONTAINER", "relationship")

AI_ENDPOINT = os.environ.get("AI_ENDPOINT")
AI_API_KEY = os.environ.get("AI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o-mini")

WORKSPACE_ID = os.environ.get("WORKSPACE_ID")
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID")

SUPPORTED_FORMATS = [".csv", ".parquet", ".json"]

# Only validate critical variables
if not STORAGE_ACCOUNT_NAME:
    logger.warning("STORAGE_ACCOUNT_NAME not set - this may cause issues")
if not STORAGE_ACCOUNT_KEY:
    logger.warning("STORAGE_ACCOUNT_KEY not set - this may cause issues")


# At the top of shared.py, after other environment variables
BLOB_CONN_STR = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT_NAME};"
    f"AccountKey={STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

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


# ====================================================================
# BLOB OPERATIONS
# ====================================================================
def read_blob_if_exists(blob_name):
    """Read a blob if it exists, otherwise return None."""
    account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    account_key = os.environ["STORAGE_ACCOUNT_KEY"]

    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix=core.windows.net"
    )

    service = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(METADATA_CONTAINER)
    client = container.get_blob_client(blob_name)

    try:
        data = client.download_blob().readall().decode()
        return json.loads(data)
    except Exception:
        return None


def save_to_blob(content, blob_name, container_name):
    """
    Save content to Azure Blob Storage.
    
    Args:
        content: String content to save (should already be JSON serialized)
        blob_name: Name of the blob
        container_name: Name of the container
    """
    account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    account_key = os.environ["STORAGE_ACCOUNT_KEY"]

    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix=core.windows.net"
    )

    blob_service_client = BlobServiceClient.from_connection_string(conn_str)

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    blob_client.upload_blob(content, overwrite=True)


# ====================================================================
# RELATIONSHIP DETECTION
# ====================================================================
import logging
import re


def detect_relationships(schemas):
    """
    Detect primary key and foreign key relationships between tables using AI.
    Enhanced with better table name normalization and fuzzy matching.
    """
    
    logging.info(f"🤖 Using AI to detect relationships for {len(schemas)} tables")
    
    client = AzureOpenAI(
        api_key=AI_API_KEY,
        api_version="2024-05-01-preview",
        azure_endpoint=AI_ENDPOINT
    )
    
    # Format schemas for the AI prompt
    formatted_schemas = []
    for schema in schemas:
        table_name = schema.get("table_name", "")
        columns = schema.get("columns", [])
        
        formatted_columns = [
            f"{col.get('column_name', '')} {col.get('data_type', '')}"
            for col in columns
        ]
        
        formatted_schemas.append({
            "schema_name": table_name,
            "columns": formatted_columns
        })
    
    # IMPROVED PROMPT with explicit examples
    prompt = (
        "You are an Intelligent Semantic Relationship Finder. You will be given multiple database schemas.\n\n"
        "Your tasks:\n"
        "1. Detect Primary Keys (PK) using these rules:\n"
        "   - Look for columns ending with '_id' (e.g., customer_id, order_id)\n"
        "   - Column named 'id' is always a PK\n"
        "   - High cardinality (near 100%) + NOT NULL indicates PK\n"
        "   - Ignore table name prefixes/suffixes like ' - Copy'\n\n"
        "2. Detect Foreign Keys (FK) when:\n"
        "   - A column like 'customer_id' exists in one table\n"
        "   - Another table has 'customer_id' as PK (or similar pattern)\n"
        "   - Handle pluralization: 'customer_id' can reference 'customers' table\n"
        "   - Ignore case and special characters in table names\n\n"
        "3. Example:\n"
        "   Table: 'customers' or 'customers - Copy'\n"
        "   - PK: customer_id\n"
        "   Table: 'orders'\n"
        "   - PK: order_id\n"
        "   - FK: customer_id → customers.customer_id (1:M relationship)\n\n"
        "4. Output ONLY valid JSON with this exact structure:\n"
        "{\n"
        '  "primary_keys": { "orders": ["order_id"], "customers": ["customer_id"] },\n'
        '  "foreign_keys": { "orders": { "customer_id": "customers(customer_id)" } },\n'
        '  "relationships": [\n'
        '    {\n'
        '      "from": "orders",\n'
        '      "to": "customers",\n'
        '      "type": "M:1",\n'
        '      "cardinality": "Many orders to One customer",\n'
        '      "from_column": "customer_id",\n'
        '      "to_column": "customer_id"\n'
        '    }\n'
        '  ],\n'
        '  "fact_tables": ["orders"],\n'
        '  "dimension_tables": ["customers"]\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "- Normalize table names (remove ' - Copy', lowercase)\n"
        "- Match 'customer_id' with 'customer' OR 'customers' tables\n"
        "- No explanation. No markdown. JSON ONLY.\n\n"
        "SCHEMAS:\n"
        f"{json.dumps(formatted_schemas, indent=2, cls=NumpyEncoder)}"
    )
    
    try:
        logging.info("📤 Sending schemas to Azure OpenAI...")
        
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
                ai_result = _extract_json_from_text(content)
                
                logging.info("✅ AI analysis complete")
                
                # Transform AI output to match the expected return format
                return _transform_ai_result_to_standard_format(ai_result, schemas)
                
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
                        logging.warning(f"⚠️ Rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logging.error("❌ Max retries reached. Using fallback.")
                        raise
                else:
                    raise
            
    except Exception as e:
        logging.exception("❌ AI relationship detection failed. Using rule-based fallback.")
        return _fallback_relationship_detection(schemas)
        

def _transform_ai_result_to_standard_format(ai_result, schemas):
    """
    Transform AI's output format to match the original function's return format.
    
    AI Format:
    {
        "primary_keys": {"table": ["pk1"]},
        "foreign_keys": {"table": {"fk": "ref_table(pk)"}},
        "relationships": [{"from": "...", "to": "...", "type": "1:M"}]
    }
    
    Standard Format:
    {
        "tables": [{"table": "name", "primary_keys": [...], "foreign_keys": [...]}],
        "relationships": [{"from_table": "...", "to_table": "...", ...}]
    }
    """
    
    primary_keys_map = ai_result.get("primary_keys", {})
    foreign_keys_map = ai_result.get("foreign_keys", {})
    ai_relationships = ai_result.get("relationships", [])
    
    tables_info = []
    relationships = []
    
    # Build tables_info
    table_names = set()
    for schema in schemas:
        table_name = schema.get("table_name", "").lower().strip()
        if table_name:
            table_names.add(table_name)
    
    for table_name in table_names:
        primary_keys = primary_keys_map.get(table_name, [])
        
        # Transform foreign keys format
        fk_dict = foreign_keys_map.get(table_name, {})
        foreign_keys = []
        
        for fk_column, fk_reference in fk_dict.items():
            # Parse "ref_table(pk_column)" format
            if "(" in fk_reference and ")" in fk_reference:
                ref_table = fk_reference.split("(")[0]
                ref_column = fk_reference.split("(")[1].rstrip(")")
                
                foreign_keys.append({
                    "column": fk_column,
                    "references_table": ref_table,
                    "references_column": ref_column
                })
        
        tables_info.append({
            "table": table_name,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        })
    
    # Transform relationships format
    for rel in ai_relationships:
        relationship = {
            "from_table": rel.get("from", ""),
            "from_column": "",  # AI doesn't always provide column details
            "to_table": rel.get("to", ""),
            "to_column": "",
            "relationship_type": rel.get("type", "many-to-one"),
            "cardinality": rel.get("cardinality", "")
        }
        relationships.append(relationship)
    
    # Logging summary
    logging.info("=" * 60)
    logging.info(f"✅ AI Relationship Detection Complete:")
    logging.info(f"   - Tables analyzed: {len(tables_info)}")
    logging.info(f"   - Relationships found: {len(relationships)}")
    logging.info(f"   - Tables with PKs: {len([t for t in tables_info if t['primary_keys']])}")
    logging.info(f"   - Tables with FKs: {len([t for t in tables_info if t['foreign_keys']])}")
    
    if "fact_tables" in ai_result:
        logging.info(f"   - Fact tables: {', '.join(ai_result['fact_tables'])}")
    if "dimension_tables" in ai_result:
        logging.info(f"   - Dimension tables: {', '.join(ai_result['dimension_tables'])}")
    
    logging.info("=" * 60)
    
    return {
        "tables": tables_info,
        "relationships": relationships,
        "fact_tables": ai_result.get("fact_tables", []),
        "dimension_tables": ai_result.get("dimension_tables", [])
    }


def _find_matching_table(column_name, table_names):
    """
    Find best matching table for a foreign key column.
    
    Handles:
    - Pluralization: customer_id → customers
    - Direct match: order_id → orders
    - Fuzzy match: cust_id → customers
    
    Args:
        column_name: FK column name (e.g., "customer_id")
        table_names: List of normalized table names
        
    Returns:
        str: Best matching table name or None
    """
    if not column_name.endswith('_id'):
        return None
    
    # Extract base name
    base_name = column_name.replace('_id', '')
    
    # Try exact match first
    if base_name in table_names:
        return base_name
    
    # Try pluralization patterns
    plural_patterns = [
        base_name + 's',           # customer → customers
        base_name + 'es',          # address → addresses
        base_name[:-1] + 'ies',    # category → categories (if ends in 'y')
    ]
    
    for pattern in plural_patterns:
        if pattern in table_names:
            return pattern
    
    # Try singular if column might be plural
    if base_name.endswith('s'):
        singular = base_name[:-1]
        if singular in table_names:
            return singular
    
    # Fuzzy match: check if base_name is substring
    for table in table_names:
        if base_name in table or table in base_name:
            return table
    
    return None

def _normalize_table_name(table_name):
    """
    Normalize table names by removing common suffixes and standardizing format.
    
    Examples:
        "customers - Copy" → "customers"
        "Orders2" → "orders2"
        "CUSTOMER_DATA" → "customer_data"
    """
    name = table_name.lower().strip()
    
    # Remove common suffixes
    suffixes_to_remove = [" - copy", "_copy", " copy", "_backup", " backup"]
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove special characters except underscore
    name = re.sub(r'[^a-z0-9_]', '', name)
    
    return name

def _fallback_relationship_detection(schemas):
    """
    IMPROVED rule-based fallback with better table name handling.
    """
    logging.info("⚙️ Using IMPROVED rule-based fallback for relationship detection")
    
    relationships = []
    tables_info = []
    
    # Build table-to-columns mapping with normalized names
    table_columns = {}
    original_to_normalized = {}  # Track original names
    
    for schema in schemas:
        original_name = schema.get("table_name", "").strip()
        normalized_name = _normalize_table_name(original_name)
        columns = schema.get("columns", [])
        
        if normalized_name and columns:
            table_columns[normalized_name] = columns
            original_to_normalized[normalized_name] = original_name
            logging.info(f"📋 Mapped: '{original_name}' → '{normalized_name}'")
    
    if not table_columns:
        logging.warning("⚠️ No valid tables found")
        return {"tables": [], "relationships": []}
    
    # Get list of all normalized table names
    all_table_names = list(table_columns.keys())
    
    # Step 1: Identify primary keys
    pk_map = {}
    for table_name, columns in table_columns.items():
        for col in columns:
            col_name = col.get("column_name", "").lower().strip()
            is_potential_key = col.get("is_potential_key", False)
            
            # Enhanced PK detection
            if is_potential_key:
                # Check if it ends with _id
                if col_name.endswith('_id'):
                    # Extract base: customer_id → customer
                    base = col_name.replace('_id', '')
                    
                    # Check if base matches table name (with normalization)
                    if base == table_name or base + 's' == table_name or base == table_name + 's':
                        pk_map[table_name] = col_name
                        logging.info(f"✓ Primary Key: {table_name}.{col_name}")
                        break
                
                # Fallback: use 'id' column
                elif col_name == "id":
                    pk_map[table_name] = col_name
                    logging.info(f"✓ Primary Key: {table_name}.{col_name}")
                    break
    
    logging.info(f"📊 Found {len(pk_map)} primary keys")
    
    # Step 2: Detect foreign keys with improved matching
    for table_name, columns in table_columns.items():
        primary_keys = [pk_map.get(table_name)] if pk_map.get(table_name) else []
        foreign_keys = []
        
        for col in columns:
            col_name = col.get("column_name", "").lower().strip()
            
            # Skip if this is the primary key
            if col_name == pk_map.get(table_name):
                continue
            
            # Check if column ends with _id (potential FK)
            if col_name.endswith("_id"):
                # Use improved matching function
                referenced_table = _find_matching_table(col_name, all_table_names)
                
                if referenced_table and referenced_table != table_name:
                    referenced_pk = pk_map.get(referenced_table)
                    
                    if referenced_pk:
                        fk_info = {
                            "column": col_name,
                            "references_table": referenced_table,
                            "references_column": referenced_pk
                        }
                        foreign_keys.append(fk_info)
                        
                        relationship = {
                            "from_table": table_name,
                            "from_column": col_name,
                            "to_table": referenced_table,
                            "to_column": referenced_pk,
                            "relationship_type": "many-to-one",
                            "cardinality": f"Many {table_name} to One {referenced_table}"
                        }
                        relationships.append(relationship)
                        
                        logging.info(f"🔗 Relationship: {table_name}.{col_name} → {referenced_table}.{referenced_pk}")
        
        # Use ORIGINAL table name in output
        original_name = original_to_normalized.get(table_name, table_name)
        tables_info.append({
            "table": original_name,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        })
    
    logging.info("=" * 60)
    logging.info(f"✅ Fallback complete:")
    logging.info(f"   - Tables: {len(tables_info)}")
    logging.info(f"   - Relationships: {len(relationships)}")
    logging.info(f"   - Tables with PKs: {len([t for t in tables_info if t['primary_keys']])}")
    logging.info(f"   - Tables with FKs: {len([t for t in tables_info if t['foreign_keys']])}")
    logging.info("=" * 60)
    
    return {
        "tables": tables_info,
        "relationships": relationships,
        "fact_tables": [t["table"] for t in tables_info if t["foreign_keys"]],
        "dimension_tables": [t["table"] for t in tables_info if not t["foreign_keys"] and t["primary_keys"]]
    }

def prepare_schemas_for_relationship_detection(schema_files, ddl_result=None):
    """
    Prepare schemas from metadata files for relationship detection.
    If DDL analysis is available, use it to get proper column structure.
    
    Args:
        schema_files: List of schema JSON file paths
        ddl_result: Optional DDL analysis result with correct structure
        
    Returns:
        List of schemas for detect_relationships()
    """
    schemas = []
    
    for schema_file in schema_files:
        with open(schema_file, 'r') as f:
            schema_data = json.load(f)
        
        table_name = schema_data.get("table_name", "")
        columns_data = schema_data.get("columns", [])
        
        # Check if we have DDL result to get proper structure
        if ddl_result and "ddl_scripts" in ddl_result:
            ddl_script = ddl_result["ddl_scripts"].get(table_name, "")
            
            if ddl_script:
                # Extract columns from DDL
                columns = extract_columns_from_ddl(ddl_script)
            else:
                # Use original schema
                columns = [
                    {
                        "name": col.get("column_name", ""),
                        "type": col.get("data_type", "")
                    }
                    for col in columns_data
                ]
        else:
            # Use original schema
            columns = [
                {
                    "name": col.get("column_name", ""),
                    "type": col.get("data_type", "")
                }
                for col in columns_data
            ]
        
        schemas.append({
            "table_name": table_name,
            "columns": columns
        })
    
    return schemas


def extract_columns_from_ddl(ddl_script):
    """
    Extract column information from DDL CREATE TABLE statement.
    
    Example DDL:
    CREATE TABLE dbo.orders2(
        order_id INT NOT NULL PRIMARY KEY,
        customer_id INT NOT NULL,
        order_date STRING NOT NULL,
        ...
    );
    
    Returns:
        List of {name, type} dictionaries
    """
    import re
    
    columns = []
    
    # Extract content between parentheses
    match = re.search(r'\((.*)\)', ddl_script, re.DOTALL)
    if not match:
        return columns
    
    columns_text = match.group(1)
    
    # Split by comma (but not commas inside STRUCT<>)
    column_definitions = []
    current_def = ""
    bracket_depth = 0
    
    for char in columns_text:
        if char in '<[':
            bracket_depth += 1
        elif char in '>]':
            bracket_depth -= 1
        elif char == ',' and bracket_depth == 0:
            column_definitions.append(current_def.strip())
            current_def = ""
            continue
        current_def += char
    
    if current_def.strip():
        column_definitions.append(current_def.strip())
    
    # Parse each column definition
    for col_def in column_definitions:
        # Pattern: column_name TYPE [NOT NULL] [PRIMARY KEY]
        parts = col_def.split()
        if len(parts) >= 2:
            col_name = parts[0]
            
            # Extract type (everything until NOT, PRIMARY, or end)
            type_parts = []
            for part in parts[1:]:
                if part.upper() in ['NOT', 'PRIMARY', 'UNIQUE', 'NULL']:
                    break
                type_parts.append(part)
            
            col_type = ' '.join(type_parts)
            
            # Simplify complex types for relationship detection
            if col_type.startswith('ARRAY') or col_type.startswith('STRUCT'):
                col_type = col_type.split('<')[0]  # Keep base type
            
            columns.append({
                "name": col_name,
                "type": col_type
            })
    
    return columns


# Usage in your pipeline
def run_relationship_detection(metadata_folder, ddl_analysis_file):
    """
    Run relationship detection using metadata and DDL analysis.
    """
    # Load DDL analysis
    with open(ddl_analysis_file, 'r') as f:
        ddl_result = json.load(f)
    
    # Find all schema files
    schema_files = [
        os.path.join(metadata_folder, f)
        for f in os.listdir(metadata_folder)
        if f.startswith('schema_') and f.endswith('.json')
    ]
    
    # Prepare schemas using DDL information
    schemas = prepare_schemas_for_relationship_detection(schema_files, ddl_result)
    
    # Log the prepared schemas
    logging.info("=" * 60)
    logging.info("📋 Prepared Schemas for Relationship Detection:")
    for schema in schemas:
        logging.info(f"\nTable: {schema['table_name']}")
        for col in schema['columns']:
            logging.info(f"  - {col['name']}: {col['type']}")
    logging.info("=" * 60)
    
    # Detect relationships
    result = detect_relationships(schemas)
    
    return result




def _datatypes_compatible(d1, d2):
    """Basic datatype compatibility check."""
    d1 = d1.lower()
    d2 = d2.lower()

    if ("int" in d1 and "int" in d2): return True
    if ("float" in d1 and "float" in d2): return True
    if ("decimal" in d1 and "decimal" in d2): return True
    if ("nvarchar" in d1 and "nvarchar" in d2): return True

    return False


# ====================================================================
# JSON EXTRACTION (Robust AI Cleanup)
# ====================================================================
def _extract_json_from_text(text):
    """Extracts JSON from noisy AI output."""
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass

    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError("No JSON object found in AI response")

    candidate = max(matches, key=len)
    return json.loads(candidate)


# ====================================================================
# LOAD SCHEMAS ONLY FOR CURRENT BATCH
# ====================================================================
def load_batch_schemas():
    """
    Loads ONLY schema files created after batch_start timestamp.
    Ensures old schemas from previous uploads are ignored.
    """
    account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    account_key = os.environ["STORAGE_ACCOUNT_KEY"]

    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix=core.windows.net"
    )
    service = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(METADATA_CONTAINER)

    # ---------------------
    # Load batch start time
    # ---------------------
    batch_info_raw = container.get_blob_client("batch_info.json") \
        .download_blob().readall().decode()

    batch_start_str = json.loads(batch_info_raw)["batch_start"]
    batch_start = datetime.fromisoformat(batch_start_str)

    # Ensure timestamps are timezone-aware
    if batch_start.tzinfo is None:
        batch_start = batch_start.replace(tzinfo=timezone.utc)

    schemas = []

    # ---------------------
    # Iterate over schema_ files
    # ---------------------
    for blob in container.list_blobs(name_starts_with="schema_"):

        try:
            ts_str = blob.name.rsplit("_", 1)[-1].replace(".json", "")
            file_ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            file_ts = file_ts.replace(tzinfo=timezone.utc)
        except:
            continue

        if file_ts >= batch_start:
            data = container.get_blob_client(blob.name).download_blob().readall()
            schemas.append(json.loads(data))

    return schemas


# ====================================================================
# SCHEMA EXTRACTION FROM DATAFRAME
# ====================================================================

def extract_schema_from_json_file(data_bytes, file_name, file_path):
    """
    Extract schema from JSON file with proper nested structure handling.
    
    Args:
        data_bytes: Raw bytes from blob
        file_name: Name of the file
        file_path: Full path to the file
        
    Returns:
        dict: Schema metadata or None if failed
    """
    try:
        content = data_bytes.decode('utf-8')
        data = json.loads(content)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Check for wrapper pattern like {"orders": [...]}
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    logging.info(f"📦 Found nested array '{key}' with {len(value)} records")
                    # Use the key as table name, not the filename
                    df = pd.json_normalize(value, max_level=0)
                    return extract_schema_metadata(df, key, file_path)
            
            # Single object - treat as one-row DataFrame
            logging.info(f"📄 Found single JSON object")
            df = pd.json_normalize([data], max_level=0)
            table_name = file_name.rsplit('.', 1)[0]
            return extract_schema_metadata(df, table_name, file_path)
        
        elif isinstance(data, list):
            # Direct array of objects
            logging.info(f"📦 Found JSON array with {len(data)} records")
            df = pd.json_normalize(data, max_level=0)
            table_name = file_name.rsplit('.', 1)[0]
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
    """
    Enhanced datatype inference including arrays and structs.
    
    Args:
        series: pandas Series
        column_name: Name of the column
        
    Returns:
        str: Data type string (including complex types like array<struct<...>>)
    """
    # Check first non-null value
    non_null = series.dropna()
    if len(non_null) == 0:
        return "string"
    
    first_value = non_null.iloc[0]
    
    # Check for lists (arrays)
    if isinstance(first_value, list):
        if len(first_value) > 0:
            first_elem = first_value[0]
            if isinstance(first_elem, dict):
                # Array of objects - build struct definition
                fields = []
                for k, v in first_elem.items():
                    field_type = infer_simple_type(v)
                    fields.append(f"{k} {field_type}")
                return f"array<struct<{', '.join(fields)}>>"
            else:
                elem_type = infer_simple_type(first_elem)
                return f"array<{elem_type}>"
        return "array<string>"
    
    # Check for dicts (structs)
    elif isinstance(first_value, dict):
        fields = []
        for k, v in first_value.items():
            field_type = infer_simple_type(v)
            fields.append(f"{k} {field_type}")
        return f"struct<{', '.join(fields)}>"
    
    # Standard types
    raw_dtype = str(series.dtype).lower()
    if "int" in raw_dtype:
        return "int"
    elif "float" in raw_dtype or "double" in raw_dtype:
        return "double"
    elif "bool" in raw_dtype:
        return "boolean"
    elif "datetime" in raw_dtype:
        return "date"
    else:
        return "string"


def infer_simple_type(value):
    """Helper to infer simple types for nested structures."""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "double"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "struct"
    else:
        return "string"

def extract_schema_metadata(df, table_name, file_path):
    """
    Extract comprehensive schema metadata from a pandas DataFrame.
    
    Args:
        df: pandas DataFrame
        table_name: Name of the table/file
        file_path: Full path to the file
        
    Returns:
        dict: Schema metadata including columns, types, statistics
    """
    total_rows = len(df)
    table_name_clean = table_name.rsplit('.', 1)[0] if '.' in table_name else table_name

    schema_info = {
        "table_name": table_name_clean,
        "file_path": file_path,
        "row_count": int(total_rows),
        "column_count": int(len(df.columns)),
        "extraction_timestamp": datetime.utcnow().isoformat(),
        "columns": []
    }

    for column in df.columns:
        series = df[column]

        # ---------------------
        # Enhanced datatype detection
        # ---------------------
        dtype = infer_enhanced_datatype(series, column)

        # ======================================================
        # Handle dict/list for statistics (convert to string)
        # ======================================================
        series_for_stats = series.copy()
        if series_for_stats.apply(lambda x: isinstance(x, (dict, list))).any():
            series_for_stats = series_for_stats.apply(
                lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x
            )

        # ---------------------
        # Nulls
        # ---------------------
        null_count = int(series_for_stats.isna().sum())

        # ---------------------
        # Distinct count
        # ---------------------
        try:
            distinct_count = int(series_for_stats.nunique())
        except Exception:
            try:
                distinct_count = len(set([str(x) for x in series_for_stats.dropna().tolist()]))
            except:
                distinct_count = 0

        # ======================================================
        # Convert to native Python types
        # ======================================================
        is_nullable = bool(series_for_stats.isna().any())
        
        col_info = {
            "column_name": str(column),
            "data_type": dtype,  # Now uses enhanced inference
            "nullable": is_nullable,
            "null_count": int(null_count),
            "null_percentage": float(round((null_count / total_rows) * 100, 2)) if total_rows else 0.0,
            "distinct_count": int(distinct_count),
            "cardinality_percentage": float(round((distinct_count / total_rows) * 100, 2)) if total_rows else 0.0,
            "is_potential_key": False
        }

        # ---------------------
        # Key detection rules
        # ---------------------
        if col_info["cardinality_percentage"] == 100.0 and null_count == 0:
            if "id" in str(column).lower():
                col_info["is_potential_key"] = True
        elif col_info["cardinality_percentage"] > 95 and total_rows > 100:
            if "id" in str(column).lower() or "key" in str(column).lower():
                col_info["is_potential_key"] = True

        # ---------------------
        # Sample values (skip complex types)
        # ---------------------
        if not dtype.startswith(("array", "struct")):
            try:
                sample_vals = series_for_stats.dropna().unique()[:5].tolist()
                col_info["sample_values"] = [str(v) for v in sample_vals]
            except:
                col_info["sample_values"] = []
        else:
            col_info["sample_values"] = []

        schema_info["columns"].append(col_info)

    return schema_info

# ====================================================================
# SANITIZE SQL OUTPUT
# ====================================================================
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


# ====================================================================
# AI-BASED SCHEMA → DDL GENERATION
# ====================================================================
def analyze_schemas_with_ai(schemas):
    """
    Sends schemas + relationships to Azure OpenAI
    and forces a clean JSON-only response.
    """

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
        '     "<table>": "CREATE TABLE dbo.<table>(...);"\n'
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
                        wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
                        logging.warning(f"⚠️ Rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logging.error("❌ Max retries reached. Using fallback.")
                        raise
                else:
                    raise
    except Exception as e:
        logging.exception("AI failed. Using fallback.")
        return _local_ddl_fallback(schemas)


# ====================================================================
# LOCAL DDL FALLBACK
# ====================================================================
def _local_ddl_fallback(schemas):
    """Generate DDL scripts locally without AI."""
    ddl = {"ddl_scripts": {}}

    if isinstance(schemas, dict) and "schemas" in schemas:
        schemas = schemas["schemas"]

    for schema in schemas:
        tname = schema["table_name"]
        cols = []
        pk = None
        pk_candidates = []

        for col in schema["columns"]:
            cname = col["column_name"]
            dtype = col["data_type"].lower()
            nullable = col["nullable"]

            # Infer SQL type
            if "int" in dtype:
                sql_type = "INT"
            elif "float" in dtype:
                sql_type = "FLOAT"
            elif "date" in dtype or "time" in dtype:
                sql_type = "DATETIME2"
            elif "bool" in dtype:
                sql_type = "BIT"
            else:
                sql_type = "NVARCHAR(255)"

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


# ====================================================================
# FINAL DDL BUILDING FOR EXECUTOR
# ====================================================================
def generate_fabric_compatible_ddl(analysis):
    """Generate final DDL script with GO statements."""
    ddl_parts = []
    for tname, script in analysis.get("ddl_scripts", {}).items():
        sanitized = _sanitize_sql_script(script)
        ddl_parts.append(sanitized + "\nGO\n")
    return "".join(ddl_parts)



    """Execute DDL statements in Fabric Warehouse."""
    credential = DefaultAzureCredential()
    token = credential.get_token("https://api.fabric.microsoft.com/.default")

    url = f"https://api.fabric.microsoft.com/v1/workspaces/{WORKSPACE_ID}/warehouses/{WAREHOUSE_ID}/query"
    headers = {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}

    statements = [s.strip() for s in ddl.split("GO") if s.strip()]
    results = []

    for stmt in statements:
        resp = requests.post(url, headers=headers, json={"query": stmt})
        results.append({
            "sql": stmt,
            "success": resp.status_code == 200,
            "response": resp.text
        })

    return results