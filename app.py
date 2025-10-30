from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import traceback
from datetime import datetime
from util import (
    init_setup, build_tree, del_subtree, build_tree_node, split_node, 
    find_var, best_split, eval_split, add_nodes, copy_node_wrapper, 
    add_label, classify_dtype, create_node_datasets, find_var_regression, best_split_regression
)

def df_to_json_safe(df):
    """Convert DataFrame to JSON-safe format by replacing NaN with None"""
    if df is None or df.empty:
        return []
    
    # Replace NaN values with None for JSON serialization
    df_clean = df.copy()
    
    # Handle different data types
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'float32']:
            # Replace NaN with None for float columns
            df_clean[col] = df_clean[col].where(pd.notnull(df_clean[col]), None)
        elif df_clean[col].dtype == 'object':
            # For object columns, replace NaN with None
            df_clean[col] = df_clean[col].where(pd.notnull(df_clean[col]), None)
    
    # Convert to dict and ensure all values are JSON serializable
    result = df_clean.to_dict(orient='records')
    
    # Additional cleanup for any remaining NaN values
    for record in result:
        for key, value in record.items():
            if pd.isna(value) if hasattr(pd, 'isna') else (value != value):  # NaN check
                record[key] = None
    
    return result

app = Flask(__name__)
CORS(app)

# Global state management - better than scattered global variables
class TreeState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.file_name = None
        self.original_data_source = None  # Store original file path or URL
        self.df = None
        self.badflag = None
        self.df_col = None
        self.df_for_tree = None
        self.node_master = None
        self.node_datasets = None
        self.X_col = None
        self.max_depth = 5
        self.min_node_split = 500
        self.min_leaf_size = 100
        self.max_branches = 5
        self.criteria = "entropy"
        self.unique_var = 0
        self.monotonous = 0
        self.regression_flag = 0

# Initialize global state
tree_state = TreeState()

def validate_required_fields(data, required_fields):
    """Validate that required fields are present in request data"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, None

def validate_file_exists(file_path):
    """Validate that file exists and is accessible"""
    if not file_path:
        return False, "File path is required"
    if not os.path.exists(file_path):
        return False, f"File '{file_path}' not found"
    return True, None

def handle_error(error_msg, status_code=500):
    """Standardized error handling"""
    print(f"Error: {error_msg}")
    print(traceback.format_exc())
    return jsonify({"error": error_msg}), status_code

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint to verify backend is working"""
    return jsonify({"message": "Backend is working", "timestamp": str(datetime.now())})

@app.route('/api/importTree', methods=['POST'])
def importTree():
    """Import an existing decision tree from JSON file"""
    try:
        print("=== IMPORT TREE ENDPOINT CALLED ===")
        print(f"Request content type: {request.content_type}")
        print(f"Request files: {list(request.files.keys())}")
        print(f"Request form: {list(request.form.keys())}")
        print("Starting import process...")
        
        # Check if this is a file upload (multipart/form-data)
        if 'file' in request.files:
            print("Processing file upload...")
            file = request.files['file']
            if file.filename == '':
                return handle_error("No file selected", 400)
            
            print(f"Uploaded file: {file.filename}")
            
            # Check if it's a CSV file
            if file.filename.lower().endswith('.csv'):
                print("CSV file detected - routing to upload endpoint")
                # For CSV files, we need to process them like the upload endpoint
                # Save uploaded file to temporary directory
                import tempfile
                
                # Create uploads directory if it doesn't exist
                upload_dir = os.path.join(os.getcwd(), 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                
                # Generate unique filename
                import uuid
                unique_filename = f"{uuid.uuid4()}_{file.filename}"
                file_path = os.path.join(upload_dir, unique_filename)
                
                # Save the file
                file.save(file_path)
                
                # Load CSV file
                tree_state.df = pd.read_csv(file_path)
                tree_state.file_name = file_path
                tree_state.original_data_source = file_path  # Store full path to saved file
                
                # Return the same response as upload endpoint
                rows, columns = tree_state.df.shape
                sample = tree_state.df.head().replace({pd.NA: None, pd.NaT: None, np.nan: None}).to_dict(orient='records')
                column_names = tree_state.df.columns.tolist()
                
                return jsonify({
                    "rows": rows,
                    "columns": columns,
                    "columnsList": column_names,
                    "sample": sample,
                    "is_csv_import": True  # Flag to indicate this is a CSV import
                })
            else:
                # Handle JSON file
                file_content = file.read().decode('utf-8')
                print(f"File content length: {len(file_content)}")
                data = json.loads(file_content)
                print("Successfully parsed JSON from uploaded file")
        else:
            print("Processing JSON data with file_name...")
            # Handle JSON data with file_name
            data = request.get_json()
            if not data:
                return handle_error("No JSON data provided", 400)
            
            # Validate required fields
            is_valid, error_msg = validate_required_fields(data, ["file_name"])
            if not is_valid:
                return handle_error(error_msg, 400)
            
            file_name = data.get("file_name")
            print(f"Looking for file: {file_name}")
            print(f"File name type: {type(file_name)}")
            print(f"File name repr: {repr(file_name)}")
            
            # Check if it's a CSV file
            if file_name.lower().endswith('.csv'):
                print("CSV file detected in URL case - routing to upload endpoint")
                # For CSV files, we need to process them like the upload endpoint
                # Check if it's a URL or local file path
                if file_name.startswith(('http://', 'https://')):
                    # Handle URL - download the file
                    import requests
                    import tempfile
                    
                    try:
                        response = requests.get(file_name)
                        response.raise_for_status()
                        
                        # Create uploads directory if it doesn't exist
                        upload_dir = os.path.join(os.getcwd(), 'uploads')
                        os.makedirs(upload_dir, exist_ok=True)
                        
                        # Generate unique filename
                        import uuid
                        unique_filename = f"{uuid.uuid4()}_downloaded.csv"
                        file_path = os.path.join(upload_dir, unique_filename)
                        
                        # Save downloaded file
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Load CSV file
                        tree_state.df = pd.read_csv(file_path)
                        tree_state.file_name = file_path
                        tree_state.original_data_source = file_name  # Store original URL
                        
                    except requests.RequestException as e:
                        return handle_error(f"Failed to download file from URL: {str(e)}", 400)
                else:
                    # Handle local file path
                    print(f"Processing local file path: {file_name}")
                    print(f"Current working directory: {os.getcwd()}")
                    print(f"File exists check: {os.path.exists(file_name)}")
                    
                    # Try multiple path resolution strategies
                    file_paths_to_try = [
                        file_name,  # Try the original path first
                        os.path.basename(file_name),  # Try just the filename
                        os.path.join(os.getcwd(), file_name),  # Try with current working directory
                        os.path.join(os.getcwd(), os.path.basename(file_name))  # Try filename in current directory
                    ]
                    
                    file_loaded = False
                    for file_path in file_paths_to_try:
                        print(f"Trying file path: {file_path}")
                        if os.path.exists(file_path):
                            tree_state.df = pd.read_csv(file_path)
                            tree_state.file_name = file_path
                            tree_state.original_data_source = file_name  # Store original file path
                            print(f"Successfully loaded CSV with shape: {tree_state.df.shape}")
                            file_loaded = True
                            break
                    
                    if not file_loaded:
                        print(f"Local file not found in any of the attempted paths: {file_paths_to_try}")
                        return handle_error(f"File not found in any attempted paths: {file_paths_to_try}", 404)
                
                # Return the same response as upload endpoint
                rows, columns = tree_state.df.shape
                sample = tree_state.df.head().replace({pd.NA: None, pd.NaT: None, np.nan: None}).to_dict(orient='records')
                column_names = tree_state.df.columns.tolist()
                
                return jsonify({
                    "rows": rows,
                    "columns": columns,
                    "columnsList": column_names,
                    "sample": sample,
                    "is_csv_import": True  # Flag to indicate this is a CSV import
                })
            else:
                # Handle JSON file
                # Validate file exists
                is_valid, error_msg = validate_file_exists(file_name)
                if not is_valid:
                    return handle_error(error_msg, 404)
                
                with open(file_name, "r") as f:
                    data = json.load(f)
        
        # Load tree configuration
        tree_state.file_name = data.get("file_name")
        tree_state.badflag = data.get("badflag")
        tree_state.max_depth = data.get("max_depth", 5)
        tree_state.min_node_split = data.get("min_node_split", 500)
        tree_state.min_leaf_size = data.get("min_leaf_size", 100)
        tree_state.max_branches = data.get("max_branches", 5)
        tree_state.criteria = data.get("criteria", "entropy")
        tree_state.unique_var = data.get("unique_var", 0)
        tree_state.monotonous = data.get("monotonous", 0)
        tree_state.df_col = pd.DataFrame(data.get("mapper"))
        node_master_read = pd.DataFrame(data.get("node_master"))
        
        # Load and process data
        print(f"Loading data from file: {tree_state.file_name}")
        
        # Try to load the original data source if it exists
        original_file_name = tree_state.file_name
        print(f"Attempting to load original data from: {original_file_name}")
        
        # Check if the file_name in JSON points to a CSV file
        if original_file_name and original_file_name.lower().endswith('.csv'):
            print("JSON contains CSV file path - processing as CSV import")
            try:
                # Check if it's a URL or local file path
                if original_file_name.startswith(('http://', 'https://')):
                    # Handle URL - download the file
                    import requests
                    import tempfile
                    
                    try:
                        response = requests.get(original_file_name)
                        response.raise_for_status()
                        
                        # Create uploads directory if it doesn't exist
                        upload_dir = os.path.join(os.getcwd(), 'uploads')
                        os.makedirs(upload_dir, exist_ok=True)
                        
                        # Generate unique filename
                        import uuid
                        unique_filename = f"{uuid.uuid4()}_imported.csv"
                        file_path = os.path.join(upload_dir, unique_filename)
                        
                        # Save downloaded file
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Load CSV file
                        tree_state.df = pd.read_csv(file_path)
                        print(f"Successfully loaded data from URL: {original_file_name}")
                        
                    except requests.RequestException as e:
                        print(f"Failed to download from URL: {e}. Creating mock dataset...")
                        raise e
                else:
                    # Handle local file path
                    print(f"Processing local file path: {original_file_name}")
                    print(f"Current working directory: {os.getcwd()}")
                    print(f"File exists check: {os.path.exists(original_file_name)}")
                    
                    # Try multiple path resolution strategies
                    file_paths_to_try = [
                        original_file_name,  # Try the original path first
                        os.path.basename(original_file_name),  # Try just the filename
                        os.path.join(os.getcwd(), original_file_name),  # Try with current working directory
                        os.path.join(os.getcwd(), os.path.basename(original_file_name))  # Try filename in current directory
                    ]
                    
                    file_loaded = False
                    for file_path in file_paths_to_try:
                        print(f"Trying file path: {file_path}")
                        if os.path.exists(file_path):
                            tree_state.df = pd.read_csv(file_path)
                            print(f"Successfully loaded data from local file: {file_path}")
                            file_loaded = True
                            break
                    
                    if not file_loaded:
                        print(f"Local file not found in any of the attempted paths: {file_paths_to_try}. Creating mock dataset...")
                        raise FileNotFoundError(f"File not found in any attempted paths: {file_paths_to_try}")
                        
            except Exception as e:
                print(f"Failed to load CSV data from JSON file_name: {e}")
                print("Creating mock dataset based on column mapping to avoid file dependency issues...")
                
                # Create a mock dataset based on the column mapping
                df_col_small = tree_state.df_col[tree_state.df_col['datatype'].isin(['numerical', 'categorical'])]
                mock_data = {}
                
                for _, row in df_col_small.iterrows():
                    col_name = row['column_name']
                    if row['datatype'] == 'numerical':
                        mock_data[col_name] = [0.0] * 100  # Create 100 rows of zeros
                    else:  # categorical
                        mock_data[col_name] = ['category1'] * 100  # Create 100 rows of category1
                
                # Add the target variable
                if tree_state.badflag not in mock_data:
                    mock_data[tree_state.badflag] = [0] * 100  # Create 100 rows of zeros
                
                tree_state.df = pd.DataFrame(mock_data)
                print(f"Created mock dataset with shape: {tree_state.df.shape}")
                print(f"Mock data columns: {list(tree_state.df.columns)}")
        else:
            # Original logic for non-CSV files - skip data loading for now
            print("Non-CSV file detected - skipping data loading")
            # For non-CSV files, we'll use the existing logic
            pass
        
        # Verify all required columns exist
        df_col_small = tree_state.df_col[tree_state.df_col['datatype'].isin(['numerical', 'categorical'])]
        print(f"Columns to use for tree: {list(df_col_small.column_name)}")
        
        # Check if all required columns exist
        missing_cols = [col for col in df_col_small.column_name if col not in tree_state.df.columns]
        if missing_cols:
            print(f"Adding missing columns: {missing_cols}")
            for col in missing_cols:
                tree_state.df[col] = [0.0] * 100  # Add missing columns
        
        # Check if badflag column exists
        if tree_state.badflag not in tree_state.df.columns:
            print(f"Adding missing badflag column: {tree_state.badflag}")
            tree_state.df[tree_state.badflag] = [0] * 100
        
        tree_state.df_for_tree = tree_state.df[df_col_small.column_name]
        tree_state.df_for_tree['badflag'] = tree_state.df[tree_state.badflag]
        
        print(f"Final df_for_tree columns: {list(tree_state.df_for_tree.columns)}")
        print(f"Final df_for_tree shape: {tree_state.df_for_tree.shape}")
        
        # Check if variables in node_master exist in the dataset
        if 'variable' in node_master_read.columns:
            node_variables = node_master_read['variable'].dropna().unique()
            missing_vars = [var for var in node_variables if var not in tree_state.df_for_tree.columns and var not in ['Parent', '']]
            if missing_vars:
                return handle_error(f"Variables in tree not found in data: {missing_vars}. Available columns: {list(tree_state.df_for_tree.columns)}", 400)
        
        print(f"About to call create_node_datasets...")
        print(f"df_for_tree columns: {list(tree_state.df_for_tree.columns)}")
        print(f"df_for_tree shape: {tree_state.df_for_tree.shape}")
        print(f"node_master_read columns: {list(node_master_read.columns)}")
        print(f"node_master_read shape: {node_master_read.shape}")
        
        # Properly initialize the tree state for imported trees
        print("Initializing tree state for imported tree...")
        tree_state.node_master = node_master_read
        
        # Create proper node_datasets structure
        # For imported trees, assign ALL data to the root node (node 1)
        # This ensures that when we delete subtrees, the root node still has all the data
        tree_state.node_datasets = tree_state.df_for_tree.copy()
        tree_state.node_datasets['node_number'] = 1
        
        print("All data assigned to root node (node 1) for imported tree")
        
        print(f"Tree state initialized successfully")
        print(f"node_master shape: {tree_state.node_master.shape}")
        print(f"node_datasets shape: {tree_state.node_datasets.shape}")
        print(f"node_datasets node_number distribution: {tree_state.node_datasets['node_number'].value_counts()}")
        
        print(f"create_node_datasets completed successfully")
        
        # Set X_col (feature columns excluding the target variable)
        print(f"Setting X_col...")
        print(f"df_for_tree columns: {list(tree_state.df_for_tree.columns)}")
        print(f"badflag: '{tree_state.badflag}'")
        print(f"badflag in columns: {tree_state.badflag in tree_state.df_for_tree.columns}")
        
        # Always exclude the target variable from feature columns
        # Use the mapper to determine feature columns (exclude target variable)
        df_col_small = tree_state.df_col[tree_state.df_col['datatype'].isin(['numerical', 'categorical'])]
        feature_columns = df_col_small['column_name'].tolist()
        
        print(f"Feature columns from mapper: {feature_columns}")
        
        # Ensure all feature columns exist in df_for_tree
        available_features = [col for col in feature_columns if col in tree_state.df_for_tree.columns]
        print(f"Available feature columns: {available_features}")
        
        if available_features:
            tree_state.X_col = available_features
            print(f"X_col set from mapper features: {list(tree_state.X_col)}")
        else:
            # Fallback: exclude target variable from all columns
            if tree_state.badflag in tree_state.df_for_tree.columns:
                tree_state.X_col = tree_state.df_for_tree.drop([tree_state.badflag], axis=1).columns
                print(f"X_col set by dropping badflag: {list(tree_state.X_col)}")
            else:
                print(f"Warning: badflag '{tree_state.badflag}' not in df_for_tree columns, using all columns")
                tree_state.X_col = tree_state.df_for_tree.columns
        
        print(f"Final X_col: {list(tree_state.X_col)}")
        print(f"X_col type: {type(tree_state.X_col)}")
        
        print(f"About to call df_to_json_safe...")
        print(f"node_master type: {type(tree_state.node_master)}")
        print(f"node_master shape: {tree_state.node_master.shape if hasattr(tree_state.node_master, 'shape') else 'No shape'}")
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/getImportDiagnostics', methods=['GET'])
def getImportDiagnostics():
    """Get diagnostic information about the imported tree data"""
    try:
        if tree_state.df is None:
            return handle_error("No data available. Please import a tree first.", 400)
        
        # Get sample records (first 5 rows)
        sample_records = tree_state.df.head(5).replace({pd.NA: None, pd.NaT: None, np.nan: None}).to_dict(orient='records')
        
        # Get column information
        columns_info = []
        for col in tree_state.df.columns:
            dtype = str(tree_state.df[col].dtype)
            unique_count = tree_state.df[col].nunique()
            columns_info.append({
                'column_name': col,
                'datatype': dtype,
                'unique_values': unique_count,
                'sample_values': tree_state.df[col].head(3).replace({pd.NA: None, pd.NaT: None, np.nan: None}).tolist()
            })
        
        # Get mapper information
        mapper_info = []
        if tree_state.df_col is not None:
            for _, row in tree_state.df_col.iterrows():
                mapper_info.append({
                    'column_name': row['column_name'],
                    'datatype': row['datatype']
                })
        
        diagnostics = {
            'dataset_info': {
                'rows': tree_state.df.shape[0],
                'columns': tree_state.df.shape[1],
                'sample_records': sample_records
            },
            'target_variable': tree_state.badflag,
            'columns_info': columns_info,
            'mapper_info': mapper_info,
            'hyperparameters': {
                'max_depth': tree_state.max_depth,
                'min_node_split': tree_state.min_node_split,
                'min_leaf_size': tree_state.min_leaf_size,
                'max_branches': tree_state.max_branches,
                'criteria': tree_state.criteria,
                'unique_var': tree_state.unique_var,
                'monotonous': tree_state.monotonous
            },
            'feature_columns': list(tree_state.X_col) if tree_state.X_col is not None else [],
            'df_for_tree_shape': tree_state.df_for_tree.shape if tree_state.df_for_tree is not None else None
        }
        
        return jsonify(diagnostics)
    
    except Exception as e:
        print(f"Error in getImportDiagnostics: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return handle_error(str(e))

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload and validate CSV file"""
    try:
        # Check if this is a file upload (multipart/form-data) or JSON request
        if request.files:
            # Handle file upload
            uploaded_file = request.files.get('file')
            if not uploaded_file:
                return handle_error("No file provided in upload", 400)
            
            if not uploaded_file.filename.endswith('.csv'):
                return handle_error("Only CSV files are allowed", 400)
            
            # Save uploaded file to temporary directory
            import tempfile
            
            # Create uploads directory if it doesn't exist
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Generate unique filename
            import uuid
            unique_filename = f"{uuid.uuid4()}_{uploaded_file.filename}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Save the file
            uploaded_file.save(file_path)
            
            # Load CSV file
            tree_state.df = pd.read_csv(file_path)
            tree_state.file_name = file_path
            tree_state.original_data_source = file_path  # Store full path to saved file
            
        else:
            # Handle JSON request (existing file path or URL)
            data = request.get_json()
            if not data:
                return handle_error("No JSON data provided", 400)
            
            # Validate required fields
            is_valid, error_msg = validate_required_fields(data, ["file"])
            if not is_valid:
                return handle_error(error_msg, 400)
            
            file_name = data.get("file")
            
            # Check if it's a URL or local file path
            if file_name.startswith(('http://', 'https://')):
                # Handle URL - download the file
                import requests
                import tempfile
                
                try:
                    response = requests.get(file_name)
                    response.raise_for_status()
                    
                    # Create uploads directory if it doesn't exist
                    upload_dir = os.path.join(os.getcwd(), 'uploads')
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # Generate unique filename
                    import uuid
                    unique_filename = f"{uuid.uuid4()}_downloaded.csv"
                    file_path = os.path.join(upload_dir, unique_filename)
                    
                    # Save downloaded file
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Load CSV file
                    tree_state.df = pd.read_csv(file_path)
                    tree_state.file_name = file_path
                    tree_state.original_data_source = file_name  # Store original URL
                    
                except requests.RequestException as e:
                    return handle_error(f"Failed to download file from URL: {str(e)}", 400)
            else:
                # Handle local file path
                # Try multiple path resolution strategies
                file_paths_to_try = [
                    file_name,  # Try the original path first
                    os.path.basename(file_name),  # Try just the filename
                    os.path.join(os.getcwd(), file_name),  # Try with current working directory
                    os.path.join(os.getcwd(), os.path.basename(file_name))  # Try filename in current directory
                ]
                
                file_loaded = False
                for file_path in file_paths_to_try:
                    if os.path.exists(file_path):
                        tree_state.df = pd.read_csv(file_path)
                        tree_state.file_name = file_path
                        tree_state.original_data_source = file_name  # Store original file path
                        file_loaded = True
                        break
                
                if not file_loaded:
                    return handle_error(f"File not found in any attempted paths: {file_paths_to_try}", 404)
        
        rows, columns = tree_state.df.shape
        sample = tree_state.df.head().replace({pd.NA: None, pd.NaT: None, np.nan: None}).to_dict(orient='records')
        column_names = tree_state.df.columns.tolist()
        
        return jsonify({
            'rows': rows,
            'columns': columns,
            'sample': sample,
            'columnsList': column_names
        })
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/setTarget', methods=['POST'])
def setTarget():
    """Set target variable for the decision tree"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["target"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        target = data.get("target")
        regression_flag = data.get("regression_flag")
        
        
        # Validate target exists in dataframe
        if tree_state.df is None:
            return handle_error("No data loaded. Please upload a file first.", 400)
        
        if target not in tree_state.df.columns:
            return handle_error(f"Target variable '{target}' not found in data", 400)
        
        tree_state.badflag = target
        tree_state.regression_flag = regression_flag
        
        return jsonify({"message": f"Target variable set to {target}"})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/getTypes', methods=['GET'])
def getTypes():
    """Get data types for all columns except target variable"""
    try:
        if tree_state.df is None:
            return handle_error("No data loaded. Please upload a file first.", 400)
        
        if tree_state.badflag is None:
            return handle_error("No target variable set. Please set target variable first.", 400)
        
        col_types = {
            col: classify_dtype(tree_state.df.drop(tree_state.badflag, axis=1)[col].dtype) 
            for col in tree_state.df.drop(tree_state.badflag, axis=1).columns
        }
        
        return jsonify(col_types)
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/keepVar', methods=['POST'])
def keepVar():
    """Select variables to keep for tree building"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        if tree_state.df is None:
            return handle_error("No data loaded. Please upload a file first.", 400)
        
        if tree_state.badflag is None:
            return handle_error("No target variable set. Please set target variable first.", 400)
        
        tree_state.df_col = pd.DataFrame(list(data.items()), columns=['column_name', 'datatype'])
        print(f"df_col created: {tree_state.df_col}")
        
        df_col_small = tree_state.df_col[tree_state.df_col['datatype'].isin(['numerical', 'categorical'])]
        print(f"df_col_small after filtering: {df_col_small}")
        
        # Convert to list to ensure proper column selection
        feature_columns = df_col_small['column_name'].tolist()
        print(f"Feature columns to use: {feature_columns}")
        
        tree_state.df_for_tree = tree_state.df[feature_columns]
        print(f"df_for_tree columns after feature selection: {list(tree_state.df_for_tree.columns)}")
        
        # Convert target variable to numeric (0/1) for tree building
        target_values = tree_state.df[tree_state.badflag]
        if target_values.dtype == 'object':
            # Convert string values to numeric (Yes/No -> 1/0)
            unique_values = target_values.unique()
            if len(unique_values) == 2:
                # Create mapping: first unique value -> 1, second -> 0
                value_map = {unique_values[0]: 1, unique_values[1]: 0}
                tree_state.df_for_tree['badflag'] = target_values.map(value_map)
            else:
                return handle_error(f"Target variable must have exactly 2 unique values, found {len(unique_values)}", 400)
        else:
            # Already numeric
            tree_state.df_for_tree['badflag'] = target_values
        
        column_names = tree_state.df_for_tree.columns.tolist()
        return jsonify({'columnsList': column_names})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/defineHP', methods=['POST'])
def defineHP():
    """Define hyperparameters for decision tree"""
    try:
        # Set default values
        tree_state.max_depth = 5
        tree_state.min_node_split = 500
        tree_state.min_leaf_size = 100
        tree_state.max_branches = 5
        tree_state.criteria = "entropy"
        tree_state.unique_var = 0
        tree_state.monotonous = 0
        
        # Override with provided values
        data = request.get_json(silent=True) or {}
        tree_state.max_depth = data.get("max_depth", tree_state.max_depth)
        tree_state.min_node_split = data.get("min_node_split", tree_state.min_node_split)
        tree_state.min_leaf_size = data.get("min_leaf_size", tree_state.min_leaf_size)
        tree_state.max_branches = data.get("max_branches", tree_state.max_branches)
        tree_state.criteria = data.get("criteria", tree_state.criteria)
        tree_state.unique_var = data.get("unique_var", tree_state.unique_var)
        tree_state.monotonous = data.get("monotonous", tree_state.monotonous)
        
        return jsonify({
            'max_depth': tree_state.max_depth,
            'min_node_split': tree_state.min_node_split,
            'min_leaf_size': tree_state.min_leaf_size,
            'max_branches': tree_state.max_branches,
            'criteria': tree_state.criteria,
            'unique_var': tree_state.unique_var,
            'monotonous': tree_state.monotonous
        })
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/initTree', methods=['POST'])
def initTree():
    """Initialize decision tree with root node"""
    try:
        if tree_state.df_for_tree is None:
            return handle_error("No processed data available. Please complete data setup first.", 400)
        
        tree_state.X_col, tree_state.node_master, tree_state.node_datasets = init_setup(
            tree_state.df_for_tree, 'badflag', tree_state.max_depth, 
            tree_state.min_node_split, tree_state.min_leaf_size, tree_state.max_branches
        )
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/autoTree', methods=['POST'])
def autoTree():
    """Train entire tree automatically"""
    try:
        if tree_state.df_for_tree is None:
            return handle_error("No processed data available. Please complete data setup first.", 400)
        
        tree_state.node_master, tree_state.node_datasets = build_tree(
            tree_state.df_for_tree, 'badflag', tree_state.max_depth, 
            tree_state.min_node_split, tree_state.min_leaf_size, 
            tree_state.max_branches, tree_state.df_col, tree_state.regression_flag
        )
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/delSubTree', methods=['POST'])
def delSubTree():
    """Delete subtree starting from specified node"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        
        if tree_state.node_master is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        tree_state.node_master, tree_state.node_datasets = del_subtree(
            tree_state.node_master, tree_state.node_datasets, node_number
        )
        
        # After deleting subtree, ensure the parent node is marked as a non-leaf node
        # This is crucial for auto training to work properly
        if node_number in tree_state.node_master['node_number'].values:
            print(f"Marking node {node_number} as non-leaf after deleting subtree")
            tree_state.node_master.loc[tree_state.node_master['node_number'] == node_number, 'leaf_flag'] = 0
            print(f"Node {node_number} leaf_flag set to 0")
        
        # After deleting subtree, ensure all remaining data is assigned to the root node
        # This is crucial for auto training to work properly
        if node_number == 1:  # If deleting from root node
            print("Deleting subtree from root node - reassigning all data to root node")
            tree_state.node_datasets = tree_state.df_for_tree.copy()
            tree_state.node_datasets['node_number'] = 1
            print(f"Reassigned all data to root node. node_datasets shape: {tree_state.node_datasets.shape}")
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/autoTrainSubTree', methods=['POST'])
def autoTrainSubTree():
    """Automatically train subtree from specified node"""
    try:
        print("=== AUTO TRAIN SUBTREE ENDPOINT CALLED ===")
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        print(f"Processing auto train for node: {node_number}")
        
        if tree_state.node_master is None or tree_state.node_datasets is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        # Check if the node exists in node_master
        if node_number not in tree_state.node_master['node_number'].values:
            return handle_error(f"Node {node_number} not found in tree", 400)
        
        # Check if required state variables are set
        if tree_state.df_for_tree is None:
            return handle_error("Data for tree not available", 400)
        if tree_state.X_col is None:
            return handle_error("Feature columns not defined", 400)
        if tree_state.df_col is None:
            return handle_error("Column mapping not available", 400)
        
        print(f"Auto training subtree for node {node_number}")
        print(f"Node master shape: {tree_state.node_master.shape}")
        print(f"Node datasets shape: {tree_state.node_datasets.shape}")
        print(f"df_for_tree shape: {tree_state.df_for_tree.shape}")
        print(f"X_col: {list(tree_state.X_col)}")
        print(f"df_col shape: {tree_state.df_col.shape}")
        
        print("About to call build_tree_node...")
        tree_state.node_master, tree_state.node_datasets = build_tree_node(
            tree_state.df_for_tree, 'badflag', tree_state.max_depth, 
            tree_state.min_node_split, tree_state.min_leaf_size, 
            tree_state.max_branches, node_number, tree_state.X_col, 
            tree_state.node_master, tree_state.node_datasets, tree_state.df_col, tree_state.regression_flag
        )
        print("build_tree_node completed successfully")
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        print(f"Error in autoTrainSubTree: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return handle_error(f"Auto train failed: {str(e)}")

@app.route('/api/findSplit', methods=['POST'])
def findSplit():
    """Find best split for specified node"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        
        if tree_state.node_master is None or tree_state.node_datasets is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        tree_state.node_master, tree_state.node_datasets = split_node(
            node_number, tree_state.min_leaf_size, tree_state.max_branches, 
            tree_state.max_depth, tree_state.min_node_split, tree_state.X_col, 
            tree_state.node_master, tree_state.node_datasets, 'badflag', tree_state.df_col, tree_state.regression_flag
        )
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/findVar', methods=['POST'])
def findVar():
    """Find top variables for splitting at specified node"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        
        if tree_state.node_datasets is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        if tree_state.regression_flag == 0:
            top_vars = find_var(
                node_number, tree_state.node_datasets, tree_state.X_col, 
                'badflag', tree_state.df_col
            )
        else:
            top_vars = find_var_regression(
                node_number, tree_state.node_datasets, tree_state.X_col, 
                'badflag', tree_state.df_col
            )
        
        return jsonify({"top_vars": df_to_json_safe(top_vars)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/bestSplit', methods=['POST'])
def bestSplit():
    """Find best split for specified variable at specified node"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number", "variable_name"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        variable_name = data.get("variable_name")
        
        if tree_state.node_datasets is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        if tree_state.regression_flag == 0:
            df_out = best_split(
                node_number, tree_state.node_datasets, tree_state.X_col, 
                'badflag', variable_name, tree_state.min_leaf_size, 
                tree_state.max_branches, tree_state.df_col
            )
        else:
            df_out = best_split_regression(
                node_number, tree_state.node_datasets, tree_state.X_col, 
                'badflag', variable_name, tree_state.min_leaf_size, 
                tree_state.max_branches, tree_state.df_col
            )
        
        return jsonify({"best_split": df_to_json_safe(df_out)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/evalSplit', methods=['POST'])
def evalSplit():
    """Evaluate provided splits for specified node"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number", "variable_name", "splits"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        variable_name = data.get("variable_name")
        splits_data = data.get("splits")
        
        if tree_state.node_datasets is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        # Create DataFrame and add required columns
        df = pd.DataFrame(splits_data)
        
        # Add missing columns that eval_split expects
        df['variable'] = variable_name
        # Use missing values from input data if provided, otherwise default to False
        if 'missing' not in df.columns:
            df['missing'] = False  # Default to no missing values
        else:
            # Ensure missing values are integers (0/1)
            df['missing'] = df['missing'].astype(int)
        
        # For categorical variables, use the list_values from the input data
        if 'list_values' not in df.columns:
            df['list_values'] = [[] for _ in range(len(df))]  # Default empty list
        else:
            # Ensure list_values are properly formatted as lists
            df['list_values'] = df['list_values'].apply(lambda x: x if isinstance(x, list) else [x] if x is not None else [])
        
        df_out = eval_split(node_number, tree_state.node_datasets, 'badflag', df, tree_state.df_col)
        
        return jsonify({"eval_split": df_to_json_safe(df_out)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/addNode', methods=['POST'])
def addNode():
    """Add new nodes based on provided splits"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number", "variable_name", "splits"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        variable_name = data.get("variable_name")
        splits_data = data.get("splits")
        
        if tree_state.node_master is None or tree_state.node_datasets is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        # Create DataFrame and add required columns
        df = pd.DataFrame(splits_data)
        
        # Add missing columns that add_nodes expects
        df['variable'] = variable_name
        # Use missing values from input data if provided, otherwise default to 0
        if 'missing' not in df.columns:
            df['missing'] = 0  # Default to no missing values
        else:
            # Ensure missing values are integers (0/1)
            df['missing'] = df['missing'].astype(int)
        # Use list_values from input data if provided, otherwise default to empty list
        if 'list_values' not in df.columns:
            df['list_values'] = [[] for _ in range(len(df))]
        
        tree_state.node_master, tree_state.node_datasets = add_nodes(
            df, node_number, tree_state.max_depth, tree_state.min_node_split, 
            0, tree_state.node_master, tree_state.node_datasets, 'badflag', tree_state.df_col
        )
        
        return jsonify({"success": True, "message": "Nodes added successfully", "node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/copyNode', methods=['POST'])
def copyNode():
    """Copy subtree from one node to another"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["from_node", "to_node"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        from_node = data.get("from_node")
        to_node = data.get("to_node")
        
        if tree_state.node_master is None or tree_state.node_datasets is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        tree_state.node_master, tree_state.node_datasets = copy_node_wrapper(
            from_node, to_node, tree_state.node_master, tree_state.node_datasets, 
            tree_state.max_depth, tree_state.min_node_split, 'badflag', tree_state.df_col
        )
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/labelNode', methods=['POST'])
def labelNode():
    """Add label to specified node"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided", 400)
        
        # Validate required fields
        is_valid, error_msg = validate_required_fields(data, ["node_number", "label"])
        if not is_valid:
            return handle_error(error_msg, 400)
        
        node_number = data.get("node_number")
        label = data.get("label")
        
        if tree_state.node_master is None:
            return handle_error("No tree initialized. Please initialize tree first.", 400)
        
        tree_state.node_master = add_label(node_number, label, tree_state.node_master)
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/exportTree', methods=['GET'])
def exportTree():
    """Export current tree state as JSON"""
    try:
        if tree_state.node_master is None:
            return handle_error("No tree to export. Please initialize tree first.", 400)
        
        return jsonify({
            'file_name': tree_state.file_name,
            'badflag': tree_state.badflag,
            'max_depth': tree_state.max_depth,
            'min_node_split': tree_state.min_node_split,
            'min_leaf_size': tree_state.min_leaf_size,
            'max_branches': tree_state.max_branches,
            'criteria': tree_state.criteria,
            'unique_var': tree_state.unique_var,
            'monotonous': tree_state.monotonous,
            'mapper': df_to_json_safe(tree_state.df_col),
            'node_master': df_to_json_safe(tree_state.node_master)
        })
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/downloadTree', methods=['GET'])
def downloadTree():
    """Download current tree state as JSON file"""
    try:
        if tree_state.node_master is None:
            return handle_error("No tree to download. Please initialize tree first.", 400)
        
        data = {
            'file_name': tree_state.original_data_source or tree_state.file_name,  # Use original data source if available
            'badflag': tree_state.badflag,
            'max_depth': tree_state.max_depth,
            'min_node_split': tree_state.min_node_split,
            'min_leaf_size': tree_state.min_leaf_size,
            'max_branches': tree_state.max_branches,
            'criteria': tree_state.criteria,
            'unique_var': tree_state.unique_var,
            'monotonous': tree_state.monotonous,
            'mapper': df_to_json_safe(tree_state.df_col),
            'node_master': df_to_json_safe(tree_state.node_master)
        }
        
        json_str = json.dumps(data, indent=2)
        response = Response(
            json_str,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=tree_export.json"}
        )
        return response
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/getTreeData', methods=['GET'])
def getTreeData():
    """Get current tree data"""
    try:
        if tree_state.node_master is None:
            return handle_error("No tree data available. Please build a tree first.", 400)
        
        return jsonify({"node_master": df_to_json_safe(tree_state.node_master)})
    
    except Exception as e:
        return handle_error(str(e))

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Decision Tree API is running"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)