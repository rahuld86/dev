from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import os
import json
from util import init_setup, build_tree, del_subtree, build_tree_node, split_node, find_var, best_split, eval_split, add_nodes, copy_node_wrapper, add_label, classify_dtype, create_node_datasets

app = Flask(__name__)
CORS(app)

## Load file

@app.route('/api/importTree',methods=['POST'])
def importTree():
    
    try:
        data_api = request.get_json()
        file_name = data_api.get("file_name")
        
        with open(file_name, "r") as f:
            data = json.load(f)
        
        global badflag, max_depth, min_node_split, min_leaf_size, max_branches, criteria, unique_var, monotonous, df_col, node_master_read
        data_file = data.get("file_name")
        badflag = data.get("badflag")
        max_depth = data.get("max_depth")
        min_node_split = data.get("min_node_split")
        min_leaf_size = data.get("min_leaf_size")
        max_branches = data.get("max_branches")
        criteria = data.get("criteria")
        unique_var = data.get("unique_var")
        monotonous = data.get("monotonous")
        df_col = pd.DataFrame(data.get("mapper"))
        node_master_read = pd.DataFrame(data.get("node_master"))
        
        df = pd.read_csv(data_file)
        
        df_col_small=df_col[df_col['datatype'].isin(['numerical', 'categorical'])]
        global df_for_tree
        df_for_tree = df[df_col_small.column_name]
        df_for_tree['badflag']=df[badflag]
        
        global node_master, node_datasets
        node_master, node_datasets = create_node_datasets(df_for_tree, df_col, node_master_read)
        
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload',methods=['POST'])
def upload_csv():
    
    data = request.get_json()
    global file_name
    file_name = data.get("file")
    
    # Validate file name
    if not file_name:
        return jsonify({"error": "Please provide a file_name"}), 400
    
    # Check if file exists
    if not os.path.exists(file_name):
        return jsonify({"error": f"File '{file_name}' not found"}), 404
    
    try:
        # Try reading file into pandas dataframe
        #load_dataframe(file_name)
        global df 
        df = pd.read_csv(file_name)
        rows, columns = df.shape
        sample = df.head().where(pd.notnull(df),None).to_dict(orient='records')
        column_names = df.columns.tolist()
        return jsonify({
            'rows': rows,
            'columns':columns,
            'sample': sample,
            'columnsList': column_names})
    except Exception as e:
        return jsonify({'error':str(e)}),500

## set target variable

@app.route('/api/setTarget',methods=['POST'])
def setTarget():
    
    data = request.get_json()
    global badflag
    badflag = data.get("target")
    
    return jsonify({"message": f"target variable set to {badflag}"})

@app.route('/api/getTypes',methods=['GET'])
def getTypes():
    
    col_types = {col: classify_dtype(df.drop(badflag,axis=1)[col].dtype) for col in df.drop(badflag,axis=1).columns}
    return jsonify(col_types)

## decide variables to keep (only categorical and numerical), define global X and y

@app.route('/api/keepVar',methods=['POST'])
def keepVar():
    
    data = request.get_json()
    global df_col
    df_col = pd.DataFrame(list(data.items()), columns=['column_name', 'datatype'])
    df_col_small=df_col[df_col['datatype'].isin(['numerical', 'categorical'])]
    
    global df_for_tree
    df_for_tree = df[df_col_small.column_name]
    df_for_tree['badflag']=df[badflag]
    
    column_names = df_for_tree.columns.tolist()
    return jsonify({'columnsList': column_names})

## Initialize Decision Tree Hyperparameters

@app.route('/api/defineHP',methods=['POST'])
def defineHP():
    
    global max_depth
    global min_node_split
    global min_leaf_size 
    global max_branches
    global criteria
    global unique_var
    global monotonous
    
    max_depth = 5
    min_node_split=500
    min_leaf_size=100
    max_branches=5
    criteria="entropy"
    unique_var=0
    monotonous=0
    
    data = request.get_json(silent=True) or {}
    max_depth       = data.get("max_depth", max_depth)
    min_node_split  = data.get("min_node_split", min_node_split)
    min_leaf_size   = data.get("min_leaf_size", min_leaf_size)
    max_branches    = data.get("max_branches", max_branches)
    criteria        = data.get("criteria", criteria)
    unique_var      = data.get("unique_var", unique_var)
    monotonous      = data.get("monotonous", monotonous)
        
    return jsonify({
            'max_depth': max_depth,
            'min_node_split':min_node_split,
            'min_leaf_size': min_leaf_size,
            'max_branches': max_branches,
            'criteria': criteria,
            'unique_var': unique_var,
            'monotonous': monotonous})

# Initialize Decision Tree with Root node    

@app.route('/api/initTree',methods=['POST'])
def initTree():
    global X_col
    global node_master
    global node_datasets
    
    try:
        X_col, node_master, node_datasets = init_setup(df_for_tree, 'badflag', max_depth, min_node_split, min_leaf_size, max_branches)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Train entire tree automatically using Hyperparameters provided    

@app.route('/api/autoTree',methods=['POST'])
def autoTree():  
    global node_master
    global node_datasets
    
    try:
        node_master, node_datasets = build_tree(df_for_tree, 'badflag', max_depth, min_node_split, min_leaf_size, max_branches, df_col)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/delSubTree',methods=['POST'])
def delSubTree():
    data = request.get_json()
    node_number = data.get("node_number")
    
    global node_master
    global node_datasets
    
    try:
        node_master, node_datasets = del_subtree(node_master, node_datasets, node_number)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/autoTrainSubTree',methods=['POST'])
def autoTrainSubTree():
    data = request.get_json()
    node_number = data.get("node_number")
    
    global node_master
    global node_datasets
    
    try:
        node_master, node_datasets = build_tree_node(df_for_tree, 'badflag', max_depth, min_node_split, min_leaf_size, max_branches, node_number,X_col, node_master, node_datasets, df_col)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/findSplit',methods=['POST'])
def findSplit():
    data = request.get_json()
    node_number = data.get("node_number")
    
    global node_master
    global node_datasets
    
    try:
        node_master, node_datasets = split_node(node_number, min_leaf_size, max_branches, max_depth, min_node_split, X_col, node_master, node_datasets, 'badflag', df_col)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/findVar',methods=['POST'])
def findVar():
    data = request.get_json()
    node_number = data.get("node_number")
    
    try:
        top_vars = find_var(node_number, node_datasets, X_col, 'badflag', df_col)
        return jsonify({"top_vars":top_vars.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/bestSplit',methods=['POST'])
def bestSplit():
    data = request.get_json()
    node_number = data.get("node_number")
    variable_name = data.get("variable_name")
    
    try:
        df_out = best_split(node_number, node_datasets, X_col, 'badflag', variable_name, min_leaf_size, max_branches, df_col)
        return jsonify({"best_split":df_out.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/evalSplit',methods=['POST'])
def evalSplit():
    data = request.get_json()
    node_number = data.get("node_number")
    df = pd.DataFrame(data.get("splits"))
    
    try:
        df_out = eval_split(node_number, node_datasets, 'badflag', df, df_col)
        return jsonify({"best_split":df_out.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/addNode',methods=['POST'])
def addNode():
    data = request.get_json()
    node_number = data.get("node_number")
    df = pd.DataFrame(data.get("splits"))
    
    global node_master
    global node_datasets
    
    try:
        node_master, node_datasets = add_nodes(df, node_number, max_depth, min_node_split, 0, node_master, node_datasets, 'badflag', df_col)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/copyNode',methods=['POST'])
def copyNode():
    data = request.get_json()
    from_node = data.get("from_node")
    to_node = data.get("to_node")
    
    global node_master
    global node_datasets
    
    try:
        node_master, node_datasets = copy_node_wrapper(from_node, to_node, node_master, node_datasets, max_depth, min_node_split, 'badflag', df_col)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/labelNode',methods=['POST'])
def labelNode():
    data = request.get_json()
    node_number = data.get("node_number")
    label = data.get("label")
    
    global node_master
    
    try:
        node_master = add_label(node_number, label, node_master)
        return jsonify({"node_master":node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/exportTree',methods=['GET'])
def exportTree():
    
    try:
        return jsonify({
            'file_name': file_name,
            'badflag':badflag,
            'max_depth': max_depth,
            'min_node_split':min_node_split,
            'min_leaf_size': min_leaf_size,
            'max_branches': max_branches,
            'criteria': criteria,
            'unique_var': unique_var,
            'monotonous': monotonous,
            'mapper': df_col.to_dict(orient='records'),
            'node_master': node_master.to_dict(orient='records')})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/downloadTree',methods=['GET'])
def downloadTree():
    
    try:
        data = {
            'file_name': file_name,
            'badflag':badflag,
            'max_depth': max_depth,
            'min_node_split':min_node_split,
            'min_leaf_size': min_leaf_size,
            'max_branches': max_branches,
            'criteria': criteria,
            'unique_var': unique_var,
            'monotonous': monotonous,
            'mapper': df_col.to_dict(orient='records'),
            'node_master': node_master.to_dict(orient='records')}
        json_str = json.dumps(data, indent=2)
        response = Response(
        json_str,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment;filename=data.json"})
        return response
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
 
if __name__ == "__main__":
    app.run(debug=True)
    