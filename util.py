import pandas as pd
import numpy as np
import math
import re
import warnings
warnings.filterwarnings('ignore')

def init_setup(input_path, event_flag, max_depth, min_node_split, min_leaf_size, max_branches):
    dataset = input_path
    X_col = dataset.drop([event_flag], axis=1).columns
    node_master_fn = pd.DataFrame(columns=['node_number','parent_node','level','branch','leaf_flag','count','events','non_events','event_rate','variable','node_rule', 'node_branch', 'lower_bound', 'upper_bound', 'node_label','list_values', 'missing'])
    count = dataset.shape[0]
    events = dataset[event_flag].sum()
    non_events = count - events
    event_rate = events / count
    row_add = pd.DataFrame([1, 0, 0, 1, 0, count, events, non_events, event_rate, 'Parent', 'Parent', 'Parent', 'Parent', 'Parent','','Parent', 0]).transpose()
    row_add.columns = ['node_number','parent_node','level','branch','leaf_flag','count','events','non_events','event_rate','variable','node_rule', 'node_branch', 'lower_bound', 'upper_bound','node_label','list_values', 'missing']
    node_master_fn = pd.concat([node_master_fn, row_add], axis=0)
    node_datasets_fn = dataset.copy()
    node_datasets_fn['node_number']=1
    return X_col, node_master_fn, node_datasets_fn

def classify_dtype(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numerical'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'datetime'
    else:
        return 'categorical'

def extract_dtype(variable, mapper):
    print(f"extract_dtype called with variable: '{variable}'")
    print(f"Mapper columns: {list(mapper['column_name'])}")
    
    filtered_mapper = mapper[mapper['column_name']==variable]
    if filtered_mapper.empty:
        print(f"ERROR: Variable '{variable}' not found in mapper!")
        print(f"Available variables in mapper: {list(mapper['column_name'])}")
        print(f"Mapper shape: {mapper.shape}")
        print(f"Mapper contents: {mapper}")
        raise ValueError(f"Variable '{variable}' not found in mapper")
    return filtered_mapper['datatype'].iloc[0]

def var_key_cat(X, y, var_name):
    data=pd.DataFrame(X[var_name])
    data['badflag']=y
    df_rollup_1 = data.groupby([var_name])['badflag'].agg(events="sum", all_records="count")
    df_rollup_1['event_rate']=df_rollup_1['events']/df_rollup_1['all_records']
    df_rollup_1 = df_rollup_1.sort_values('event_rate')
    df_rollup_1 = df_rollup_1.reset_index()
    df_rollup_1 = df_rollup_1.reset_index()
    df_rollup_1 = df_rollup_1.rename(columns = {'index':'var_key'})
    data = data.merge(df_rollup_1[['var_key', var_name]], on=var_name, how='left')
    data = data.sort_values('var_key').reset_index(drop=True)
    return data, df_rollup_1[['var_key', var_name]]

def add_bins(X, y, var_name):
    data=pd.DataFrame(X[var_name])
    data['badflag']=y
    if len(pd.unique(data[var_name])) > 20:
        df1=pd.qcut(data[var_name],q=20,duplicates='drop')
        df2=pd.DataFrame(df1)
        df2.columns=['bins']
        df2=df2.join(data[['badflag',var_name]])
        if data[var_name].isna().sum() > 0:
            df2["bins"] = df2["bins"].cat.add_categories("Missing")
            df2.loc[df2['bins'].isna(), 'bins'] = 'Missing'
            new_order = ["Missing"] + list(df2["bins"].cat.categories.drop("Missing"))
            df2["bins"] = df2["bins"].cat.reorder_categories(new_order, ordered=True)
        #df2['bins']=df2['bins'].astype('string')
        df3=df2.groupby('bins')['badflag'].agg(['count', 'sum'])
        df4=df3.reset_index()
        df4['variable']=var_name
        df4.columns=['bins','count','sum','variable']
        track = 1
    else:
        df3=data.groupby(var_name)['badflag'].agg(['count', 'sum'])
        df4=df3.reset_index()
        df4['variable']=var_name
        df4.columns=['bins','count','sum','variable']
        if data[var_name].isna().sum() > 0:
            df4['bins']=df4['bins'].astype(str)
            missing_count=data[var_name].isna().sum()
            missing_sum=data[data[var_name].isna()]['badflag'].sum()
            add_record = pd.DataFrame(['Missing', missing_count, missing_sum, var_name]).transpose()
            add_record.columns = ['bins','count','sum','variable']
            df4 = pd.concat([df4, add_record], axis=0)
            df4["bins"] = pd.Categorical(df4["bins"], categories=["Missing"] + sorted(set(df4["bins"]) - {"Missing"}), ordered=True)
            #df4["bins"] = df4["bins"].cat.reorder_categories(new_order, ordered=True)
            df4=df4.sort_values('bins').reset_index(drop=True)
        track = 2
    df4['non_events']=df4['count']-df4['sum']
    df4['events_pct']=df4['sum']/df4['sum'].sum()
    df4['nonevent_pct']=df4['non_events']/df4['non_events'].sum()
    df4['woe']=df4.apply(lambda x: 0 if ((x['events_pct']== 0) or (x['nonevent_pct'] == 0)) else np.log(x['nonevent_pct']/x['events_pct']), axis=1)
    df4['for_iv']=(df4['nonevent_pct']-df4['events_pct'])*df4['woe']
    iv_calc=df4['for_iv'].sum()
    return iv_calc, df4, track

def add_bins_regression(X, y, var_name):
    data=pd.DataFrame(X[var_name])
    data['badflag']=y
    if len(pd.unique(data[var_name])) > 20:
        df1=pd.qcut(data[var_name],q=20,duplicates='drop')
        df2=pd.DataFrame(df1)
        df2.columns=['bins']
        df2=df2.join(data[['badflag',var_name]])
        if data[var_name].isna().sum() > 0:
            df2["bins"] = df2["bins"].cat.add_categories("Missing")
            df2.loc[df2['bins'].isna(), 'bins'] = 'Missing'
            new_order = ["Missing"] + list(df2["bins"].cat.categories.drop("Missing"))
            df2["bins"] = df2["bins"].cat.reorder_categories(new_order, ordered=True)
        #df2['bins']=df2['bins'].astype('string')
        df3=df2.groupby('bins')['badflag'].agg(['count', 'sum'])
        df3['average'] = df3.apply(lambda x: 0 if x['count']==0 else x['sum']/x['count'], axis=1)
        df4=df3.reset_index()
        df4['variable']=var_name
        df4.columns=['bins','count','sum','average','variable']
        df5 = pd.merge(df2, df4[['bins','average']], on='bins', how='left')
        df5['se']=(df5['average']-df5['badflag'])*(df5['average']-df5['badflag'])
        rmse = np.sqrt(df5['se'].sum()/df5['se'].count())
        track = 1
    else:
        df2=data[[var_name,'badflag']]
        df2[var_name]=df2[var_name].fillna('Missing')
        df3=df2.groupby(var_name)['badflag'].agg(['count', 'sum'])
        df3['average'] = df3.apply(lambda x: 0 if x['count']==0 else x['sum']/x['count'], axis=1)
        df4=df3.reset_index()
        df4['variable']=var_name
        df4.columns=['bins','count','sum','average','variable']
        df5 = pd.merge(df2, df4[['bins','average']], left_on = var_name, right_on='bins', how='left')
        df5['se']=(df5['average']-df5['badflag'])*(df5['average']-df5['badflag'])
        rmse = np.sqrt(df5['se'].sum()/df5['se'].count())
        track = 2
    return rmse, df4, track,  df5[[var_name, 'badflag', 'bins']]

def calc_iv(node_number, node_datasets, X_col, event_flag, mapper):
    print(f"calc_iv called with node_number: {node_number}")
    print(f"node_datasets shape: {node_datasets.shape}")
    print(f"node_datasets columns: {list(node_datasets.columns)}")
    print(f"X_col: {list(X_col)}")
    print(f"event_flag: {event_flag}")
    print(f"mapper columns: {list(mapper['column_name'])}")
    
    df_iv=pd.DataFrame([0,0]).transpose()
    df_iv.columns=['variable','iv']
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    
    print(f"X shape: {X.shape}")
    print(f"X columns: {list(X.columns)}")
    print(f"y shape: {y.shape}")
    
    if X.empty:
        print("ERROR: X is empty - no features to process!")
        raise ValueError("No features available for node processing")
    
    if len(X.columns) == 0:
        print("ERROR: X has no columns - no features to process!")
        raise ValueError("No feature columns available for node processing")
    
    for i in X.columns:
        print(f"Processing variable: '{i}'")
        if extract_dtype(i, mapper) == 'numerical':
            iv_calc, df_bins, track = add_bins(X, y, str(i))
            df_new=pd.DataFrame([i,iv_calc]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)
        elif extract_dtype(i, mapper) == 'categorical':
            df_mod, mapping = var_key_cat(X, y, str(i))
            iv_calc, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
            df_new=pd.DataFrame([i,iv_calc]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)
    df_iv=df_iv.sort_values(['iv'], ascending=[False])
    df_iv=df_iv.reset_index(drop=True)
    
    # Remove the dummy row (0,0) that was added at the beginning
    df_iv = df_iv[df_iv['variable'] != 0].reset_index(drop=True)
    
    if df_iv.empty or len(df_iv) == 0:
        print("ERROR: df_iv is empty - no variables were processed!")
        raise ValueError("No variables available for IV calculation")
    
    top_iv = df_iv.iloc[0].variable
    if extract_dtype(top_iv, mapper) == 'numerical':
        iv_calc, df_bins, track = add_bins(X, y, top_iv)
        df_bins["list_values"] = df_bins["bins"]
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index(drop=True)
        var_type = 'numerical'
    elif extract_dtype(top_iv, mapper) == 'categorical':
        df_mod, mapping = var_key_cat(X, y, top_iv)
        mapping["var_key"] = pd.to_numeric(mapping["var_key"])
        iv_calc, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
        df_bins['variable'] = top_iv
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index(drop=True)
        def get_values(row):
            return mapping[(mapping["var_key"] > row["lb_clean"]) &(mapping["var_key"] <= row["ub_clean"])][top_iv].to_list()
        df_bins["list_values"] = df_bins.apply(get_values, axis=1)
        var_type = 'categorical'
    return top_iv, iv_calc, df_bins, track, var_type

def calc_iv_regression(node_number, node_datasets, X_col, event_flag, mapper):

    df_iv=pd.DataFrame([0,0]).transpose()
    df_iv.columns=['variable','rmse']
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    
    if X.empty:
        print("ERROR: X is empty - no features to process!")
        raise ValueError("No features available for node processing")
    
    if len(X.columns) == 0:
        print("ERROR: X has no columns - no features to process!")
        raise ValueError("No feature columns available for node processing")
    
    for i in X.columns:
        if extract_dtype(i, mapper) == 'numerical':
            rmse, df_bins, track, raw_data = add_bins_regression(X, y, str(i))
            df_new=pd.DataFrame([i,rmse]).transpose()
            df_new.columns=['variable','rmse']
            df_iv=pd.concat([df_new, df_iv], axis=0)
        elif extract_dtype(i, mapper) == 'categorical':
            df_mod, mapping = var_key_cat(X, y, str(i))
            rmse, df_bins, track, raw_data = add_bins_regression(df_mod, df_mod.badflag, 'var_key')
            df_new=pd.DataFrame([i,rmse]).transpose()
            df_new.columns=['variable','rmse']
            df_iv=pd.concat([df_new, df_iv], axis=0)
    df_iv=df_iv.sort_values(['rmse'], ascending=[True]) ## need lowest RMSE for regression
    df_iv=df_iv.reset_index(drop=True)
    # Remove the dummy row (0,0) that was added at the beginning
    df_iv = df_iv[df_iv['variable'] != 0].reset_index(drop=True)
    
    if df_iv.empty or len(df_iv) == 0:
        print("ERROR: df_iv is empty - no variables were processed!")
        raise ValueError("No variables available for IV calculation")
    
    top_iv = df_iv.iloc[0].variable
    if extract_dtype(top_iv, mapper) == 'numerical':
        rmse, df_bins, track, raw_data = add_bins_regression(X, y, top_iv)
        df_bins["list_values"] = df_bins["bins"]
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index(drop=True)
        var_type = 'numerical'
    elif extract_dtype(top_iv, mapper) == 'categorical':
        df_mod, mapping = var_key_cat(X, y, top_iv)
        mapping["var_key"] = pd.to_numeric(mapping["var_key"])
        rmse, df_bins, track, raw_data = add_bins_regression(df_mod, df_mod.badflag, 'var_key')
        df_bins['variable'] = top_iv
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index(drop=True)
        def get_values(row):
            return mapping[(mapping["var_key"] > row["lb_clean"]) &(mapping["var_key"] <= row["ub_clean"])][top_iv].to_list()
        df_bins["list_values"] = df_bins.apply(get_values, axis=1)
        var_type = 'categorical'
    return top_iv, rmse, df_bins, track, var_type, raw_data

def iterate(df):
    i=2
    df['scenario']=0
    scenario = 0
    df_out = pd.DataFrame(columns = df.columns)
    while i<=df.shape[0]:
        j=df.shape[0]-1
        counter = 0
        while counter == 0:
            if df['group'][j] >= i and j > 0:
                j = j-1
            elif j == 0:
                i=i+1
                counter = 1
            else:
                if df['group'][j]==df['group'][j-1] and j-1 >= 0:
                    df.loc[j, "group"] = i
                    df_2 = df
                    df_2['scenario'] = scenario
                    df_out = pd.concat([df_out, df_2], axis = 0)
                    scenario = scenario + 1
                    counter = 1
                else:
                    i=i+1
                    counter = 1
    return df_out

def check_conditions(df, min_leaf_size, max_branches):
    df_out = pd.DataFrame(columns = df.columns)
    if df.shape[0]==0:
        return df_out
    else:
        for i in range(df['scenario'].max()+1):
            df_small = df[df.scenario == i]
            if df_small['group'].max() <= max_branches:
                branch_valid = 1
            else:
                branch_valid = 0
            df_count = df_small.groupby('group').agg({'count':"sum"})
            if df_count['count'].min() >= min_leaf_size:
                leaf_valid = 1
            else:
                leaf_valid = 0
            if branch_valid == 1 and leaf_valid == 1:
                df_out = pd.concat([df_out, df_small], axis = 0)
        return df_out

def calc_entropy(df):
    df_rollup_1 = df.groupby(['scenario', 'group']).agg({'sum':"sum"})
    df_rollup_1.columns = ['events']
    df_rollup_2 = df.groupby(['scenario', 'group']).agg({'count':"sum"})
    df_rollup_2.columns = ['all_count']
    df_rollup_3 = df_rollup_1.join(df_rollup_2)
    df_rollup_3 = df_rollup_3.reset_index()
    df_rollup_3['p0']=df_rollup_3['events']/df_rollup_3['all_count']
    df_rollup_3['p1']=1-df_rollup_3['p0']
    df_rollup_3['l_p0']=df_rollup_3['p0'].apply(lambda x: 0 if x <= 0 else -1*x*math.log2(x))
    df_rollup_3['l_p1']=df_rollup_3['p1'].apply(lambda x: 0 if x <= 0 else -1*x*math.log2(x))
    df_rollup_3['entr']=df_rollup_3['l_p0']+df_rollup_3['l_p1']
    df_scen_rollup = df.groupby(['scenario']).agg({'count':"sum"})
    df_scen_rollup.columns = ['overall_count']
    df_rollup_4=df_rollup_3.join(df_scen_rollup)
    df_rollup_4['weights']=df_rollup_4['all_count']/df_rollup_4['overall_count']
    df_rollup_4['w_entr']=df_rollup_4['entr']*df_rollup_4['weights']
    df_summ = df_rollup_4.groupby(['scenario']).agg({'w_entr':"sum"})
    min_entropy = df_summ['w_entr'].min()
    best_scenario = df_summ[df_summ['w_entr'] == min_entropy].index[0]
    df_best = df[df['scenario'] == best_scenario]
    return df_best

def calc_rmse(df, raw_data):
    df_rollup_1 = df.groupby(['scenario', 'group']).agg({'sum':"sum"})
    df_rollup_1.columns = ['events']
    df_rollup_2 = df.groupby(['scenario', 'group']).agg({'count':"sum"})
    df_rollup_2.columns = ['all_count']
    df_rollup_3 = df_rollup_1.join(df_rollup_2)
    df_rollup_3 = df_rollup_3.reset_index()
    df_rollup_3['average'] = df_rollup_3['events'] / df_rollup_3['all_count']
    scen_rmse = pd.DataFrame(columns=['scenario','rmse'])
    for i in df_rollup_3.scenario.unique():
        scen_table = df_rollup_3[df_rollup_3.scenario == i]
        bin_table = df[df.scenario == i]
        raw_data_2 = pd.merge(raw_data, bin_table[['bins','group']], on='bins', how='left')
        raw_data_3 = pd.merge(raw_data_2, scen_table[['group','average']], on='group', how='left')
        raw_data_3['se']=(raw_data_3['average']-raw_data_3['badflag'])*(raw_data_3['average']-raw_data_3['badflag'])
        rmse = np.sqrt(raw_data_3['se'].sum()/raw_data_3['se'].count())
        row_add = pd.DataFrame([i, rmse]).transpose()
        row_add.columns = ['scenario','rmse']
        scen_rmse = pd.concat([scen_rmse, row_add], axis=0)
    scen_rmse=scen_rmse.sort_values(['rmse'], ascending=[True])
    scen_rmse=scen_rmse.reset_index(drop=True)
    best_scenario = scen_rmse.iloc[0].scenario
    df_best = df[df['scenario'] == best_scenario]
    return df_best

def rollup_best(df, track, var_type):
    df_missing = df[df.bins=='Missing'].reset_index(drop = True)
    df = df[df.bins!='Missing']
    if track == 1:
        df[['lower_bound','upper_bound']] = df['bins'].astype(str).str.split(',',expand=True)
        df['lb_clean']=pd.to_numeric(df['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        df['ub_clean']=pd.to_numeric(df['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
    else:
        df['lower_bound'] = pd.to_numeric(df['bins'])-1
        df['upper_bound'] = pd.to_numeric(df['bins'])
        df['lb_clean'] = df['lower_bound']
        df['ub_clean'] = df['upper_bound']
    if var_type == 'categorical':
        df_clean=df.groupby(['group']).agg({'lb_clean':"min", 'ub_clean':"max", 'sum':"sum", 'count':"sum", 'list_values':"sum"})
    else:
        df_clean=df.groupby(['group']).agg({'lb_clean':"min", 'ub_clean':"max", 'sum':"sum", 'count':"sum"})
        df_clean["list_values"] = "(" + df_clean["lb_clean"].astype(int).astype(str) + ", " + df_clean["ub_clean"].astype(int).astype(str) + "]"
    if df.empty:
        print("ERROR: df is empty in coarse_classing!")
        raise ValueError("DataFrame is empty in coarse_classing")
    df_clean['variable']=df['variable'].iloc[0]
    df_clean['bad_rate']=df_clean['sum']/df_clean['count']
    df_clean_2=df_clean[['variable','lb_clean','ub_clean','sum','count','bad_rate', 'list_values']]
    df_clean_2.columns = ['variable','lower_bound','upper_bound', 'events', 'all_records', 'event_rate', 'list_values']
    df_clean_2['missing']=0
    if df_missing.empty:
        return df_clean_2
    else:
        group_label = df_missing['group'].iloc[0]
        if group_label in df_clean_2.index:
            df_clean_2.loc[group_label, 'missing'] = 1
            df_clean_2.loc[group_label, 'events'] = df_clean_2.loc[group_label, 'events'] + df_missing.loc[0, 'sum']
            df_clean_2.loc[group_label, 'all_records'] = df_clean_2.loc[group_label, 'all_records'] +  + df_missing.loc[0, 'count']
            df_clean_2['event_rate'] = df_clean_2['events'] / df_clean_2['all_records']
        else:
            missing_variable = df_missing['variable'].iloc[0]
            missing_lb = np.nan
            missing_ub = np.nan
            missing_events = df_missing['sum'].iloc[0]
            missing_all_records = df_missing['count'].iloc[0]
            missing_event_rate = df_missing['average'].iloc[0]
            missing_list_values = 'Missing'
            df_missing_for_append = pd.DataFrame([missing_variable, missing_lb, missing_ub, missing_events, missing_all_records, missing_event_rate, missing_list_values, 1]).transpose()
            df_missing_for_append.columns = ['variable','lower_bound','upper_bound', 'events', 'all_records', 'event_rate', 'list_values', 'missing']
            df_clean_2 = pd.concat([df_clean_2, df_missing_for_append])
        df_clean_2 = df_clean_2.reset_index(drop=True)
        return df_clean_2

def coarse_classing(df, min_leaf_size, max_branches, track, var_type):
    if df.empty:
        print("ERROR: df is empty after removing missing values!")
        raise ValueError("No valid bins available for coarse classing")
    
    df['group'] = 1
    df_scenarios = iterate(df)
    df_valid = check_conditions(df_scenarios, min_leaf_size, max_branches)
    if df_valid.shape[0] == 0:
        df['event_rate'] = df['sum'] / df['count']
        df['missing'] = df.apply(lambda x: 1 if x['bins']=='Missing' else 0, axis=1)
        df = df.rename(columns={"sum": "events", "count": "all_records"})
        return df[['variable','lower_bound','upper_bound','events','all_records','event_rate','missing','list_values']], 1, var_type
    else:
        df_best = calc_entropy(df_valid)
        df_clean = rollup_best(df_best, track, var_type)
        return df_clean[['variable','lower_bound','upper_bound','events','all_records','event_rate','missing','list_values']], 0, var_type

def coarse_classing_regression(df, min_leaf_size, max_branches, track, var_type, raw_data):
    if df.empty:
        print("ERROR: df is empty after removing missing values!")
        raise ValueError("No valid bins available for coarse classing")
        
    df['group'] = 1
    df_scenarios = iterate(df)
    
    df_valid = check_conditions(df_scenarios, min_leaf_size, max_branches)
    
    if df_valid.shape[0] == 0:
        df['event_rate'] = df['sum'] / df['count']
        df['missing'] = df.apply(lambda x: 1 if x['bins']=='Missing' else 0, axis=1)
        df = df.rename(columns={"sum": "events", "count": "all_records"})
        return df[['variable','lower_bound','upper_bound','events','all_records','event_rate','missing','list_values']], 1, var_type
    else:
        df_best, raw_data = calc_rmse(df_valid, raw_data)
        df_clean, raw_data = rollup_best(df_best, track, var_type, raw_data)
        return df_clean[['variable','lower_bound','upper_bound','events','all_records','event_rate','missing','list_values']], 0, var_type

def add_nodes(df, parent_node, max_depth, min_node_split, error, node_master, node_datasets, event_flag, mapper):
    node_master_fn = node_master.copy()
    node_datasets_fn = node_datasets.copy()
    #print(f'error:{error}')
    #print(node_master_fn)
    if error == 1:
        node_master_fn.loc[node_master_fn['node_number'] == parent_node, 'leaf_flag'] = 1
        #print(node_master_fn)
        return node_master_fn, node_datasets_fn
    else:
        node_sp = node_datasets_fn[node_datasets_fn['node_number']==parent_node]
        for i in range(df.shape[0]):
            group = i+1
            lower = df['lower_bound'].iloc[i]
            upper = df['upper_bound'].iloc[i]
            variable = df['variable'].iloc[i]
            missing = df['missing'].iloc[i]
            list_values = df['list_values'].iloc[i]
            var_type = extract_dtype(variable, mapper)
            if i == 0:
                if var_type == 'numerical':
                    if pd.isna(lower) or pd.isna(upper):
                        df_split = node_sp[node_sp[variable].isna()]
                        node_rule = "".join([variable, " == np.nan"])
                        node_branch = "".join([" == np.nan"])
                    elif missing == 1:
                        df_split = node_sp[(node_sp[variable] <= upper) | (node_sp[variable].isna())]
                        node_rule = "".join(["(", variable, " <= ", str(upper), ") | (", variable, " == np.nan)"])
                        node_branch = "".join([" <= ", str(upper), " | == np.nan"])
                    else:
                        df_split = node_sp[node_sp[variable] <= upper]
                        node_rule = "".join([variable, " <= ", str(upper)])
                        node_branch = "".join([" <= ", str(upper)])
                elif var_type == 'categorical':
                    if pd.isna(lower) or pd.isna(upper):
                        df_split = node_sp[node_sp[variable].isna()]
                        node_rule = "".join([variable, " == np.nan"])
                        node_branch = "".join([" == np.nan"])
                    elif missing == 1:
                        df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                        node_rule = "".join(["(", variable, " in ", str(list_values), ") | (", variable, " == np.nan)"])
                        node_branch = "".join([" in ", str(list_values), " | == np.nan"])
                    else:
                        df_split = node_sp[node_sp[variable].isin(list_values)]
                        node_rule = "".join([variable, " in ", str(list_values)])
                        node_branch = "".join([" in ", str(list_values)])
                lower_bound = lower
                upper_bound = upper
            elif i == df.shape[0]:
                if var_type == 'numerical':
                    if pd.isna(lower) or pd.isna(upper):
                        df_split = node_sp[node_sp[variable].isna()]
                        node_rule = "".join([variable, " == np.nan"])
                        node_branch = "".join([" == np.nan"])
                    elif missing == 1:
                        df_split = node_sp[(node_sp[variable] > lower) | (node_sp[variable].isna())]
                        node_rule = "".join(["(", variable, " > ", str(lower), ") | (", variable, " == np.nan)"])
                        node_branch = "".join([" > ", str(lower), " | == np.nan"])
                    else:
                        df_split = node_sp[node_sp[variable] > lower]
                        node_rule = "".join([variable, " > ", str(lower)])
                        node_branch = "".join([" > ", str(lower)])
                elif var_type == 'categorical':
                    if pd.isna(lower) or pd.isna(upper):
                        df_split = node_sp[node_sp[variable].isna()]
                        node_rule = "".join([variable, " == np.nan"])
                        node_branch = "".join([" == np.nan"])
                    elif missing == 1:
                        df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                        node_rule = "".join(["(", variable, " in ", str(list_values), ") | (", variable, " == np.nan)"])
                        node_branch = "".join([" in ", str(list_values), " | == np.nan"])
                    else:
                        df_split = node_sp[node_sp[variable].isin(list_values)]
                        node_rule = "".join([variable, " in ", str(list_values)])
                        node_branch = "".join([" in ", str(list_values)])
                lower_bound = lower
                upper_bound = upper
            else:
                if var_type == 'numerical':
                    if pd.isna(lower) or pd.isna(upper):
                        df_split = node_sp[node_sp[variable].isna()]
                        node_rule = "".join([variable, " == np.nan"])
                        node_branch = "".join([" == np.nan"])
                    elif missing == 1:
                        df_split = node_sp[((node_sp[variable] > lower) & (node_sp[variable] <= upper)) | (node_sp[variable].isna())]
                        node_rule = "".join(["(", variable, " > ", str(lower), " and ", variable, " <= ", str(upper), ") | (", variable, " == np.nan)"])
                        node_branch = "".join(["(", str(lower), ", ", str(upper), "]", " | == np.nan"])
                    else:
                        df_split = node_sp[(node_sp[variable] > lower) & (node_sp[variable] <= upper)]
                        node_rule = "".join([variable, " > ", str(lower), " and ", variable, " <= ", str(upper)])
                        node_branch = "".join(["(", str(lower), ", ", str(upper), "]"])
                elif var_type == 'categorical':
                    if pd.isna(lower) or pd.isna(upper):
                        df_split = node_sp[node_sp[variable].isna()]
                        node_rule = "".join([variable, " == np.nan"])
                        node_branch = "".join([" == np.nan"])
                    elif missing == 1:
                        df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                        node_rule = "".join(["(", variable, " in ", str(list_values), ") | (", variable, " == np.nan)"])
                        node_branch = "".join([" in ", str(list_values), " | == np.nan"])
                    else:
                        df_split = node_sp[node_sp[variable].isin(list_values)]
                        node_rule = "".join([variable, " in ", str(list_values)])
                        node_branch = "".join([" in ", str(list_values)])
                lower_bound = lower
                upper_bound = upper
            last_node_number = node_master_fn['node_number'].max()
            new_node_number = last_node_number + 1
            parent_level = node_master_fn.loc[node_master_fn['node_number'] == parent_node, 'level'].values[0]
            child_level = parent_level + 1
            parent_branch = node_master_fn.loc[node_master_fn['node_number'] == parent_node, 'branch'].values[0]
            # Fix floating-point precision issues in branch calculation
            child_branch = round(parent_branch + (group/pow(10, child_level)), child_level)
            count = df_split.shape[0]
            events = df_split[event_flag].sum()
            non_events = count - events
            event_rate = events / count
            if child_level < max_depth and count >= min_node_split:
                leaf_flag = 0
            else:
                leaf_flag = 1
            for_node_append = pd.DataFrame([new_node_number, parent_node, child_level, child_branch, leaf_flag, count, events, non_events, event_rate, variable, node_rule, node_branch, lower_bound, upper_bound, list_values, missing]).transpose()
            for_node_append.columns = ['node_number','parent_node','level','branch','leaf_flag','count','events','non_events','event_rate', 'variable', 'node_rule', 'node_branch', 'lower_bound', 'upper_bound', 'list_values', 'missing']
            node_master_fn = pd.concat([node_master_fn, for_node_append])
            df_split.loc[:, 'node_number'] = new_node_number
            node_datasets_fn = pd.concat([node_datasets_fn, df_split])
        return node_master_fn, node_datasets_fn

def split_node(node_number, min_leaf_size, max_branches, max_depth, min_node_split, X_col, node_master, node_datasets, event_flag, mapper, regression_flag):
    if regression_flag == 0:
        top_iv, iv_calc, df_iv, track, var_type = calc_iv(node_number, node_datasets, X_col, event_flag, mapper)
        df_out, error, var_type = coarse_classing(df_iv, min_leaf_size, max_branches, track, var_type)
        node_master_fn, node_datasets_fn = add_nodes(df_out, node_number, max_depth, min_node_split, error, node_master, node_datasets, event_flag, mapper)
        return node_master_fn, node_datasets_fn
    else:
        top_iv, iv_calc, df_iv, track, var_type, raw_data = calc_iv_regression(node_number, node_datasets, X_col, event_flag, mapper)
        df_out, error, var_type = coarse_classing_regression(df_iv, min_leaf_size, max_branches, track, var_type, raw_data)
        node_master_fn, node_datasets_fn = add_nodes(df_out, node_number, max_depth, min_node_split, error, node_master, node_datasets, event_flag, mapper)
        return node_master_fn, node_datasets_fn

def next_node_to_split(node_master, max_depth):
    node_to_split = 0
    parent_nodes = set(node_master['parent_node'].values)
    for idx, row in node_master.iterrows():
        if row['leaf_flag'] == 0 and row['level'] < max_depth and row['node_number'] not in parent_nodes:
            node_to_split = row['node_number']
            break
    return node_to_split    

def build_tree(input_path, event_flag, max_depth, min_node_split, min_leaf_size, max_branches, mapper, regression_flag):
    X_col, node_master, node_datasets = init_setup(input_path, event_flag, max_depth, min_node_split, min_leaf_size, max_branches)
    while next_node_to_split(node_master, max_depth) != 0:
        node_number = next_node_to_split(node_master, max_depth)
        node_master, node_datasets = split_node(node_number, min_leaf_size, max_branches, max_depth, min_node_split, X_col, node_master, node_datasets, event_flag, mapper, regression_flag)
    return node_master, node_datasets
	
def del_subtree(df, df_node, node_number):
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    df_copy['branch_str'] = df_copy['branch'].apply(lambda x: format(x, 'g'))
    filtered_df = df_copy[df_copy['node_number']==node_number]
    if filtered_df.empty:
        print(f"ERROR: Node {node_number} not found in del_subtree!")
        print(f"Available nodes: {list(df_copy['node_number'])}")
        raise ValueError(f"Node {node_number} not found in del_subtree")
    str_del = filtered_df['branch_str'].iloc[0]
    df_copy['mark']=df_copy.apply(lambda x: 1 if (x.branch_str.startswith(str_del)) and (x.branch_str != str_del) else 0, axis=1)
    keep_list = df_copy[df_copy['mark'].isin([0])].node_number
    df_node_return = df_node[df_node['node_number'].isin(keep_list)].reset_index(drop=True)
    df_filtered = df_copy[df_copy['mark'].isin([0])].reset_index(drop=True)
    return df_filtered.drop(['branch_str','mark'], axis=1), df_node_return

def next_node_to_split_node(df, max_depth, node_number): ## df is node_master
    df['branch_str'] = df['branch'].apply(lambda x: format(x, 'g')) ## convert float to string
    filtered_df = df[df['node_number']==node_number]
    if filtered_df.empty:
        print(f"ERROR: Node {node_number} not found in next_node_to_split_node!")
        print(f"Available nodes: {list(df['node_number'])}")
        raise ValueError(f"Node {node_number} not found in next_node_to_split_node")
    str_del = filtered_df['branch_str'].iloc[0] ## identify the branch of the parent node to be split
    df['mark']=df.apply(lambda x: 1 if x.branch_str.startswith(str_del) else 0, axis=1) ## mark node_master for the correct branch only
    df_filtered = df[df['mark'].isin([1])] ## take only the marked nodes
    node_to_split = 0
    parent_nodes = set(df_filtered['parent_node'].values)
    for idx, row in df_filtered.iterrows():
        if row['leaf_flag'] == 0 and row['level'] < max_depth and row['node_number'] not in parent_nodes:
            node_to_split = row['node_number']
            break
    return node_to_split  

def build_tree_node(input_path, event_flag, max_depth, min_node_split, min_leaf_size, max_branches, node_to_split, X_col, node_master, node_datasets, mapper, regression_flag):
    iteration_count = 0
    while next_node_to_split_node(node_master, max_depth, node_to_split) != 0:
        iteration_count += 1
        node_number = next_node_to_split_node(node_master, max_depth, node_to_split)
        node_master, node_datasets = split_node(node_number, min_leaf_size, max_branches, max_depth, min_node_split, X_col, node_master, node_datasets, event_flag, mapper, regression_flag)
        if iteration_count > 10:  # Safety break to prevent infinite loops
            print("WARNING: Too many iterations, breaking loop")
            break
    
    print(f"build_tree_node completed after {iteration_count} iterations")
    print(f"Final node_master shape: {node_master.shape}")
    return node_master, node_datasets

def find_var(node_number, node_datasets, X_col, event_flag, mapper):
    df_iv=pd.DataFrame([0,0]).transpose()
    df_iv.columns=['variable','iv']
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    for i in X.columns:
        if extract_dtype(i, mapper) == 'numerical':
            iv_calc, df_bins, track = add_bins(X, y, str(i))
            df_new=pd.DataFrame([i,iv_calc]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)
        elif extract_dtype(i, mapper) == 'categorical':
            df_mod, mapping = var_key_cat(X, y, str(i))
            iv_calc, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
            df_new=pd.DataFrame([i,iv_calc]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)    
    df_iv=df_iv.sort_values(['iv'], ascending=[False])
    df_iv=df_iv.reset_index(drop=True)
    return df_iv[['variable','iv']]

def find_var_regression(node_number, node_datasets, X_col, event_flag, mapper):
    df_iv=pd.DataFrame([0,0]).transpose()
    df_iv.columns=['variable','rmse']
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    for i in X.columns:
        if extract_dtype(i, mapper) == 'numerical':
            rmse, df_bins, track, raw_data = add_bins_regression(X, y, str(i))
            df_new=pd.DataFrame([i,rmse]).transpose()
            df_new.columns=['variable','rmse']
            df_iv=pd.concat([df_new, df_iv], axis=0)
        elif extract_dtype(i, mapper) == 'categorical':
            df_mod, mapping = var_key_cat(X, y, str(i))
            rmse, df_bins, track, raw_data = add_bins_regression(df_mod, df_mod.badflag, 'var_key')
            df_new=pd.DataFrame([i,rmse]).transpose()
            df_new.columns=['variable','rmse']
            df_iv=pd.concat([df_new, df_iv], axis=0)    
    df_iv=df_iv.sort_values(['rmse'], ascending=[True])
    df_iv=df_iv.reset_index(drop=True)
    return df_iv[df_iv['rmse']!=0][['variable','rmse']]

def best_split(node_number, node_datasets, X_col, event_flag, variable_name, min_leaf_size, max_branches, mapper):
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    if extract_dtype(variable_name, mapper) == 'numerical':
        iv_calc, df_bins, track = add_bins(X, y, variable_name)
        df_bins["list_values"] = df_bins["bins"]
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index()
        var_type = 'numerical'
    elif extract_dtype(variable_name, mapper) == 'categorical':
        df_mod, mapping = var_key_cat(X, y, variable_name)
        mapping["var_key"] = pd.to_numeric(mapping["var_key"])
        iv_calc, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
        df_bins['variable'] = variable_name
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index(drop=True)
        def get_values(row):
            return mapping[(mapping["var_key"] > row["lb_clean"]) &(mapping["var_key"] <= row["ub_clean"])][variable_name].to_list()
        df_bins["list_values"] = df_bins.apply(get_values, axis=1)
        var_type = 'categorical'
    df_out, error, var_type = coarse_classing(df_bins, min_leaf_size, max_branches, track, var_type)
    return df_out

def best_split_regression(node_number, node_datasets, X_col, event_flag, variable_name, min_leaf_size, max_branches, mapper):
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    if extract_dtype(variable_name, mapper) == 'numerical':
        rmse, df_bins, track, raw_data = add_bins_regression(X, y, variable_name)
        df_bins["list_values"] = df_bins["bins"]
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index()
        var_type = 'numerical'
    elif extract_dtype(variable_name, mapper) == 'categorical':
        df_mod, mapping = var_key_cat(X, y, variable_name)
        mapping["var_key"] = pd.to_numeric(mapping["var_key"])
        rmse, df_bins, track, raw_data = add_bins_regression(df_mod, df_mod.badflag, 'var_key')
        df_bins['variable'] = variable_name
        if track == 1:
            df_bins[['lower_bound','upper_bound']] = df_bins['bins'].astype(str).str.split(',',expand=True)
            df_bins[['lower_bound','upper_bound']] = df_bins[['lower_bound','upper_bound']].fillna("")
            df_bins['lb_clean']=pd.to_numeric(df_bins['lower_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
            df_bins['ub_clean']=pd.to_numeric(df_bins['upper_bound'].apply(lambda x: re.sub(r'[^-.0-9\s]','',x)))
        else:
            df_bins['lower_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")-1
            df_bins['upper_bound'] = pd.to_numeric(df_bins['bins'], errors="coerce")
            df_bins['lb_clean'] = df_bins['lower_bound']
            df_bins['ub_clean'] = df_bins['upper_bound']
        df_bins=df_bins.sort_values(['lb_clean'], ascending=[True])
        df_bins=df_bins.reset_index(drop=True)
        def get_values(row):
            return mapping[(mapping["var_key"] > row["lb_clean"]) &(mapping["var_key"] <= row["ub_clean"])][variable_name].to_list()
        df_bins["list_values"] = df_bins.apply(get_values, axis=1)
        var_type = 'categorical'
    df_out, error, var_type = coarse_classing_regression(df_bins, min_leaf_size, max_branches, track, var_type, raw_data)
    return df_out

def eval_split(node_number, node_datasets, event_flag, df, mapper):
    node_sp = node_datasets[node_datasets["node_number"] == node_number]
    df_summ=pd.DataFrame(columns=['all_records', 'event_rate', 'events', 'lower_bound', 'upper_bound', 'variable'])

    for i in range(df.shape[0]):
        lower = df['lower_bound'].iloc[i]
        upper = df['upper_bound'].iloc[i]
        variable = df['variable'].iloc[i]
        missing = df['missing'].iloc[i]
        list_values = df['list_values'].iloc[i]
        var_type = extract_dtype(variable, mapper)
        if i == 0:
            if var_type == 'numerical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] <= upper) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] <= upper]
            elif var_type == 'categorical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        elif i == df.shape[0]:
            if var_type == 'numerical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] > lower) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] > lower]
            elif var_type == 'categorical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        else:
            if var_type == 'numerical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[((node_sp[variable] > lower) & (node_sp[variable] <= upper)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[(node_sp[variable] > lower) & (node_sp[variable] <= upper)]
            elif var_type == 'categorical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        all_records = df_split.shape[0]
        events = df_split[event_flag].sum()
        if all_records == 0:
            event_rate = 0
        else:
            event_rate = events / all_records
        df_summ_append = pd.DataFrame([all_records, event_rate, events, lower, upper, variable, list_values, missing]).transpose()
        df_summ_append.columns=['all_records', 'event_rate', 'events', 'lower_bound', 'upper_bound', 'variable', 'list_values', 'missing']
        df_summ=pd.concat([df_summ, df_summ_append], axis=0)
        
    return df_summ

def copy_node(from_node, to_node, node_master, node_datasets, max_depth, min_node_split, target, mapper):
    node_master['branch_str'] = node_master['branch'].apply(lambda x: format(x, 'g'))
    
    # Check if from_node exists
    from_node_exists = node_master[node_master.node_number==from_node].shape[0] > 0
    if not from_node_exists:
        print(f"Error: from_node {from_node} does not exist")
        return node_master, node_datasets
    
    # Check if to_node exists
    to_node_exists = node_master[node_master.node_number==to_node].shape[0] > 0
    if not to_node_exists:
        print(f"Error: to_node {to_node} does not exist")
        return node_master, node_datasets
    
    if node_master[node_master.parent_node==from_node].shape[0] == 0:
        return node_master, node_datasets
    else:
        df = node_master[node_master.parent_node == from_node][['variable','lower_bound', 'upper_bound', 'list_values', 'missing']]
        node_master, node_datasets = add_nodes(df, to_node, max_depth, min_node_split, 0, node_master, node_datasets, target, mapper)
        node_master['branch_str'] = node_master['branch'].apply(lambda x: format(x, 'g'))
        children = node_master[node_master.parent_node==from_node].node_number.tolist()
        
        # Get branch strings with error checking
        from_branch_df = node_master[node_master.node_number==from_node][['branch_str']]
        to_branch_df = node_master[node_master.node_number==to_node][['branch_str']]
        
        if from_branch_df.shape[0] == 0:
            print(f"Error: Could not find branch_str for from_node {from_node}")
            return node_master, node_datasets
            
        if to_branch_df.shape[0] == 0:
            print(f"Error: Could not find branch_str for to_node {to_node}")
            return node_master, node_datasets
            
        from_branch = from_branch_df.iloc[0, 0]
        to_branch = to_branch_df.iloc[0, 0]
        
        for i in children:
            if node_master[node_master.parent_node==i].shape[0] > 0:
                child_branch_df = node_master[node_master.node_number==i][['branch_str']]
                if child_branch_df.shape[0] == 0:
                    print(f"Error: Could not find branch_str for child node {i}")
                    continue
                    
                child_branch = child_branch_df.iloc[0, 0]
                new_child_branch = child_branch.replace(from_branch, to_branch)
                print("new_child_branch", new_child_branch)
                
                new_node_df = node_master[node_master.branch_str==new_child_branch][['node_number']]
                if new_node_df.shape[0] == 0:
                    print(f"Error: Could not find new_node for branch_str {new_child_branch}")
                    continue
                    
                new_node = new_node_df.iloc[0, 0]
                print(i, new_node)
                node_master, node_datasets = copy_node(i, new_node, node_master, node_datasets, max_depth, min_node_split, target, mapper)
        return node_master, node_datasets
    
def copy_node_wrapper(from_node, to_node, node_master, node_datasets, max_depth, min_node_split, target, mapper):
    node_master, node_datasets = copy_node(from_node, to_node, node_master, node_datasets, max_depth, min_node_split, target, mapper)
    
    # Update leaf_flag for the target node - if it now has children, it's no longer a leaf
    has_children = node_master[node_master.parent_node == to_node].shape[0] > 0
    if has_children:
        node_master.loc[node_master['node_number'] == to_node, 'leaf_flag'] = 0
    
    # Clean up any temporary columns that might have been created
    columns_to_drop = ['branch_str', 'mark']
    existing_columns_to_drop = [col for col in columns_to_drop if col in node_master.columns]
    
    if existing_columns_to_drop:
        node_master = node_master.drop(existing_columns_to_drop, axis=1)
    
    return node_master, node_datasets

def add_label(node_number, label, node_master):
    node_master.loc[node_master['node_number'] == node_number, 'node_label'] = label
    return node_master

def add_node_init(df, parent_node, df_for_tree, node_datasets_new, mapper):
    print(f"add_node_init called with parent_node: {parent_node}")
    print(f"node_datasets_new columns: {list(node_datasets_new.columns)}")
    print(f"node_datasets_new shape: {node_datasets_new.shape}")
    
    node_sp = node_datasets_new[node_datasets_new['node_number']==parent_node]
    print(f"node_sp columns: {list(node_sp.columns)}")
    print(f"node_sp shape: {node_sp.shape}")
    print(f"df columns: {list(df.columns)}")
    print(f"df shape: {df.shape}")
    
    for i in range(df.shape[0]):
        lower = df['lower_bound'].iloc[i]
        upper = df['upper_bound'].iloc[i]
        variable = df['variable'].iloc[i]
        missing = df['missing'].iloc[i]
        list_values = df['list_values'].iloc[i]
        var_type = extract_dtype(variable, mapper)
        node_number = df['node_number'].iloc[i]
        
        print(f"Processing row {i}: variable='{variable}', var_type='{var_type}', node_number={node_number}")
        print(f"node_sp columns: {list(node_sp.columns)}")
        
        if variable not in node_sp.columns:
            print(f"ERROR: Variable '{variable}' not found in node_sp columns: {list(node_sp.columns)}")
            raise ValueError(f"Variable '{variable}' not found in node_sp columns")
        if i == 0:
            if var_type == 'numerical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] <= upper) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] <= upper]
            elif var_type == 'categorical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        elif i == df.shape[0]:
            if var_type == 'numerical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] > lower) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] > lower]
            elif var_type == 'categorical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        else:
            if var_type == 'numerical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[((node_sp[variable] > lower) & (node_sp[variable] <= upper)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[(node_sp[variable] > lower) & (node_sp[variable] <= upper)]
            elif var_type == 'categorical':
                if pd.isna(lower) or pd.isna(upper):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        df_split['node_number'] = node_number
        node_datasets_new = pd.concat([node_datasets_new, df_split])
    return node_datasets_new

def create_node_datasets(df_for_tree, mapper, node_master):
    try:
        print(f"create_node_datasets called with:")
        print(f"  df_for_tree columns: {list(df_for_tree.columns)}")
        print(f"  df_for_tree shape: {df_for_tree.shape}")
        print(f"  node_master columns: {list(node_master.columns)}")
        print(f"  node_master shape: {node_master.shape}")
        print(f"  mapper columns: {list(mapper.columns)}")
        
        node_datasets = df_for_tree.copy()
        node_datasets['node_number']=1
        print(f"Created node_datasets with node_number column")
        
        # Check if required columns exist in node_master
        required_cols = ['node_number','variable','lower_bound','upper_bound','list_values','missing']
        missing_cols = [col for col in required_cols if col not in node_master.columns]
        if missing_cols:
            print(f"Warning: Missing columns in node_master: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                if col == 'lower_bound' or col == 'upper_bound':
                    node_master[col] = 0
                elif col == 'missing':
                    node_master[col] = 0
                elif col == 'list_values':
                    node_master[col] = ''
                else:
                    node_master[col] = ''
        
        print(f"Processing parent nodes...")
        parent_nodes = node_master['parent_node'].unique().tolist()
        print(f"Parent nodes to process: {parent_nodes}")
        
        for i in parent_nodes:
            if i > 0:
                try:
                    print(f"Processing parent node {i}...")
                    add_node_df = node_master[node_master['parent_node']==i][['node_number','variable','lower_bound','upper_bound','list_values','missing']]
                    print(f"add_node_df shape: {add_node_df.shape}")
                    print(f"add_node_df columns: {list(add_node_df.columns)}")
                    node_datasets = add_node_init(add_node_df, i, df_for_tree, node_datasets, mapper)
                    print(f"Successfully processed parent node {i}")
                except Exception as e:
                    print(f"Error processing parent node {i}: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    continue
        print(f"create_node_datasets completed successfully")
        return node_master, node_datasets
    except Exception as e:
        print(f"Error in create_node_datasets: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Return minimal datasets
        node_datasets = df_for_tree.copy()
        node_datasets['node_number'] = 1
        return node_master, node_datasets