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
    return mapper[mapper['column_name']==variable]['datatype'].iloc[0]

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
    df4['woe']=df4.apply(lambda x: 0 if ((x['events_pct']== 0)|(x['nonevent_pct'] == 0)) else np.log(x['nonevent_pct']/x['events_pct']), axis=1)
    df4['for_iv']=(df4['nonevent_pct']-df4['events_pct'])*df4['woe']
    iv_calc=df4['for_iv'].sum()
    return iv_calc, df4, track

def calc_iv(node_number, node_datasets, X_col, event_flag, mapper):
    df_iv=pd.DataFrame([0,0]).transpose()
    df_iv.columns=['variable','iv']
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    for i in X.columns:
        if extract_dtype(i, mapper) == 'numerical':
            iv, df_bins, track = add_bins(X, y, str(i))
            df_new=pd.DataFrame([i,iv]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)
        elif extract_dtype(i, mapper) == 'categorical':
            df_mod, mapping = var_key_cat(X, y, str(i))
            iv_calc, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
            df_new=pd.DataFrame([i,iv]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)
    df_iv=df_iv.sort_values(['iv'], ascending=[False])
    df_iv=df_iv.reset_index()
    top_iv = df_iv.iloc[0].variable
    if extract_dtype(i, mapper) == 'numerical':
        iv, df_bins, track = add_bins(X, y, top_iv)
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
    elif extract_dtype(i, mapper) == 'categorical':
        df_mod, mapping = var_key_cat(X, y, top_iv)
        mapping["var_key"] = pd.to_numeric(mapping["var_key"])
        iv, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
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
    return top_iv, iv, df_bins, track, var_type

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
    df_rollup_3.reset_index()
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
    best_scenario = df_summ[df_summ['w_entr'] == df_summ['w_entr'].min()].index[0]
    df_best = df[df['scenario'] == best_scenario]
    return df_best

def rollup_best(df, track, var_type):
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
    df_clean['variable']=df['variable'].iloc[0]
    df_clean['bad_rate']=df_clean['sum']/df_clean['count']
    df_clean_2=df_clean[['variable','lb_clean','ub_clean','sum','count','bad_rate', 'list_values']]
    df_clean_2.columns = ['variable','lower_bound','upper_bound', 'events', 'all_records', 'event_rate', 'list_values']
    return df_clean_2

def add_missing(df, df_missing):
    missing_events = df_missing.loc[0, "sum"]
    missing_all_records = df_missing.loc[0, "count"]
    df_compiled = df.copy()
    df_compiled['missing'] = 0
    df_compiled['scenario'] = -999
    for i in df.index:
        df_test = df.copy()
        df_test['missing']=0
        df_test['scenario']=i
        df_test.loc[i, "events"] = df_test.loc[i, "events"]+missing_events
        df_test.loc[i, "all_records"] = df_test.loc[i, "all_records"]+missing_all_records
        df_test.loc[i, "missing"] = 1
        df_compiled = pd.concat([df_compiled, df_test])
    # Missing as a separate scenario
    df_missing_for_append = df_missing[['variable', 'sum', 'count']]
    df_missing_for_append['lower_bound'] = np.nan
    df_missing_for_append['upper_bound'] = np.nan
    df_missing_for_append['event_rate'] = df_missing_for_append['sum'] / df_missing_for_append['count']
    df_missing_for_append['missing'] = 1
    df_missing_for_append['scenario'] = -999
    df_missing_for_append['group'] = 0
    df_missing_for_append = df_missing_for_append.set_index('group')
    df_compiled.columns = ['variable','lower_bound','upper_bound','sum','count','event_rate','list_values','missing','scenario']
    df_compiled = pd.concat([df_compiled, df_missing_for_append])
    df_best = calc_entropy(df_compiled)
    return df_best.sort_index()

def coarse_classing(df, min_leaf_size, max_branches, track, var_type):
    df_original = df.copy()
    df_missing = df[df['bins']=='Missing']
    df = df[df['bins']!='Missing'].reset_index(drop=True)
    df['group'] = 1
    df_scenarios = iterate(df)
    df_valid = check_conditions(df_scenarios, min_leaf_size, max_branches)
    if df_valid.shape[0] == 0:
        df_original['event_rate'] = df_original['sum'] / df_original['count']
        df_original['missing'] = df_original.apply(lambda x: 1 if x['bins']=='Missing' else 0, axis=1)
        df_original = df_original.rename(columns={"sum": "events", "count": "all_records"})
        return df_original[['variable','lower_bound','upper_bound','events','all_records','event_rate','missing','list_values']], 1, var_type
    else:
        df_best = calc_entropy(df_valid)
        df_clean = rollup_best(df_best, track, var_type)
        if df_missing.shape[0] > 0:
            df_clean_m=add_missing(df_clean, df_missing.reset_index(drop=True))
            df_clean_m = df_clean_m.rename(columns={"sum": "events", "count": "all_records"})
            return df_clean_m.drop(['scenario'],axis=1)[['variable','lower_bound','upper_bound','events','all_records','event_rate','missing','list_values']], 0, var_type
        else:
            df_clean['missing'] = 0
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
                    if (pd.isna(lower)) | (pd.isna(upper)):
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
                    if (pd.isna(lower)) | (pd.isna(upper)):
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
                    if (pd.isna(lower)) | (pd.isna(upper)):
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
                    if (pd.isna(lower)) | (pd.isna(upper)):
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
                    if (pd.isna(lower)) | (pd.isna(upper)):
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
                    if (pd.isna(lower)) | (pd.isna(upper)):
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
            child_branch = parent_branch + (group/pow(10, child_level))
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

def split_node(node_number, min_leaf_size, max_branches, max_depth, min_node_split, X_col, node_master, node_datasets, event_flag, mapper):
    top_iv, iv, df_iv, track, var_type = calc_iv(node_number, node_datasets, X_col, event_flag, mapper)
    df_out, error, var_type = coarse_classing(df_iv, min_leaf_size, max_branches, track, var_type)
    node_master_fn, node_datasets_fn = add_nodes(df_out, node_number, max_depth, min_node_split, error, node_master, node_datasets, event_flag, mapper)
    return node_master_fn, node_datasets_fn

def next_node_to_split(node_master, max_depth):
    node_to_split = 0
    for idx, row in node_master.iterrows():
        if row['leaf_flag'] == 0 and row['level'] < max_depth and row['node_number'] not in node_master['parent_node'].values:
            node_to_split = row['node_number']
            break
    return node_to_split    

def build_tree(input_path, event_flag, max_depth, min_node_split, min_leaf_size, max_branches, mapper):
    X_col, node_master, node_datasets = init_setup(input_path, event_flag, max_depth, min_node_split, min_leaf_size, max_branches)
    while next_node_to_split(node_master, max_depth) != 0:
        node_number = next_node_to_split(node_master, max_depth)
        node_master, node_datasets = split_node(node_number, min_leaf_size, max_branches, max_depth, min_node_split, X_col, node_master, node_datasets, event_flag, mapper)
    return node_master, node_datasets
	
def del_subtree(df, df_node, node_number):
    df['branch_str'] = df['branch'].apply(lambda x: format(x, 'g'))
    str_del = df[df['node_number']==node_number]['branch_str'].iloc[0]
    df['mark']=df.apply(lambda x: 1 if (x.branch_str.startswith(str_del)) & (x.branch_str != str_del) else 0, axis=1)
    keep_list = df[df['mark'].isin([0])].node_number
    df_node_return = df_node[df_node['node_number'].isin(keep_list)].reset_index()
    df_filtered = df[df['mark'].isin([0])].reset_index()
    return df_filtered.drop(['branch_str','mark', 'index'], axis=1), df_node_return

def next_node_to_split_node(df, max_depth, node_number): ## df is node_master
    df['branch_str'] = df['branch'].apply(lambda x: format(x, 'g')) ## convert float to string
    str_del = df[df['node_number']==node_number]['branch_str'].iloc[0] ## identify the branch of the parent node to be split
    df['mark']=df.apply(lambda x: 1 if x.branch_str.startswith(str_del) else 0, axis=1) ## mark node_master for the correct branch only
    df_filtered = df[df['mark'].isin([1])] ## take only the marked nodes
    node_to_split = 0
    for idx, row in df_filtered.iterrows():
        if row['leaf_flag'] == 0 and row['level'] < max_depth and row['node_number'] not in df_filtered['parent_node'].values:
            node_to_split = row['node_number']
            break
    return node_to_split  

def build_tree_node(input_path, event_flag, max_depth, min_node_split, min_leaf_size, max_branches, node_to_split, X_col, node_master, node_datasets, mapper):
    while next_node_to_split_node(node_master, max_depth, node_to_split) != 0:
        node_number = next_node_to_split_node(node_master, max_depth, node_to_split)
        node_master, node_datasets = split_node(node_number, min_leaf_size, max_branches, max_depth, min_node_split, X_col, node_master, node_datasets, event_flag, mapper)
    return node_master, node_datasets

def find_var(node_number, node_datasets, X_col, event_flag, mapper):
    df_iv=pd.DataFrame([0,0]).transpose()
    df_iv.columns=['variable','iv']
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    for i in X.columns:
        if extract_dtype(i, mapper) == 'numerical':
            iv, df_bins, track = add_bins(X, y, str(i))
            df_new=pd.DataFrame([i,iv]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)
        elif extract_dtype(i, mapper) == 'categorical':
            df_mod, mapping = var_key_cat(X, y, str(i))
            iv_calc, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
            df_new=pd.DataFrame([i,iv]).transpose()
            df_new.columns=['variable','iv']
            df_iv=pd.concat([df_new, df_iv], axis=0)    
    df_iv=df_iv.sort_values(['iv'], ascending=[False])
    df_iv=df_iv.reset_index(drop=True)
    return df_iv[['variable','iv']]

def best_split(node_number, node_datasets, X_col, event_flag, variable_name, min_leaf_size, max_branches, mapper):
    X = node_datasets[node_datasets.node_number==node_number][X_col]
    y = node_datasets[node_datasets.node_number==node_number][event_flag]
    if extract_dtype(variable_name, mapper) == 'numerical':
        iv, df_bins, track = add_bins(X, y, variable_name)
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
        iv, df_bins, track = add_bins(df_mod, df_mod.badflag, 'var_key')
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
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] <= upper) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] <= upper]
            elif var_type == 'categorical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        elif i == df.shape[0]:
            if var_type == 'numerical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] > lower) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] > lower]
            elif var_type == 'categorical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        else:
            if var_type == 'numerical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[((node_sp[variable] > lower) & (node_sp[variable] <= upper)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[(node_sp[variable] > lower) & (node_sp[variable] <= upper)]
            elif var_type == 'categorical':
                if (pd.isna(lower)) | (pd.isna(upper)):
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
    if node_master[node_master.parent_node==from_node].shape[0] == 0:
        return node_master, node_datasets
    else:
        df = node_master[node_master.parent_node == from_node][['variable','lower_bound', 'upper_bound', 'list_values', 'missing']]
        node_master, node_datasets = add_nodes(df, to_node, max_depth, min_node_split, 0, node_master, node_datasets, target, mapper)
        node_master['branch_str'] = node_master['branch'].apply(lambda x: format(x, 'g'))
        children = node_master[node_master.parent_node==from_node].node_number.tolist()
        from_branch=node_master[node_master.node_number==from_node][['branch_str']].iloc[0, 0]
        to_branch=node_master[node_master.node_number==to_node][['branch_str']].iloc[0, 0]
        for i in children:
            if node_master[node_master.parent_node==i].shape[0] > 0:
                child_branch = node_master[node_master.node_number==i][['branch_str']].iloc[0, 0]
                new_child_branch = child_branch.replace(from_branch, to_branch)
                print("new_child_branch", new_child_branch)
                new_node = node_master[node_master.branch_str==new_child_branch][['node_number']].iloc[0, 0]
                print(i, new_node)
                node_master, node_datasets = copy_node(i, new_node, node_master, node_datasets, max_depth, min_node_split, target, mapper)
        return node_master, node_datasets
    
def copy_node_wrapper(from_node, to_node, node_master, node_datasets, max_depth, min_node_split, target, mapper):
    node_master, node_datasets = copy_node(from_node, to_node, node_master, node_datasets, max_depth, min_node_split, target, mapper)
    return node_master.drop(['branch_str'], axis=1), node_datasets

def add_label(node_number, label, node_master):
    node_master.loc[node_master['node_number'] == node_number, 'node_label'] = label
    return node_master

def add_node_init(df, parent_node, df_for_tree, node_datasets_new, mapper):
    node_sp = node_datasets_new[node_datasets_new['node_number']==parent_node]
    print(parent_node)
    print(node_sp)
    for i in range(df.shape[0]):
        lower = df['lower_bound'].iloc[i]
        upper = df['upper_bound'].iloc[i]
        variable = df['variable'].iloc[i]
        missing = df['missing'].iloc[i]
        list_values = df['list_values'].iloc[i]
        var_type = extract_dtype(variable, mapper)
        node_number = df['node_number'].iloc[i]
        print(node_number)
        if i == 0:
            if var_type == 'numerical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] <= upper) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] <= upper]
            elif var_type == 'categorical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        elif i == df.shape[0]:
            if var_type == 'numerical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable] > lower) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable] > lower]
            elif var_type == 'categorical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        else:
            if var_type == 'numerical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[((node_sp[variable] > lower) & (node_sp[variable] <= upper)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[(node_sp[variable] > lower) & (node_sp[variable] <= upper)]
            elif var_type == 'categorical':
                if (pd.isna(lower)) | (pd.isna(upper)):
                    df_split = node_sp[node_sp[variable].isna()]
                elif missing == 1:
                    df_split = node_sp[(node_sp[variable].isin(list_values)) | (node_sp[variable].isna())]
                else:
                    df_split = node_sp[node_sp[variable].isin(list_values)]
        df_split['node_number'] = node_number
        node_datasets_new = pd.concat([node_datasets_new, df_split])
    return node_datasets_new

def create_node_datasets(df_for_tree, mapper, node_master):
    node_datasets = df_for_tree.copy()
    node_datasets['node_number']=1
    for i in node_master['parent_node'].unique().tolist():
        if i > 0:
            add_node_df = node_master[node_master['parent_node']==i][['node_number','variable','lower_bound','upper_bound','list_values','missing']]
            node_datasets = add_node_init(add_node_df, i, df_for_tree, node_datasets, mapper)
    return node_master, node_datasets