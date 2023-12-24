import json
import sys, os
import pandas as pd

dir_path = './'

test_results = []

for path, subdirs, subfiles in os.walk(dir_path):
    if len(list(filter(lambda x: x.startswith('metrics.json'), subfiles))) > 0:
        try:
            a= path+ '/' + str(list(filter(lambda x: x.startswith('metrics.json'), subfiles)).pop())
        except:
            continue
        test_results.append(a)
    else:
        continue

print(test_results)
print('-'*83)
deli_df = pd.DataFrame()
bib_df = pd.DataFrame()
for path in test_results:
    if path == 0:
        continue
    with open(path) as f:
        dic = json.load(f)

    results = pd.DataFrame.from_dict(dic, orient='index').T[['test_MAP', 'test_fixed_f1', 'test_relaxed_f1']]
    print(results)
    path = path.split('/')[1]
    if 'results' in path:
        continue
    try:
        file_name = path.split('_')
        task = file_name[0]
        data = file_name[1]
    except:
        continue
    print(file_name)

    score_net = '_'.join(file_name[file_name.index('s'): file_name.index('t')][1:])
    task_net = '_'.join(file_name[file_name.index('t'): file_name.index('unwei0.1')][1:])

    weights = '_'.join(file_name[file_name.index('unwei0.1'): file_name.index('unwei0.1')+2])
    import re
    label_batch = re.sub('[A-Z, a-z]', '', file_name[-2]) 
    unlabel_batch = re.sub('[A-Z, a-z]', '', file_name[-1]) 

    results['path'] = path
    results['task'] = task
    results['data'] = data
    results['score_net'] = score_net
    results['task_net'] = task_net
    results['weights'] = weights
    results['label'] = label_batch
    results['unlabel'] = unlabel_batch

    if data == 'deli':
        deli_df = pd.concat((deli_df, results))
    elif 'bib' in data:
        bib_df = pd.concat((bib_df, results))

print(deli_df)
print(bib_df)
deli_df = deli_df.sort_values(by=['unlabel'])
bib_df = bib_df.sort_values(by=['unlabel'])
deli_df.to_csv('./deli-overall-summary.csv', sep='\t')
bib_df.to_csv('./bib-overall-summary.csv', sep='\t')
raise ValueError


inputio = sys.argv
print(inputio)
dir_path = inputio[1]
data = inputio[2]
print(dir_path)
print(data)

test_results = [file for file in os.listdir(dir_path) if file.startswith('test_metric')]

results = {}
for test in test_results:
    print(test)
    with open(os.path.join(dir_path, test)) as f:
        results[test.split('_')[-1].split('.')[0]] = json.load(f)
        print(test.split('_')[-1])    
results = pd.DataFrame.from_dict(results)

if data == 'nyt':
    results = results[['tot-unseen', 'trg']].T
else:
    results = results[['tot-unseen', 'trg', 'src']].T

results.round(4).to_csv(os.path.join(dir_path, "summary.csv"), sep='\t')

files = sorted(os.listdir('./'))
files = list(filter(lambda x : x if x.startswith('srl_') else None, files))
df = pd.DataFrame()
for file in files:
    if not os.path.exists(file + '/summary.csv'):
        continue
    tmp = pd.read_csv(file+'/summary.csv', sep='\t')
    tmp = tmp.reset_index()
    tmp[['index']] = '_'.join(file.split("_")[1:])
    with open(file+'/config.json', 'rb') as f:
        try:
            dicts = json.load(f)
        except json.decoder.JSONDecodeError:
            print(file)
    accum = dicts['trainer'].get('num_gradient_accumulation_steps', 2)
    try:
        tmp['labeled_batch'] = dicts['data_loader']['scheduler']['batch_size']['labeled'] * accum / 2
        tmp['unlabeled_batch'] = dicts['data_loader']['scheduler']['batch_size']['unlabeled'] * accum / 2 
    except:
        tmp['labeled_batch'] = dicts['data_loader']['batch_sampler']['batch_size'] * accum
    df = pd.concat((df, tmp))
df = df.drop(columns='loss')
df['data'] = df['index'].str.split('_', n=1, expand=True)[0]
df['training_method'] = df['index'].str.split('_', n=3, expand=True)[1] + '_' + df['index'].str.split('_', n=3, expand=True)[2]

df.to_csv('./overall-summary.csv', sep='\t')
df.round(4).reset_index().sort_values(by=['index', 'training_method','level_0', 'unlabeled_batch']).to_csv('./overall_sorted.csv', sep='\t')
