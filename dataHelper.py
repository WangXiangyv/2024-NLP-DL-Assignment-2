import json
import jsonlines
import copy
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value

implemented_datasets = {
    'restaurant_sup',
    'laptop_sup',
    'acl_sup',
    'agnews_sup',
    'restaurant_fs',
    'laptop_fs',
    'acl_fs',
    'agnews_fs',
}

datasets_class_num = {
    'restaurant_sup':3,
    'laptop_sup':3,
    'acl_sup':6,
    'agnews_sup':4,
    'restaurant_fs':3,
    'laptop_fs':3,
    'acl_fs':6,
    'agnews_fs':4,
}


def load_ABSA_json(path, sep_token):
    label2idx = {'positive':0, 'neutral':1, 'negative':2}
    with open(path, 'r') as f:
        raw_dict = json.load(f)
    texts, labels = zip(*[(v['term']+sep_token+v['sentence'], label2idx[v['polarity']]) for v in raw_dict.values()])
    ds = Dataset.from_dict({'text':texts, 'label':labels})
    return ds

def load_ABSA_dataset(folder, sep_token):
    train_ds = load_ABSA_json(folder + '\\train.json', sep_token)
    test_ds = load_ABSA_json(folder + '\\test.json', sep_token)
    ds = DatasetDict({'train':train_ds, 'test':test_ds})
    return ds

def load_acl_jsonl(path):
    label2idx = {'Uses':0, 'Future':1, 'CompareOrContrast':2, 'Motivation':3, 'Extends':4, 'Background':5}
    data = {'text':[], 'label':[]}
    with open(path, 'r') as f:
        for item in jsonlines.Reader(f):
            data['text'].append(item['text'])
            data['label'].append(label2idx[item['label']])
    return Dataset.from_dict(data)

def get_restaurant_sup(sep_token, *arg):
    return load_ABSA_dataset('data\\SemEval14-res', sep_token)

def get_laptop_sup(sep_token, *arg):
    return load_ABSA_dataset('data\\SemEval14-laptop', sep_token)

def get_acl_sup(*arg):
    train_file = 'data\\ACL-ARC\\train.jsonl'
    test_file = 'data\\ACL-ARC\\test.jsonl'
    train_ds = load_acl_jsonl(train_file)
    test_ds = load_acl_jsonl(test_file)
    return DatasetDict({'train':train_ds, 'test':test_ds})

def get_agnews_sup(*arg):
    data_file="data\\agnews\\test.parquet"
    raw_ds = Dataset.from_parquet(data_file)
    raw_ds = raw_ds.map(features=Features({'text':Value(dtype='string', id=None), 'label':Value(dtype='int64', id=None)}))
    ds = raw_ds.train_test_split(test_size=0.1, seed=2022, shuffle=True)
    return ds

def to_fs_dataset(ds:DatasetDict, seed=2022):
    ds = copy.copy(ds)
    num_labels = max(ds['train']['label']) + 1
    if num_labels <= 4:
        ds['train'] = ds['train'].shuffle(seed=seed)
        ds['train'] = ds['train'].select(range(32))
    else:
        ds['train'] = ds['train'].shuffle(seed=seed)
        _idx = [[] for _ in range(num_labels)]
        for idx, label in enumerate(ds['train']['label']):
            if len(_idx[label]) < 8:
                _idx[label].append(idx)
        idx_lst = [i for item in _idx for i in item]
        ds['train'] = ds['train'].select(idx_lst).shuffle(seed=seed)
    return ds

def to_fs_function(sup_func):
    def wrapper(*args):
        return to_fs_dataset(sup_func(*args))
    return wrapper

get_restaurant_fs = to_fs_function(get_restaurant_sup)

get_laptop_fs = to_fs_function(get_laptop_sup)

get_acl_fs = to_fs_function(get_acl_sup)

get_agnews_fs = to_fs_function(get_agnews_sup)

def aggregate_dataset(ds_list, num_list, shuffle=True, seed=2022):
    assert len(ds_list) == len(num_list)
    if len(ds_list) == 1:
        return ds_list[0]
    else:
        add = 0
        new_ds_list = [ds_list[0]]
        def shift_label(example):
            example['label'] += add
            return example
        for i in range(1, len(ds_list)):
            add += num_list[i - 1]
            new_ds_list.append(ds_list[i].map(shift_label))
        concat_train = concatenate_datasets([d['train'] for d in new_ds_list])
        concat_test = concatenate_datasets([d['test'] for d in new_ds_list])
        if shuffle:
            concat_train = concat_train.shuffle(seed=seed)
            concat_test = concat_test.shuffle(seed=seed)
        return DatasetDict({'train':concat_train, 'test':concat_test})

def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    def name_to_dataset(ds_name)->DatasetDict:
        assert ds_name in implemented_datasets
        return globals()['get_' + ds_name](sep_token)
    
    if isinstance(dataset_name, str):
        return name_to_dataset(dataset_name)
    else:
        ds_list = [name_to_dataset(i) for i in dataset_name]
        num_list = [datasets_class_num[i] for i in dataset_name]
        return aggregate_dataset(ds_list, num_list)
    
# if __name__ == '__main__':
#     a = get_dataset("agnews_sup", '')
#     print(a["train"].unique("label"))
#     b = get_dataset("restaurant_sup","")
#     print(b["train"].unique("label"))
#     c = get_dataset("acl_sup", "")
#     print(c["train"].unique("label"))
#     d = get_dataset(["agnews_sup", "restaurant_sup", "acl_sup"], sep_token='')
#     print(d['train'].unique('label'))