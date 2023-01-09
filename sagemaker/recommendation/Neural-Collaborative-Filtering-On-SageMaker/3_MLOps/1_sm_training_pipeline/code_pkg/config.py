# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'TBD']

# model name 
model = 'NeuMF-end'

main_path = '../../data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
