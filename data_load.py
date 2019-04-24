import json  # we need to use the JSON package to load the data, since the data is stored in JSON format
from data_process import split_data, preprocess, load_data, save_data

with open("data/reddit.json") as fp:
    data = json.load(fp)

# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment

# Example:
data_point = data[0]  # select the first data point in the dataset

# Now we print all the information about this datapoint
for info_name, info_value in data_point.items():
    print(info_name + " : " + str(info_value))

features = ['text', 'is_root', 'controversiality', 'children']  # list of features to preprocess
train, val, test = split_data(data)

train_ = preprocess(train, feature_list=features, max=500)
val_ = preprocess(val, feature_list=features)
test_ = preprocess(test, feature_list=features)

save_data(train_, val_, test_)
