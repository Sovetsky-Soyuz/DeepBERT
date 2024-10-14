import csv
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ast
import numpy as np
from math import sqrt
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from helper.general_functions import create_and_write_csv
from combine_review_rating import Calculate_Deep, mergeReview_Rating
from init import args
# from data_processing import TransformLabel, merge_csv_columns
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from review_processing.merge_senmatic_review import extract_features, initialize_features
from helper.utils import read_data, setup_path, word_segment, convert_string_to_float_list
# from review_processing.coarse_gain import get_word2vec_model
# from data_processing import TransformLabel_Deep
from rating_processing.svd import initialize_svd
from rating_processing.factorization_machine import run

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path) # best model
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def reprocess_input(data):
    user_idx = torch.tensor([int(x) for x in data['reviewerID']], dtype=torch.long)
    item_idx = torch.tensor([int(x) for x in data['itemID']], dtype=torch.long)
    rating = torch.tensor([float(x) for x in data['overall']], dtype=torch.float32)
    item_bias = torch.tensor([float(x) for x in data['item_bias']], dtype=torch.float32)
    user_bias = torch.tensor([float(x) for x in data['user_bias']], dtype=torch.float32)

    user_feature = []
    for item in data['Udeep']:
        if isinstance(item, str):
            user_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            user_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            user_feature.append(item.float())
    
    item_feature = []
    for item in data['Ideep']:
        if isinstance(item, str):
            item_feature.append(torch.tensor(ast.literal_eval(item), dtype=torch.float32))
        elif isinstance(item, np.ndarray):
            item_feature.append(torch.tensor(item, dtype=torch.float32))
        else:
            item_feature.append(item.float())
    
    user_feature = torch.stack(user_feature)
    item_feature = torch.stack(item_feature)
    
    return user_idx, item_idx, rating, user_feature, item_feature, item_bias, user_bias


def calculate_rmse(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    squared_errors = (y_true_np - y_pred_np) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

# Define the model
class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(FullyConnectedModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_indices, item_indices, user_features, item_features, item_bias, user_bias):
        # user_features = torch.tensor(user_features)
        # item_features = torch.tensor(item_features)
        interaction = user_features * item_features
        interaction_sum = interaction.sum(dim=1)
        # print("interaction_sum: ", interaction_sum.size())
        if(len(interaction_sum.size()) != self.input_dim):
            self.fc = nn.Linear(len(interaction_sum), len(interaction_sum), bias=True)
            
        # Multiply by weights
        prediction = self.fc(interaction_sum.to(dtype=torch.float32))
        prediction += self.global_bias + item_bias.squeeze()  + user_bias.squeeze() 
        return prediction.squeeze()

    
def train_deepbert(train_data_loader, valid_data_loader, num_factors, batch_size, epochs, method_name, log_interval=100):
    print("=================== Training DeepCGSR model ============================")
    model = FullyConnectedModel(input_dim=batch_size, output_dim=num_factors)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.09)
    early_stopper = EarlyStopper(num_trials=5, save_path=f'model/DeepCGSR/chkpt/{method_name}.pt')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_data_loader):
            try:
                user_idx, item_idx, rating, user_feature, item_feature, item_bias, user_bias = reprocess_input({
                    'reviewerID': batch[0],
                    'itemID': batch[1],
                    'overall': batch[2],
                    'Udeep': batch[3],
                    'Ideep': batch[4],
                    'item_bias': batch[5],
                    'user_bias': batch[6],
                })
                
                predictions = model(user_idx, item_idx, user_feature, item_feature, item_bias, user_bias)
                print("predictions: ", predictions)
                # predictions = torch.clamp(predictions, min=1.0, max=5.0)
                loss = criterion(predictions, rating)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Train Epoch: {epoch+1} [{batch_idx * len(batch[0])}/{len(train_data_loader.dataset)} "
                          f"({100. * batch_idx / len(train_data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                    
            except Exception as e:
                print("Error: ", e)

        auc = test(model, valid_data_loader) 
        # print(f"Validation AUC: {auc}")
        
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    return model



def test(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            user_idx, item_idx, target, udeep, ideep, item_bias, user_bias = reprocess_input(data)
            udeep = torch.tensor(udeep, dtype=torch.float32) if isinstance(udeep, list) else udeep
            ideep = torch.tensor(ideep, dtype=torch.float32) if isinstance(ideep, list) else ideep

            y = model(user_idx, item_idx, udeep, ideep, item_bias, user_bias)
            
            targets.extend(target)
            predicts.extend([round(float(pred)) for pred in y.flatten().cpu().numpy()])

    new_targets = [-1 if i < 4 else 1 for i in targets]
    new_predicts = [-1 if i < 4 else 1 for i in predicts]

    accuracy = accuracy_score(new_targets, new_predicts)
    return accuracy

def test_rsme(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            data = {
                'reviewerID': batch[0],
                'itemID': batch[1],
                'overall': batch[2],
                'Udeep': batch[3],
                'Ideep': batch[4],
                'item_bias': batch[5],
                'user_bias': batch[6],
            }
            
            user_idx, item_idx, target, udeep, ideep, item_bias, user_bias = reprocess_input(data)
            y = model(user_idx, item_idx, udeep, ideep, item_bias, user_bias)
            targets.extend(target)
            predicts.extend([float(pred) for pred in y.flatten().cpu().numpy()])

    new_targer = []
    new_predict = []
    new_targer = targets
    new_predict = new_predict

    print("rsme raw: ", calculate_rmse(targets, predicts))
    mae_value = mean_absolute_error(targets, predicts)
    print("MAE: ", mae_value)
    return calculate_rmse(targets, predicts), mae_value

def format_array(arr):
    return "[" + ", ".join(map(str, arr)) + "]"

def map_and_add_column(df1, df2, column_df1, column_df2=None, column_to_map=None, new_column_name='new_column'):
    if isinstance(df2, pd.DataFrame):
        if column_df2 is None or column_to_map is None:
            raise ValueError("Cần chỉ định column_df2 và column_to_map khi df2 là DataFrame")
        map_dict = df2.set_index(column_df2)[column_to_map].to_dict()
    elif isinstance(df2, dict):
        map_dict = df2
    else:
        raise ValueError("df2 phải là DataFrame hoặc dict")

    df1[new_column_name] = df1[column_df1].map(map_dict)
    df1[new_column_name] = df1[new_column_name].apply(
        lambda x: format_array(x) if isinstance(x, (list, np.ndarray)) else x
    )
    
    return df1

def calulate_user_item_bias(allFeatureReviews):
    print("allFeatureReviews: ", allFeatureReviews)
    allFeatureReviews['Ideep'] = allFeatureReviews['Ideep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
    allFeatureReviews['Udeep'] = allFeatureReviews['Udeep'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
    
    item_features = np.vstack(allFeatureReviews['Ideep'].tolist())
    user_features = np.vstack(allFeatureReviews['Udeep'].tolist())

    ratings = np.array(allFeatureReviews['overall'].tolist()).astype(np.float64)
    ratings = np.nan_to_num(ratings, nan=0.0)

    item_bias = calculate_bias(item_features, ratings)
    user_bias = calculate_bias(user_features, ratings)    
    
    return item_bias, user_bias

def calculate_bias(feature_vectors, ratings):    
    model = LinearRegression()
    model.fit(feature_vectors, ratings)
    predicted_ratings = model.predict(feature_vectors)
    bias = ratings - predicted_ratings
    return bias


def parse_array_from_string(array_string):
    try:
        if isinstance(array_string, (int, float)):
            return [float(array_string)]

        array_string = array_string.strip()
        array_string = re.sub(r'(?<![\d.])e[\d.]+', '', array_string)
        return ast.literal_eval(array_string)
    except (ValueError, SyntaxError):
        return []

def csv_to_dataloader(csv_link, batch_size, shuffle=True):
    df = pd.read_csv(csv_link)

    df['Udeep'] = df['Udeep'].apply(parse_array_from_string)
    df['Ideep'] = df['Ideep'].apply(parse_array_from_string)
    
    df['reviewerID'] = df['reviewerID'].astype(int)
    df['itemID'] = df['itemID'].astype(int)
    
    reviewerID_tensor = torch.tensor(df['reviewerID'].values, dtype=torch.long)
    itemID_tensor = torch.tensor(df['itemID'].values, dtype=torch.long)
    overall_tensor = torch.tensor(df['overall'].values, dtype=torch.float32)
    Udeep_tensor = torch.tensor(df['Udeep'].tolist(), dtype=torch.float32)
    Ideep_tensor = torch.tensor(df['Ideep'].tolist(), dtype=torch.float32)
    itembias_tensor = torch.tensor(df['item_bias'].values, dtype=torch.float32)
    userbias_tensor = torch.tensor(df['user_bias'].values, dtype=torch.float32)
    
    dataset = TensorDataset(reviewerID_tensor, itemID_tensor, overall_tensor, Udeep_tensor, Ideep_tensor, itembias_tensor, userbias_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)
    
    return dataloader


def encode_and_save_csv(df, output_path, columns_to_encode):
    label_encoders = {}
    for column in columns_to_encode:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    df.to_csv(output_path, index=False)
    return label_encoders


# DeepCGSR
def DeepBERT(dataset_df, num_factors, num_words, filename, method_name="DeepCGSR", is_switch_data=False):

    train_data_list = dataset_df["filteredReviewText"].tolist() 
    allreviews_path, reviewer_path, item_path, _, _, _, _, final_data_path, svd_path, checkpoint_path, sparse_matrix_path = setup_path(method_name)   
    split_data = []

    for i in train_data_list:
        split_data.append(word_segment(i))


    allFeatureReviews = extract_features(dataset_df, split_data, num_factors, num_words, filename, method_name, is_switch_data)
    reviewer_feature_dict, item_feature_dict = initialize_features(filename, num_factors, method_name)
    
    svd = initialize_svd(allreviews_path + filename + ".csv", num_factors, svd_path + filename +'.pt')
    z_item = mergeReview_Rating(item_path + filename +".csv", "z_item_" + filename, svd, reviewer_feature_dict, item_feature_dict, "item", method_name)
    z_review = mergeReview_Rating(reviewer_path + filename +".csv", "z_reviewer_" + filename, svd, reviewer_feature_dict, item_feature_dict, "reviewer", method_name)
    
    v_reviewer_list = []
    v_item_list = []
    fm = run(allreviews_path + filename +".csv", num_factors * 2, checkpoint_path + filename +'.pkl', sparse_matrix_path + filename +'.npz')
    for name in z_review.items():
        v_reviewer_list.append(fm.get_embedding('reviewerID_' + name[0]))

    for name in z_item.items():
        v_item_list.append(fm.get_embedding('itemID_' + name[0]))
        
        
    print("================")
    u_deep = {}
    i_deep = {}

    for (z_name, z_value), v_value  in zip(z_review.items(), v_reviewer_list):
        u_deep[z_name] = Calculate_Deep(z_value, v_value)

    for (z_name, z_value), v_value in zip(z_item.items(), v_item_list):
        i_deep[z_name] = Calculate_Deep(z_value, v_value)
        
    create_and_write_csv("u_deep_" + filename, u_deep, method_name)
    create_and_write_csv("i_deep_" + filename, i_deep, method_name)

    allFeatureReviews = allFeatureReviews[['reviewerID', 'itemID', 'overall']]
    allFeatureReviews = map_and_add_column(allFeatureReviews, u_deep, 'reviewerID', 'Key', 'Array', 'Udeep')
    allFeatureReviews = map_and_add_column(allFeatureReviews, i_deep, 'itemID', 'Key', 'Array', 'Ideep')
 
    item_bias, user_bias = calulate_user_item_bias(allFeatureReviews)
    allFeatureReviews['item_bias'] = item_bias
    allFeatureReviews['user_bias'] = user_bias
    
    encode_and_save_csv(allFeatureReviews, final_data_path + method_name + "_" + filename +".csv", ['reviewerID', 'itemID'])
    


