import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import bz2
import pickletools  

class ZomatoModel:

    def __init__(self):
        self.rf = None
        self.rft = None
        self.ann = None
        self.sc = None
        self.mlb = MultiLabelBinarizer()
        self.label_encoder = LabelEncoder()
        self.order_book_map = {"No": 0, "Yes": 1}
        self.columns_to_encode = ['cuisines', 'location']
        self.sort_fimp = None
        self.top_columns = None
        self.traindf = None
        self.testdf = None
        self.le_file = None
        self.loc_freq_map = None
        self.cuisines_freq_map = None

    def encode_online_book(self, train, test):
        with open('online_order_mapping.pkl', 'wb') as mapping_order_table:
            pickle.dump(self.order_book_map, mapping_order_table)

        train['online_order'] = train['online_order'].map(self.order_book_map)
        test['online_order'] = test['online_order'].map(self.order_book_map)

        train['book_table'] = train['book_table'].map(self.order_book_map)
        test['book_table'] = test['book_table'].map(self.order_book_map)

    def encode_rest_type(self, train, test):
        rest_type_df = pd.DataFrame(self.mlb.fit_transform(train['rest_type'].str.split(', ')),
                                    columns=self.mlb.classes_, index=train.index)
        rest_type_df = rest_type_df.add_suffix('_rest_type')

        rest_type_test_df = pd.DataFrame(self.mlb.transform(test['rest_type'].str.split(', ')),
                                         columns=self.mlb.classes_, index=test.index)
        rest_type_test_df = rest_type_test_df.add_suffix('_rest_type')

        train = train.join(rest_type_df)
        test = test.join(rest_type_test_df)
        return train,test

    def compute_location_freq_map(self, train):
        self.loc_freq_map = train['location'].value_counts().to_dict()
        self.cuisines_freq_map = train['cuisines'].value_counts().to_dict()
        return self.loc_freq_map, self.cuisines_freq_map
    
    def freq_encode_columns(self, train, test):
        for column in self.columns_to_encode:
            if column == 'location':
                train[column] = train[column].map(self.loc_freq_map)
                test[column] = test[column].map(self.loc_freq_map)
                
            if column == 'cuisines':
                train[column] = train[column].map(self.cuisines_freq_map)
                test[column] = test[column].map(self.cuisines_freq_map)
   
    def label_encode_type(self, train, test):
        train['type'] = self.label_encoder.fit_transform(train['type'])
        test['type'] = self.label_encoder.fit_transform(test['type'])

    def drop_columns(self, train, test):
        train.drop(['rest_type', 'name'], axis=1, inplace=True)
        test.drop(['rest_type', 'name'], axis=1, inplace=True)

    def train_and_evaluate_model(self, train, test):
        X = train.drop(['rate'], axis=1)
        y = train['rate']
        x_train, x_eval, y_train, y_eval = train_test_split(X, y, random_state=2, test_size=0.2, stratify=train.rate)

        self.rf = rfr(criterion='squared_error', max_depth=30, min_samples_split=2, n_estimators=150)
        self.rf.fit(x_train, y_train)

        pred = self.rf.predict(x_eval)
        y_test_pred = self.rf.predict(test.drop(['rate'], axis=1))

        return x_train

    def get_feature_importance(self,x_train):
        if self.rf is not None:
            print(x_train.columns)
            fimp = pd.Series(self.rf.feature_importances_, index=x_train.columns)
            self.sort_fimp = fimp.sort_values(ascending=False)
            return self.sort_fimp
        else:
            return None
        
    def train_model_on_ft_importance(self,sort_fimp,train,test):
        self.top_columns = list(sort_fimp.head(7).index)
        
        # Convert the index (self.top_columns) to a list of column names
        # self.top_columns = train.columns[self.top_columns]

        # Keep only the top columns in the train dataset
        self.traindf = train[self.top_columns].copy()  # Create a copy to avoid the warning
        # Add the 'rate' column back to the train dataset
        self.traindf['rate'] = train['rate'].copy()  # Create a copy to avoid the warning

        # Keep only the top columns in the test dataset
        self.testdf = test[self.top_columns].copy()  # Create a copy to avoid the warning
        # Add the 'rate' column back to the test dataset
        self.testdf['rate'] = test['rate'].copy()  # Create a copy to avoid the warning
        pd.set_option('display.max_columns',None)
                
        X1 = self.traindf.drop(['rate'], axis=1)
        y1 = self.traindf['rate']
        x_train, x_eval, y_train, y_eval = train_test_split(X1, y1, random_state=2, test_size=0.2, stratify=y1)
        
        self.rft = rfr(criterion = 'squared_error', max_depth= 30, min_samples_split= 2, n_estimators= 150)
        self.rft.fit(x_train, y_train)

        pred = self.rft.predict(x_eval)
    
    def pickle_encodings(self):
        with open('label_encoder.pkl', 'wb') as le_file:
            pickle.dump(self.label_encoder, le_file)

        with open('multi_label_binarizer.pkl', 'wb') as mlb_file:
            pickle.dump(self.mlb, mlb_file)

    # def pickle_model(self):
    #     if self.rft is not None:
    #         with open('random_forest_model.pkl', 'wb') as model_file:
    #             pickle.dump(self.rf, model_file)
                
    def pickle_model(self):
        if self.rft is not None:
            with bz2.BZ2File('random_forest_model.pkl.bz2', 'wb') as model_file:
                pickle.dump(self.rft, model_file)
                
    def unpickle_model(self):
        try:
            with bz2.BZ2File('random_forest_model.pkl.bz2', 'rb') as model_file:
                self.rft = pickle.load(model_file)
        except FileNotFoundError:
            print("Model file not found. Please make sure 'random_forest_model.pkl.bz2' exists.")
    
    # def unpickle_model(self):
    #     try:
    #         with open('random_forest_model.pkl', 'rb') as model_file:
    #             self.rf = pickle.load(model_file)
    #     except FileNotFoundError:
    #         print("Model file not found. Please make sure 'random_forest_model.pkl' exists.")

    def unpickle_encodings(self):
        try:
            with open('label_encoder.pkl', 'rb') as le_file:
                self.label_encoder = pickle.load(le_file)

            with open('multi_label_binarizer.pkl', 'rb') as mlb_file:
                self.mlb = pickle.load(mlb_file)

            with open('online_order_mapping.pkl', 'rb') as mapping_order_table:
                self.order_book_map = pickle.load(mapping_order_table)

        except FileNotFoundError:
            print("One or more encoding files not found. Make sure the necessary files exist.")

    def input_encodings(self,online_order,table_booking,location,cuisines): 
        online_order = self.order_book_map[online_order]
        table_booking = self.order_book_map[table_booking]
        location = self.loc_freq_map[location]
        cuisines = self.cuisines_freq_map[cuisines]
        return online_order,table_booking, location,cuisines
    
    def train_ann_model(self,train,test):
        X = train.drop(['rate'], axis=1)
        y = train['rate']
        
        x_train,x_eval,y_train,y_eval = train_test_split(X,y, random_state=2,test_size=0.2,stratify=train.rate)
        
        self.sc = StandardScaler()
        X_train = self.sc.fit_transform(x_train)
        X_eval = self.sc.transform(x_eval)
        pd.set_option('display.max_columns',None)
        print(x_train.head(2))
        
        self.ann = tf.keras.models.Sequential()
        self.ann.add(tf.keras.layers.Dense(units=4, activation='tanh'))
        self.ann.add(tf.keras.layers.Dense(units=1, activation='tanh'))
        self.ann.add(tf.keras.layers.Dense(units=1))
        self.ann.compile(optimizer = 'sgd', loss = 'mean_squared_error')
        self.ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
      
    def pickle_ann_model(self):
        try:
            with open('ann_model.pkl', 'wb') as ann_model_file:
                pickle.dump(self.ann, ann_model_file)
            print(f"ANN model and StandardScaler have been pickled and saved successfully.")
        except Exception as e:
            print(f"An error occurred while pickling the ANN model: {str(e)}")

    def pickle_sc_model(self):
        try:
            with open('sc_model.pkl', 'wb') as sc_model_file:
                pickle.dump(self.sc, sc_model_file)  # Pickle the StandardScaler too
            print(f"ANN model and StandardScaler have been pickled and saved successfully.")
        except Exception as e:
            print(f"An error occurred while pickling the ANN model: {str(e)}")
    
    def unpickle_ann_model(self):
        try:
            with open('ann_model.pkl', 'rb') as ann_model_file:
                self.ann = pickle.load(ann_model_file)
            print(f"ANN model has been successfully unpickled.")
        except FileNotFoundError:
            print("ANN model file 'ann_model.pkl' not found. Please make sure it exists.")
        except Exception as e:
            print(f"An error occurred while unpickling the ANN model: {str(e)}")
    
    def unpickle_sc_model(self):
        try:
            with open('sc_model.pkl', 'rb') as sc_model_file:
                self.sc = pickle.load(sc_model_file)  # Unpickle the StandardScaler too
            print(f"StandardScaler has been successfully unpickled.")
        except FileNotFoundError:
            print("ANN model file 'ann_model.pkl' not found. Please make sure it exists.")
        except Exception as e:
            print(f"An error occurred while unpickling the ANN model: {str(e)}")            
        
        
        
def main():
    train = pd.read_csv('zomato_train.csv')
    test = pd.read_csv('zomato_test.csv')
    pd.set_option('display.max_rows',None)
        
    zomato_model = ZomatoModel()

    zomato_model.encode_online_book(train, test)
    train,test = zomato_model.encode_rest_type(train, test)
    zomato_model.compute_location_freq_map(train)
    zomato_model.freq_encode_columns(train, test)
    zomato_model.label_encode_type(train, test)
    zomato_model.drop_columns(train, test)
    
    x_train = zomato_model.train_and_evaluate_model(train, test)
    zomato_model.train_ann_model(train,test)
    
    sort_fimp = zomato_model.get_feature_importance(x_train)
    zomato_model.train_model_on_ft_importance(sort_fimp,train,test)
    
    zomato_model.pickle_encodings()
    zomato_model.pickle_ann_model()
    zomato_model.pickle_sc_model()
    zomato_model.pickle_model()

if __name__ == "__main__":
    main()
