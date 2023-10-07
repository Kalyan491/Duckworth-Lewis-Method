import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        pred=Z_0 * (1-(np.exp(-(self.L * X)/Z_0))) 
        return pred

    

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        return np.mean((self.get_predictions(X,self.Z0[w-1],w)-Y)**2)
        
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """

        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    data=pd.read_csv(data_path)
    return data
    

def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    #(i)Removing unnecessary columns
    columns_preserved = [1,2,3,4,5,6,7,11,24]
    df_necessary_columns = data.iloc[:, columns_preserved]

    #(ii)loading date in proper format DD-MM-YYYY
    dictionary_for_months = {'Jan': '01','Feb': '02','Mar': '03','Apr': '04','May': '05','Jun': '06',
    'Jul': '07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}
    for index, str in data['Date'].items():
        if '/' in str:
            date, month, year = str.split('/')
        else:
            month, date, year = str.split(' ')
            month = dictionary_for_months [month]
            date,_ = date.split('-')
        df_necessary_columns.at[index, 'Date'] = f'{date}-{month}-{year}'

    #(iii)Removing the rows with missing values
    if df_necessary_columns.isna().any().any():
        df_necessary_columns=df_necessary_columns.dropna()

    #(iv)Getting training data
    training_data=[]
    for Wickets_in_Hand in range(1,11):
        filtered_df = data[(data['Wickets.in.Hand'] == Wickets_in_Hand) & (data['Innings'] == 1)]
        double_filtered_df = data[(data['Over'] == 1) & (data['Innings'] == 1)]
        tuples_list_data = [[row['Total.Overs']-row['Over'], row['Runs.Remaining']] for index, row in filtered_df.iterrows()]
        new_list_data=[]
        if Wickets_in_Hand==10:
            new_list_data=[[row['Total.Overs'], row['Innings.Total.Runs']] for index, row in double_filtered_df.iterrows()]
            for each in new_list_data:
                tuples_list_data.append(each)
        training_data.append(tuples_list_data)
        
    training_data=np.array(training_data,dtype=object)
    return training_data

def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """

    #converting data into (x,y) training pairs
    training_data=list(data)
    def func(x, training_pairs):
        s=0
        totalpoints=0
        for i in range(len(training_pairs)):
            totalpoints+=1
            ind=training_pairs[i][1]-1
            s+=((x[ind] * (1-(np.exp(-(x[10] * training_pairs[i][0][0])/x[ind]))))-training_pairs[i][0][1])**2
        return s/totalpoints

    training_pairs=[]
    for i in range(1,11):
        for j in range(len(training_data[i-1])):
            samples=[training_data[i-1][j],i]
            training_pairs.append(samples)

    x=[1]*10        #trainable parameters Z_list
    x.append(10)    #trainable parameter L
    res = sp.optimize.minimize(func, x,training_pairs)

    model.Z0=res.x[:10]
    model.L=res.x[10]
    
    return model


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    def func(x, Z0):
        return Z0 * (1-(np.exp(-(model.L * x)/Z0)))
    x = np.arange(0, 51)
    for i in range(10):
        name="w="+str(i+1)
        plt.plot(x,func(x,model.Z0[i]),label=name)
    plt.legend()
    plt.xlabel("Overs Remaining")
    plt.ylabel("Average Runs obtainable")
    plt.savefig(plot_path)
    del os.environ['QT_QPA_PLATFORM']
    
def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    params=[]
    for i in model.Z0:
        params.append(i)
    params.append(model.L)
    print("Model Parameters (Z_0(1), ..., Z_0(10)):",params[:10])
    print("Model Parameter L:",params[10])
    return params
    


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    

    losses=[]
    total_points=0
    for i in range(1,11):
        ans=np.array(data[i-1])
        total_points+=ans.shape[0]
        loss=model.calculate_loss([], ans[:,0], ans[:,1],i)*ans.shape[0]
        losses.append(loss)

    loss=sum(losses)/total_points
    print("Normalised squared error Loss is:",loss)
    return loss


def main(args):
    """Main Function"""
    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model
    model.save(args['model_path'])  # Saving the model
    
    plot(model, args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
