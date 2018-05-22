# coding: utf-8

# # Stockex LSTM predictive model

# In[1]:
import pandas as pd
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import numpy as np
import math
import math
import pandas as pd
from IPython.display import display
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math
import time
import pandas as pd
import numpy as np
from IPython.display import display
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
# # Historical data fetch function using google finance
# We've chosen Google Finanace because of the ability to handle closing prices adjustment
class StockModel:
    # In[13]:
    def get_historical_data(self, symbol, start_date, end_date):
        ''' Daily quotes from Google. Date format='yyyy-mm-dd' '''
        symbol = symbol.upper()
        start = datetime.date(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
        end = datetime.date(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))
        url_string = "https://finance.google.co.uk/bctzjpnsun/historical?q=NASDAQ:{0}".format(symbol)
        url_string += "&startdate={0}&enddate={1}&num={0}&ei=KKltWZHCBNWPuQS9147YBw&output=csv".format(
            start.strftime('%b%d,%Y'), end.strftime('%b%d,%Y'), 4000)

        col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        stocks = pd.read_csv(url_string, header=0, names=col_names)

        df = pd.DataFrame(stocks)
        return df


    # In[16]:
    def remove_data(self, data):
        """
        Remove columns from the data
        :param data: a record of all the stock prices with columns as  ['Date','Open','High','Low','Close','Volume']
        :return: a DataFrame with columns as  ['index','Open','Close','Volume']
        """
        # Define columns of data to keep from historical stock data
        item = []
        open = []
        close = []
        volume = []

        # Loop through the stock data objects backwards and store factors we want to keep
        i_counter = 0
        for i in range(len(data) - 1, -1, -1):
            item.append(i_counter)
            open.append(data['Open'][i])
            close.append(data['Close'][i])
            volume.append(data['Volume'][i])
            i_counter += 1

        # Create a data frame for stock data
        stocks = pd.DataFrame()


        # Add factors to data frame
        stocks['Item'] = item
        stocks['Open'] = open
        stocks['Close'] = pd.to_numeric(close)
        stocks['Volume'] = pd.to_numeric(volume)

        # return new formatted data
        return stocks


    # In[19]:
    def price(self, x):
        """
        format the coords message box
        :param x: data to be formatted
        :return: formatted data
        """
        return '$%1.2f' % x


    # # In[20]:
    # def plot_basic(self, stocks, title='Google Trading', y_label='Price USD', x_label='Trading Days'):
    #     """
    #     Plots basic pyplot
    #     :param stocks: DataFrame having all the necessary data
    #     :param title:  Title of the plot
    #     :param y_label: yLabel of the plot
    #     :param x_label: xLabel of the plot
    #     :return: prints a Pyplot againts items and their closing value
    #     """
    #     fig, ax = plt.subplots()
    #     ax.plot(stocks['Item'], stocks['Close'], '#0A7388')
    #
    #     ax.format_ydata = price
    #     ax.set_title(title)
    #
    #     # Add labels
    #     plt.ylabel(y_label)
    #     plt.xlabel(x_label)
    #
    #     plt.show()


    # In[21]:
    def plot_prediction(self, actual, prediction, title='Google Trading vs Prediction', y_label='Price USD',
                        x_label='Trading Days'):
        """
        Plots train, test and prediction
        :param actual: DataFrame containing actual data
        :param prediction: DataFrame containing predicted values
        :param title:  Title of the plot
        :param y_label: yLabel of the plot
        :param x_label: xLabel of the plot
        :return: prints a Pyplot againts items and their closing value
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Add labels
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        # Plot actual and predicted close values

        plt.plot(actual, '#00FF00', label='Adjusted Close')
        plt.plot(prediction, '#0000FF', label='Predicted Close')

        # Set title
        ax.set_title(title)
        ax.legend(loc='upper left')

        plt.show()


    # In[22]:
    def plot_lstm_prediction(self, actual, prediction, title='Google Trading vs Prediction', y_label='Price USD',
                             x_label='Trading Days'):
        """
        Plots train, test and prediction
        :param actual: DataFrame containing actual data
        :param prediction: DataFrame containing predicted values
        :param title:  Title of the plot
        :param y_label: yLabel of the plot
        :param x_label: xLabel of the plot
        :return: prints a Pyplot againts items and their closing value
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Add labels
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        # Plot actual and predicted close values

        plt.plot(actual, '#00FF00', label='Adjusted Close')
        plt.plot(prediction, '#0000FF', label='Predicted Close')

        # Set title
        ax.set_title(title)
        ax.legend(loc='upper left')

        plt.show()
        print("EYYY: "+str(len(prediction) - len(actual)))#actual.tail(1)[list(actual)[0]])
        # list(prediction)[0]
        # print("PRE-SCORE: "+)

    #plot_basic(stocks)
    #In[24]:
    def get_normalised_data(self, data):
        """
        Normalises the data values using MinMaxScaler from sklearn
        :param data: a DataFrame with columns as  ['index','Open','Close','Volume']
        :return: a DataFrame with normalised value for all the columns except index
        """

        scaler = preprocessing.MinMaxScaler()
        # Initialize a scaler, then apply it to the features
        scaler = MinMaxScaler()
        numerical = ['Open', 'Close', 'Volume']
        data[numerical] = scaler.fit_transform(data[numerical])

        return data


    # In[29]:
    def scale_range(self, x, input_range, target_range):
        """

        Rescale a numpy array from input to target range
        :param x: data to scale
        :param input_range: optional input range for data: default 0.0:1.0
        :param target_range: optional target range for data: default 0.0:1.0
        :return: rescaled array, incoming range [min,max]
        """

        range = [np.amin(x), np.amax(x)]
        x_std = (x - input_range[0]) / (1.0 * (input_range[1] - input_range[0]))
        x_scaled = x_std * (1.0 * (target_range[1] - target_range[0])) + target_range[0]
        return x_scaled, range


    # In[30]:
    def train_test_split_linear_regression(self, stocks):
        """
            Split the data set into training and testing feature for Linear Regression Model
            :param stocks: whole data set containing ['Open','Close','Volume'] features
            :return: X_train : training sets of feature
            :return: X_test : test sets of feature
            :return: y_train: training sets of label
            :return: y_test: test sets of label
            :return: label_range: scaled range of label used in predicting price,
        """
        # Create numpy arrays for features and targets
        feature = []
        label = []

        # Convert dataframe columns to numpy arrays for scikit learn
        for index, row in stocks.iterrows():
            # print([np.array(row['Item'])])
            feature.append([(row['Item'])])
            label.append([(row['Close'])])

        # Regularize the feature and target arrays and store min/max of input data for rescaling later
        feature_bounds = [min(feature), max(feature)]
        feature_bounds = [feature_bounds[0][0], feature_bounds[1][0]]
        label_bounds = [min(label), max(label)]
        label_bounds = [label_bounds[0][0], label_bounds[1][0]]

        feature_scaled, feature_range = self.scale_range(np.array(feature), input_range=feature_bounds, target_range=[-1.0, 1.0])
        label_scaled, label_range = self.scale_range(np.array(label), input_range=label_bounds, target_range=[-1.0, 1.0])

        # Define Test/Train Split 80/20
        split = .315
        split = int(math.floor(len(stocks['Item']) * split))

        # Set up training and test sets
        X_train = feature_scaled[:-split]
        X_test = feature_scaled[-split:]

        y_train = label_scaled[:-split]
        y_test = label_scaled[-split:]

        return X_train, X_test, y_train, y_test, label_range


    # In[31]:
    # TODO: HOW TO DETERMINE TEST_DATA_SIZE AND UNROLL_LENGTH, PREDICTION_TIME?
    def train_test_split_lstm(self, stocks, prediction_time=1, test_data_size=80, unroll_length=20):
        """
            Split the data set into training and testing feature for Long Short Term Memory Model
            :param stocks: whole data set containing ['Open','Close','Volume'] features
            :param prediction_time: no of days
            :param test_data_size: size of test data to be used
            :param unroll_length: how long a window should be used for train test split
            :return: X_train : training sets of feature
            :return: X_test : test sets of feature
            :return: y_train: training sets of label
            :return: y_test: test sets of label
        """
        # training data
        test_data_cut = test_data_size + unroll_length + 1

        x_train = stocks[0:-prediction_time - test_data_cut].as_matrix()
        y_train = stocks[prediction_time:-test_data_cut]['Close'].as_matrix()

        # test data
        x_test = stocks[0 - test_data_cut:-prediction_time].as_matrix()
        print("prediction_time:-test_data_cut  {}  :  {}  ".format(prediction_time,-test_data_cut))
        y_test = stocks[prediction_time - test_data_cut:]['Close'].as_matrix()
        # print("Y_TRAIN: " + str(y_train))
        return x_train, x_test, y_train, y_test


    # In[32]:
    def unroll(self, data, sequence_length=24):
        """
        use different windows for testing and training to stop from leak of information in the data
        :param data: data set to be used for unrolling
        :param sequence_length: window length
        :return: data sets with different window.
        """
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)


    # In[34]:
    def build_model(self, X, y):
        """
        build a linear regression model using sklearn.linear_model
        :param X: Feature dataset
        :param y: label dataset
        :return: a linear regression model
        """
        linear_mod = linear_model.LinearRegression()  # defining the linear regression model
        X = np.reshape(X, (X.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))
        linear_mod.fit(X, y)  # fitting the data points in the model

        return linear_mod


    # In[35]:
    def build_model(self, X, y):
        """
        build a linear regression model using sklearn.linear_model
        :param X: Feature dataset
        :param y: label dataset
        :return: a linear regression model
        """
        linear_mod = linear_model.LinearRegression()  # defining the linear regression model
        X = np.reshape(X, (X.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))
        linear_mod.fit(X, y)  # fitting the data points in the model

        return linear_mod


    # In[36]:
    def predict_prices(self, model, x, label_range):
        """
        Predict the label for given test sets
        :param model: a linear regression model
        :param x: testing features
        :param label_range: normalised range of label data
        :return: predicted labels for given features
        """
        x = np.reshape(x, (x.shape[0], 1))
        predicted_price = model.predict(x)
        predictions_rescaled, re_range = self.scale_range(predicted_price, input_range=[-1.0, 1.0], target_range=label_range)

        return predictions_rescaled.flatten()


    # In[44]:
    def build_basic_model(self, input_dim, output_dim, return_sequences):
        """
        Builds a basic lstm model
        :param input_dim: input dimension of the model
        :param output_dim: output dimension of the model
        :param return_sequences: return sequence of the model
        :return: a basic lstm model with 3 layers.
        """
        model = Sequential()
        model.add(LSTM(
            input_shape=(None, input_dim),
            units=output_dim,
            return_sequences=return_sequences))

        model.add(LSTM(
            100,
            return_sequences=False))

        model.add(Dense(
            units=1))
        model.add(Activation('linear'))

        return model


    # In[45]:
    def build_improved_model(self, input_dim, output_dim, return_sequences):
        """
        Builds an improved Long Short term memory model using keras.layers.recurrent.lstm
        :param input_dim: input dimension of model
        :param output_dim: ouput dimension of model
        :param return_sequences: return sequence for the model
        :return: a 3 layered LSTM model
        """
        model = Sequential()
        model.add(LSTM(
            input_shape=(None, input_dim),
            units=output_dim,
            return_sequences=return_sequences))

        model.add(Dropout(0.2))

        model.add(LSTM(
            128,
            return_sequences=False))

        model.add(Dropout(0.2))

        model.add(Dense(
            units=1))
        model.add(Activation('linear'))

        return model



    def start(self, symbol, start_date, end_date):
        # In[276]:
        data = self.get_historical_data(symbol, start_date, end_date)

        # In[277]:
        data.to_csv('./csv/'+symbol+'.csv', index=False)

        # Calculate Mean, Std , Min, Max for current dataset
        # In[14]:

        data = pd.read_csv('./csv/'+symbol+'.csv')
        print(data.head())
        print(data.tail())

        print("\n")
        print("Open   --- mean :", np.mean(data['Open']), "  \t Std: ", np.std(data['Open']), "  \t Max: ",
              np.max(data['Open']), "  \t Min: ", np.min(data['Open']))
        print("High   --- mean :", np.mean(data['High']), "  \t Std: ", np.std(data['High']), "  \t Max: ",
              np.max(data['High']), "  \t Min: ", np.min(data['High']))
        print("Low    --- mean :", np.mean(data['Low']), "  \t Std: ", np.std(data['Low']), "  \t Max: ", np.max(data['Low']),
              "  \t Min: ", np.min(data['Low']))
        print("Close  --- mean :", np.mean(data['Close']), "  \t Std: ", np.std(data['Close']), "  \t Max: ",
              np.max(data['Close']), "  \t Min: ", np.min(data['Close']))
        print("Volume --- mean :", np.mean(data['Volume']), "  \t Std: ", np.std(data['Volume']), "  \t Max: ",
              np.max(data['Volume']), "  \t Min: ", np.min(data['Volume']))

        # # Preprocessing # #

        # In[15]:
        stocks = self.remove_data(data)

        # Print the dataframe head and tail
        print(stocks.head())
        print("---")
        print(stocks.tail())

        # Remove least prominent features - Date, Low and High value
        # In[17]:
        stocks = self.remove_data(data)

        # Print the dataframe head and tail
        print(stocks.head())
        print("---")
        print(stocks.tail())

        # # Plotting and Visualization

        # In[18]:

        plt.rcParams['figure.figsize'] = (18, 12)
        # Raw plotting

        # In[23]:
        # Normalize the data
        # In[25]:
        stocks = self.get_normalised_data(stocks)
        print(stocks.head())

        print("\n")
        print("Open   --- mean :", np.mean(stocks['Open']), "  \t Std: ", np.std(stocks['Open']), "  \t Max: ",
              np.max(stocks['Open']), "  \t Min: ", np.min(stocks['Open']))
        print("Close  --- mean :", np.mean(stocks['Close']), "  \t Std: ", np.std(stocks['Close']), "  \t Max: ",
              np.max(stocks['Close']), "  \t Min: ", np.min(stocks['Close']))
        print("Volume --- mean :", np.mean(stocks['Volume']), "  \t Std: ", np.std(stocks['Volume']), "  \t Max: ",
              np.max(stocks['Volume']), "  \t Min: ", np.min(stocks['Volume']))

        # In[26]:
        #plot_basic(stocks)

        # In[27]:
        stocks.to_csv('./csv/'+symbol+'_preprocessed.csv', index=False)

        # # Stock Data Manipulation

        # In[28]:

        # # linear Regression Benchmark Model

        # In[33]:

        # Load the preprocessed data

        # In[37]:

        stocks = pd.read_csv('./csv/'+symbol+'_preprocessed.csv')
        display(stocks.head())

        # Split data into train and test pairs

        # In[38]:
        X_train, X_test, y_train, y_test, label_range = self.train_test_split_linear_regression(stocks)

        print("x_train", X_train.shape)
        print("y_train", y_train.shape)
        print("x_test", X_test.shape)
        print("y_test", y_test.shape)

        # Train a Linear regressor model on training set and get prediction

        # In[39]:
        model = self.build_model(X_train, y_train)

        # Get prediction on test set

        # In[40]:
        predictions = self.predict_prices(model, X_test, label_range)

        # Plot the predicted values against actual

        # In[41]:
        self.plot_prediction(y_test, predictions)

        # measure accuracy of the prediction

        # In[42]:
        trainScore = mean_squared_error(X_train, y_train)
        print('Train Score: %.4f MSE (%.4f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = mean_squared_error(predictions, y_test)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

        # ## Long-Sort Term Memory Model
        #
        # LSTM  train and test phases

        # In[43]:

        # In[46]:

        stocks = pd.read_csv('./csv/'+symbol+'_preprocessed.csv')
        stocks_data = stocks.drop(['Item'], axis=1)

        display(stocks_data.head())

        # Split train and test data sets and Unroll train and test data for lstm model

        # In[47]:
        z = self.train_test_split_lstm(stocks_data)
        X_train, X_test, y_train, y_test = z

        print("PRINT MODEL:" + str(y_train))

        unroll_length = 50
        # X_train = unroll(X_train, unroll_length)
        # X_test = unroll(X_test, unroll_length)
        y_train = y_train[-X_train.shape[0]:]
        y_test = y_test[-X_test.shape[0]:]

        # y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
        # y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        #
        # y_train = np.reshape(y_train, (-X_train.shape[0], 1, y_train.shape[1]))
        # y_test = np.reshape(y_test, (-X_test.shape[0], 1, y_test.shape[1]))


        print("x_train", X_train.shape)
        print("y_train", y_train.shape)
        print("x_test", X_test.shape)
        print("y_test", y_test.shape)

        #  Build a basic Long-Short Term Memory mode

        # In[48]:
        # build basic lstm model
        model = self.build_basic_model(input_dim=X_train.shape[-1], output_dim=unroll_length, return_sequences=True)

        # Compile the model
        start = time.time()
        model.compile(loss='mean_squared_error', optimizer='adam')
        print('compilation time : ', time.time() - start)

        # Train the model
        # In[49]:
        model.fit(
            X_train,
            y_train,
            batch_size=1,
            epochs=1,
            validation_split=0.05) # TODO: WHAT DOES THIS DO

        # Predict

        # In[149]:
        '''data = pd.read_csv('goog.csv')
        print(data.head())
        print(data.tail())
        
        stocks = remove_data(data)
        stocks = get_normalised_data(stocks)
        stocks = stocks.drop(['Item'], axis = 1)
        
        #Print the dataframe head and tail
        print(stocks.head())
        
        X = stocks[:].as_matrix()
        Y = stocks[:]['Close'].as_matrix()
        X = sd.unroll(X,1)
        Y = Y[-X.shape[0]:]
        
        print(X.shape)
        print(Y.shape)
        
        # Generate predictions 
        predictions = model.predict(X)
        
        #get the test score
        testScore = model.evaluate(X, Y, verbose=0)
        print('Test Score: %.4f MSE (%.4f RMSE)' % (testScore, math.sqrt(testScore)))'''

        predictions = model.predict(X_test)

        # Plot results

        # In[150]:
        print("PRINTGIN BLUE")
        self.plot_lstm_prediction(predictions, y_test)

        # Get Test Scores

        # In[151]:
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

        # Improved LSTM Model
        # Step 1: Build an improved LSTM model

        # In[152]:
        # Set up hyperparameters
        batch_size = 512
        epochs = 20

        # build improved lstm model
        model = self.build_improved_model(X_train.shape[-1], output_dim=unroll_length, return_sequences=True)

        start = time.time()
        # final_model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile(loss='mean_squared_error', optimizer='adam')
        print('compilation time : ', time.time() - start)

        # Train improved LSTM model

        # In[153]:
        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_split=0.05 # WHAT DOES THIS DO
                  )

        # Make prediction on improved LSTM model

        # In[159]:
        # Generate predictions
        predictions = model.predict(X_test, batch_size=batch_size)

        # In[160]:
        self.plot_lstm_prediction(predictions, y_test)

        # Get test score

        # In[161]:
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

        # In[162]:
        range = [np.amin(stocks_data['Close']), np.amax(stocks_data['Close'])]

        # Calculate the stock price delta in $

        true_delta = testScore * (range[1] - range[0])
        print('Delta Price: %.6f - RMSE * Adjusted Close Range' % true_delta)



sm = StockModel()
sm.start('AUTO', '2017-01-01', '2018-05-22')