import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.datasets import boston_housing
from sklearn.datasets import load_diabetes
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
#from scipy.stats import uniform, randint  # Importing randint and uniform
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

 


#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Power Generation Predictor App',
    layout='wide')

#---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    estimator= GradientBoostingRegressor()    # gradient model to be used


    rf = RandomizedSearchCV(estimator=estimator,param_distributions=param_dist,
                            random_state=random_state,
        n_jobs=n_jobs)
    
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

      # Plot predictions and residuals
    plot_prediction_distribution(Y_test, Y_pred_test)
    plot_residuals(Y_test, Y_pred_test)
         
  
#---------------------------------#
st.write("""
# The Machine Learning App

In this implementation, the *GradientBoostingRegressor()* function is used in this app for build a regression model using the **Random SearchCV** algorithm.

Try adjusting the hyperparameters!

""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")



# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)


with st.sidebar.subheader('2.1. Learning Parameters'):
    
    # User inputs for hyperparameters
    n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=132, step=10)
    learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.07451727904094499, step=0.001)
    max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3, step=1)
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, step=1)
    subsample = st.slider("Subsample", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

    # Define the parameter grid
    param_dist = {
        'n_estimators': [n_estimators],  # User-provided value
        'learning_rate': [learning_rate],  # User-provided value
        'max_depth': [max_depth],  # User-provided value
        'min_samples_split': [min_samples_split],  # User-provided value
        'min_samples_leaf': [min_samples_leaf],  # User-provided value
        'subsample': [subsample]  # User-provided value
    }
    



with st.sidebar.subheader('2.2. General Parameters'):
    random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
 #   parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
 #   parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])




def plot_prediction_distribution(Y_true, Y_pred):
    plt.figure(figsize=(10, 6))
    sns.histplot(Y_true, color='blue', label='True Values', kde=True)
    sns.histplot(Y_pred, color='red', label='Predicted Values', kde=True)
    plt.title('Distribution of True vs. Predicted Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    st.subheader('Prediction Distribution')
    st.pyplot(plt)
    plt.close()

def plot_residuals(Y_true, Y_pred):
    residuals = Y_true - Y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y_pred, y=residuals, color='purple')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    st.subheader('Residuals Plot')
    st.pyplot(plt)
    plt.close()



#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df=df.dropna()
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Diabetes dataset is used as the example.')
        st.write(df.head(5))

        


        build_model(df)
   
        
   
    
   
    
   
    
   
    
       # Boston housing dataset
       #boston = load_boston_housing()
       #X = pd.DataFrame(boston.data, columns=boston.feature_names)
       #Y = pd.Series(boston.target, name='response')
       #df = pd.concat( [X,Y], axis=1 )

       #st.markdown('The Boston housing dataset is used as the example.')
       #st.write(df.head(5)) 
        
        
        
        
        
        
        #     n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 100, 200, 150)
        #     learning_rate = st.slider('Learning Rate', 0.01, 0.1, 0.05)
        #     max_depth =st.slider('Max Depth', 3, 4, 3)
        #    # max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
        #     min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        #     min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)



        # # Define the parameter grid storing learning parameters.
        # param_dist = {
        #     'n_estimators': n_estimators,  # Number of boosting stages to be run
        #     'learning_rate': learning_rate,  # Step size for each iteration
        #     'max_depth': max_depth,  # Maximum depth of the trees
        #     'min_samples_split': min_samples_split,  # Minimum number of samples required to split an internal node
        #     'min_samples_leaf': min_samples_leaf,  # Minimum number of samples required to be at a leaf node
        # #    'max_features': max_features  # Number of features to consider at each split (e.g., ['auto', 'sqrt', 'log2', None])
        # }



        #st.title("Hyperparameter Tuning for Gradient Boosting Classifier")
