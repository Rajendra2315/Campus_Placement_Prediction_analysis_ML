import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV

df = pd.read_csv('campus.csv')

df['workexp'] = df['workexp'].fillna(value = 'Yes')
df['num_of_projects'] = df['num_of_projects'].fillna(value = np.round(np.mean(df['num_of_projects'])))
df['num_of_global_certifications'] = df['num_of_global_certifications'].fillna(value = np.round(np.mean(df['num_of_global_certifications'])))
df['github_repo'] = df['github_repo'].fillna(value = np.round(np.mean(df['github_repo'])))
df['status'] = df['status'].map({'Placed':1, 'Not Placed':0})

num_col=['ssc_cgpa','hsc_cgpa','ug_cgpa','num_of_projects','num_of_global_certifications','github_repo']
cat_col=df[['gender','branch','workexp','github_acc']]

encoder = LabelEncoder()
cat_col['gender']= encoder.fit_transform (cat_col['gender'])
cat_col['branch']= encoder.fit_transform (cat_col['branch'])
cat_col['workexp']= encoder.fit_transform (cat_col['workexp'])
cat_col['github_acc']= encoder.fit_transform (cat_col['github_acc'])

df.drop(cat_col,axis=1,inplace=True)
new_data = pd.concat([cat_col,df],axis='columns')

x = new_data.drop('status',axis=1)
y = new_data['status']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Model = KNN

KNNClassifier = KNeighborsClassifier()
KNNClassifier.fit(x_train, y_train)

pickle.dump(KNNClassifier,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
