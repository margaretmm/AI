import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv('data_train.csv')  # 读取train数据
train_y = train.OnlineBugNum
predictor_x = ['ComplexLevel', 'DesignDocErrNum', 'CodeModLineNum', 'HistoryTestDay', 'BugFatalNum','BugErrorNum']  # 特征
train_x = train[predictor_x]
rf1=DecisionTreeRegressor()
rf2=RandomForestRegressor(n_estimators=1000)           #一般来说n_estimators越大越好，运行结果呈现出的两种结果该值分别是10和1000rf3=ExtraTreesRegressor()

#my_model = RandomForestRegressor()  # 随机森林模型

rf2.fit(train_x, train_y)  # fit
test = pd.read_csv('data_test.csv')  # 读取test数据
test_x = test[predictor_x]
test_y=test.OnlineBugNum
pre_test_y_rf1 = rf1.fit(test_x,test_y).predict(test_x)
pre_test_y_rf2 = rf2.fit(test_x,test_y).predict(test_x)
print(pre_test_y_rf1)

my_submission = pd.DataFrame({'Predict_OnlineBugNum': pre_test_y_rf1,'Actual OnlineBugNum': test_y,'loss':pre_test_y_rf1-test_y})  # 建csv
my_submission.to_csv('submission1.csv', index=False)

my_submission = pd.DataFrame({'Predict_OnlineBugNum': pre_test_y_rf2,'Actual OnlineBugNum': test_y,'loss':pre_test_y_rf2-test_y})  # 建csv
my_submission.to_csv('submission2.csv', index=False)
