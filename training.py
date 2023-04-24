# load dependencies
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from hmmlearn import hmm
from sklearn.model_selection import LeaveOneOut, GridSearchCV, RepeatedStratifiedKFold, train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import VotingRegressor
import eli5
import seaborn as sns


# *** models:
# general linear model (GLM) ensembles 
# partial least squares (PLS) regression
# LASSO (least absolute shrinkage and selection operator) penalised regression *
# ElasticNet *
# SVR *
# XGB *
# Bayesian *
# HMM
# RNN *
# Linear regression *
# principle component regression *

# *** metric
# mean absolute error (MAE)

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# load data

f = r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\data\a Processed data 2023_2.xlsx"
xl = pd.ExcelFile(f)


# training
list_region = []
list_pred = []
list_pred_el = []
list_pred_la = []
list_pred_svr = []
list_pred_glm = []
list_pred_bayes = []
list_pred_olr = []
list_pred_pcr = []
list_pred_er = []
list_pred_gbm = []
for name in xl.sheet_names:
# for name in ['Bundy', 'Lismore', 'SEQ']:
    # data 
    df = xl.parse(name)
    
    df_1 = df[(df.Year != 2023) & (df.Year != 2022)]
    # df_1 = df_1.drop(['ClimLabel!'], axis = 1)
    train, test = train_test_split(df_1, test_size=0.2, random_state=42, shuffle=True)
    label_x = train[' Dev%']
    # label_x = train[' Tonnes']
    data_x = train.drop(['Year', ' Tonnes', ' Dev%'], axis = 1)
    label_y = test[' Dev%']
    # label_y = test[' Tonnes']
    data_y = test.drop(['Year', ' Tonnes', ' Dev%'], axis = 1)

    dtrain = xgb.DMatrix(data_x, label=label_x)
    dtest = xgb.DMatrix(data_y, label=label_y)
    # setting parameters
    param = {'max_depth': 5, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    # ==== training ===
    # XGB
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)
    # ElasticNet
    regr = ElasticNet(random_state=0)
    regr.fit(data_x, label_x)
    # Lasso
    # clf_lasso = Lasso(alpha=0.2)
    # clf_lasso.fit(data_x, label_x)
    search = GridSearchCV(Lasso(),
                      {'alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )
    search.fit(data_x, label_x)
    # coefficients = search.best_estimator_.coef_
    # lasso_importance = np.abs(coefficients)
    # print(lasso_importance)
    # SVR
    svr = SVR(C=1.0, epsilon=0.2)
    svr.fit(data_x, label_x)
    # GLM
    # glm = linear_model.GammaRegressor()
    # glm.fit(data_x, np.log(label_x+10))
    # glm_importance = glm.coef_

    # Bayesian Ridge
    bayes = linear_model.BayesianRidge()
    bayes.fit(data_x, label_x)
    # HMM
    # clf_hmm = hmm.PoissonHMM(n_components = 3,  n_iter = 10, random_state = 42)
    # print(label_x[:,None])
    # print(np.log(label_x[:,None]))
    # clf_hmm.fit(label_x[:,None])
    # Z = clf_hmm.predict(label_x[:,None])
    # states = pd.unique(Z)
    # linear regression
    olr = linear_model.LinearRegression().fit(data_x, label_x)
    # grid_lr = GridSearchCV(estimator=linear_model.LinearRegression(), param_grid = parameters, cv = 2, n_jobs=-1)
    # principle component regression
    pcr = make_pipeline(PCA(n_components=1), linear_model.LinearRegression())
    pcr.fit(data_x, label_x)
    # lightGBM
    lgbm = lgb.LGBMRegressor(task = 'train', objective = "regression", boosting = "gbdt",  num_leaves = 40,
        learning_rate = 0.05, feature_fraction = 1, bagging_fraction = 1, lambda_l1 = 5, lambda_l2 = .1, max_depth = 5,
        min_child_weight = 1, min_split_gain = 0.001, num_boost_round=1, verbose= 100)
    lgbm.fit(data_x.values, label_x.values)
    # Ridge
    rd = Ridge(alpha=1.0)
    rd.fit(data_x, label_x)

    # # plot model states over time
    # fig, ax = plt.subplots()
    # ax.plot(clf_hmm.lambdas_[states], ".-", ms=6, mfc="orange")
    # ax.plot(label_x)
    # ax.set_title('States compared to generated')
    # ax.set_xlabel('State')
    # plt.show()

    # ensemble
    # er = VotingRegressor([('XGB', bst), ('ElasticNet', regr), ('Lasso', clf_lasso),
    #                     ('SVR', svr), ('Bayesian', bayes), ('OLR', olr), ('PCR', pcr)])
    er = VotingRegressor([('ElasticNet', regr), ('Lasso', search.best_estimator_), ('SVR', svr), ('Bayesian', bayes), ('OLR', olr), ('PCR', pcr)])
    er.fit(data_x, label_x)
 
    # === predicting ===
    ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    ypred_el = regr.predict(data_y)
    ypred_la = search.best_estimator_.predict(data_y)
    # ypred_la = clf_lasso.predict(data_y)
    ypred_svr = svr.predict(data_y)
    # ypred_glm = glm.predict(data_y)
    # ypred_glm = np.exp(ypred_glm)-10
    ypred_bayes = bayes.predict(data_y)
    ypred_olr = olr.predict(data_y)
    y_pred_pcr = pcr.predict(data_y)
    y_pred_er = er.predict(data_y)
    y_pred_gbm = lgbm.predict(data_y.values)
    y_pred_rd = rd.predict(data_y)
    
    # === accuracy ===
    err = mean_squared_error(label_y, ypred)
    err_el = mean_squared_error(label_y, ypred_el)
    err_la = mean_squared_error(label_y, ypred_la)
    err_svr = mean_squared_error(label_y, ypred_svr)
    # err_glm = mean_squared_error(label_y, ypred_glm)
    err_bayes = mean_squared_error(label_y, ypred_bayes)
    err_olr = mean_squared_error(label_y, ypred_olr)
    err_pcr = mean_squared_error(label_y, y_pred_pcr)
    err_er = mean_squared_error(label_y, y_pred_er)
    err_gbm = mean_squared_error(label_y.values, y_pred_gbm)
    err_rd = mean_squared_error(label_y, y_pred_rd)


    # === year 2023 ===
    # print('%s%s%f', (name, 'MSE', err))
    df_2023 = df.loc[df['Year'] == 2023]
    x_2023 = df_2023.drop(['Year', ' Tonnes', ' Dev%'], axis = 1)
    dm_2023 = xgb.DMatrix(x_2023)
    ypred_2023 = bst.predict(dm_2023, iteration_range=(0, bst.best_iteration + 1))
    ypred_el_2023 = regr.predict(x_2023)
    ypred_la_2023 = search.best_estimator_.predict(x_2023)
    # ypred_la_2023 = clf_lasso.predict(x_2023)
    ypred_svr_2023 = svr.predict(x_2023)
    # ypred_glm_2022 = glm.predict(x_2022)
    # print('%s%s%f', (name, 'Pred Dev%', ypred_2022[0]))
    ypred_bayes_2023 = bayes.predict(x_2023)
    ypred_olr_2023 = olr.predict(x_2023)
    ypred_pcr_2023 = pcr.predict(x_2023)
    ypred_er_2023 = er.predict(x_2023)
    ypred_gbm_2023 = lgbm.predict(x_2023)
    ypred_rd_2023 = rd.predict(x_2023)

    # year 2022
    df_2022 = df.loc[df['Year'] == 2022]
    x_2022 = df_2022.drop(['Year', ' Tonnes', ' Dev%'], axis = 1)
    dm_2022 = xgb.DMatrix(x_2022)
    ypred_2022 = bst.predict(dm_2022, iteration_range=(0, bst.best_iteration + 1))
    ypred_el_2022 = regr.predict(x_2022)
    ypred_la_2022 = search.best_estimator_.predict(x_2022)
    ypred_svr_2022 = svr.predict(x_2022)
    ypred_bayes_2022 = bayes.predict(x_2022)
    ypred_olr_2022 = olr.predict(x_2022)
    ypred_pcr_2022 = pcr.predict(x_2022)
    ypred_er_2022 = er.predict(x_2022)
    ypred_gbm_2022 = lgbm.predict(x_2022)
    ypred_rd_2022 = rd.predict(x_2022)

    # SHAP
    # explainer = shap.Explainer(xgb)
    # shap_values = explainer(data_x)
    # clust = shap.utils.hclust(data_x, label_x, linkage="single")
    # shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1)
    # shap.plots.scatter(shap_values, ylabel="SHAP value\n(higher means more likely to impact mac production)")
    # plt.show()

    # Eli5 explanation
    df_ElasticNet = eli5.format_as_dataframe(eli5.explain_weights(regr, top=-1, feature_names = data_x.columns.tolist()))
    # df_ElasticNet.to_csv(r'C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\output\ElasticNet_features.csv', index=False)

    # df_xgb = eli5.format_as_dataframe(eli5.xgboost.explain_weights_xgboost(xgb, top=-1, feature_names = data_x.columns.tolist()))
    df_lasso = eli5.format_as_dataframe(eli5.explain_weights(search.best_estimator_, top=-1, feature_names = data_x.columns.tolist()))
    df_svr = eli5.format_as_dataframe(eli5.explain_weights(svr, top=-1, feature_names = data_x.columns.tolist()))
    df_bayes = eli5.format_as_dataframe(eli5.explain_weights(bayes, top=-1, feature_names = data_x.columns.tolist()))
    df_lr = eli5.format_as_dataframe(eli5.explain_weights(olr, top=-1, feature_names = data_x.columns.tolist()))
    df_rd = eli5.format_as_dataframe(eli5.explain_weights(rd, top=-1, feature_names = data_x.columns.tolist()))
    # print(eli5.format_as_text(eli5.explain_weights(pcr, top=-1, feature_names = data_x.columns.tolist())))
    # print(df_xgb)
    #create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(r'C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\output\Selected_features.xlsx', engine='xlsxwriter')

    #write each DataFrame to a specific sheet
    df_lr.to_excel(writer, sheet_name='LinearRegression', index=False)
    df_ElasticNet.to_excel(writer, sheet_name='ElasticNet', index=False)
    df_lasso.to_excel(writer, sheet_name='Lasso', index=False)
    df_rd.to_excel(writer, sheet_name='Ridge', index=False)
    # df_xgb.to_excel(writer, sheet_name='XGB', index=False) 
    # df_svr.to_excel(writer, sheet_name='SVR', index=False)
    # df_bayes.to_excel(writer, sheet_name='Bayesian', index=False)
    

    #close the Pandas Excel writer and output the Excel file
    writer.save()


    # === plot feature importance ===
    # # XGB
    # ax_imp = xgb.plot_importance(bst)
    # ax_imp.figure.tight_layout()
    # ax_imp.figure.savefig(r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\pic\XGB_%s.png"%name)
    # plt.close()
    # # ElasticNet (linear regressor)
    # fig = plt.figure()
    # feature_importance_Ela = pd.Series(index = data_x.columns, data = np.abs(regr.coef_))
    # feature_importance_Ela.sort_values().tail(20).plot(kind = 'bar', figsize = (20,12))
    # fig.savefig(r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\pic\ElasNet_%s.png"%name)
    # plt.close()
    # # LASSO
    # fig = plt.figure()
    # feature_importance_la = pd.Series(index = data_x.columns, data = lasso_importance)
    # feature_importance_la.sort_values().tail(20).plot(kind = 'bar', figsize = (20,12))
    # fig.savefig(r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\pic\LASSO_%s.png"%name)
    # plt.close()
    # GLM
    # fig = plt.figure()
    # feature_importance_glm = pd.Series(index = data_x.columns, data = glm_importance)
    # feature_importance_glm.sort_values().tail(20).plot(kind = 'bar', figsize = (20,12))
    # fig.savefig(r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\pic\GLM_%s.png"%name)
    # plt.close()
    # XGB feature plot
    ax = xgb.plot_importance(bst)
    ax.figure.savefig(r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\pic\XGB_%s.png"%name)
    # lightGBM feature plot
    feature_imp = pd.DataFrame({'Value':lgbm.feature_importances_,'Feature':data_x.columns})
    plt.figure(figsize=(40, 20))
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:20])
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig(r"C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\pic\lightGBM_%s.png"%name)


    list_region.append(name)
    list_pred.append(ypred_2023[0])
    list_pred_el.append(ypred_el_2023[0])
    list_pred_la.append(ypred_la_2023[0])
    list_pred_svr.append(ypred_svr_2023[0])
    # list_pred_glm.append(np.exp(ypred_glm_2022[0])-10)
    list_pred_bayes.append(ypred_bayes_2023[0])
    list_pred_olr.append(ypred_olr_2023[0])
    list_pred_pcr.append(ypred_pcr_2023[0])
    list_pred_er.append(ypred_er_2023[0])
    list_pred_gbm.append(ypred_gbm_2023[0])

# save prediction
data = {'Region': list_region,
        'XGB(Dev%)': list_pred,
        'ElasNet(Dev%)':list_pred_el,
        'LASSO(Dev%)': list_pred_la,
        'SVR(Dev%))':list_pred_svr,
        'Bayes(Dev%))':list_pred_bayes,
        'LR(Dev%)':list_pred_olr,
        'PCR(Dev%)': list_pred_pcr,
        'LightGBM(Dev%)': list_pred_gbm,
        'Ensemble(Dev%)':list_pred_er}
df = pd.DataFrame(data)
df.to_csv(r'C:\Users\lim\Documents\Projects\Main\DavidMayer\Macadamia\2022\code\output\predictions_2023.csv', index=False)