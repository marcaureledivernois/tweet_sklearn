import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#df = pd.read_pickle('df_clean.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_1.pkl')
#df = pd.read_pickle('P:\\df_withcorona_clean_2.pkl')
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_new.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_new.pkl')])
#df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')])  #sent with opti thresholds

df = pd.concat([pd.read_pickle('P:\\df_withcorona_clean_1_with_proba_opti_and_hour.pkl'), pd.read_pickle('P:\\df_withcorona_clean_2_with_proba_opti_and_hour.pkl')])  #sent with opti thresholds


#=========================== for training, keep only labeled twits then format =========================================

dftrain = df.loc[((df['sent']==-1) | (df['sent']==1)) & (df['clean_text']!='')]
dftrain = dftrain.reset_index(drop=True)

dftrain = dftrain.drop_duplicates(subset="id_msg")
dftrain = dftrain.reset_index(drop=True)

#=================================== original pipeline =================================================================

#as empirical findings, keeping the stop words improve the model performance
#original and ros : max_features = 100000, ngram = (1,2)
#slightly better in precision : max_features = 1000000, ngram = (1,3)

def lr_cv(splits, X, Y, pipeline, average_method):

    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X, Y):
        lr_fit = pipeline.fit(X[train], Y[train])
        prediction = lr_fit.predict(X[test])
        scores = lr_fit.score(X[test], Y[test])

        accuracy.append(scores * 100)
        precision.append(precision_score(Y[test], prediction, average=average_method) * 100)
        print('              negative    positive')
        print('precision:', precision_score(Y[test], prediction, average=None))
        recall.append(recall_score(Y[test], prediction, average=average_method) * 100)
        print('recall:   ', recall_score(Y[test], prediction, average=None))
        f1.append(f1_score(Y[test], prediction, average=average_method) * 100)
        print('f1 score: ', f1_score(Y[test], prediction, average=None))
        print('-' * 50)

    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
    return lr_fit


tvec = TfidfVectorizer(stop_words=None, max_features=100000, ngram_range=(1, 3))
rf = RandomForestClassifier(n_estimators=200, random_state=0)
lr = LogisticRegression(max_iter = 4000)

original_pipeline = Pipeline([('vectorizer', tvec),('classifier', lr)])

rf_pipeline = Pipeline([
    ('vectorizer', tvec),
    ('classifier', rf)
])

lr_fit = lr_cv(5, dftrain.clean_text, dftrain.sent, original_pipeline, 'macro')
rf_fit = lr_cv(5, dftrain.clean_text, dftrain.sent, rf_pipeline, 'macro')

#==================================== save the model to disk ===========================================================
# filename = 'C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\original_fit.sav'
# pickle.dump(lr_fit, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))

#===================================== predict_proba ===================================================================

# lr_fit.predict_proba(['this stock is overvalued'])
# lr_fit.predict_proba(['this stock is undervalued'])
# lr_fit.predict_proba(['bull trap'])
# lr_fit.predict_proba(['bear trap'])
# lr_fit.predict_proba(['probably go bankrupt soon'])

#====================================== RandomOverSampler pipeline =====================================================

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

ROS_pipeline = make_pipeline(tvec, RandomOverSampler(random_state=777),lr)

original_pipeline = Pipeline([
    ('vectorizer', tvec),
    ('classifier', lr)
])

lr_fit_ros = lr_cv(5, dftrain.clean_text, dftrain.sent, ROS_pipeline, 'macro')

##### save the model to disk
filename = 'C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\ros_fit_ngram3_maxft1000000.sav'
pickle.dump(lr_fit_ros, open(filename, 'wb'))

#======================================  Compare models ================================================================

filename = 'C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\original_fit.sav'
orig_mod = pickle.load(open(filename, 'rb'))

filename = 'C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\ros_fit.sav'
ros_mod_2gram_100k = pickle.load(open(filename, 'rb'))

filename = 'C:\\Users\\divernoi\\Dropbox\\PycharmProjects2\\Preprocessing_tweets\\ros_fit_ngram3_maxft1000000.sav'
ros_mod_3gram_1000k = pickle.load(open(filename, 'rb'))

unclassified_tweets = df.head(1000).reset_index()

def compare(text):
    print(text)
    print('P(sent=-1) on original model', orig_mod.predict_proba([text])[0][0])
    print('P(sent=-1) on ros_2gram_100k model     ', ros_mod_2gram_100k.predict_proba([text])[0][0])
    #print('P(sent=-1) on ros_3gram_1000k model     ', ros_mod_3gram_1000k.predict_proba([text])[0][0])

compare(unclassified_tweets.clean_text[0])

#======================================= Classification ================================================================

#TODO : when i will do the time serie for global sentiment (all firms combined) be careful of duplicates!
#       for time serie of individual firm the duplicates are not a problem because they will appear once for each firm anyway
#TODO : use followers and ideas variable as a weighting scheme   ----------> how ?

def bullish_proba(text,model):
    return model.predict_proba([text])[0][1]

df['bullish_proba'] = df['clean_text'].apply(bullish_proba, model = orig_mod)

#========================================== Compute E ==================================================================

def efunc(x):
    if x['sent'] == 1:
        E = x['bullish_proba']
    elif x['sent'] == -1:
        E = 1-x['bullish_proba']
    else:
        E = np.nan
    return E

E = df.apply(efunc, axis=1)
E2 = E.dropna()

def plotE(alpha):
    print(np.quantile(E2, 1 - alpha))
    return np.quantile(E2, 1 - alpha)

alp = np.linspace(0,1,21)
lamb = plotE(alp)

plt.plot(alp,lamb)
plt.xlabel('alpha')
plt.ylabel('lambda')
plt.show()


#pickle.dump(E, open('E.pkl', 'wb'))
Eload = pickle.load(open('E.pkl', 'rb'))

#==================================== Classify sentiment given the probability =========================================
#RUN THIS
def classify_3c(proba_bullish, lambda1, lambda2):
    # epsilon : parameter to control for neutral sentiment classification
    # let P be the output proba for negative sentiment of the model. The classification is the following :
    # P in [0, 0.5-epsilon] :          positive
    # P in [0.5-epsilon,0.5+epsilon] : neutral
    # P in [0.5+epsilon,1] :           negative

    # proba_bullish = model.predict_proba([text])[0][1]
    if 0 <= proba_bullish <= lambda1:
        c = -1
    elif lambda1 < proba_bullish <= lambda2:
        c = 0
    elif lambda2 < proba_bullish <= 1:
        c = 1
    return c

df['sent_new_noros'] = df['bullish_proba'].apply(classify_3c, lambda1=0.5, lambda2=0.7)

#def classify_3c(text,epsilon=0.1,model=orig_mod):
#    # epsilon : parameter to control for neutral sentiment classification
#    # let P be the output proba for negative sentiment of the model. The classification is the following :
#    # P in [0, 0.5-epsilon] :          positive
#    # P in [0.5-epsilon,0.5+epsilon] : neutral
#    # P in [0.5+epsilon,1] :           negative
#    proba_negative = model.predict_proba([text])[0][0]
#    if  0 <= proba_negative <= 0.5-epsilon:
#        c = 1
#    elif 0.5-epsilon <= proba_negative <= 0.5+epsilon:
#        c = 0
#    elif 0.5+epsilon <= proba_negative <= 1:
#        c = -1
#    return c

#df['sent_new_noros'] = df['clean_text'].apply(classify_3c)

#======================================= optimal thresholds ============================================================

df_labeled = df.loc[((df['sent']==-1) | (df['sent']==1)) & (df['clean_text']!='')]
df_labeled = df_labeled.reset_index(drop=True)
df_labeled_train ,df_labeled_test = train_test_split(df_labeled,test_size=0.2)


def perf_measures(true,pred):
    measures = dict()
    measures['TP'] = 1*((true==1) & (pred==1))
    measures['FP'] = 1*((true==-1) & (pred==1))
    measures['FN'] = 1*((true==1) &  (pred==-1))
    measures['TN'] = 1*((true==-1) & (pred==-1) )
    measures['NeutralBullish'] = 1 * ((true == 1) & (pred == 0))
    measures['NeutralBearish'] = 1 * ((true == -1) & (pred == 0))
    measures['precision'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FP'].sum().sum())
    measures['recall'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FN'].sum().sum())
    measures['specificity'] = measures['TN'].sum().sum() / (measures['TN'].sum().sum() + measures['FP'].sum().sum())
    measures['f1score'] = 2 * measures['precision'] * measures['recall'] / (measures['precision'] + measures['recall'])
    measures['accuracy'] =  (measures['TP'].sum().sum() + measures['TN'].sum().sum())/\
                            (measures['TP'].sum().sum() + measures['TN'].sum().sum() + measures['FP'].sum().sum() + measures['FN'].sum().sum())
    print('accuracy : ' , measures['accuracy'],', precision : ' , measures['precision'] , ', recall : ' , measures['recall'], ' , specificity : ' , measures['specificity'], 'f1 score : ', measures['f1score'])
    return measures

def classif(probas, thresholds):
    if probas < thresholds[0]:
        c = -1
    elif thresholds[0] < probas < thresholds[1]:
        c = 0
    elif probas > thresholds[1]:
        c = 1
    return c

obs = []
for thres1 in np.linspace(0.1,0.5,9):
    for thres2 in np.linspace(0.5,0.9,9):
        predictions = df_labeled['bullish_proba'].apply(classif, thresholds = [thres1,thres2])
        perf = perf_measures(df_labeled['sent'], predictions)
        obs.append([thres1,thres2,perf['f1score']])


fig = plt.figure()
ax = plt.axes(projection='3d')

surf = ax.plot_trisurf(np.array([item[0] for item in obs]), np.array([item[1] for item in obs]), np.array([item[2] for item in obs]),cmap='viridis', edgecolor='none')
ax.set_title('F1-score surface')
ax.set_xlabel('lower threshold')
ax.set_ylabel('upper threshold')
ax.set_zlabel('F1-score')
fig.colorbar(surf)
plt.show()


def classif2(probas, threshold, classif_type):
    if classif_type == 'Bullish':
        if probas > threshold:
            c = 1
        else:
            c = -1
    elif classif_type == 'Bearish':
        if probas < threshold:
            c = -1
        else:
            c = 1
    return c

obs1  = []
for thresh in  np.linspace(0.1,0.9,19):
    predictions = df_labeled['bullish_proba'].apply(classif2, threshold=thresh, classif_type = 'Bullish')

    TP = 1*((df_labeled['sent']==1) & (predictions==1))
    FP = 1*((df_labeled['sent']==-1) & (predictions==1))
    FN = 1*((df_labeled['sent']==1) &  (predictions==-1))
    TN = 1*((df_labeled['sent']==-1) & (predictions==-1) )
    pre = TP.sum().sum() / (TP.sum().sum() + FP.sum().sum())
    rec = TP.sum().sum() / (TP.sum().sum() + FN.sum().sum())
    spec = TN.sum().sum() / (TN.sum().sum() + FP.sum().sum())
    f1 = 2 * pre * rec / (pre + rec)
    obs1.append([thresh, f1])

obs2  = []
for thresh in  np.linspace(0.1,0.9,19):
    predictions = df_labeled['bullish_proba'].apply(classif2, threshold=thresh, classif_type = 'Bearish')
    TP = 1 * ((df_labeled['sent'] == -1) & (predictions == -1))
    FP = 1 * ((df_labeled['sent'] == 1) & (predictions == -1))
    FN = 1 * ((df_labeled['sent'] == -1) & (predictions == 1))
    TN = 1 * ((df_labeled['sent'] == 1) & (predictions == 1))
    pre = TP.sum().sum() / (TP.sum().sum() + FP.sum().sum())
    rec = TP.sum().sum() / (TP.sum().sum() + FN.sum().sum())
    spec = TN.sum().sum() / (TN.sum().sum() + FP.sum().sum())
    f1 = 2 * pre * rec / (pre + rec)
    obs2.append([thresh, f1])

plt.style.use('seaborn-whitegrid')
plt.plot([item[0] for item in obs1], [item[1] for item in obs1], '-', color='green',label ='Bullish')
plt.plot([item[0] for item in obs2], [item[1] for item in obs2], '-', color='red', label='Bearish')
plt.scatter(0.5, 0.9286812026620501, s=80, facecolors='none', edgecolors='blue')
plt.scatter(0.72, 0.6394655410250576, s=80, facecolors='none', edgecolors='blue')
plt.axvline(x=0.5,  ymax=0.94, color = 'blue',  linestyle='--')
plt.axvline(x=0.72, ymax= 0.558, color = 'blue',  linestyle='--')
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('Optimal classification thresholds')
#plt.savefig(vartoplot[:2] + '_' + jmptype + '.jpg')
plt.show()

#======================================= Merged sentiment ================================================================
# if tweet was labeled already, take that label, otherwise take the classified sentiment
#RUN THIS
def sentmerged(pre,post):
    if pre == 0:
        out = post
    else:
        out = pre
    return out

df['sent_merged'] = df.apply(lambda x: sentmerged(x.sent, x.sent_new_noros), axis=1)

#overwrite empty messages with no label
df.loc[((df['sent']==0)) & (df['clean_text']==''), 'sent_merged'] = 0

#============================= save ====================================================================================

#df[:45000000].to_pickle('P:\\df_withcorona_clean_1_with_proba_opti.pkl')
#df[45000000:].to_pickle('P:\\df_withcorona_clean_2_with_proba_opti.pkl')


#============================ TFIDF representation =====================================================================

tfvecarr = ros_mod_2gram_100k.named_steps.tfidfvectorizer.fit_transform(unclassified_tweets['clean_text']).toarray()  #replace unclassified_tweets by dftrain
tvec.get_feature_names()

#============================ access coefficients =====================================================================
coef = ros_mod_2gram_100k.named_steps.logisticregression.coef_
voc = ros_mod_2gram_100k.named_steps.tfidfvectorizer.vocabulary_

coefdf = pd.DataFrame(coef).T
vocdf = pd.DataFrame(voc.items())
vocdf = vocdf.set_index(1)
merged_coef = vocdf.merge(coefdf, left_index=True, right_index=True)
merged_coef = merged_coef.set_index('0_x')

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# recompute manually probability
sigmoid( np.dot(ros_mod_2gram_100k.named_steps.tfidfvectorizer.transform(['hello']).toarray(), ros_mod_2gram_100k.named_steps.logisticregression.coef_.T) + ros_mod_2gram_100k.named_steps.logisticregression.intercept_)
ros_mod_2gram_100k.predict_proba(['hello'])

#clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)


