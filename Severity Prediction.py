from flask import Flask,request,render_template,redirect,url_for
app=Flask(__name__)
@app.route('/',methods=["GET","POST"])
def home():
    if request.method=="POST":
        log=request.form["login"]
        paswd=request.form["password"]
        if log=="admin" and paswd=="password":
            return render_template('PredictionPage.html')
    return render_template("login.html")

@app.route('/prediction',methods=["GET","POST"])
def prediction():
    if request.method=="POST":
        inp1 = request.form["input1"]
        inp2 = request.form["input2"]
        inp3 = request.form["input3"]
        inp4 = request.form["input4"]
        inp5 = request.form["input5"]
        inp6 = request.form["input6"]
        inp7 = request.form["input7"]
        inp8 = request.form["input8"]
        inp9 = request.form["input9"]
        inp10=request.form["input10"]
        inp11=request.form["input11"]
        inp12=request.form["input12"]
        inp13=request.form["input13"]
        inp14=request.form["input14"]
        inp15=request.form["input15"]
        inp16=request.form["input16"]
        inp17=request.form["input17"]
        import numpy as np
        import pandas as pd
        # import matplotlib.pyplot as plt
        np.set_printoptions(suppress=True)
        covid = pd.read_excel("covid_kaggle.xlsx")
        print("Size Of the Dataset before preprocessing is ")
        # print(covid.shape)
        #  1.DataWash
        covid = covid.drop(
            ['Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63',
             'Parainfluenza 1', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E',
             'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Rhinovirus/Enterovirus',
             'Coronavirus HKU1', 'Parainfluenza 3', 'Influenza B, rapid test', 'Influenza A, rapid test'], axis=1)
        covid = covid.drop(['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)',
                            'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                            'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1)
        urine_features = ['Urine - Esterase', 'Urine - Aspect', 'Urine - pH', 'Urine - Hemoglobin',
                          'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Nitrite', 'Urine - Density',
                          'Urine - Urobilinogen', 'Urine - Protein', 'Urine - Sugar', 'Urine - Leukocytes',
                          'Urine - Crystals', 'Urine - Red blood cells', 'Urine - Hyaline cylinders',
                          'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']
        covid = covid.drop(urine_features, axis=1)
        arterial_blood_gas_features = ['Hb saturation (arterial blood gases)', 'pCO2 (arterial blood gas analysis)',
                                       'Base excess (arterial blood gas analysis)', 'pH (arterial blood gas analysis)',
                                       'Total CO2 (arterial blood gas analysis)', 'HCO3 (arterial blood gas analysis)',
                                       'pO2 (arterial blood gas analysis)', 'Arteiral Fio2', 'Phosphor',
                                       'ctO2 (arterial blood gas analysis)']
        covid = covid.drop(arterial_blood_gas_features, axis=1)
        print(covid.shape)
        i = 0
        for column in covid:
            if (covid[column].count() < 100):
                # print(column, covid[column].count())
                covid = covid.drop(column, axis=1)

        covid = covid.loc[:, covid.apply(pd.Series.nunique) != 1]
        features = list(covid.columns)
        sorted_features = [x for _, x in sorted(zip(covid[features].count(), features))]

        covid_init = covid[sorted_features[-1]]
        # for i in reversed(range(0, len(sorted_features))):
        # print(sorted_features[i], covid[sorted_features[i]].count())

        removed_features = ['Lactic Dehydrogenase', 'Creatine phosphokinase\xa0(CPK)\xa0',
                            'International normalized ratio (INR)', 'Base excess (venous blood gas analysis)',
                            'HCO3 (venous blood gas analysis)', 'Hb saturation (venous blood gas analysis)',
                            'Total CO2 (venous blood gas analysis)', 'pCO2 (venous blood gas analysis)',
                            'pH (venous blood gas analysis)', 'pO2 (venous blood gas analysis)', 'Alkaline phosphatase',
                            'Gamma-glutamyltransferase\xa0', 'Direct Bilirubin', 'Indirect Bilirubin',
                            'Total Bilirubin', 'Serum Glucose', 'Alanine transaminase', 'Aspartate transaminase',
                            'Strepto A', 'Sodium', 'Potassium', 'Urea', 'Creatinine']

        covid = covid.drop(removed_features, axis=1)

        # Drop patients that have less than 10 records

        for index, row in covid.iterrows():
            if row.count() < 10:
                covid.drop(index, inplace=True)

        features = list(covid.columns)
        sorted_features = [x for _, x in sorted(zip(covid[features].count(), features))]
        # for i in reversed(range(0, len(sorted_features))):
        # print(sorted_features[i], covid[sorted_features[i]].count())

        # Drop NaN

        covid = covid.dropna()

        # set poitive as 1 and negative as 0
        covid['SARS-Cov-2 exam result'] = covid['SARS-Cov-2 exam result'].map({'positive': 1, 'negative': 0})

        # we consider 420 rows + 18 Coloumns
        print("Size of Dataset after preprocessing is ")
        print(covid.shape)

        from sklearn import preprocessing
        from sklearn.model_selection import train_test_split

        # 2.Test Train Split

        y = covid["SARS-Cov-2 exam result"].to_numpy()

        X = covid
        X = X.drop(["SARS-Cov-2 exam result"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=1)

        # # 3.4 Neural Networks
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        import keras.losses
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import cross_val_score
        import keras.activations
        import keras.metrics
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import cross_val_score
        nn = Sequential()
        nn.add(Dense(activation='relu', input_dim=X_train.shape[1], units=10))
        nn.add(Dropout(rate=0.1))
        nn.add(Dense(kernel_initializer="uniform", activation='relu', units=15))
        nn.add(Dropout(rate=0.1))
        nn.add(Dense(kernel_initializer="uniform", activation='relu', units=5))
        nn.add(Dropout(rate=0.1))
        nn.add(Dense(kernel_initializer='uniform', activation='sigmoid', units=1))
        nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        nn.fit(X_train, y_train, batch_size=50, epochs=200, validation_split=0.2)
        nn_predi = nn.predict([[float(inp1),float(inp2),float(inp3),float(inp4),float(inp5),float(inp6),float(inp7),float(inp8),
                               float(inp9),float(inp10),float(inp11),float(inp12),float(inp13),float(inp14),float(inp15),float(inp16),
                               float(inp17)]])
        print(nn_predi)
        import keras.metrics
        for nn_pred in nn_predi:
            if nn_pred > 0 or nn_pred <= 0.3:
                abc = f"Level-1--{nn_pred}"
                print(nn_pred)
                return render_template('prediction_result.html',metric=abc)
            elif nn_pred > 0.3 or nn_pred <= 0.6:
                abc = f"Level-2-->{nn_pred}"
                print(nn_pred)
                return render_template('prediction_result.html',metric=abc)
            elif nn_pred > 0.6 or nn_pred <= 1.0:
                abc = f"Level-3-->{nn_pred}"
                print(nn_pred)
                return render_template('prediction_result.html',metric=abc)
   
            
   
@app.route('/comparision',methods=["GET","POST"])
def comparision():
     return render_template('comparision.html')
   
  
if __name__ == '__main__':
    app.run()