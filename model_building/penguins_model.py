try:
    import pandas as pd
    penguins = pd.read_csv('data/penguins_cleaned.csv')
    print('Imported the datafile.')
except:
    print('Something went wrong with the data import')


try:
    df = penguins.copy()

    target = 'species'
    encode = ['sex', 'island']

    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]

    targetmap = {'Adelie': 0, 'Chinstrap':1, 'Gentoo':2}

    def target_encoding(val):
        return targetmap[val]

    df.species = df.species.apply(target_encoding)

    X = df.drop('species', axis=1)
    y = df["species"]

    print('Successfully created X and y.')
except:
    print('Something went wrong with the data encoding.')

# create model
try:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X,y)
    print('Successfully trained a classifier.')
except:
    print('Something went wrong with training.')

try:
    import pickle
    with open('penguin_clf.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print('Saved the trained model.')
except:
    print('Something went wrong with saving the model.')
