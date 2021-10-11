import pickle
details = pickle.load(open('./RES/details_SF-MA.pkl', 'rb'))
for d in details:
    print(d)