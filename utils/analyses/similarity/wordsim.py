from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np 

def calc_pair_sim(df_vec, concept1, concept2, out_of_category_concepts):
    print(f'{concept1} {concept2} {out_of_category_concepts}')
    df1 = pd.DataFrame(df_vec.loc[concept1]).transpose()
    df2 = pd.DataFrame(df_vec.loc[concept2]).transpose()
    sim = cosine_similarity(df1, df2)[0][0]
    
    out_sims = []
    for out in out_of_category_concepts:
        df1 = pd.DataFrame(df_vec.loc[concept1]).transpose()
        df2 = pd.DataFrame(df_vec.loc[concept2]).transpose()
        df3 = pd.DataFrame(df_vec.loc[out]).transpose()
        out_sim = cosine_similarity(df1, df3)
        out_sims.append(out_sim)
        out_sim = cosine_similarity(df2, df3)
        out_sims.append(out_sim)
    
    mean = np.asarray(out_sims).mean()
    return sim - mean