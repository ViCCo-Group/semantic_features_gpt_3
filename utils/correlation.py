from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from scipy.spatial.distance import squareform
import math 
import pandas as pd 

def get_similiarity_vector(df_vec, name=""):
    # Build cosine similarity matrices of features and flatten triangular part
    #data_dir = './evaluation/check_data'
    pred_sim_matrix = cosine_similarity(df_vec, df_vec)
    #pd.DataFrame(pred_sim_matrix).to_csv('%s/similarity_matrix_%s.csv' % (data_dir, name))
    sim = squareform(pred_sim_matrix, force='tovector', checks=False)
    return sim

def calc_semi_partial_correlation(r_12, r_13, r_23):
    semipartial_23 = (r_12 - (r_13*r_23)) / (math.sqrt(1-(r_23 ** 2)))
    return semipartial_23 

def calc_correlation(gpt_vec, mc_vec, behv_sim, cslb_vec):
    r_gpt_behav, r_cslb_behav, r_mc_behav, r_gpt_mc, r_cslb_gpt = None, None, None, None, None
    
    gpt_sim = get_similiarity_vector(gpt_vec, 'gpt')
    behv_sim = squareform(behv_sim, force='tovector', checks=False)

    # Calc Correlation
    r_gpt_behav = np.corrcoef(gpt_sim, behv_sim)[1][0]
    #r_gpt_behav = spearmanr(gpt_sim, behv_sim).correlation
    print('Correlation {} and {}: {:.4f}'.format('GPT', 'THINGS', r_gpt_behav))

    print('\n')
    if mc_vec is not None:
        mc_sim = get_similiarity_vector(mc_vec, 'mcrae')
        r_gpt_mc = np.corrcoef(gpt_sim, mc_sim)[1][0]
        
        print('Correlation {} and {}: {:.4f}'.format('GPT', 'Mc', r_gpt_mc))
        r_mc_behav = np.corrcoef(mc_sim, behv_sim)[1][0]
        #r_mc_behav = spearmanr(mc_sim, behv_sim).correlation
        print('Correlation {} and {}: {:.4f}'.format('Mc', 'THINGS', r_mc_behav))
        
        # variance partitioning analysis
        explained_variance_gpt = calc_semi_partial_correlation(r_gpt_behav, r_mc_behav, r_gpt_mc)
        print('unique variance GPT (partial out McRae): {:.4f}'.format(explained_variance_gpt ** 2))

        explained_variance_mc = calc_semi_partial_correlation(r_mc_behav, r_gpt_behav, r_gpt_mc)
        print('unique variance McRae (partial out GPT): {:.4f}'.format(explained_variance_mc ** 2))

        shared_variance = (r_mc_behav ** 2) - (explained_variance_mc ** 2)
        print('shared variance between GPT and McRae: {:.4f}'.format(shared_variance))
        print('\n')

    if cslb_vec is not None:
        cslb_sim = get_similiarity_vector(cslb_vec, 'cslb')
        r_cslb_behav = np.corrcoef(cslb_sim, behv_sim)[1][0]
        #r_cslb_behav = spearmanr(cslb_sim, behv_sim).correlation
        print('Correlation {} and {}: {:.4f}'.format('CSLB', 'THINGS', r_cslb_behav))

        r_cslb_gpt = np.corrcoef(cslb_sim, gpt_sim)[1][0]
        print('Correlation {} and {}: {:.4f}'.format('CSLB', 'GPT', r_cslb_gpt))

        explained_variance_csbl = calc_semi_partial_correlation(r_cslb_behav, r_gpt_behav, r_cslb_gpt)
        print('unique variance CSLB (partial out GPT): {:.4f}'.format(explained_variance_csbl ** 2))

        explained_variance_gpt = calc_semi_partial_correlation(r_gpt_behav, r_cslb_behav, r_cslb_gpt)
        print('unique variance GPT (partial out CSLB): {:.4f}'.format(explained_variance_gpt ** 2))
        
        shared_variance = (r_cslb_behav ** 2) - (explained_variance_csbl ** 2)
        print('shared variance between GPT and CSLB: {:.4f}'.format(shared_variance))
        print('\n')

    if cslb_vec is not None and mc_vec is not None:
        r_cslb_mc = np.corrcoef(cslb_sim, mc_sim)[1][0]
        print('Correlation {} and {}: {:.4f}'.format('CSLB', 'McRae', r_cslb_mc))

        explained_variance_mc = calc_semi_partial_correlation(r_mc_behav, r_cslb_behav, r_cslb_mc)
        print('unique variance McRae (partial out CSLB): {:.4f}'.format(explained_variance_mc ** 2))

        explained_variance_cslb = calc_semi_partial_correlation(r_cslb_behav, r_mc_behav, r_cslb_mc)
        print('unique variance CSLB (partial out McRae): {:.4f}'.format(explained_variance_cslb ** 2))
        
        shared_variance = (r_cslb_behav ** 2) - (explained_variance_cslb ** 2)
        print('shared variance between McRae and CSLB: {:.4f}'.format(shared_variance))

    return r_gpt_behav, r_cslb_behav, r_mc_behav, r_gpt_mc, r_cslb_gpt


def calc_correlation2(feature_norms_vec, things_similarity, method='pearson'):
    values = {
        "THINGS": squareform(things_similarity, force='tovector', checks=False)
    }    

    for name, feature_norm_vec in feature_norms_vec.items():
        similarity_vector = get_similiarity_vector(feature_norm_vec)
        values[name] = similarity_vector

    return pd.DataFrame(values).corr(method)

