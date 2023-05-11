### Correlation with THINGS on categories

for category, category_concepts, category_plot_name in categories:
    print(f'Category: {category}')
    gpt_vec_cat = feature_norms_vec['GPT3-davinci-McRae'].loc[category_concepts]
    gpt_sim = cosine_similarity(gpt_vec_cat, gpt_vec_cat)
    #sns.heatmap(gpt_sim, yticklabels=True, xticklabels=True)
    mc_vec_cat = feature_norms_vec['McRae'].loc[category_concepts]
    cslb_vec_cat = feature_norms_vec['CSLB'].loc[category_concepts]
    behv_sim_cat = match_behv_sim(behv_sim, category_concepts, load_sorting())
    
    sns.heatmap(behv_sim_cat, yticklabels=True, xticklabels=True)
    r_gpt_behav, r_cslb_behav, r_mc_behav, r_gpt_mc, r_cslb_gpt = calc_correlation(gpt_vec_cat, mc_vec_cat, behv_sim_cat, cslb_vec_cat, None)
    print(f'Corr GPT: {r_gpt_behav}')
    print(f'Corr CSLB: {r_cslb_behav}')
    print(f'Corr McRae: {r_mc_behav}')
    