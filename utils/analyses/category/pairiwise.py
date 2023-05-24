from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
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
   
def calc_sim(gpt_vec, cslb_vec, mc_vec, categories):
    sims_gpt = {}
    sims_cslb = {}
    sims_mc = {}

    for category, category_concepts, category_plot_name in categories:
        out_of_category = []
        for new_category, new_category_concepts, new_category_plot_name in categories:
            if new_category != category:
                out_of_category += new_category_concepts
        print(category)
        sims_gpt[category] = []
        sims_cslb[category] = []
        sims_mc[category] = []

        for i, concept1 in enumerate(category_concepts):
            for concept2 in category_concepts[i+1:]:
                sim_gpt = calc_pair_sim(gpt_vec, concept1, concept2, out_of_category)
                sim_cslb = calc_pair_sim(cslb_vec, concept1, concept2, out_of_category)
                sim_mc = calc_pair_sim(mc_vec, concept1, concept2, out_of_category)

                sims_gpt[category].append(sim_gpt)
                sims_cslb[category].append(sim_cslb)
                sims_mc[category].append(sim_mc)
    
        
    return sims_gpt, sims_cslb, sims_mc

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def set_violin_color(parts, color):
    for pc in parts['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.3)
    for part in ['cbars', 'cmeans', 'cmins', 'cmaxes']:
        parts[part].set_color(color)
        parts[part].set_alpha(0.5)

def plot_violin(ax, sims_gpt, sims_cslb, sims_mc, categories):
    ticks = []
    data_gpt = []
    data_cslb = []
    data_mc = []

    for category, category_concepts, category_plot_name in categories:
        ticks.append(f'{category_plot_name} \n n={len(category_concepts)}')
        data_gpt.append(sims_gpt[category])
        data_cslb.append(sims_cslb[category])
        data_mc.append(sims_mc[category])

    bpl_mc = ax.violinplot(data_mc, positions=np.array(range(len(data_mc)))*2.0-0.5, widths=0.4, showmeans=True)
    bpl_cslb = ax.violinplot(data_cslb, positions=np.array(range(len(data_cslb)))*2.0, widths=0.4, showmeans=True)
    bpl_gpt= ax.violinplot(data_gpt, positions=np.array(range(len(data_gpt)))*2.0+0.5,widths=0.4, showmeans=True)

    set_violin_color(bpl_gpt, '#D7191C') 
    set_violin_color(bpl_cslb, '#2C7BB6')
    set_violin_color(bpl_mc, '#006400')

    ax.plot([], c='#006400', label='McRae')
    ax.plot([], c='#2C7BB6', label='CSLB')
    ax.plot([], c='#D7191C', label='GPT')

    ax.legend()

    ax.set_xticks(range(0, len(ticks) * 2, 2))
    ax.set_xticklabels(ticks)
    ax.set_xlim(-2, len(ticks)*2)
    ax.set_ylabel('within-between similarity')
    ax.set_xlabel('category with number of concepts')