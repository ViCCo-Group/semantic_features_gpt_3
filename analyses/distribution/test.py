
gpt_contigence = gpt_mcrae_label_df.groupby('label').count()
cslb_contigence = cslb_label_df.groupby('label').count()

a = scipy.stats.chisquare(gpt_contigence, cslb_contigence)
