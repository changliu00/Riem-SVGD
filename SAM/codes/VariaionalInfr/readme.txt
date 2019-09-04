# MATLAB codes for (full-batch) variational baselines on SAM inference
# Modified by Chang Liu (chang-li14@mails.tsinghua.edu.cn)
  based on the MATLAB codes by Reisinger et al.
  (Spherical Topic Models, ICML-14; joeraii@cs.utaxes.edu)
  for the mean-field variational inference of the SAM model.

# Files with name containing sub-string "CompWithSmp" is our added files.

# "CompWithSmp_diff.m" is the entrance for running
  variational inference on "20News-different" dataset.
# Perplexity evaluation of trained models is done in "../SAM_GMC_seq/", using the command
  "./samgmc ts [model_filename] settings_vi.txt [other_options]".

