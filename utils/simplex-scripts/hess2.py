import pandas as pd
import numpy as np
import os
import time
import json
import sys
import pickle
from cmdstanpy import CmdStanModel
import argparse
import arviz as az
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from scipy.stats import norm, entropy
import json
import sys
import os.path
# sys.path.insert(1, '../../')
# from sample import sample
rng = np.random.default_rng(12345)

output_dir='/mnt/home/mjhajaria/ceph/sampling_results/simplex'
with open('data/dirichletsymmetric.json') as f:
    datajson = json.load(f)

n_repeat=100
import bridgestan as bs
bs.set_bridgestan_path('/mnt/home/mjhajaria/.bridgestan/bridgestan-2.0.0')



for transform in ['ALR']:
    for datakey in ['2', '5', '8']:
        if os.path.isfile(f"{output_dir}/{transform}/DirichletSymmetric/ad_cond_{datakey}_{n_repeat}.npy")==False:

            stan_filename=f'stan_models/{transform}_DirichletSymmetric.stan'
            with open(stan_filename, 'w') as f:
                f.write(f'#include target_densities/DirichletSymmetric.stan{os.linesep}#include transforms/simplex/{transform}.stan{os.linesep}')
                f.close()
            output_file_name=f'{output_dir}/{transform}/DirichletSymmetric/draws_{datakey}_{n_repeat}.nc'
            alpha=datajson[datakey]
            data={'alpha': alpha, 'N': len(alpha)}
            data = json.dumps(data)
            try:
                idata = az.from_netcdf(output_file_name)

                try:
                    bsmodel = bs.StanModel.from_stan_file(stan_filename, data, 
                    stanc_args=[f"--include-paths='/mnt/home/mjhajaria/transforms/'"],
                    make_args=[ "BRIDGESTAN_AD_HESSIAN=true", "STAN_THREADS=true"])
                    n=bsmodel.param_unc_num()
                    print(bsmodel.param_num(), n)
                    if 'BRIDGESTAN_AD_HESSIAN=true' in bsmodel.model_info():
                        cond_array=np.asarray([])
                        data= idata.posterior.y.values.reshape(400000,n)
                        x= list(rng.choice(400000, 40000, replace=False))
                        data=data[x]
                        for idx, row in tqdm(enumerate(data)):
                            theta = bsmodel.param_unconstrain(row)
                            lp, grad, hessian = bsmodel.log_density_hessian(theta)
                            try:
                                cond_array= np.append(cond_array, np.linalg.cond(hessian, 2))
                            except:
                                cond_array= np.append(cond_array, np.nan)

                        np.save(f"{output_dir}/{transform}/DirichletSymmetric/ad_cond_{datakey}_{n_repeat}.npy",cond_array)
                except:
                    print(f"AD failed for {transform} {datakey}")
            except OSError:
                print("file not found error")