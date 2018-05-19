Path_MySrc="src/"
#
push!(LOAD_PATH,Path_MySrc)
# include -> can't use "struct BGMM" on 2 times ...unknown?

import GaussianMixtureModel
#Pkg.add("Plots")
using Plots
#Pkg.add("GR")
#Plots.gr()

function test()
    D = 2
    K = 5
    alpha = 100.0 * ones(K) # related with componets weights 
    beta = 0.02 # ?
    m = zeros(D)
    nu = D + 3.0
    W = eye(D)
    cmp = [GaussianMixtureModel.GW(beta, m, nu, W) for _ in 1:K]
    bgmm = GaussianMixtureModel.BGMM(D, K, alpha, cmp)

    N = 5000
    gmm = GaussianMixtureModel.sample_GMM(bgmm)
    X, S = GaussianMixtureModel.sample_data(gmm,N)

    max_iter = 10
    gmm,last_ELBO,gmm_best,ELBO_best = GaussianMixtureModel.learn_em(X, K, max_iter)
    println(gmm_best.Gcmp[1].mu)
    println(gmm_best.Gcmp[2].mu)
    println(gmm_best.Gcmp[3].mu)
    println(gmm_best.Gcmp[4].mu)
    println(gmm_best.Gcmp[5].mu)
    return X, gmm,gmm_best
end

X,gmm,gmm_best=test()
scatter(X[1,:],X[2,:],title="test"); savefig("test.png")
mus=[gmm.Gcmp[1].mu[1] gmm.Gcmp[1].mu[2];
     gmm.Gcmp[2].mu[1] gmm.Gcmp[2].mu[2];
     gmm.Gcmp[3].mu[1] gmm.Gcmp[3].mu[2];
     gmm.Gcmp[4].mu[1] gmm.Gcmp[4].mu[2];
     gmm.Gcmp[5].mu[1] gmm.Gcmp[5].mu[2]]
mus2=[gmm_best.Gcmp[1].mu[1] gmm_best.Gcmp[1].mu[2];
      gmm_best.Gcmp[2].mu[1] gmm_best.Gcmp[2].mu[2];
      gmm_best.Gcmp[3].mu[1] gmm_best.Gcmp[3].mu[2];
      gmm_best.Gcmp[4].mu[1] gmm_best.Gcmp[4].mu[2];
      gmm_best.Gcmp[5].mu[1] gmm_best.Gcmp[5].mu[2]]
scatter!(mus[:,1],mus[:,2])
scatter!(mus2[:,1],mus2[:,2])
savefig("test4.png")



