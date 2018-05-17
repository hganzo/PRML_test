Path_MySrc="../src/"
#
push!(LOAD_PATH,Path_MySrc)
# include -> can't use "struct BGMM" on 2 times ...unknown? 
import GaussianMixtureModel
using Plots
gr()

function test()
    D = 2
    K = 3
    alpha = 100.0 * ones(K) # related with componets weights 
    beta = 0.01 # ?
    m = zeros(D)
    nu = D + 3.0
    W = eye(D)
    cmp = [GaussianMixtureModel.GW(beta, m, nu, W) for _ in 1:K]
    bgmm = GaussianMixtureModel.BGMM(D, K, alpha, cmp)

    N = 3000
    gmm = GaussianMixtureModel.sample_GMM(bgmm)
    X, S = GaussianMixtureModel.sample_data(gmm,N)

    max_iter = 10

    test = GaussianMixtureModel.learn_em(X, K, max_iter)
    println(test)
#    plot(X[1,:],X[2,:],marker=:circle,linewidth=0)
end

test()



