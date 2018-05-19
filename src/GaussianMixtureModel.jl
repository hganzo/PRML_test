### GaussianMixtureModel ###

# EM algo 
module GaussianMixtureModel
#Pkg.add("Distributions")
using Distributions
srand(2)
using PDMats

# global structure
export GW, BGMM, Gauss, GMM
# global function 
export sample_GMM, sample_data
# learn function
export learn_em

struct GW
    # Parameters of Gauss Wisahrt distribution
    beta::Float64
    m::Vector{Float64}
    nu::Float64
    W::Matrix{Float64}
end

struct BGMM
    # Parameters of Bayesian Gaussian Mixture Model 
    D::Int  # Dimension of data
    K::Int  # Mixted Components
    alpha::Vector{Float64} # related with components weight (Dirichlet parameter)
    Wcmp::Vector{GW} #Wishart parameters struct
end

struct Gauss
    # Parameters of Gauss Distribution
    mu::Vector{Float64} # Gassian mean
    Lambda::Matrix{Float64} # Gaussian covariance
end

struct GMM
    # Parameters of Gauss Mixture Model
    D::Int # data dimension
    K::Int # N components
    phi::Vector{Float64} # components parameter
    Gcmp::Vector{Gauss} # Gaussian parameter struct
end


# Wishart:
# D dim Gaussian prior (Lamba known)
# MvNormal:
# D dim Gaussian prior (mu known)


function sample_GMM(bgmm::BGMM)
# sampling gaussian parameters
#    srand(1)
    cmp = Vector{Gauss}()
    for c in bgmm.Wcmp
        Lambda = rand(Wishart(c.nu, PDMats.PDMat(Symmetric(c.W))))
        mu = rand(MvNormal(c.m, PDMats.PDMat(Symmetric(inv(c.beta*Lambda)))))
        push!(cmp,Gauss(mu,Lambda))
    end
    phi = rand(Dirichlet(bgmm.alpha)) 
    return GMM(bgmm.D, bgmm.K, phi, cmp)
end

function sample_data(gmm::GMM, N::Int)
# sampling the MvNormal data
#    srand(1)
    X = zeros(gmm.D, N) # data
    S = categorical_sample(gmm.phi, N) # X components
    for n in 1 : N
        k = indmax(S[:, n])
        X[:,n] = rand(MvNormal(gmm.Gcmp[k].mu,PDMats.PDMat(Symmetric(pinv(gmm.Gcmp[k].Lambda)))))
        end
    return X, S
end


# The case of 1 dim (N is blank)
categorical_sample(p::Vector{Float64}) = categorical_sample(p, 1)[:,1]

function categorical_sample(p::Vector{Float64}, N::Int)
    K = length(p)
    S = zeros(Int8, K, N)
    S_tmp = rand(Categorical(p), N)
    for k in 1 : K
        S[k,find(S_tmp.==k)] = 1
    end
    return S
end

function learn_em(X::Matrix{Float64}, K::Int, max_iter::Int)
    N = size(X, 2) # data dims
    D = size(X, 1)

    # initialize
    phi = rand(Dirichlet(100.0*ones(K)))
    S = categorical_sample(phi, N)    
    pi_k = sum(S,2)[:,1]/sum(S)
    cmp = Vector{Gauss}()
    for c in 1 : K
        mu = vec(mean(X[:,find(S[c,:].==1)], 2))
        Lambda = cov(X[:,find(S[c,:].==1)]') #/2.
#        println(mu,Lambda)
        push!(cmp, Gauss(mu, Lambda))
    end
    gmm=GMM(D, K , pi_k, cmp)
    gamma_kn = zeros(K,N)
    # initialize fin.
    calc_ELBO = -Inf
    local gmm_best::GMM, ELBO_best::Float64
    @time for iter in 1:max_iter
        for c in 1 : K
            gamma_kn[c,:] .= pi_k[c]*pdf(MvNormal(gmm.Gcmp[c].mu, PDMats.PDMat(Symmetric(pinv(gmm.Gcmp[c].Lambda)))), X)
        end
        
        if calc_ELBO<sum(log.(sum(gamma_kn,1))) 
            gmm_best=gmm
            ELBO_best=sum(log.(sum(gamma_kn,1)))
        end
        calc_ELBO=sum(log.(sum(gamma_kn,1)))
        
        gamma_kn .= gamma_kn ./ sum(gamma_kn,1)
        Nk=sum(gamma_kn,2)[:,1]

        # calc mu_new
        for c in 1 : K
            gmm.Gcmp[c].mu.=zeros(D,1)[:,1]
            for n in 1:N
                gmm.Gcmp[c].mu .+= gamma_kn[c,n]*X[:,n]
            end
            gmm.Gcmp[c].mu ./= Nk[c]
        
        end

        # calc Lambda_new
        for c in 1 : K
            gmm.Gcmp[c].Lambda.=zeros(D, D)
            for n in 1:N
                gmm.Gcmp[c].Lambda .+= gamma_kn[c,n] * (X[:,n].-gmm.Gcmp[c].mu)*(X[:,n].-gmm.Gcmp[c].mu)'
            end
            gmm.Gcmp[c].Lambda ./=  Nk[c]
        end
        pi_k .= Nk./sum(Nk)
    end
    return gmm,calc_ELBO, gmm_best,ELBO_best
end

end

