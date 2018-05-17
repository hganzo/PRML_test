### GaussianMixtureModel ###

# EM algo 
module GaussianMixtureModel
#Pkg.add("Distributions")
using Distributions
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
        X[:,n] = rand(MvNormal(gmm.Gcmp[k].mu, PDMats.PDMat(Symmetric(inv(gmm.Gcmp[k].Lambda)))))
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
    
    println("test")
    N = size(X, 2) # data dims
# initialize
    phi = rand(Dirichlet(100.0*ones(K)))
    S = categorical_sample(phi,N)    
    pi_k = sum(S,2)/sum(S)
    cmp = Vector{Gauss}()
    for c in 1:3
        mu = vec(mean(X[:,find(S[c,:].==1)],2))
        Lambda = cov(X[:,find(S[c,:].==1)]')
        println(mu,Lambda)
        push!(cmp, Gauss(mu, Lambda))
    end
    gmm=GMM(size(X,1), K , pi_k, cmp)
# initialize fin.

#    gammas=zeros(max_iter,3)
#    mus=zeros(max_iter,3)
#    GMM(size(X,1), K, )

#    for i in 1 : max_iter
#        for c in gmm.Gcmp
#            gammas[1,i]=pdf(gmm,X)
#        end
#    end

   
    return 111
end


end

