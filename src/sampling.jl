import Distributions
using LinearAlgebra
using Match
using Random
import Plots
import Statistics: std
import Evolutionary
using BenchmarkTools

Random.seed!(123)

num_funds = 9
vols = repeat([0.10, 0.05, 0.03], 3)
sharpe = 0.8
μ = vols * sharpe 
cor_mat = Matrix(I*1.0, num_funds, num_funds)
for i = 1:num_funds
    for j = (i+1):num_funds
        new_cor = @match (i, j) begin
        (1:3, 1:3) => 0.7
        (4:6, 4:6) => 0.4
        (7:9, 7:9) => 0.1
        _          => continue
        end 
        print(i)
        cor_mat[i, j] = cor_mat[j, i] = new_cor
    end
end 

Σ = diagm(vols) * cor_mat * diagm(vols)

mvn = Distributions.MultivariateNormal(μ, Σ)
num_return_samples = 1000
returns = rand(mvn, num_return_samples)


weight_distribution = Distributions.Dirichlet(num_funds, 1)
num_weight_samples = 50000
rand_weights = rand(weight_distribution, num_weight_samples)

function compute_portfolio_returns(weights, returns)
    return transpose(transpose(weights) * returns)
end

portfolio_returns = compute_portfolio_returns(rand_weights, returns)

function compute_portfolio_volatility(portfolio_returns)
    ncol = size(portfolio_returns)[2]
    result = Matrix(undef, 1, ncol)
    for i in 1:ncol
        result[i] = std(portfolio_returns[:, i])
    end
    return result
end

function compute_return_threshold_likelihood(
    ;
    portfolio_returns,
    threshold
)
    (nrow, ncol) = size(portfolio_returns)
    result = Matrix(undef, 1, ncol)
    for i in 1:ncol
        result[i] = sum(x-> (x > threshold), portfolio_returns[:, i]) / nrow
    end
    return result
end


function objective(;weights, returns)::Matrix{Float64}
    portfolio_returns = compute_portfolio_returns(weights, returns) 
    threshold_likelihood = compute_return_threshold_likelihood(
        portfolio_returns=portfolio_returns,
        threshold=0.10,
    )
    portfolio_vol = compute_portfolio_volatility(portfolio_returns)
    print(size(portfolio_returns))
    print(size(portfolio_vol))
    return vcat(-threshold_likelihood, portfolio_vol)
end

obj = objective(weights=rand_weights, returns=returns)

Plots.scatter(obj[1, :], obj[2, :], xlabel="Return Threshold Likelihood", ylabel="Portfolio Volatility", label="dominated")

function dominates2(x, y)
    strict_inequality_found = false
    for i in eachindex(x)
        y[i] < x[i] && return false
        strict_inequality_found |= x[i] < y[i]
    end
    return strict_inequality_found
end


function naive_pareto(xs::Matrix{Float64}, ys::Matrix{Float64})
    nondominated_idx = Vector{Int64}()
    for (i, y) in enumerate(eachcol(ys))
        if !any(dominates2(y′, y) for y′ in eachcol(ys))
            push!(nondominated_idx, i)
        end
    end
    return (xs[:, nondominated_idx], ys[:, nondominated_idx])
end

@time par_weights, par_obj = naive_pareto(rand_weights, obj)
@btime par_weights, par_obj = naive_pareto(rand_weights, obj)
Plots.scatter!(par_obj[1, :], par_obj[2, :], label="nondominated")

