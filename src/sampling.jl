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
        cor_mat[i, j] = cor_mat[j, i] = new_cor
    end
end 

Σ = diagm(vols) * cor_mat * diagm(vols)

mvn = Distributions.MultivariateNormal(μ, Σ)
num_return_samples = 1000
returns = rand(mvn, num_return_samples)


# weight_distribution = Distributions.Dirichlet(num_funds, 1)
# num_weight_samples = 5000
# rand_weights = rand(weight_distribution, num_weight_samples)



function compute_portfolio_returns(weights, returns)
    return transpose(transpose(weights) * returns)
end

# portfolio_returns = compute_portfolio_returns(rand_weights, returns)

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
    return vcat(-threshold_likelihood, portfolio_vol)
end

#obj = objective(weights=rand_weights, returns=returns)

#plt = Plots.scatter(obj[1, :], obj[2, :], xlabel="Return Threshold Likelihood", ylabel="Portfolio Volatility", label="dominated")

function dominates(x, y)
    strict_inequality_found = false
    for i in eachindex(x)
        y[i] < x[i] && return false
        strict_inequality_found |= x[i] < y[i]
    end
    return strict_inequality_found
end


function naive_pareto(ys::Matrix{Float64})
    nondominated_idx = Vector{Int64}()
    for (i, y) in enumerate(eachcol(ys))
        if !any(dominates(y′, y) for y′ in eachcol(ys))
            push!(nondominated_idx, i)
        end
    end
    return nondominated_idx
end

#par_weights, par_obj = naive_pareto(rand_weights, obj)
#@profview par_weights, par_obj = naive_pareto(rand_weights, obj)
# Plots.scatter!(par_obj[1, :], par_obj[2, :], label="nondominated")

function get_non_domination_levels(ys)
    ys_col = eachcol(ys)
    L, m = 0, length(ys_col)
    levels = zeros(Int, m)
    while minimum(levels) == 0
        L += 1
        for (i, y) in enumerate(ys_col)
            if levels[i] == 0 
                dominator_found = false
                for j in 1:m
                    if (levels[j] == 0 || levels[j] == L) && dominates(ys_col[j], y)
                        dominator_found = true
                        break
                    end
                end
                (!dominator_found) && (levels[i] = L)
            end
        end
    end
    return levels
end 

#=
@profview levels = @inbounds get_non_domination_levels(obj)
@btime  levels = @inbounds get_non_domination_levels(obj)

max_level = 5
for level in 1:max_level
    idx = findall(==(level), levels)
    print(level)
    # print(idx)
    Plots.scatter!(plt, obj[1, idx], obj[2, idx], label="Level $level")
end
plt

num_parents = 200
num_generations = 1000
every = num_generations / 5
population = rand(weight_distribution, num_parents)
obj = objective(weights=population, returns=returns)
plt = Plots.scatter(obj[1, :], obj[2, :], xlabel="Return Threshold Likelihood", ylabel="Portfolio Volatility", label="dominated")


for gen in 1:num_generations
    new_population = similar(population)
    for i in 1:num_parents
        new_population[:, i] = shuffle(population[:, i])
    end
    population = hcat(population, new_population)
    obj = objective(weights=population, returns=returns)
    levels = get_non_domination_levels(obj)
    surviving_idx = sortperm(levels)[1:num_parents]
    population = population[:, surviving_idx]
    obj = obj[:, surviving_idx]
    if gen % every == 0
        par_weights, par_obj = naive_pareto(population, obj)
        Plots.scatter!(plt, par_obj[1, :], par_obj[2, :], label="Generation $gen")
    end
end 


plt
=#