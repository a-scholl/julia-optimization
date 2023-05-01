include("simplex.jl")
include("sampling.jl")
include("chromosome.jl")

import IterTools: partition

pop_size = 100
num_included = num_funds # TODO: allow for subset selection
num_bits = num_funds * 64
initial_ys = [sample_big_int(num_bits=num_bits) for _ in 1:pop_size]

initial_pop = [Chromosome(y, randperm(num_funds)[1:num_included], BigInt(num_funds)) for y in initial_ys]

function chr_objective_function(pop_chromosomes::Array{Chromosome}, asset_returns)
    pop_weights = reduce(hcat, get_weights_from_chromosome.(pop_chromosomes))
    obj_vals = objective(weights=pop_weights, returns=asset_returns)
    return obj_vals
end

obj = chr_objective_function(initial_pop, returns)

nondominated_idx = naive_pareto(obj)

Plots.scatter(obj[1,:], obj[2,:])
Plots.scatter!(obj[1,nondominated_idx], obj[2,nondominated_idx])


function optimization_step(population, asset_returns, num_bits)
    pop_size = length(population)
    new_pop = []
    for (idx1, idx2) in partition(1:length(population), 2)
        parent1, parent2 = population[[idx1, idx2]]
        append!(new_pop, single_point_crossover(parent1, parent2, num_bits))
    end
    population = copy(population)
    append!(population, new_pop)
    obj = chr_objective_function(population, asset_returns)
    nondom_levels = get_non_domination_levels(obj)
    surviving_idx = sortperm(nondom_levels)[1:pop_size]
    return (population[ surviving_idx], obj[:, surviving_idx])
end

new_pop = []
for (idx1, idx2) in partition(1:length(population), 2)
    parent1, parent2 = population[[idx1, idx2]]
    append!(new_pop, single_point_crossover(parent1, parent2, num_bits))
end

pop = [Chromosome(y, randperm(num_funds)[1:num_included], BigInt(num_funds)) for y in initial_ys]
obj = chr_objective_function(pop, returns)
plt = Plots.scatter(obj[1, :], obj[2, :])
num_epochs = 4
num_generations = 100
for e in 1:num_epochs
    println(e)
    for _ in 1:num_generations
        pop, obj = optimization_step(pop, returns, num_bits)
    end
    Plots.scatter!(plt, obj[1, :], obj[2, :])
end
plt