include("simplex.jl")

struct Chromosome
    y::BigInt
    included::Array{Integer}
    n::Integer
end


function get_weights_from_chromosome(chromosome::Chromosome)
    layer_idx = get_layer_indices(y=chromosome.y, n=BigInt(chromosome.n))
    coords = get_coords_from_layer_indices(layer_idx)
    weights = zeros(chromosome.n)
    weights[chromosome.included] = coords
    return weights 
end


function create_mask(num_bits)
    result = BigInt(0)
    unit_bit_size = 128
    num_complete_chunks = num_bits ÷ unit_bit_size
    num_excess_bits = num_bits % unit_bit_size
    chunks = rand(UInt128, num_complete_chunks)
    for chunk in chunks
        result <<= unit_bit_size
        result |= typemax(UInt128)
    end
    if num_excess_bits > 0
        result <<= num_excess_bits
        result |= typemax(UInt128)
    end
    return result
end


function single_point_crossover(parent1::Chromosome, parent2::Chromosome, num_bits) 
    mask = create_mask(num_bits)
    num_lower_bits = rand(1:num_bits)
    upper_mask = mask & (mask << num_lower_bits)
    lower_mask = mask & (mask >> (num_bits - num_lower_bits))
    c1_y = (parent1.y & upper_mask) | (parent2.y & lower_mask)
    c2_y = (parent2.y & upper_mask) | (parent1.y & lower_mask)
    
    # TODO: implement included OX crossover
    
    return [Chromosome(c1_y, parent1.included, parent1.n),
            Chromosome(c2_y, parent2.included, parent2.n)]
end 