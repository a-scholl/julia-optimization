import Plots
import LinearAlgebra: I
using PlotlyJS
import Random: rand
using BitIntegers
using BenchmarkTools

dim = 3
corners = Matrix(I*1.0, dim, dim)

function plot3d(weights)
    plot(scatter(
        x=weights[:,1],
        y=weights[:,2], 
        z=weights[:,3],
        mode="markers",
        type="scatter3d",
        marker=attr(
            size=4,
        )
    ))
end

plot3d(corners)

function sample_big_int(;num_bits)
    result = BigInt(0)
    unit_bit_size = 128
    num_complete_chunks = num_bits ÷ unit_bit_size
    num_excess_bits = num_bits % unit_bit_size
    chunks = rand(UInt128, num_complete_chunks)
    for chunk in chunks
        result <<= unit_bit_size
        result |= chunk
    end
    if num_excess_bits > 0
        result <<= num_excess_bits
        chunk = rand(UInt128)
        chunk >>= (unit_bit_size - num_excess_bits)
        result |= chunk
    end
    return result
end

y = sample_big_int(num_bits=64 * 3)

content(n::BigInt) = sqrt(n) / factorial(n-1)

height(n) = sqrt(n/(n-1))

function calculate_cube_edge_length(;n, float_bits=64)
    Vₙ₋₁ = content(n)
    denom = BigFloat(2) ^ (n * float_bits)
    return (Vₙ₋₁ / denom) ^ (1/(n-1))
end

function get_layer_indices(;y::BigInt, n::BigInt)
    ks = []
    for i in 1:(n-1)
        j = n - i + 1 
        V = content(j)
        H = height(j)
        k = ceil( ((y / V) ^ (1/(j-1))) * H )
        append!(ks, k)
        y = y - (((k-1)/H) ^ (j-1)) * V
        println(y)
    end
    return ks
end

ks = get_layer_indices(y=y, n=BigInt(40))