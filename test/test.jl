module Test
    include("../src/sample.jl")
    import .Sample

    print(Sample.sample())

end