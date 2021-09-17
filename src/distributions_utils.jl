export GDistr, DistrIterator
export SMvNormal, SMvUniform, SUniform, SNormal

import Distributions: rand, logpdf
import Bijectors: AbstractBijector, bijector
import StatsFuns


"""
A wrapper of a sequence of distributions. This can be used to sample or compute
the corresponding logpdf of a sequence of values.

The type parameter `Iter` should be an iterators of distributions, e.g., 
`Vector{Normal}` or `Tuple{Uniform, MvNormal}`.

## Examples
```julia
julia> di = DistrIterator(fill(Normal(), 4))
Main.SEDL.DistrIterator{Vector{Distributions.Normal{Float64}}}(...)

julia> xs = rand(di)
4-element Vector{Float64}:
 -0.49010706913108426
  0.04667852819261214
  1.4516555944579874
 -1.3288896894564055

julia> logpdf(di, xs)
-5.733571930754293

julia> nested = DistrIterator((di, di));
julia> rand(nested)
([-0.3008128094801595, 0.7595305727349723, 0.0731633860379036, -0.15532821730030588], [-0.5856011152398879, -1.3111512044326252, 1.1532299087076099, 1.5392183048453147])

julia> logpdf(nested, rand(nested))
-13.87393093934596

# Some bijector interface is also supported.
julia> di = DistrIterator(fill(Uniform(0, 1), 4))
Main.SEDL.DistrIterator{Vector{Distributions.Uniform{Float64}}}(Distributions.Uniform{Float64}[Distributions.Uniform{Float64}(a=0.0, b=1.0), Distributions.Uniform{Float64}(a=0.0, b=1.0), Distributions.Uniform{Float64}(a=0.0, b=1.0), Distributions.Uniform{Float64}(a=0.0, b=1.0)])

julia> xs = rand(di);

julia> di_bj = bijector(di);

julia> inv(di_bj)(di_bj(xs)) ≈ xs
true
```
"""
struct DistrIterator{Iter}
    distributions::Iter

    DistrIterator(distributions::Iter) where Iter = begin
        foreach(distributions) do d
            d isa GDistr || throw(
                ArgumentError("Expect value of type `$GDistr`, but got: $d"))
        end
        new{Iter}(distributions)
    end
end

rand(rng::Random.AbstractRNG, diter::DistrIterator) = 
    map(d -> rand(rng, d), diter.distributions)
logpdf(diter::DistrIterator, xs) = 
    sum(logpdf(d, x) for (d, x) in zip(diter.distributions, xs))

Base.show(io::IO, di::DistrIterator) = 
    if di.distributions isa AbstractVector
        print(io, "DistrIterator([$(join(di.distributions, ","))])")
    else
        print(io, "DistrIterator($(di.distributions))")
    end

"""
An iterator of bijectors, used to transform [`DistrIterator`](@ref).
"""
struct BijectorIterator{Iter} <: AbstractBijector
    bijectors::Iter
    BijectorIterator(bjs::Iter) where Iter = begin
        eltype(bjs) <: AbstractBijector || throw(
            ArgumentError("Expect an iterator of bijectors, but got $bjs."))
        new{Iter}(bjs)
    end
end

(bit::BijectorIterator)(x) = zipmap(bit.bijectors, x)

function Bijectors.bijector(di::DistrIterator)
    BijectorIterator(bijector.(di.distributions))
end

function Bijectors.inv(bit::BijectorIterator)
    BijectorIterator(inv.(bit.bijectors))
end

"""
Distributions that returns static vectors.
"""
abstract type StaticDistr end

"""
Generalized distributions that support sampling (and evaluating the probability of) 
structured data. 
"""
const GDistr = Union{Distribution, StaticDistr, DistrIterator}

SMvNormal(μ::AbstractVector, σ::Real) = let 
    n = length(μ)
    SMvNormal(SVector{n}(μ), SVector{n}(σ for _ in 1:n))
end
SMvNormal(μ::AbstractVector, σ::AbstractVector) = let 
    n = length(μ)
    @assert length(σ) == n
    SMvNormal(SVector{n}(μ), SVector{n}(σ))
end
SMvNormal(μ::SVector{n}, σ::SVector{n}) where n = let
    normals = SNormal.(μ, σ)
    DistrIterator(normals)
end

const SNormal = Normal
const SUniform = Uniform

struct SMvUniform{n, T} <: StaticDistr
    uniforms::StaticVector{n, Uniform{T}}
end

SMvUniform(ranges::Tuple{Real, Real}...) = let
    uniforms = map(ranges) do (l, u) 
        SUniform(l, u)
    end
    SMvUniform(SVector{length(ranges)}(uniforms))
end

rand(rng::Random.AbstractRNG, d::SMvUniform) = let
    map(d -> rand(rng, d), d.uniforms)
end

logpdf(d::SMvUniform, xs) =
    sum(logpdf(d, x) for (d, x) in zip(d.uniforms, xs))

function Bijectors.bijector(d::SMvUniform)
    BijectorIterator(bijector.(d.uniforms))
end

# This fixes the numerical error when y is a Dual and has Inf partial derivatives.
function Bijectors.truncated_invlink(y, a, b)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    v = if lowerbounded && upperbounded
        (b - a) * StatsFuns.logistic(y) + a
    elseif lowerbounded
        exp(y) + a
    elseif upperbounded
        b - exp(y)
    else
        y
    end
    fix_nan_dual(v)
end