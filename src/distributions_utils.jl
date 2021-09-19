export GDistr, DistrIterator
export SMvNormal, SMvUniform, SUniform, SNormal
export score

import Distributions: rand, logpdf, extrema
import Bijectors: AbstractBijector, bijector
import StatsFuns

## simplify the distribution displays
Base.show(io::IO, d::Normal) = print(io, "Normal(μ=$(d.μ), σ=$(d.σ))")
Base.show(io::IO, ::Type{<:Normal}) = print(io, "Normal{...}")

Base.show(io::IO, d::Uniform) = print(io, "Uniform(lb=$(d.a), ub=$(d.b))")
Base.show(io::IO, ::Type{<:Uniform}) = print(io, "Uniform{...}")


"""
    score(distribution, value) -> log_p_score
Returns a score equals to `logpdf` plus a constant. During optimization, this can be 
used in place of `logpdf` to reduce the computational cost.
"""
function score end

score(d::Distribution, x) = logpdf(d, x)

score(d::Normal, x) = let
    (; μ, σ) = d
    -abs2((x - μ) / σ)/2 - log(σ)
end

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

logpdf(diter::DistrIterator, xs) = let
    if diter isa DistrIterator{<:NamedTuple} && xs isa NamedTuple
        @assert keys(diter.distributions) == keys(xs) "keys do not match:\n\
            distributions: $(diter.distributions)\nvalues:$xs"
    end
    sum(logpdf(d, x)::Real for (d, x) in zip(diter.distributions, xs))::Real
end

score(diter::DistrIterator, xs) = let
    if diter isa DistrIterator{<:NamedTuple} && xs isa NamedTuple
        @assert keys(diter.distributions) == keys(xs) "keys do not match:\n\
            distributions: $(diter.distributions)\nvalues:$xs"
    end
    sum(score(d, x)::Real for (d, x) in zip(diter.distributions, xs))::Real
end

Base.show(io::IO, di::DistrIterator) = 
    if di.distributions isa AbstractVector
        print(io, "DistrIterator([$(join(di.distributions, ","))])")
    else
        print(io, "DistrIterator($(di.distributions))")
    end

Base.show(io::IO, ::Type{<:DistrIterator}) = print(io, "DistrIterator{...}")

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
    BijectorIterator(map(bijector, di.distributions))
end

function Bijectors.inv(bit::BijectorIterator)
    BijectorIterator(map(inv, bit.bijectors))
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

SNormal(μ, σ) = let 
    σ >= zero(σ) || throw(OverflowError("Normal σ = $σ"))
    Normal(μ, σ)
end
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

score(d::SMvUniform, xs) = 0.0

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

"""
Like a normal distribution, but always warp values to [0, 2π). 
"""
struct CircularNormal{T1, T2} <: ContinuousUnivariateDistribution
    μ::T1
    σ::T2
end
Base.show(io::IO, ::Type{<:CircularNormal}) = print(io, "CircularNormal{...}")
Base.show(io::IO, d::CircularNormal) = 
    print(io, "CircularNormal(μ=$(d.μ), σ=$(d.σ))")

extrema(::CircularNormal) = (0.0, 2π)
rand(rng::AbstractRNG, d::CircularNormal) = 
    warp_angle(rand(rng, truncated(Normal(0.0, d.σ), -π, π)) + d.μ)
logpdf(d::CircularNormal, x) = let
    (0 <= x <= 2π) || return -Inf
    dis = warp_angle(x - d.μ)
    dis = min(dis, 2π-dis)
    logpdf(truncated(Normal(0.0, d.σ), -π, π), dis)
end
score(d::CircularNormal, x) = let
    (0 <= x <= 2π) || return -Inf
    dis = warp_angle(x - d.μ)
    dis = min(dis, 2π-dis)
    score(Normal(0.0, d.σ), dis)
end

# copied from https://github.com/oxinabox/ProjectManagement.jl/blob/da3de128ebc031b695bcb1795b53bcfeba617d87/src/timing_distributions.jl
"""
    PertBeta(a, b, c) <: ContinuousUnivariateDistribution
The [PERT Beta distribution](https://en.wikipedia.org/wiki/PERT_distribution).
 - `a`: the minimum value of the support
 - `b`: the mode
 - `c`: the maximum value of the support
"""
struct PertBeta{T<:Real} <: ContinuousUnivariateDistribution
    a::T # min
    b::T # mode
    c::T # max
end

function beta_dist(dd::PertBeta)
    α = (4dd.b + dd.c - 5dd.a)/(dd.c - dd.a)
    β = (5dd.c - dd.a - 4dd.b)/(dd.c - dd.a)
    return Beta(α, β)
end

# Shifts x to the domain of the beta_dist
input_shift(dd::PertBeta, x) = (x - dd.a)/(dd.c - dd.a)
# Shifts y from the domain of the beta_dist
output_shift(dd::PertBeta, y) = y*(dd.c - dd.a) + dd.a

Distributions.mode(dd::PertBeta) = dd.b
Base.minimum(dd::PertBeta) = dd.a
Base.maximum(dd::PertBeta) = dd.c
Statistics.mean(dd::PertBeta) = (dd.a + 4dd.b + dd.c)/6
Statistics.var(dd::PertBeta) = ((mean(dd) - dd.a) * (dd.c - mean(dd)))/7
Distributions.insupport(dd::PertBeta, x) = dd.a < x < dd.c

for f in (:skewness, :kurtosis)
    @eval Distributions.$f(dd::PertBeta) = $f(beta_dist(dd))
end
for f in (:pdf, :cdf, :logpdf, :score)
    @eval Distributions.$f(dd::PertBeta, x::Real) = $f(beta_dist(dd), input_shift(dd, x))
end

Statistics.quantile(dd::PertBeta, x) = output_shift(dd, quantile(beta_dist(dd), x))
Base.rand(rng::AbstractRNG, dd::PertBeta) = output_shift(dd, rand(rng, beta_dist(dd)))