export GDistr, DistrIterator
export SMvNormal, SMvUniform, SUniform, SNormal, CircularNormal
export PertBeta, DistrTransform, OptionalDistr
export score

import Distributions: rand, logpdf, extrema, mean
import Bijectors: AbstractBijector, bijector
import StatsFuns

## simplify the distribution displays
Base.show(io::IO, d::Normal) = print(io, "Normal(μ=$(d.μ), σ=$(d.σ))")
Base.show(io::IO, ::Type{<:Normal}) = print(io, "Normal{...}")

Base.show(io::IO, d::Uniform) = print(io, "Uniform(lb=$(d.a), ub=$(d.b))")
Base.show(io::IO, ::Type{<:Uniform}) = print(io, "Uniform{...}")


"""
    log_score(distribution, value) -> log_p_score
Returns a score equals to `logpdf` plus a constant. During optimization, this can be 
used in place of `logpdf` to reduce the computational cost.
"""
function log_score end

function log_score(d::Distribution, x, ::Type{T})::T where T 
    logpdf(d, x)
end

function log_score(d::Dirac, x, ::Type{T})::T where T
    zero(T)
end

function log_score(d, x, tv::Val{T}) where T
    log_score(d, x, T)
end

function log_score(d::Normal, x, ::Type{T})::T where T 
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
    core::Iter

    DistrIterator(distributions::Iter) where Iter = begin 
        if Iter <: AbstractVector
            eltype(distributions) <: GDistr || throw(ArgumentError(
                "Expect value of type `$GDistr`, but got eltype: $(eltype(distributions))"
            ))
        else
            all(map(d -> d isa GDistr, distributions)) ||
                foreach(distributions) do d
                    d isa GDistr || throw(
                        ArgumentError("Expect value of type `$GDistr`, but got: $d"))
                end
        end
        new{Iter}(distributions)
    end
end

rand(rng::Random.AbstractRNG, diter::DistrIterator) = 
    map(d -> rand(rng, d), diter.core)

logpdf(diter::DistrIterator, xs) = let
    if diter isa DistrIterator{<:NamedTuple} && xs isa NamedTuple
        @assert keys(diter.core) == keys(xs) "keys do not match:\n\
            distributions: $(diter.core)\nvalues:$xs"
    end
    sum(map(logpdf, diter.core, xs))::Real
end

mean(diter::DistrIterator) = map(mean, diter.core)

@generated function sum2_static(
    f, xs, ys, len::Val{N}
) where {T, N}
    if N == 0
        :(0.0)
    else
        exs = map(i -> :(f(xs[$i], ys[$i])), 1:N)
        Expr(:call, :+, exs...)
    end
end

# @generated function log_score_static(
#     distributions, values, type::Type{T}, len::Val{N}
# ) where {T, N}
#     if N == 0 
#         :(zero($T))
#     else
#         exs = map(1:N) do i 
#             :(log_score(distributions[$i], values[$i], $T)::$T)
#         end
#         Expr(:call, :+, exs...)
#     end
# end

function log_score_static(
     distributions, values, type::Val{T}, len::Val{N},
) where {T, N}
    sum(map((d, v) -> log_score(d, v, type) , distributions, values))
    # sum(map((d, v) -> logpdf(d, v)::Real, distributions, values))
end

function log_score(diter::DistrIterator, xs, ::Type{T})::T where T
    if diter isa DistrIterator{<:NamedTuple} && xs isa NamedTuple
        @assert keys(diter.core) == keys(xs) "keys do not match:\n\
            distributions: $(diter.core)\nvalues:$xs"
    end
    @assert length(diter.core) == length(xs)

    if xs isa Union{Tuple, NamedTuple, StaticArray}
        log_score_static(diter.core, xs, Val{T}(), Val{length(xs)}())
    else
        s::T = 0
        for (d, x) in zip(diter.core, xs)
            s += log_score(d, x, T)::T
        end
        s
    end
end

Base.show(io::IO, di::DistrIterator) = 
    if di.core isa AbstractVector
        print(io, "DistrIterator([$(join(di.core, ","))])")
    else
        print(io, "DistrIterator($(di.core))")
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
    BijectorIterator(map(bijector, di.core))
end

function Bijectors.inv(bit::BijectorIterator)
    BijectorIterator(map(inv, bit.bijectors))
end

"""
Parent type for the extended distributions.
"""
abstract type ExtDistr end

"""
Generalized distributions that support sampling (and evaluating the probability of) 
structured data. 
"""
const GDistr = Union{Distribution, ExtDistr, DistrIterator}

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

struct SMvUniform{n, T} <: ExtDistr
    uniforms::StaticVector{n, Uniform{T}}
end

Base.eltype(::SMvUniform{n, T}) where {n, T} = T

SMvUniform(ranges::Tuple{Real, Real}...) = let
    uniforms = map(ranges) do (l, u) 
        SUniform(l, u)
    end
    SMvUniform(SVector{length(ranges)}(uniforms))
end

rand(rng::Random.AbstractRNG, d::SMvUniform) = let
    map(d -> rand(rng, d), d.uniforms)
end

logpdf(d::SMvUniform, xs) = sum(map(logpdf, d.uniforms, xs))

mean(d::SMvUniform) = map(mean, d.uniforms)


eltype(Normal(Dual(1.0)))
function log_score(d::SMvUniform, xs, ::Type{T})::T where T
    zero(T)
end

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
    CircularNormal(μ::T1, σ::T2) where {T1, T2} = new{T1, T2}(warp_angle(μ), σ)
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
function log_score(d::CircularNormal, x, ::Type{T})::T where T
    (0 <= x <= 2π) || return -Inf
    dis = warp_angle(x - d.μ)
    dis = min(dis, 2π-dis)
    log_score(Normal(0.0, d.σ), dis, T)
end

"""
An distribution wrapper that optionally samples the wrapped distribution. It 
first samples from a Bernoulli with parameter `p`, and depending on the outcome,
either samples from `core` or samples `missing`.

## Examples
```
julia> od = OptionalDistr(0.5, Normal())
       [rand(od) for _ in 1:10]
10-element Vector{Union{Missing, Float64}}:
-0.6809494860705712
    missing
-0.4665500454017262
0.541651591598847
    missing
    missing
-0.2091150884206728
-1.7821543135478637
-0.2681990522044774
    missing
```
"""
struct OptionalDistr{R<:Real, Core<:GDistr} <: ExtDistr
    p::R
    core::Core
    OptionalDistr(p::R, core::Core) where {R, Core} = begin
        0 <= p <= 1 || throw(ArgumentError(
            "The data probability p = $p. It should be between 0 and 1."))
        new{R, Core}(p, core)
    end
end

function rand(rng::AbstractRNG, d::OptionalDistr)
    b = rand(rng, Bernoulli(d.p))::Bool
    if b
        rand(rng, d.core)
    else
        missing
    end
end

function logpdf(d::OptionalDistr, x)
    if x === missing
        log(1-d.p)
    else
        log(d.p) + logpdf(d.core, x)
    end
end

function log_score(d::OptionalDistr, x)
    if x === missing
        log(1-d.p)
    else
        log(d.p) + log_score(d.core, x)
    end
end

"""
Perform a bijective transformation to a distribution to obtain a new one.
Should implement the following
- `forward_transform(::DistrMap, x) -> y`: performs the forward transformation to a sample.
- `inverse_transform(::DistrMap, y) -> x`: performs the inverse transformation to a sample.
- `DistrMap.core::GDistr`: get the underlying distribution being transformed.
"""
abstract type DistrTransform <: ExtDistr end

function forward_transform end
function inverse_transform end

rand(rng::AbstractRNG, d::DistrTransform) = forward_transform(d, rand(rng, d.core))

logpdf(d::DistrTransform, x) = logpdf(d.core, inverse_transform(d, x))

log_score(d::DistrTransform, x, ::Type{T}) where T = 
    log_score(d.core, inverse_transform(d, x), T)::T

"""
Rotate a 2D-distribution counter-clockwise by `θ`.
"""
struct Rotate2dDistr{N<:Real, D<:GDistr} <: DistrTransform
    θ::N
    core::D
end

Rotate2dDistr(θ, d::Rotate2dDistr) = Rotate2dDistr(θ+d.θ, d.core)

Base.show(io::IO, ::Type{<:Rotate2dDistr}) = print(io, "Rotate2dDistr{...}")
Base.show(io::IO, d::Rotate2dDistr) = 
    print(io, "rotate2d($(d.θ), $(d.core))")

forward_transform(d::Rotate2dDistr, x) = rotate2d(d.θ, x)
inverse_transform(d::Rotate2dDistr, x) = rotate2d(-d.θ, x)

mean(d::Rotate2dDistr) = rotate2d(d.θ, mean(d.core))

"""
Rotate a 2D-distribution counter-clockwise by `θ`.
"""
rotate2d(θ, d::GDistr) = Rotate2dDistr(θ, d)


"""
Shift the mean of a distribution by `x`.
"""
struct ShiftDistr{N, D<:GDistr} <: DistrTransform
    shift::N
    core::D
end

ShiftDistr(shift, core::ShiftDistr) = ShiftDistr(shift + core.shift, core.core) 

Base.show(io::IO, ::Type{<:ShiftDistr}) = print(io, "ShiftDistr{...}")
Base.show(io::IO, d::ShiftDistr) = 
    print(io, "ShiftDistr($(d.shift), $(d.core))")

forward_transform(d::ShiftDistr, x) = x + d.shift
inverse_transform(d::ShiftDistr, x) = x - d.shift

mean(d::ShiftDistr) = d.shift + mean(d.core)

Base.:+(d::GDistr, x::Union{Real, AbstractVector}) = ShiftDistr(x, d) 
Base.:+(x::Union{Real, AbstractVector}, d::GDistr) = ShiftDistr(x, d) 
Base.:+(d::MultivariateDistribution, x::Union{Real, AbstractVector}) = ShiftDistr(x, d) 


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
    PertBeta(a, b, c) = begin
        @assert a <= b <= c
        (a1, b1, c1) = promote(a, b, c)
        new{typeof(a1)}(a1,b1,c1)
    end
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