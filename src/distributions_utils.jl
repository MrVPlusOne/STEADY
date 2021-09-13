export GDistr, DistrIterator
export DistrWrapper, SVecDistr, SMvNormal

import Distributions: rand, logpdf
import Bijectors: AbstractBijector, bijector

"""
Generalized distributions that support sampling (and evaluating the probability of) 
structured data. 
"""
abstract type GDistr end

struct DistrWrapper{D, F, Inv} <: GDistr
    distr::D
    sample_transform::F
    sample_inv::Inv

    DistrWrapper(
        distr::D, 
        sample_transform::F=identity,
        sample_inv::Inv=identity,
    ) where {D <: Union{Distribution, GDistr}, F, Inv} = 
        new{D, F, Inv}(distr, sample_transform, sample_inv)
end

rand(rng::AbstractRNG, d::DistrWrapper) = rand(rng, d.distr) |> d.sample_transform
logpdf(d::DistrWrapper, x) = logpdf(d.distr, d.sample_inv(x))
bijector(d::DistrWrapper) = bijector(d.distr)  # this only works for simple sample_transform
Base.show(io::IO, dw::DistrWrapper) =
    print(io, "DistrWrapper(`$(dw.distr) |> $(dw.sample_transform)`)")


SVecDistr(distr::Distribution{Multivariate}) = DistrWrapper(distr, to_svec)
const SMvNormal = SVecDistr ∘ MvNormal
const SNormal = DistrWrapper ∘ Normal
const SUniform = DistrWrapper ∘ Uniform

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
struct DistrIterator{Iter} <: GDistr
    distributions::Iter

    DistrIterator(distributions::Iter) where Iter = begin
        eltype(distributions) <: GDistr || throw(
            ArgumentError("Expect an iterator of `GDistr`, but got $distributions"))
        new{Iter}(distributions)
    end
end


rand(rng::Random.AbstractRNG, diter::DistrIterator) = 
    map(d -> rand(rng, d), diter.distributions)
logpdf(diter::DistrIterator, xs) = 
    sum(logpdf(d, x) for (d, x) in zip(diter.distributions, xs))

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
