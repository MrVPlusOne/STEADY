
specific_elems(xs) = identity.(xs)

@inline rotation2D(θ) = @SArray(
    [cos(θ) -sin(θ)
     sin(θ)  cos(θ)]
)

rotate2d(θ, v) = rotation2D(θ) * v

to_measurement(values) = begin
    μ = mean(values)
    σ = std(values)
    μ ± σ
end

max_by(f) = xs -> begin
    ys = map(f, xs)
    _, i = findmax(ys)
    xs[i]
end

sort_by(f) = xs -> sort(xs, by=f)

"""
Apply a tuple of functions to a tuple of corresponding arguments. The result is also a 
tuple.

Currently, the result type is restricted to be the same type as `xs` to aid type inference
when `length(xs)` is large.
"""
@inline function zipmap(fs::Tuple, xs::X)::X where {X<:Tuple}
    @assert length(fs) == length(xs) "Need the same number of functions and values"
    ntuple(i -> fs[i](xs[i]), length(xs))
end

"""
Apply a tuple of functions to a NamedTuple of corresponding arguments. The result is a 
NamedTuple.
"""
@inline function zipmap(fs::Tuple, xs::X)::X where {X<:NamedTuple}
    @assert length(fs) == length(xs) "Need the same number of functions and values"
    t = ntuple(i -> fs[i](xs[i]), length(xs))
    NamedTuple{keys(xs)}(t)
end

subtuple(xs::NamedTuple, keys::Tuple) = begin
    NamedTuple{keys}(map(k -> getfield(xs, k), keys))
end

"""
Lightweight version of @timed.
"""
macro ltimed(ex)
    quote
        t0 = time()
        v = $(esc(ex))
        (time=time()-t0, value=v)
    end
end

function optimize_no_tag(loss, x₀, optim_options)
    # drop tag to reduce JIT compilation time and avoid tag checking
    cfg = ForwardDiff.GradientConfig(nothing, x₀) 
    function fg!(F, G, x)
        (G === nothing) && return loss(x)

        gr = ForwardDiff.DiffResult(first(x), (G,))
        ForwardDiff.gradient!(gr, loss, x, cfg)
        if F !== nothing
            return gr.value
        end
    end
    Optim.optimize(Optim.only_fg!(fg!), x₀, Optim.LBFGS(), optim_options)
end