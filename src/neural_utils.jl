##-----------------------------------------------------------
using Flux: Flux
using Flux: Dense, Chain, LayerNorm, SkipConnection, BatchNorm
using Flux: @functor, softplus
using Adapt: Adapt
using Setfield: @set
using Zygote: Zygote

export TensorConfig, tensor_type, check_type
"""
A configuration specifying which type (cpu or gpu) and floating type should be used
for array/tensor operations. 

## Usages
- Use `tconf(x)` to convert an object `x` (e.g. a number or an array) to having the 
desired configuration `tconf`.
- Use `check_type(tconf, x)` to check if the object `x` has the desired configuration.
"""
struct TensorConfig{on_gpu,ftype} end
TensorConfig(on_gpu, ftype=Float32) = TensorConfig{on_gpu,ftype}()

Base.show(io::IO, ::TensorConfig{on_gpu,ftype}) where {on_gpu,ftype} =
    print(io, (; on_gpu, ftype))

function Base.getproperty(::TensorConfig{on_gpu,ftype}, s::Symbol) where {on_gpu,ftype}
    if s === :on_gpu
        return on_gpu
    elseif s === :ftype
        return ftype
    else
        error("Unknown property: $s ")
    end
end

@inline function tensor_type(::TensorConfig{on_gpu,ftype}) where {on_gpu,ftype}
    (on_gpu ? CUDA.CuArray : Array){ftype}
end

new_array(tconf::TensorConfig{on_gpu,ftype}, sizes::Int...) where {on_gpu,ftype} =
    tensor_type(tconf)(undef, sizes...)

Adapt.adapt_storage(::Flux.FluxCPUAdaptor, tf::TensorConfig{true,ftype}) where {ftype} =
    TensorConfig{false,ftype}()

Adapt.adapt_storage(::Flux.FluxCUDAAdaptor, tf::TensorConfig{false,ftype}) where {ftype} =
    TensorConfig{true,ftype}()

(tconf::TensorConfig{on_gpu,ftype})(v::Real) where {on_gpu,ftype} = convert(ftype, v)
(tconf::TensorConfig{on_gpu,ftype})(a::AbstractArray{<:Real}) where {on_gpu,ftype} =
    convert(tensor_type(tconf), a)

function check_type(tconf::TensorConfig{on_gpu,ftype}, x::Real) where {on_gpu,ftype}
    @smart_assert x isa ftype
end

function check_type(tconf::TensorConfig{on_gpu,ftype}, x) where {on_gpu,ftype}
    desired_type = tensor_type(tconf)
    @smart_assert typeof(x) <: desired_type
end

##-----------------------------------------------------------
export BatchTuple, common_batch_size, inflate_batch, split_components
"""
A wrapper over a batch of data with named components.

## Usages
Assuming `batch::BatchTuple`:
- Use `batch.val` to access the underlying NamedTuple, in which each 
component is a batch of data, with the last dimension having the batch size.
- Use `tconf(batch)` to change the batch to a different `TensorConfig`. 
- Use `batch[idx]` to access a subset of the batch, e.g., `tconf[1]` or `tconf[3:6]`, 
both returns a new `BatchTuple`.
- Use `inflate_batch(batch)` to expand the underlying value to full batch size (if the batch 
dimension is 1).
- Use `common_batch_size(batchs...)` to get the common batch size of a list of `BatchTuple`.
"""
struct BatchTuple{C<:TensorConfig,V<:NamedTuple}
    tconf::C
    batch_size::Int
    val::V
    BatchTuple(
        tconf::TensorConfig{on_gpu}, batch_size::Int, val::NamedTuple
    ) where {on_gpu} = begin
        foreach(keys(val)) do k
            v = getproperty(val, k)
            check_type(tconf, v)
            if v isa Real
                return nothing
            else
                @smart_assert(
                    ndims(v) >= 2,
                    "Batch values should have at least 2 dimensions, with the last one \
                    being the batch dimension."
                )
                size(v)[end] == 1 ||
                    size(v)[end] == batch_size ||
                    error("Element $k has size $(size(v)), but batch size is $batch_size")
            end
        end
        new{typeof(tconf),typeof(val)}(tconf, batch_size, val)
    end
end
@use_short_show BatchTuple

Base.map(f, batch::BatchTuple) =
    BatchTuple(batch.tconf, batch.batch_size, map(f, batch.val))

"""
Create from a batch of values.
"""
function BatchTuple(tconf::TensorConfig, values::Vector{<:NamedTuple})
    BatchTuple(tconf, length(values), map(tconf, hcatreduce(values)))
end

function Base.length(batch::BatchTuple)
    batch.batch_size
end

function Base.lastindex(batch::BatchTuple)
    batch.batch_size
end

function Base.getindex(batch::BatchTuple, ids)
    new_val = map(v -> batch_subset(v, ids), batch.val)
    BatchTuple(batch.tconf, length(ids), new_val)
end

function (tconf::TensorConfig)(batch::BatchTuple)
    if tconf == batch.tconf
        batch
    else
        new_val = map(tconf, batch.val)
        BatchTuple(tconf, batch.batch_size, new_val)
    end
end

function batch_subset(m::Union{Real,AbstractArray}, ids::Union{Integer, UnitRange})
    r = if m isa Real
        m
    else
        s = size(m)
        if s[end] == 1
            m
        else
            colons = map(_ -> :, s)
            ids1 = ids isa Integer ? (ids:ids) : ids
            m[colons[1:(end - 1)]..., ids1]
        end
    end
    r::typeof(m)
end
##-----------------------------------------------------------
# Batch utils
common_batch_size(sizes::Integer...) = begin
    max_size = max(sizes...)
    all(sizes) do s
        s == max_size || s == 1 || error("Inconsistent batch sizes: $sizes")
    end
    max_size
end

common_batch_size(bts::BatchTuple...) = begin
    common_batch_size(map(x -> x.batch_size, bts)...)
end

function Base.repeat(batch::BatchTuple, n::Int; inflate=false)
    @smart_assert batch.batch_size == 1
    nb = @set batch.batch_size = n
    inflate ? inflate_batch(nb) : nb
end

"""
Split a matrix into a named tuple of matrices according to the given sizes.

## Examples
```jldoctest
julia> split_components(ones(4, 5), (a=2, b=1, c=1))
(a = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], b = [1.0 1.0 … 1.0 1.0], c = [1.0 1.0 … 1.0 1.0])
```
"""
function split_components(x::AbsMat, comp_sizes::NamedNTuple{names,Int}) where {names}
    @smart_assert size(x, 1) == sum(comp_sizes)

    i = 1
    comps = map(names) do k
        x[i:((i += comp_sizes[k]) - 1), :]
    end
    NamedTuple{names}(comps)
end

"""
Foreach component whose batch dimension size is 1, pad it along the batch dimension 
to grow to the full batch size.
"""
function inflate_batch(batch::BatchTuple)
    bs = batch.batch_size
    @set batch.val = map(batch.val) do v
        if size(v)[end] == 1
            repeat(v, ntuple(Returns(1), ndims(v) - 1)..., bs)
        else
            @smart_assert size(v)[end] == bs
            v
        end
    end
end

function check_components(
    batch::BatchTuple, comp_sizes::NamedNTuple{names,Int}
) where {names}
    Zygote.ignore() do
        @smart_assert keys(batch.val) == keys(comp_sizes)
        foreach(batch.val, comp_sizes) do v, dim
            @smart_assert size(v, 1) == dim
        end
    end
end

function check_approx(batch1::BatchTuple, batch2::BatchTuple; kargs...)
    @smart_assert keys(batch1.val) == keys(batch2.val)
    foreach(keys(batch1.val)) do k
        isapprox(batch1.val[k], batch2.val[k]; kargs...) || error(
            "Faild for componenet $k.\n\tLeft = $(batch1.val[k])\n\tRight = $(batch2.val[k])",
        )
    end
end


export plot_batched_series
"""
A general plotting function that plots a time series of `BatchedTuple`s. Each batch 
component will be plotted inside a separate subplot.

## Examples
```julia
let
    tconf = TensorConfig(false)
    times = 0:0.1:5
    series = [
        BatchTuple(tconf, 
            [
                (
                    pos=[randn(Float32), randn(Float32)+2], 
                    θ=[randn(Float32) * sin(Float32(t))],
                ) 
                for b in 1:3
            ]) 
        for t in times
    ]

    plot_batched_series(times, series)
end
```
"""
function plot_batched_series(
    times,
    series::TimeSeries{<:BatchTuple};
    truth::Union{Nothing,TimeSeries{<:BatchTuple}}=nothing,
    plot_width=600,
    plot_args...,
)
    to_plot_data(ys::TimeSeries{<:Real}) = (; xs=times, ys, alpha=1.0)
    to_plot_data(ys::TimeSeries{<:AbsMat{<:Real}}) = begin
        local (dim, batch_size) = size(ys[1])
        local Num = eltype(ys[1])
        local T = length(ys)
        local x_out = repeat([times; NaN], batch_size)
        local y_out = Matrix{Num}(undef, batch_size * (T + 1), dim)
        for i in 1:batch_size
            for t in 1:T
                y_out[(i - 1) * (T + 1) + t, :] = ys[t][:, i]
            end
            y_out[i * (T + 1), :] .= NaN
        end
        (; xs=x_out, ys=y_out, alpha=1.0 / sqrt(batch_size))
    end

    series = TensorConfig(false).(series)
    template = series[1]
    if truth !== nothing
        truth = TensorConfig(false).(truth)
        truth_temp = truth[1]
        @smart_assert truth_temp.batch_size == 1
        @smart_assert keys(truth_temp.val) == keys(template.val)
    end

    subplots = []
    for k in keys(template.val)
        xs, ys, linealpha = to_plot_data((b -> b.val[k]).(series))
        n_comps = size(template.val[k], 1)
        linecolor = hcatreduce(1:n_comps)
        label = ["$k[$i]" for i in 1:n_comps] |> hcatreduce
        push!(subplots, plot(xs, ys; linealpha, linecolor, label))
        if truth !== nothing
            txs, tys, _ = to_plot_data((b -> b.val[k]).(truth))
            plot!(
                txs, tys; label=hcatreduce(["$k[$i] (truth)" for i in 1:n_comps]), linecolor
            )
        end
    end
    plot(
        subplots...;
        layout=(length(subplots), 1),
        size=(plot_width, 0.6plot_width * length(subplots)),
        left_margin=1.5cm,
        plot_args...,
    )
end
##-----------------------------------------------------------
export FluxLayer
"""
    FluxLayer(forward, Val(layer_name), trainable, [layer_info=trainable])

An flexible wrapper to represent an arbitrary trainable computation. This can be 
used to implement new forward computation logic without declaring a new struct.

- `forward::Function` should have the signature `forward(trainable)(args...) -> result`. i.e., 
it should be a higher-order function that first takes in the trainable parameters and then 
the input arguments of the layer. See the examples below.
- `trainable` can be either a Tuple or NamedTuple. 
- `layer_info=trainable` can be anything and will only be used to display the layer.

## Examples
```julia
julia> layer = FluxLayer(
           Val(:MyBilinear), 
           (w=rand(2,3),), 
           info=(x_dim = 3, y_dim = 2)
       ) do trainable
           (x, y) -> y' * trainable.w * x
       end
FluxLayer{MyBilinear}((x_dim = 3, y_dim = 2))

julia> layer(ones(3), ones(2))
2.7499240871541684

julia> Flux.params(layer)
Params([[0.6650776972679167 0.004201998819206465 0.7122138939453165; 
0.2132508860808494 0.17734661433873555 0.977832996702144]])
```
"""
struct FluxLayer{name,PS<:Union{Tuple,NamedTuple},F<:Function,Info}
    forward::F
    layer_name::Val{name}
    trainable::PS
    info::Info
end
Flux.@functor FluxLayer

FluxLayer(forward::Function, layer_name, trainable; info=trainable) =
    FluxLayer(forward, layer_name, trainable, info)

function Base.show(io::IO, ::Type{<:FluxLayer{name}}) where {name}
    print(io, "FluxLayer{$name, ...}")
end

function Base.show(io::IO, l::FluxLayer{name}) where {name}
    print(io, "FluxLayer{$name}($(l.info))")
end

function (l::FluxLayer)(args...)
    l.forward(l.trainable)(args...)
end

function Flux.trainable(l::FluxLayer)
    Flux.trainable(l.trainable)
end
##-----------------------------------------------------------
"""
A wrapper over a trainalbe array parameter that carries the information about whether it 
requires regularization.
"""
struct Trainable{regular,M<:AbstractArray}
    is_regular::Val{regular}
    array::M
end
Flux.@functor Trainable
Trainable(regular::Bool, a::AbstractArray) = Trainable(Val(regular), a)

Base.show(io::IO, x::Trainable{regular}) where {regular} = begin
    print(io, "Trainable(regular=$(regular), $(summary(x.array)))")
end

"""
Return an iterator of trainable parameters that require regularization.
You can also wrap an array `x` inside a [`Trainale`](@ref), which would allow you 
to specify whether `x` needs regularization.
"""
function regular_params end

regular_params(layer::Trainable{true}) = (layer.array,)
regular_params(layer::Trainable{false}) = tuple()


regular_params(layer::Union{FluxLayer,Chain,SkipConnection}) =
    Iterators.flatten(map(regular_params, Flux.trainable(layer)))

regular_params(layer::Dense) = (layer.weight,)

regular_params(layer::Union{Flux.GRUCell,Flux.LSTMCell}) = (layer.Wi, layer.Wh)

regular_params(tp::Union{Tuple,NamedTuple}) = Iterators.flatten(map(regular_params, tp))

regular_params(::Union{Function,BatchNorm,Nothing}) = tuple()
