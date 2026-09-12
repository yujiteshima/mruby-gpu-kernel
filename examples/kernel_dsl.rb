# mruby-gpu-narray -- turning a Ruby block into a GPU kernel.
#
# The block below is never run as Ruby arithmetic. It is called once with a
# proxy object, which records the operators applied to it, and the resulting
# expression tree is printed as GLSL, compiled to SPIR-V, and dispatched as a
# compute shader. On a Raspberry Pi 5 that shader runs on the VideoCore VII.

puts "device : #{GPU.info[:device]}"
puts "backend: #{GPU.info[:backend]} (Vulkan API #{GPU.info[:api_version]})"
puts

# ---- 1. what tracing produces --------------------------------------------
puts "-- the block --"
puts "  a.map { |x| x * 2 + sin(x) }"
puts
puts "-- becomes this shader --"
puts GPU.kernel_source { |x| x * 2 + sin(x) }

# ---- 2. running it --------------------------------------------------------
a = GPU::SFloat.new(8).seq
puts "-- and runs on the GPU --"
puts "  a                        = #{a.inspect}"
puts "  a.map { x * 2 + sin(x) } = #{a.map { |x| x * 2 + sin(x) }.inspect}"
puts

# Anything GLSL has an element-wise built-in for is available inside the block:
# sqrt, exp, log, abs, min, max, pow, clamp, floor, tanh, ...
signal = GPU::SFloat.new(8).seq(-4, 1)          # -4 .. 3
puts "  signal                   = #{signal.inspect}"
puts "  clamp(x, -1, 1)          = #{signal.map { |x| clamp(x, -1, 1) }.inspect}"
puts "  sqrt(abs(x))             = #{signal.map { |x| sqrt(abs(x)) }.inspect}"
puts

# ---- 3. why it is worth doing --------------------------------------------
# Each operator is its own dispatch. With GPU.sync_mode = :eager that is also
# a submit and a fence wait per operator; by default mruby-gpu-narray records
# the four dispatches and submits them together when a result is read. The
# traced kernel is one dispatch either way -- and one pass over memory instead
# of four, which is what still matters once the arrays are large.
def bench(iters)
  yield
  t0 = Time.now.to_f
  iters.times { yield }
  ((Time.now.to_f - t0) * 1000.0) / iters
end

def ms(x)
  ((x * 1000).round.to_f / 1000).to_s
end

big = GPU::SFloat.new(1024).seq
fused = GPU.kernel { |x| x * 2 + 1 - 3 + 4 }

GPU.sync_mode = :eager
eager_ms    = bench(100) { big * 2 + 1 - 3 + 4 }
GPU.sync_mode = :deferred
deferred_ms = bench(100) { (big * 2 + 1 - 3 + 4); GPU.sync }
fused_ms    = bench(100) { fused.call(big); GPU.sync }

puts "-- four operators, 1024 elements --"
puts "  a * 2 + 1 - 3 + 4      #{ms(eager_ms)} ms   (4 dispatches, 4 waits -- GPU.sync_mode = :eager)"
puts "  same, one batch        #{ms(deferred_ms)} ms   (4 dispatches, 1 wait)"
puts "  one traced kernel      #{ms(fused_ms)} ms   (1 dispatch,  1 wait)"
puts "  speedup                #{ms(eager_ms / fused_ms)}x over eager, #{ms(deferred_ms / fused_ms)}x over the batch"
