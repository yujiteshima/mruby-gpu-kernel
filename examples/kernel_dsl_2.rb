# mruby-gpu-kernel -- two inputs, reductions, and control flow.
#
# Continues examples/kernel_dsl.rb. Everything below is still one traced block
# per dispatch: two input arrays, a reduction fused with its expression, a
# branch (where) and a loop (iterate) inside the shader.

puts "device : #{GPU.info[:device]}"
puts

# ---- 1. two inputs -------------------------------------------------------
a = GPU::SFloat[1, 2, 3, 4]
b = GPU::SFloat[10, 20, 30, 40]
puts "-- map2: a.map2(b) { |x, y| x * y + 1 } --"
puts "  #{a.map2(b) { |x, y| x * y + 1 }.inspect}"
puts

# ---- 2. the expression and the reduction in one dispatch ------------------
sig = GPU::SFloat.new(1024).seq
puts "-- sum { |x| x * 2 + 1 } -- one shader: evaluate, then reduce in shared memory --"
puts "  #{sig.sum { |x| x * 2 + 1 }}"
puts "-- a.dot(b) --"
puts "  #{a.dot(b)}"
puts "-- kernel(reduce: :max) { |x| abs(x) } --"
puts "  #{GPU.kernel(reduce: :max) { |x| abs(x) }.call(GPU::SFloat[-5, 3, 2])}"
puts

# ---- 3. a branch: where(cond, a, b) --------------------------------------
# Ruby's `if` cannot be traced (the proxy is always truthy), so the branch is
# a function. Conditions combine with & | ! -- parenthesise them.
v = GPU::SFloat.new(9).seq(-4, 1)          # -4 .. 4
puts "-- where((x > -2) & (x < 2), x, 0): keep the middle, zero the rest --"
puts "  #{v.map { |x| where((x > -2) & (x < 2), x, 0) }.inspect}"
puts

# ---- 4. a loop: iterate(init, count) { |acc, i| ... } ----------------------
# A real `for` in the shader. Newton's method for sqrt, 20 rounds:
puts "-- iterate(x, 20) { |g, i| (g + x / g) * 0.5 } -- Newton's sqrt --"
puts "  #{GPU::SFloat[2, 9, 100].map { |x| iterate(x, 20) { |g, i| (g + x / g) * 0.5 } }.inspect}"
puts "  sqrt(x) for comparison: #{GPU::SFloat[2, 9, 100].map { |x| sqrt(x) }.inspect}"
puts
puts "-- and the shader it became --"
puts GPU.kernel_source { |x| iterate(x, 20) { |g, i| (g + x / g) * 0.5 } }
