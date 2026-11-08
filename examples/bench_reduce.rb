# How much a fused reduction saves: a.map { }.sum is two dispatches (and two
# intermediate buffers -- the mapped array and the partials); a.sum { } is one.
#
#   mruby examples/bench_reduce.rb

def bench(iters)
  yield
  t0 = Time.now.to_f
  iters.times { yield }
  ((Time.now.to_f - t0) * 1000.0) / iters
end

def ms(x)
  ((x * 1000).round.to_f / 1000).to_s
end

puts "device: #{GPU.info[:device]}"
puts
# Both sides trace their block on every call, so the comparison is fair. The
# third column reuses a kernel traced once (GPU.kernel outside the loop): the
# gap between columns two and three is the cost of tracing + cache lookup in
# mruby, which is why a hot loop should hoist its kernel.
puts "size        map{}.sum (2 dispatch)   sum{} (1 dispatch)   hoisted kernel   speedup (col 2 / col 3)"
[1024, 1_048_576].each do |n|
  a = GPU::SFloat.new(n).seq
  iters = n > 100_000 ? 30 : 200
  k = GPU.kernel(reduce: :sum) { |x| x * 2 + 1 }
  two = bench(iters) { a.map { |x| x * 2 + 1 }.sum }
  one = bench(iters) { a.sum { |x| x * 2 + 1 } }
  hoisted = bench(iters) { k.call(a) }
  puts "#{n.to_s.rjust(9)}   #{ms(two).rjust(10)} ms           #{ms(one).rjust(8)} ms      #{ms(hoisted).rjust(8)} ms     #{ms(two / one)}x / #{ms(two / hoisted)}x"
end
puts
puts "dot product, 1M elements"
a = GPU::SFloat.new(1_048_576).fill(1.0)
b = GPU::SFloat.new(1_048_576).fill(2.0)
two = bench(30) { (a * b).sum }
one = bench(30) { a.dot(b) }
puts "  (a * b).sum   #{ms(two)} ms   (2 dispatches)"
puts "  a.dot(b)      #{ms(one)} ms   (1 dispatch)   #{ms(two / one)}x"
