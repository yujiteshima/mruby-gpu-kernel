# mruby-gpu-kernel -- known-value tests. Needs a Vulkan device and
# glslangValidator on PATH.
# Run: mruby test/kernel_test.rb

$pass = 0
$fail = 0

def ok(label)
  $pass += 1
  puts "PASS #{label}"
end

def ng(label, detail)
  $fail += 1
  puts "FAIL #{label}: #{detail}"
end

def assert_ary(label, expected, actual, eps = 1e-3)
  unless actual.is_a?(Array)
    return ng(label, "expected Array, got #{actual.class}")
  end
  if expected.size != actual.size
    return ng(label, "size #{expected.size} != #{actual.size}")
  end
  expected.each_index do |i|
    if (expected[i] - actual[i]).abs > eps
      return ng(label, "index #{i}: expected #{expected[i]}, got #{actual[i]}")
    end
  end
  ok(label)
end

def assert_near(label, expected, actual, eps = 1e-3)
  if (expected - actual).abs <= eps
    ok(label)
  else
    ng(label, "expected #{expected}, got #{actual}")
  end
end

def assert_true(label, cond, detail = "false")
  cond ? ok(label) : ng(label, detail)
end

def assert_raise(label, klass)
  begin
    yield
  rescue => e
    return e.is_a?(klass) ? ok(label) : ng(label, "expected #{klass}, got #{e.class}: #{e.message}")
  end
  ng(label, "expected #{klass}, but nothing was raised")
end

puts "device: #{GPU.info[:device]} (#{GPU.info[:backend]})"
puts

# ---- values ----
assert_ary("map { x * 2 + 1 }", [1.0, 3.0, 5.0, 7.0],
           GPU::SFloat[0, 1, 2, 3].map { |x| x * 2 + 1 }.to_a)
assert_ary("map agrees with the operator chain",
           (GPU::SFloat.new(64).seq * 2 + 1).to_a,
           GPU::SFloat.new(64).seq.map { |x| x * 2 + 1 }.to_a)
assert_ary("map { sqrt(x) }", [1.0, 2.0, 3.0, 4.0],
           GPU::SFloat[1, 4, 9, 16].map { |x| sqrt(x) }.to_a)
assert_ary("map { x * 2 + sin(x) }",
           [0.0, 1.0, 2.0, 3.0].map { |x| x * 2 + Math.sin(x) },
           GPU::SFloat[0, 1, 2, 3].map { |x| x * 2 + sin(x) }.to_a)
assert_ary("map { min(abs(x), 2) }", [2.0, 2.0, 1.0, 0.0],
           GPU::SFloat[-2, 2, 1, 0].map { |x| min(abs(x), 2) }.to_a)
assert_ary("map { clamp(x, -1, 1) }", [-1.0, -1.0, 0.0, 1.0, 1.0],
           GPU::SFloat[-9, -1, 0, 1, 9].map { |x| clamp(x, -1, 1) }.to_a)
assert_ary("map { -x }", [-1.0, 2.0, -3.0],
           GPU::SFloat[1, -2, 3].map { |x| -x }.to_a)
assert_ary("block that ignores x", [7.0, 7.0],
           GPU::SFloat[1, 2].map { 7 }.to_a)
assert_ary("nested calls", [3.0, 3.0, 3.0],
           GPU::SFloat[1, 2, 3].map { |x| max(min(x, 3), 3) }.to_a)
# n not a multiple of the 256-wide workgroup: exercises the tail guard.
assert_near("map over 1000 elements", 1000.0,
            GPU::SFloat.new(1000).fill(3.0).map { |x| x / 3 }.sum)

# ---- the generated shader is public surface ----
assert_true("kernel_source emits the traced expression",
            GPU.kernel_source { |x| x * 2 }.include?("b[idx] = (x * 2.0);"),
            GPU.kernel_source { |x| x * 2 })
assert_true("a kernel keeps its GLSL source",
            GPU.kernel { |x| x * 2 }.glsl.include?("#version 450"))
assert_true("identical kernels are compiled once",
            GPU.kernel { |x| x + 1 }.equal?(GPU.kernel { |x| x + 1 }),
            "got two distinct kernel objects")
assert_true("a kernel is reusable",
            GPU.kernel { |x| x + 1 }.call(GPU::SFloat[1, 2]).to_a == [2.0, 3.0])

# ---- the traced tree ----
tree = GPU::Expr.wrap(
  GPU::KernelContext.new.instance_exec(GPU::Expr.new(:var, ["x"])) { |x| x * 2 + 1 }
)
assert_true("tracing builds an expression tree", tree.inspect == "(+ (* x 2.0) 1.0)",
            tree.inspect)

# ---- errors ----
assert_raise("2 * x inside a kernel -> TypeError", TypeError) do
  GPU::SFloat[1, 2].map { |x| 2 * x }
end
assert_raise("non-numeric inside a kernel -> TypeError", TypeError) do
  GPU::SFloat[1, 2].map { |x| x * "nope" }
end
assert_raise("unknown function -> NoMethodError", NoMethodError) do
  GPU::SFloat[1, 2].map { |x| wobble(x) }
end
assert_raise("Kernel#call on a non-array -> TypeError", TypeError) do
  GPU.kernel { |x| x }.call([1, 2, 3])
end
assert_raise("no block -> ArgumentError", ArgumentError) do
  GPU.kernel
end

# ---- two inputs ----
assert_ary("map2 { x * y + 1 }", [11.0, 41.0, 91.0],
           GPU::SFloat[1, 2, 3].map2(GPU::SFloat[10, 20, 30]) { |x, y| x * y + 1 }.to_a)
assert_ary("kernel { |x, y| } takes two arrays (arity from the block)", [-9.0, -18.0, -27.0],
           GPU.kernel { |x, y| x - y }.call(GPU::SFloat[1, 2, 3], GPU::SFloat[10, 20, 30]).to_a)
assert_ary("kernel(arity: 2) stated explicitly", [11.0, 22.0],
           GPU.kernel(arity: 2) { |x, y| x + y }.call(GPU::SFloat[1, 2], GPU::SFloat[10, 20]).to_a)
assert_true("two-input source loads y from the second buffer",
            GPU.kernel_source { |x, y| x + y }.include?("float y = b[idx];") &&
            GPU.kernel_source { |x, y| x + y }.include?("c[idx] = (x + y);"))
assert_raise("map2 with mismatched sizes -> ArgumentError", ArgumentError) do
  GPU::SFloat[1, 2, 3].map2(GPU::SFloat[1, 2]) { |x, y| x + y }
end
assert_raise("two-input kernel called with one array -> ArgumentError", ArgumentError) do
  GPU.kernel { |x, y| x + y }.call(GPU::SFloat[1, 2])
end
assert_raise("one-input kernel called with two arrays -> ArgumentError", ArgumentError) do
  GPU.kernel { |x| x }.call(GPU::SFloat[1, 2], GPU::SFloat[1, 2])
end

# ---- reductions: the expression and the sum in one dispatch ----
assert_near("sum { x * 2 + 1 } over 1000 elements", 1_000_000.0,
            GPU::SFloat.new(1000).seq.sum { |x| x * 2 + 1 })
assert_near("sum without a block still works", 3.0, GPU::SFloat[1, 2].sum)
assert_near("dot product", 32.0, GPU::SFloat[1, 2, 3].dot(GPU::SFloat[4, 5, 6]))
assert_near("kernel(reduce: :sum) { |x, y| } is the dot product", 32.0,
            GPU.kernel(reduce: :sum) { |x, y| x * y }.call(GPU::SFloat[1, 2, 3], GPU::SFloat[4, 5, 6]))
assert_near("reduce: :max", 5.0,
            GPU.kernel(reduce: :max) { |x| abs(x) }.call(GPU::SFloat[-5, 3, 2]))
assert_near("reduce: :min", -5.0,
            GPU.kernel(reduce: :min) { |x| x }.call(GPU::SFloat[-5, 3, 2]))
# 1000 elements = 4 workgroups, so the host really combines several partials.
assert_near("reduce: :max across workgroups", 999.0,
            GPU.kernel(reduce: :max) { |x| x }.call(GPU::SFloat.new(1000).seq))
assert_near("sum of an empty array is 0", 0.0, GPU::SFloat.new(0).sum { |x| x })
assert_true("reduce source has the shared-memory tree",
            GPU.kernel_source(reduce: :sum) { |x| x }.include?("shared float sdata[256];"))
assert_raise("unknown reduction -> ArgumentError", ArgumentError) do
  GPU.kernel(reduce: :product) { |x| x }
end

# ---- control flow: where, & | !, iterate ----
assert_ary("where(x > 0, x, 0) is relu", [0.0, 0.0, 0.0, 1.0, 2.0],
           GPU::SFloat[-2, -1, 0, 1, 2].map { |x| where(x > 0, x, 0) }.to_a)
assert_ary("select is where", [0.0, 1.0],
           GPU::SFloat[-1, 1].map { |x| select(x > 0, 1, 0) }.to_a)
assert_ary("(a) & (b)", [0.0, 1.0, 1.0, 1.0, 0.0],
           GPU::SFloat[0, 1, 2, 3, 4].map { |x| where((x >= 1) & (x <= 3), 1, 0) }.to_a)
assert_ary("(a) | (b) and !", [0.0, 1.0, 2.0, -1.0, 4.0],
           GPU::SFloat[0, 1, 2, 3, 4].map { |x| where(!(x > 2) | (x == 4), x, -1) }.to_a)
assert_ary("== and !=", [100.0, 2.0, 100.0],
           GPU::SFloat[1, 2, 3].map { |x| where(x != 2, 100, x) }.to_a)
assert_true("where becomes GLSL's ternary",
            GPU.kernel_source { |x| where(x > 0, x, 0) }.include?("((x > 0.0) ? x : 0.0)"))
assert_ary("iterate: Newton's method matches sqrt(x)",
           GPU::SFloat[1, 4, 9, 2, 100].map { |x| sqrt(x) }.to_a,
           GPU::SFloat[1, 4, 9, 2, 100].map { |x| iterate(x, 20) { |g, i| (g + x / g) * 0.5 } }.to_a)
assert_ary("iterate: i counts from 0", [0.0 + 1 + 2 + 3] * 2,
           GPU::SFloat[0, 0].map { |x| iterate(0, 4) { |acc, i| acc + i } }.to_a)
assert_ary("nested iterate", [18.0, 18.0],
           GPU::SFloat[0, 0].map { |x|
             iterate(0, 3) { |acc, i| acc + iterate(0, 4) { |acc2, j| acc2 + i * j } }
           }.to_a)
assert_ary("a loop result used twice is one loop", [16.0, 32.0],
           GPU::SFloat[1, 2].map { |x| s = iterate(x, 3) { |a2, i| a2 * 2 }; s + s }.to_a)
assert_true("iterate emits a real GLSL for loop",
            GPU.kernel_source { |x| iterate(x, 5) { |acc, i| acc * 2 } }.include?("for (int i2 = 0; i2 < 5; i2++) {"),
            GPU.kernel_source { |x| iterate(x, 5) { |acc, i| acc * 2 } })
assert_ary("a Ruby loop with a known count is unrolled", [8.0, 16.0],
           GPU::SFloat[1, 2].map { |x| y = x; 3.times { y = y * 2 }; y }.to_a)
assert_true("...and leaves no loop in the shader",
            !GPU.kernel_source { |x| y = x; 3.times { y = y * 2 }; y }.include?("for ("))
assert_true("tracing shows the loop node",
            GPU::KernelContext.new.instance_exec(GPU::Expr.new(:var, ["x"])) { |x|
              iterate(x, 2) { |acc, i| acc + 1 }
            }.inspect == "(iterate x 2 (+ acc 1.0))")

# ---- control-flow errors are Ruby errors, not glslang's ----
assert_raise("a condition in arithmetic -> TypeError", TypeError) do
  GPU::SFloat[1, 2].map { |x| (x > 1) + 1 }
end
assert_raise("where with a number as the condition -> TypeError", TypeError) do
  GPU::SFloat[1, 2].map { |x| where(x, 1, 0) }
end
assert_raise("& on numbers -> TypeError", TypeError) do
  GPU::SFloat[1, 2].map { |x| where(x & x, 1, 0) }
end
assert_raise("iterate count must be an Integer (got Float) -> TypeError", TypeError) do
  GPU::SFloat[1, 2].map { |x| iterate(x, 2.5) { |acc, i| acc } }
end
assert_raise("iterate count must be known at trace time (got x) -> TypeError", TypeError) do
  GPU::SFloat[1, 2].map { |x| iterate(0, x) { |acc, i| acc } }
end
assert_raise("negative iterate count -> ArgumentError", ArgumentError) do
  GPU::SFloat[1, 2].map { |x| iterate(0, -1) { |acc, i| acc } }
end
assert_raise("loop variable escaping its block -> ArgumentError", ArgumentError) do
  GPU::SFloat[1, 2].map { |x| leaked = nil; iterate(0, 2) { |acc, i| leaked = i; acc }; leaked }
end

# ---- summary ----
puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
puts($fail > 0 ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
