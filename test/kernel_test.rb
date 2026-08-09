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

# ---- summary ----
puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
puts($fail > 0 ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
