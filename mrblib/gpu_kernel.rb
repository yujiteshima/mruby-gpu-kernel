# Turn a Ruby block into a GPU kernel, on top of mruby-gpu-narray.
#
#   na.map { |x| x * 2 + sin(x) }
#   na.map2(nb) { |x, y| x * y + 1 }
#   na.sum { |x| x * 2 + 1 }
#
# mruby has no way to read a block's AST, so we trace instead. The block is
# called once with a proxy (Expr) in place of the data. Every operator and math
# function on that proxy records itself and returns a new node, so calling the
# block builds an expression tree instead of computing anything. The tree is
# printed as GLSL, compiled to SPIR-V by glslangValidator, and run as a compute
# shader -- see src/gpu_kernel.c for that half.
#
# Consequences worth knowing:
#   * The block runs on a KernelContext (instance_exec), which is what lets you
#     write sin(x) rather than x.sin. Methods of the enclosing object are not
#     visible inside; local variables still are.
#   * mruby has no numeric coercion, so `2 * x` cannot work -- same rule as the
#     arrays themselves: keep the expression on the left, as in `x * 2`.
#   * Ruby's own `if`, `while`, `&&` and `||` cannot be traced: the proxy is
#     always truthy, so `if x > 0` takes the same branch for every element.
#     Use where(cond, a, b) for a branch and iterate(init, n) { |acc, i| } for
#     a loop; combine conditions with `&`, `|` and `!`. A Ruby loop with a
#     count known at trace time (3.times { ... }) is simply unrolled.

module GPU
  # A node in the traced expression tree.
  class Expr
    # Nodes whose GLSL type is bool rather than float.
    BOOL_OPS = [:>, :<, :>=, :<=, :==, :!=, :and, :or, :not, :bconst].freeze

    def initialize(op, args = [])
      @op = op
      @args = args
    end
    attr_reader :op, :args

    def bool?
      BOOL_OPS.include?(@op)
    end

    def self.wrap(value)
      return value if value.is_a?(Expr)
      return new(:bconst, [value]) if true.equal?(value) || false.equal?(value)
      unless value.is_a?(Numeric)
        raise TypeError,
              "a GPU kernel expression takes Numeric or Expr, got #{value.class}"
      end
      new(:const, [value.to_f])
    end

    # wrap, then insist on a float-typed node. `what` names the operation for
    # the error message.
    def self.numeric(value, what)
      e = wrap(value)
      if e.bool?
        raise TypeError,
              "#{what} needs a number but got a condition; " \
              "use where(cond, a, b) to turn a condition into a number"
      end
      e
    end

    # wrap, then insist on a bool-typed node.
    def self.boolean(value, what)
      e = wrap(value)
      unless e.bool?
        raise TypeError,
              "#{what} needs a condition (x > 0, x == 1, ...) but got a number"
      end
      e
    end

    # GLSL reads a bare 2 as an int, which will not implicitly convert in every
    # position, so every literal is emitted as a float.
    def self.literal(f)
      s = f.to_s
      (s.include?(".") || s.include?("e")) ? s : "#{s}.0"
    end

    [:+, :-, :*, :/].each do |operator|
      define_method(operator) do |other|
        Expr.new(operator, [Expr.numeric(self, operator.to_s),
                            Expr.numeric(other, operator.to_s)])
      end
    end

    # Comparisons produce bool nodes: feed them to where / & / | / !.
    [:>, :<, :>=, :<=, :==, :!=].each do |operator|
      define_method(operator) do |other|
        Expr.new(operator, [Expr.numeric(self, operator.to_s),
                            Expr.numeric(other, operator.to_s)])
      end
    end

    # && and || cannot be overloaded, so the logical operators are the
    # bitwise ones. They bind tighter than comparisons in Ruby, so
    # parenthesise: (x > 1) & (x < 3).
    def &(other)
      Expr.new(:and, [Expr.boolean(self, "&"), Expr.boolean(other, "&")])
    end

    def |(other)
      Expr.new(:or, [Expr.boolean(self, "|"), Expr.boolean(other, "|")])
    end

    def !
      Expr.new(:not, [Expr.boolean(self, "!")])
    end

    def -@
      Expr.new(:neg, [Expr.numeric(self, "-")])
    end

    # GLSL for a tree without loops. Kernel.source_for handles the general
    # case, where a loop needs statements as well as an expression.
    def to_glsl
      Codegen.new.expr(self)
    end

    # S-expression view of the tree, handy when showing what tracing produced.
    def inspect
      case @op
      when :var    then @args[0]
      when :const  then Expr.literal(@args[0])
      when :bconst then @args[0].to_s
      when :acc    then "acc"
      when :idx    then "i"
      when :and    then "(& #{@args[0].inspect} #{@args[1].inspect})"
      when :or     then "(| #{@args[0].inspect} #{@args[1].inspect})"
      when :not    then "(! #{@args[0].inspect})"
      when :loop
        init, count, _acc, _idx, body = @args
        "(iterate #{init.inspect} #{count} #{body.inspect})"
      else "(#{@op} #{@args.map { |a| a.inspect }.join(' ')})"
      end
    end
    alias to_s inspect
  end

  # Expression tree -> GLSL. An expression may need statements ahead of it (a
  # loop is a `for` in GLSL, not an expression), so the generator returns the
  # expression string and collects those statements in #lines, indented for
  # the block they belong in.
  class Codegen
    attr_reader :lines

    def initialize(depth = 1)
      @lines = []
      @depth = depth
      @tmp = 0
      @scopes = [{}]   # per open block: loop node -> its accumulator name
      @names = {}      # acc / idx node -> GLSL name, set when its loop is emitted
    end

    def expr(e)
      case e.op
      when :var    then e.args[0]
      when :const  then Expr.literal(e.args[0])
      when :bconst then e.args[0] ? "true" : "false"
      when :acc, :idx
        @names[e.object_id] ||
          raise(ArgumentError, "a loop variable was used outside its iterate block")
      when :neg   then "(-#{expr(e.args[0])})"
      when :not   then "(!#{expr(e.args[0])})"
      when :and   then "(#{expr(e.args[0])} && #{expr(e.args[1])})"
      when :or    then "(#{expr(e.args[0])} || #{expr(e.args[1])})"
      when :+, :-, :*, :/, :>, :<, :>=, :<=, :==, :!=
        "(#{expr(e.args[0])} #{e.op} #{expr(e.args[1])})"
      when :where
        "(#{expr(e.args[0])} ? #{expr(e.args[1])} : #{expr(e.args[2])})"
      when :loop  then loop(e)
      else "#{e.op}(#{e.args.map { |a| expr(a) }.join(', ')})"
      end
    end

    private

    def emit(line)
      @lines << ("    " * @depth) + line
    end

    def fresh(prefix)
      @tmp += 1
      "#{prefix}#{@tmp}"
    end

    # iterate(init, count) { |acc, i| body } becomes
    #     float accN = <init>;
    #     for (int iN = 0; iN < count; iN++) { accN = <body>; }
    # and the expression is accN. A node reached twice inside one block is
    # emitted once; reached again in a different block it is emitted afresh
    # there, because the first accumulator is out of scope.
    def loop(e)
      @scopes.reverse_each { |s| return s[e.object_id] if s[e.object_id] }
      init, count, acc, idx, body = e.args
      acc_name = fresh("acc")
      i_name = fresh("i")
      init_s = expr(init)
      emit "float #{acc_name} = #{init_s};"
      @names[acc.object_id] = acc_name
      @names[idx.object_id] = "float(#{i_name})"
      emit "for (int #{i_name} = 0; #{i_name} < #{count}; #{i_name}++) {"
      @depth += 1
      @scopes.push({})
      body_s = expr(body)
      emit "#{acc_name} = #{body_s};"
      @scopes.pop
      @depth -= 1
      emit "}"
      @scopes.last[e.object_id] = acc_name
      acc_name
    end
  end

  # The block is instance_exec'd here, so these become the vocabulary available
  # inside a kernel. Every name below is a GLSL built-in of the same name,
  # plus where / iterate for control flow.
  class KernelContext
    UNARY = [:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh,
             :exp, :exp2, :log, :log2, :sqrt, :inversesqrt, :abs, :sign,
             :floor, :ceil, :fract, :radians, :degrees].freeze
    BINARY = [:min, :max, :pow, :mod, :step].freeze
    TERNARY = [:clamp, :mix, :smoothstep].freeze

    UNARY.each do |f|
      define_method(f) { |a| Expr.new(f, [Expr.numeric(a, f.to_s)]) }
    end
    BINARY.each do |f|
      define_method(f) do |a, b|
        Expr.new(f, [Expr.numeric(a, f.to_s), Expr.numeric(b, f.to_s)])
      end
    end
    TERNARY.each do |f|
      define_method(f) do |a, b, c|
        Expr.new(f, [Expr.numeric(a, f.to_s), Expr.numeric(b, f.to_s),
                     Expr.numeric(c, f.to_s)])
      end
    end

    # where(cond, a, b) -> a where cond holds, else b. GLSL's `cond ? a : b`.
    # This is the branch: Ruby's `if` cannot be traced.
    def where(cond, a, b)
      Expr.new(:where, [Expr.boolean(cond, "where"),
                        Expr.numeric(a, "where"), Expr.numeric(b, "where")])
    end
    alias select where

    # iterate(init, count) { |acc, i| ... } -> the value of acc after `count`
    # rounds of acc = block(acc, i). A real GLSL `for` loop; `count` must be
    # an Integer at trace time, `i` is the round as a float (0, 1, ...).
    # Nests freely. This is the loop: Ruby's `while` cannot be traced.
    def iterate(init, count, &block)
      raise ArgumentError, "iterate requires a block" unless block
      unless count.is_a?(Integer)
        raise TypeError,
              "iterate needs an Integer count known when the block is traced, " \
              "got #{count.class}"
      end
      raise ArgumentError, "iterate count must be >= 0, got #{count}" if count < 0
      acc = Expr.new(:acc, [])
      idx = Expr.new(:idx, [])
      body = Expr.numeric(instance_exec(acc, idx, &block), "iterate body")
      Expr.new(:loop, [Expr.numeric(init, "iterate init"), count, acc, idx, body])
    end
  end

  class Kernel
    # Reductions: GLSL identity, GLSL combine step, and the code src/gpu_kernel.c
    # uses to combine the per-workgroup partials on the host.
    REDUCE = {
      sum: ["0.0",                          "sdata[tid] + sdata[tid + s]",   1],
      min: ["uintBitsToFloat(0x7F800000u)", "min(sdata[tid], sdata[tid + s])", 2],
      max: ["uintBitsToFloat(0xFF800000u)", "max(sdata[tid], sdata[tid + s])", 3]
    }.freeze

    # Set by GPU::Kernel.compile in C: the exact GLSL that was compiled.
    # Showing it is half the fun.
    def glsl
      @glsl
    end

    # 1 or 2 input arrays.
    def inputs
      @inputs
    end

    # nil for an element-wise kernel, else :sum / :min / :max.
    def reduce
      @reduce
    end

    def self.source_for(tree, inputs = 1, reduce = nil)
      reduce ? reduce_source(tree, inputs, reduce) : map_source(tree, inputs)
    end

    # Element-wise: one thread per element, out[idx] = expr.
    def self.map_source(tree, inputs)
      gen = Codegen.new(1)
      result = gen.expr(tree)
      prelude = gen.lines.map { |l| "#{l}\n" }.join
      loads = "    float x = a[idx];\n"
      loads += "    float y = b[idx];\n" if inputs == 2
      out = inputs == 2 ? "c" : "b"
      <<~GLSL
        #version 450

        layout(local_size_x = 256) in;

        #{bindings(inputs, out)}

        layout(push_constant) uniform PushConstants { uint n; } pc;

        // generated by GPU.kernel
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= pc.n) { return; }
        #{loads}#{prelude}    #{out}[idx] = #{result};
        }
      GLSL
    end

    # Reduction: expr per element, then a tree reduction in shared memory, one
    # partial per workgroup. The host combines the partials (gpu_kernel.c).
    # The bounds check is an `if`, not a `return`: every thread must reach
    # the barriers.
    def self.reduce_source(tree, inputs, reduce)
      identity, combine, _code = REDUCE[reduce]
      gen = Codegen.new(2)
      result = gen.expr(tree)
      prelude = gen.lines.map { |l| "#{l}\n" }.join
      loads = "        float x = a[idx];\n"
      loads += "        float y = b[idx];\n" if inputs == 2
      <<~GLSL
        #version 450

        layout(local_size_x = 256) in;

        #{bindings(inputs, "partial")}

        layout(push_constant) uniform PushConstants { uint n; } pc;

        shared float sdata[256];

        // generated by GPU.kernel (reduce: #{reduce})
        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint idx = gl_GlobalInvocationID.x;
            float v = #{identity};
            if (idx < pc.n) {
        #{loads}#{prelude}        v = #{result};
            }
            sdata[tid] = v;
            barrier();
            for (uint s = 128u; s > 0u; s >>= 1u) {
                if (tid < s) { sdata[tid] = #{combine}; }
                barrier();
            }
            if (tid == 0u) { partial[gl_WorkGroupID.x] = sdata[0]; }
        }
      GLSL
    end

    def self.bindings(inputs, out)
      lines = ["layout(set = 0, binding = 0) buffer BufA { float a[]; };"]
      lines << "layout(set = 0, binding = 1) buffer BufB { float b[]; };" if inputs == 2
      slot = inputs == 2 ? 2 : 1
      name = out == "partial" ? "BufOut" : "Buf#{out.upcase}"
      lines << "layout(set = 0, binding = #{slot}) buffer #{name} { float #{out}[]; };"
      lines.join("\n")
    end
  end

  # Run the block once over proxies and return the GLSL it traces to.
  # `inputs` is 1 (|x|) or 2 (|x, y|).
  def self.trace(inputs, reduce, &block)
    vars = [Expr.new(:var, ["x"])]
    vars << Expr.new(:var, ["y"]) if inputs == 2
    tree = Expr.numeric(KernelContext.new.instance_exec(*vars, &block), "a kernel body")
    Kernel.source_for(tree, inputs, reduce)
  end

  def self.kernel_inputs(arity, block)
    inputs = arity
    if inputs.nil?
      inputs = block.respond_to?(:arity) && block.arity > 1 ? 2 : 1
    end
    unless inputs == 1 || inputs == 2
      raise ArgumentError, "a kernel takes 1 or 2 input arrays, got arity: #{inputs.inspect}"
    end
    inputs
  end

  def self.check_reduce(reduce)
    return if reduce.nil? || Kernel::REDUCE.key?(reduce)
    raise ArgumentError,
          "unknown reduction #{reduce.inspect}; use :sum, :min or :max"
  end

  # GPU.kernel { |x| ... }                    -> GPU::Kernel, element-wise
  # GPU.kernel { |x, y| ... }                 -> two inputs (from the block's arity)
  # GPU.kernel(arity: 2) { ... }              -> two inputs, stated explicitly
  # GPU.kernel(reduce: :sum) { |x| ... }      -> Kernel#call returns a Float
  #
  # Traces the block and returns a reusable, compiled kernel. Identical source
  # is compiled once and cached, so calling this in a loop is cheap.
  def self.kernel(arity: nil, reduce: nil, &block)
    raise ArgumentError, "GPU.kernel requires a block" unless block
    inputs = kernel_inputs(arity, block)
    check_reduce(reduce)
    source = trace(inputs, reduce, &block)
    @kernel_cache ||= {}
    @kernel_cache[source] ||= begin
      # mruby-gpu-narray brings Vulkan up lazily on its first operation and
      # keeps that entry point private to its C code, so reach it through a
      # public call before compiling. Only runs on a cache miss.
      GPU.info
      k = Kernel.compile(source, inputs, reduce ? Kernel::REDUCE[reduce][2] : 0)
      k.instance_variable_set(:@inputs, inputs)
      k.instance_variable_set(:@reduce, reduce)
      k
    end
  end

  # Trace the block into GLSL without compiling it -- for tests and for showing
  # the generated shader on a slide.
  def self.kernel_source(arity: nil, reduce: nil, &block)
    raise ArgumentError, "GPU.kernel_source requires a block" unless block
    check_reduce(reduce)
    trace(kernel_inputs(arity, block), reduce, &block)
  end

  class NArray
    # na.map { |x| x * 2 + sin(x) } -- the block becomes one GPU kernel, so the
    # whole expression is a single dispatch no matter how long it is.
    def map(&block)
      GPU.kernel(arity: 1, &block).call(self)
    end

    # na.map2(nb) { |x, y| x * y + 1 } -- two inputs, one dispatch.
    def map2(other, &block)
      GPU.kernel(arity: 2, &block).call(self, other)
    end

    # na.sum { |x| x * 2 + 1 } -- expression and reduction in one dispatch.
    # Without a block this is the C implementation from mruby-gpu-narray.
    alias sum_without_block sum
    def sum(&block)
      block ? GPU.kernel(arity: 1, reduce: :sum, &block).call(self) : sum_without_block
    end

    # na.dot(nb) -- inner product, one dispatch. The block captures nothing,
    # so the kernel is traced once and kept: tracing a block and looking its
    # source up in the cache costs a few hundred microseconds of Ruby time,
    # which matters in a loop. (a.sum { ... } traces on every call, because
    # its block may capture a local that changes.)
    def dot(other)
      GPU::NArray.dot_kernel.call(self, other)
    end

    def self.dot_kernel
      @dot_kernel ||= GPU.kernel(arity: 2, reduce: :sum) { |x, y| x * y }
    end
  end
end
